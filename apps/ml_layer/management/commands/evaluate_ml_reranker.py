import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.ml_layer.training.build_dataset import build_ml_training_dataset
from apps.ml_layer.training.splitters import make_stratified_participant_level_split
from apps.ml_layer.training.evaluate import evaluate_reranker_predictions
from apps.ml_layer.services.ml_reranking.service import run_ml_reranking_layer


class Command(BaseCommand):
    help = "Evaluate rules-only vs ML-only vs hybrid reranking on a held-out participant-level test split."

    def add_arguments(self, parser):
        parser.add_argument(
            "--safety-context",
            type=str,
            default="apps/recs_csl/output/safety_context.jsonl",
        )
        parser.add_argument(
            "--ranked",
            type=str,
            default="apps/ranking_layer/output/ranked_recommendations.jsonl",
        )
        parser.add_argument(
            "--participant-features",
            type=str,
            default="nhanes_step1_out/NHANES_Master_engineered.parquet",
        )
        parser.add_argument(
            "--top-k-positive",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
        )
        parser.add_argument(
            "--out-json",
            type=str,
            default="apps/ml_layer/output/evaluation_metrics.json",
        )
        parser.add_argument(
            "--out-csv",
            type=str,
            default="apps/ml_layer/output/evaluation_per_participant.csv",
        )
        parser.add_argument(
            "--label-scheme",
            type=str,
            default="rule_top_k",
            choices=[
                "rule_top_k",
                "rule_top_k_with_min_score",
                "rule_top_k_plus_bundle_retention",
            ],
        )
        parser.add_argument(
            "--min-score-norm",
            type=float,
            default=0.80,
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)

        safety_path = base_dir / options["safety_context"]
        ranked_path = base_dir / options["ranked"]
        participant_features_path = base_dir / options["participant_features"]
        out_json_path = base_dir / options["out_json"]
        out_csv_path = base_dir / options["out_csv"]

        self.stdout.write("Loading evaluation inputs...")
        safety_df = pd.read_json(safety_path, lines=True)
        ranked_df = pd.read_json(ranked_path, lines=True)
        participant_features_df = pd.read_parquet(participant_features_path)

        self.stdout.write("Rebuilding supervised dataset for held-out split...")
        supervised_df = build_ml_training_dataset(
            safety_context_df=safety_df,
            ranked_df=ranked_df,
            participant_features_df=participant_features_df,
            top_k_positive=int(options["top_k_positive"]),
            exclude_triage=True,
            label_scheme=options["label_scheme"],
            min_score_norm=float(options["min_score_norm"]),
        )

        self.stdout.write("Creating participant-level stratified split...")
        split = make_stratified_participant_level_split(
            df=supervised_df,
            stratify_col="feature_tier",
            participant_col="ParticipantKey",
            seed=int(options["seed"]),
        )

        test_participants = set(split.test_keys)

        safety_test_df = safety_df.loc[safety_df["ParticipantKey"].astype(str).isin(test_participants)].copy()
        ranked_test_df = ranked_df.loc[ranked_df["ParticipantKey"].astype(str).isin(test_participants)].copy()
        participant_features_test_df = participant_features_df.loc[
            participant_features_df["ParticipantKey"].astype(str).isin(test_participants)
        ].copy()

        self.stdout.write("Running live ML reranking on held-out test split...")
        ml_out = run_ml_reranking_layer(
            safety_context_df=safety_test_df,
            ranked_df=ranked_test_df,
            participant_features_df=participant_features_test_df,
        )
        reranked_test_df = ml_out["reranked_recommendations"].copy()

        self.stdout.write("Joining labels onto reranked test output...")
        labels_test_df = split.test_df[["ParticipantKey", "action_code", "label", "feature_tier"]].copy()
        if "Severity_band" in split.test_df.columns:
            labels_test_df["Severity_band"] = split.test_df["Severity_band"]

        eval_df = reranked_test_df.merge(
            labels_test_df.drop_duplicates(subset=["ParticipantKey", "action_code"]),
            on=["ParticipantKey", "action_code"],
            how="left",
            validate="1:1",
        )

        missing_labels = int(eval_df["label"].isna().sum())
        if missing_labels > 0:
            self.stdout.write(
                self.style.WARNING(
                    f"{missing_labels} rows missing labels after merge; filling with 0 for evaluation."
                )
            )
            eval_df["label"] = eval_df["label"].fillna(0)

        self.stdout.write("Computing evaluation metrics...")
        bundle = evaluate_reranker_predictions(
            df=eval_df,
            participant_col="ParticipantKey",
            label_col="label",
            rule_score_col="total_score",
            ml_score_col="ml_score",
            final_score_col="final_score",
            ks=(1, 3, 5),
        )

        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        model_version = "unknown"
        if "ml_model_version" in reranked_test_df.columns:
            non_null = reranked_test_df["ml_model_version"].dropna().astype(str)
            if not non_null.empty:
                model_version = non_null.iloc[0]

        schema_version = "unknown"
        if "ml_schema_version" in reranked_test_df.columns:
            non_null = reranked_test_df["ml_schema_version"].dropna().astype(str)
            if not non_null.empty:
                schema_version = non_null.iloc[0]

        calibration_version = "unknown"
        if "ml_calibration_version" in reranked_test_df.columns:
            non_null = reranked_test_df["ml_calibration_version"].dropna().astype(str)
            if not non_null.empty:
                calibration_version = non_null.iloc[0]

        payload = {
            "overall": bundle.overall,
            "subgroup_results": bundle.subgroup_results,
            "ml_runtime_manifest": ml_out.get("manifest", {}),
            "model_metadata": {
                "model_version": model_version,
                "schema_version": schema_version,
                "calibration_version": calibration_version,
            },
            "evaluation_config": {
                "top_k_positive": int(options["top_k_positive"]),
                "label_scheme": options["label_scheme"],
                "min_score_norm": float(options["min_score_norm"]),
                "seed": int(options["seed"]),
            },
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        bundle.per_participant_metrics.to_csv(out_csv_path, index=False)

        self.stdout.write(self.style.SUCCESS("ML reranker evaluation complete."))
        self.stdout.write(f"Saved summary metrics to: {out_json_path}")
        self.stdout.write(f"Saved per-participant metrics to: {out_csv_path}")

        self.stdout.write("\nOverall metrics:")
        for k, v in payload["overall"].items():
            self.stdout.write(f"  - {k}: {v}")
