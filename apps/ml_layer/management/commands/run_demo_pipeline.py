import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.recs_csl.services.cold_start.service import run_cold_start_layer
from apps.ranking_layer.services.ranking.service import run_ranking_layer
from apps.ml_layer.services.ml_reranking.service import run_ml_reranking_layer
from apps.bundling_layer.services.bundling.service import run_bundling_layer
from apps.explainability_layer.services.explainability.service import run_explainability_layer


class Command(BaseCommand):
    help = "Run a fast demo/test version of the hybrid pipeline on one participant or a small subset."

    def add_arguments(self, parser):
        parser.add_argument(
            "--participants",
            nargs="*",
            default=[],
            help="Specific ParticipantKey values to run.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="If no participant keys are given, run only the first N participants.",
        )
        parser.add_argument(
            "--participant-features",
            type=str,
            default="nhanes_step1_out/NHANES_Master_engineered.parquet",
        )
        parser.add_argument(
            "--safe-candidates",
            type=str,
            default="rules_safety_engine/output/safe_recommendation_candidates.jsonl",
        )
        parser.add_argument(
            "--safety-context",
            type=str,
            default="apps/recs_csl/output/safety_context.jsonl",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="apps/ml_layer/output/demo_pipeline",
        )
        parser.add_argument(
            "--export-case-study",
            action="store_true",
            help="Also export case study report for the first selected participant.",
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)

        participant_features_path = base_dir / options["participant_features"]
        safe_candidates_path = base_dir / options["safe_candidates"]
        safety_context_path = base_dir / options["safety_context"]
        output_dir = base_dir / options["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Loading demo inputs...")
        participant_features_df = pd.read_parquet(participant_features_path)
        safe_candidates_df = pd.read_json(safe_candidates_path, lines=True)
        safety_context_df = pd.read_json(safety_context_path, lines=True)

        requested_participants = [str(x) for x in options["participants"] if str(x).strip()]

        if requested_participants:
            selected_keys = requested_participants
        else:
            selected_keys = (
                safety_context_df["ParticipantKey"]
                .astype(str)
                .drop_duplicates()
                .head(int(options["limit"]))
                .tolist()
            )

        self.stdout.write(f"Selected participants: {len(selected_keys)}")

        safety_context_demo = safety_context_df.loc[
            safety_context_df["ParticipantKey"].astype(str).isin(selected_keys)
        ].copy()

        participant_features_demo = participant_features_df.loc[
            participant_features_df["ParticipantKey"].astype(str).isin(selected_keys)
        ].copy()

        safe_candidates_demo = safe_candidates_df.loc[
            safe_candidates_df["ParticipantKey"].astype(str).isin(selected_keys)
        ].copy()

        self.stdout.write("Running cold-start layer...")
        cold_result = run_cold_start_layer(
            safety_context_df=safety_context_demo,
            safe_candidates_df=safe_candidates_demo,
        )
        gated_df = cold_result["gated_candidates"]

        self.stdout.write("Running ranking layer...")
        ranking_result = run_ranking_layer(
            safety_context_df=safety_context_demo,
            gated_candidates_df=gated_df,
        )
        ranked_df = ranking_result["ranked_recommendations"]

        self.stdout.write("Running ML reranking layer...")
        ml_result = run_ml_reranking_layer(
            safety_context_df=safety_context_demo,
            ranked_df=ranked_df,
            participant_features_df=participant_features_demo,
        )
        reranked_df = ml_result["reranked_recommendations"]

        self.stdout.write("Running bundling layer...")
        bundling_result = run_bundling_layer(
            safety_context_df=safety_context_demo,
            ranked_df=reranked_df,
        )
        bundles_df = bundling_result["final_recommendations"]

        self.stdout.write("Running explainability layer...")
        explain_result = run_explainability_layer(
            safety_context_df=safety_context_demo,
            final_recommendations_df=bundles_df,
            reranked_df=reranked_df,
            participant_features_df=participant_features_demo,
        )
        explained_df = explain_result["explained_recommendations"]

        self.stdout.write("Saving demo outputs...")
        safety_context_demo.to_json(output_dir / "safety_context_demo.jsonl", orient="records", lines=True, force_ascii=False)
        gated_df.to_json(output_dir / "gated_candidates_demo.jsonl", orient="records", lines=True, force_ascii=False)
        ranked_df.to_json(output_dir / "ranked_before_ml_demo.jsonl", orient="records", lines=True, force_ascii=False)
        reranked_df.to_json(output_dir / "reranked_recommendations_demo.jsonl", orient="records", lines=True, force_ascii=False)
        bundles_df.to_json(output_dir / "final_recommendations_after_ml_demo.jsonl", orient="records", lines=True, force_ascii=False)
        explained_df.to_json(output_dir / "explained_recommendations_demo.jsonl", orient="records", lines=True, force_ascii=False)

        manifest = {
            "selected_participants": selected_keys,
            "n_participants": len(selected_keys),
            "rows": {
                "safety_context": int(len(safety_context_demo)),
                "gated_candidates": int(len(gated_df)),
                "ranked": int(len(ranked_df)),
                "reranked": int(len(reranked_df)),
                "bundles": int(len(bundles_df)),
                "explained": int(len(explained_df)),
            },
            "layer_manifests": {
                "cold_start": cold_result.get("manifest", {}),
                "ranking": ranking_result.get("manifest", {}),
                "ml_reranking": ml_result.get("manifest", {}),
                "bundling": bundling_result.get("manifest", {}),
                "explainability": explain_result.get("manifest", {}),
            },
        }

        with open(output_dir / "demo_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS("Demo pipeline complete."))
        self.stdout.write(f"Participants run: {len(selected_keys)}")
        self.stdout.write(f"Saved to: {output_dir}")

        if options["export_case_study"] and selected_keys:
            self.stdout.write(
                self.style.WARNING(
                    f"Now run: python manage.py export_case_study_report --participant-key {selected_keys[0]}"
                )
            )
