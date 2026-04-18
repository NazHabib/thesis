import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Export thesis-ready summary tables from reranking, bundling, evaluation, and selected SHAP reports."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reranked",
            type=str,
            default="apps/ml_layer/output/reranked_recommendations.jsonl",
        )
        parser.add_argument(
            "--bundles",
            type=str,
            default="apps/ml_layer/output/final_recommendations_after_ml.jsonl",
        )
        parser.add_argument(
            "--evaluation-json",
            type=str,
            default="apps/ml_layer/output/evaluation_metrics.json",
        )
        parser.add_argument(
            "--shap-dir",
            type=str,
            default="apps/explainability_layer/output/shap",
        )
        parser.add_argument(
            "--participants",
            nargs="*",
            default=[],
            help="Participant keys to include from SHAP reports.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="apps/ml_layer/output/thesis_tables",
        )
        parser.add_argument(
            "--manifest-paths",
            nargs="*",
            default=[],
            help="Optional list of manifest JSON paths to include in experiment comparison export.",
        )        
        

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)
        out_dir = base_dir / options["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        
        

        reranked_path = base_dir / options["reranked"]
        bundles_path = base_dir / options["bundles"]
        evaluation_json_path = base_dir / options["evaluation_json"]
        shap_dir = base_dir / options["shap_dir"]
        
        manifest_paths = [
            (base_dir / p) if not Path(p).is_absolute() else Path(p)
            for p in options["manifest_paths"]
        ]

        self.stdout.write("Loading reranked recommendations...")
        reranked_df = pd.read_json(reranked_path, lines=True)

        self.stdout.write("Loading bundled recommendations...")
        bundles_df = pd.read_json(bundles_path, lines=True)
        
        followup_window_distribution_df = self._build_followup_window_distribution_df(bundles_df)

        evaluation_metrics = {}
        if evaluation_json_path.exists():
            self.stdout.write("Loading evaluation metrics...")
            with open(evaluation_json_path, "r", encoding="utf-8") as f:
                evaluation_metrics = json.load(f)

        self.stdout.write("Building summary tables...")
        summary_metrics_df = self._build_summary_metrics_df(
            reranked_df=reranked_df,
            bundles_df=bundles_df,
            evaluation_metrics=evaluation_metrics,
        )
        fallback_counts_df = self._build_fallback_counts_df(reranked_df)
        lambda_tier_counts_df = self._build_lambda_tier_counts_df(reranked_df)
        confidence_distribution_df = self._build_confidence_distribution_df(bundles_df)
        bundle_family_distribution_df = self._build_bundle_family_distribution_df(bundles_df)
        ranking_impact_metrics_df = self._build_ranking_impact_metrics_df(evaluation_metrics)
        followup_urgency_distribution_df = self._build_followup_urgency_distribution_df(bundles_df)
        temporal_risk_tier_distribution_df = self._build_temporal_risk_tier_distribution_df(bundles_df)
        followup_horizon_summary_df = self._build_followup_horizon_summary_df(bundles_df)
        shap_top_features_df = self._build_shap_top_features_case_studies_df(
            shap_dir=shap_dir,
            participant_keys=options["participants"],
        )
        experiment_comparison_df = self._build_experiment_comparison_df(
            evaluation_metrics=evaluation_metrics,
            manifest_paths=manifest_paths,
        )

        summary_metrics_df.to_csv(out_dir / "summary_metrics.csv", index=False)
        ranking_impact_metrics_df.to_csv(out_dir / "ranking_impact_metrics.csv", index=False)
        fallback_counts_df.to_csv(out_dir / "fallback_counts.csv", index=False)
        lambda_tier_counts_df.to_csv(out_dir / "lambda_tier_counts.csv", index=False)
        confidence_distribution_df.to_csv(out_dir / "confidence_distribution.csv", index=False)
        bundle_family_distribution_df.to_csv(out_dir / "bundle_family_distribution.csv", index=False)
        followup_urgency_distribution_df.to_csv(out_dir / "followup_urgency_distribution.csv", index=False)
        temporal_risk_tier_distribution_df.to_csv(out_dir / "temporal_risk_tier_distribution.csv", index=False)
        followup_horizon_summary_df.to_csv(out_dir / "followup_horizon_summary.csv", index=False)
        shap_top_features_df.to_csv(out_dir / "shap_top_features_case_studies.csv", index=False)
        experiment_comparison_df.to_csv(out_dir / "experiment_comparison.csv", index=False)
        followup_window_distribution_df.to_csv(out_dir / "followup_window_distribution.csv", index=False)

        self.stdout.write(self.style.SUCCESS("Thesis tables exported."))
        self.stdout.write(f"Saved to: {out_dir}")

    @staticmethod
    def _build_summary_metrics_df(
        reranked_df: pd.DataFrame,
        bundles_df: pd.DataFrame,
        evaluation_metrics: dict,
    ) -> pd.DataFrame:
        total_rows = len(reranked_df)
        ml_reranked_rows = int((reranked_df["rerank_status"] == "ml_reranked").sum())
        fallback_rows = int((reranked_df["rerank_status"] == "rules_only_fallback").sum())

        rows = [
            {"metric": "reranked_rows_total", "value": total_rows},
            {"metric": "ml_reranked_rows", "value": ml_reranked_rows},
            {"metric": "rules_only_fallback_rows", "value": fallback_rows},
            {
                "metric": "ml_reranked_pct",
                "value": round((ml_reranked_rows / total_rows) * 100, 4) if total_rows else 0.0,
            },
            {
                "metric": "rules_only_fallback_pct",
                "value": round((fallback_rows / total_rows) * 100, 4) if total_rows else 0.0,
            },
            {
                "metric": "mean_ml_contrib",
                "value": round(float(reranked_df["ml_contrib"].mean()), 6)
                if "ml_contrib" in reranked_df.columns else 0.0,
            },
            {
                "metric": "median_ml_contrib",
                "value": round(float(reranked_df["ml_contrib"].median()), 6)
                if "ml_contrib" in reranked_df.columns else 0.0,
            },
            {
                "metric": "mean_lambda_u",
                "value": round(float(reranked_df["lambda_u"].mean()), 6)
                if "lambda_u" in reranked_df.columns else 0.0,
            },
            {"metric": "final_bundle_rows", "value": len(bundles_df)},
        ]

        overall = evaluation_metrics.get("overall", {}) if isinstance(evaluation_metrics, dict) else {}
        for key, value in overall.items():
            if isinstance(value, (int, float)):
                rows.append({"metric": key, "value": value})

        return pd.DataFrame(rows)

    @staticmethod
    def _build_fallback_counts_df(reranked_df: pd.DataFrame) -> pd.DataFrame:
        if "fallback_reason" not in reranked_df.columns:
            return pd.DataFrame(columns=["fallback_reason", "count", "pct"])

        vc = reranked_df["fallback_reason"].fillna("none").value_counts(dropna=False)
        total = len(reranked_df)
        return pd.DataFrame({
            "fallback_reason": vc.index.astype(str),
            "count": vc.values,
            "pct": [round((v / total) * 100, 4) if total else 0.0 for v in vc.values],
        })

    @staticmethod
    def _build_lambda_tier_counts_df(reranked_df: pd.DataFrame) -> pd.DataFrame:
        if "lambda_u" not in reranked_df.columns:
            return pd.DataFrame(columns=["lambda_u", "count", "pct"])

        vc = reranked_df["lambda_u"].fillna(-1).value_counts(dropna=False).sort_index()
        total = len(reranked_df)
        return pd.DataFrame({
            "lambda_u": vc.index,
            "count": vc.values,
            "pct": [round((v / total) * 100, 4) if total else 0.0 for v in vc.values],
        })

    @staticmethod
    def _build_confidence_distribution_df(bundles_df: pd.DataFrame) -> pd.DataFrame:
        if "confidence_label" not in bundles_df.columns:
            return pd.DataFrame(columns=["confidence_label", "count", "pct"])

        vc = bundles_df["confidence_label"].fillna("unknown").value_counts(dropna=False)
        total = len(bundles_df)
        return pd.DataFrame({
            "confidence_label": vc.index.astype(str),
            "count": vc.values,
            "pct": [round((v / total) * 100, 4) if total else 0.0 for v in vc.values],
        })

    @staticmethod
    def _build_bundle_family_distribution_df(bundles_df: pd.DataFrame) -> pd.DataFrame:
        if "bundle_family" not in bundles_df.columns:
            return pd.DataFrame(columns=["bundle_family", "count", "pct"])

        vc = bundles_df["bundle_family"].fillna("unknown").value_counts(dropna=False)
        total = len(bundles_df)
        return pd.DataFrame({
            "bundle_family": vc.index.astype(str),
            "count": vc.values,
            "pct": [round((v / total) * 100, 4) if total else 0.0 for v in vc.values],
        })

    @staticmethod
    def _build_followup_urgency_distribution_df(bundles_df: pd.DataFrame) -> pd.DataFrame:
        if "followup_urgency" not in bundles_df.columns:
            return pd.DataFrame(columns=["followup_urgency", "count", "pct"])

        vc = bundles_df["followup_urgency"].fillna("unknown").value_counts(dropna=False)
        total = len(bundles_df)
        return pd.DataFrame({
            "followup_urgency": vc.index.astype(str),
            "count": vc.values,
            "pct": [round((v / total) * 100, 4) if total else 0.0 for v in vc.values],
        })

    @staticmethod
    def _build_temporal_risk_tier_distribution_df(bundles_df: pd.DataFrame) -> pd.DataFrame:
        if "temporal_risk_tier" not in bundles_df.columns:
            return pd.DataFrame(columns=["temporal_risk_tier", "count", "pct"])

        vc = bundles_df["temporal_risk_tier"].fillna("unknown").value_counts(dropna=False)
        total = len(bundles_df)
        return pd.DataFrame({
            "temporal_risk_tier": vc.index.astype(str),
            "count": vc.values,
            "pct": [round((v / total) * 100, 4) if total else 0.0 for v in vc.values],
        })

    @staticmethod
    def _build_followup_horizon_summary_df(bundles_df: pd.DataFrame) -> pd.DataFrame:
        if "followup_horizon_days" not in bundles_df.columns:
            return pd.DataFrame(columns=["metric", "value"])

        s = pd.to_numeric(bundles_df["followup_horizon_days"], errors="coerce").dropna()
        if s.empty:
            return pd.DataFrame(columns=["metric", "value"])

        return pd.DataFrame([
            {"metric": "min_followup_horizon_days", "value": float(s.min())},
            {"metric": "median_followup_horizon_days", "value": float(s.median())},
            {"metric": "mean_followup_horizon_days", "value": round(float(s.mean()), 4)},
            {"metric": "max_followup_horizon_days", "value": float(s.max())},
        ])

    @staticmethod
    def _build_shap_top_features_case_studies_df(shap_dir: Path, participant_keys: list[str]) -> pd.DataFrame:
        rows = []

        if not shap_dir.exists():
            return pd.DataFrame(columns=[
                "participant_key", "scope", "rank", "feature", "feature_label", "feature_group", "mean_abs_shap"
            ])

        if participant_keys:
            json_paths = [shap_dir / f"{pk.replace('/', '_')}_shap_report.json" for pk in participant_keys]
        else:
            json_paths = sorted(shap_dir.glob("*_shap_report.json"))

        for path in json_paths:
            if not path.exists():
                continue

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            participant_key = data.get("participant_key", path.stem.replace("_shap_report", ""))

            for scope_name, key in [
                ("all", "global_feature_importance"),
                ("clinical_only", "global_feature_importance_clinical_only"),
            ]:
                features = data.get(key, [])[:10]
                for rank, item in enumerate(features, start=1):
                    rows.append({
                        "participant_key": participant_key,
                        "scope": scope_name,
                        "rank": rank,
                        "feature": item.get("feature"),
                        "feature_label": item.get("feature_label"),
                        "feature_group": item.get("feature_group"),
                        "mean_abs_shap": item.get("mean_abs_shap"),
                    })

        return pd.DataFrame(rows)

    @staticmethod
    def _build_ranking_impact_metrics_df(evaluation_metrics: dict) -> pd.DataFrame:
        overall = evaluation_metrics.get("overall", {}) if isinstance(evaluation_metrics, dict) else {}

        keys = [
            "mean_top1_changed",
            "mean_top3_changed",
            "mean_any_rank_changed",
            "mean_mean_absolute_rank_shift",
            "mean_median_absolute_rank_shift",
            "mean_max_absolute_rank_shift",
            "mean_mean_signed_rank_shift",
            "mean_promoted_count",
            "mean_demoted_count",
            "mean_unchanged_count",
        ]

        rows = []
        for key in keys:
            if key in overall:
                rows.append({"metric": key, "value": overall[key]})

        return pd.DataFrame(rows)

    @staticmethod
    def _build_experiment_comparison_df(
        evaluation_metrics: dict,
        manifest_paths: list[Path] | None = None,
    ) -> pd.DataFrame:
        manifest_paths = manifest_paths or []
        rows = []

        # Manifest-driven multi-experiment comparison
        for manifest_path in manifest_paths:
            if not manifest_path.exists():
                continue

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            training = manifest.get("training", {})
            metrics = manifest.get("metrics", {})

            rows.append({
                "experiment_name": manifest.get("model_version", "unknown"),
                "model_version": manifest.get("model_version", "unknown"),
                "feature_profile": training.get("feature_profile"),
                "schema_version": manifest.get("feature_schema_version", "unknown"),
                "calibration_version": manifest.get("calibration_version", "unknown"),
                "calibration_method": manifest.get("calibration_method"),
                "top_k_positive": training.get("top_k_positive"),
                "label_scheme": training.get("label_scheme"),
                "min_score_norm": training.get("min_score_norm"),
                "temporal_feature_count": training.get("temporal_feature_count", 0),
                "train_rows": training.get("train_rows"),
                "val_rows": training.get("val_rows"),
                "test_rows": training.get("test_rows"),
                "val_log_loss": metrics.get("val_log_loss"),
                "val_roc_auc": metrics.get("val_roc_auc"),
                "val_brier_raw": metrics.get("val_brier_raw"),
                "val_brier_calibrated": metrics.get("val_brier_calibrated"),
                "val_log_loss_calibrated": metrics.get("val_log_loss_calibrated"),
            })

        if rows:
            df = pd.DataFrame(rows)

            profile_order = {
                "experiment_c": 1,
                "no_temporal": 2,
                "temporal_only": 3,
                "temporal_full": 4,
            }
            if "feature_profile" in df.columns:
                df["profile_order"] = df["feature_profile"].map(profile_order).fillna(99)
                df = df.sort_values(
                    by=["profile_order", "model_version"],
                    ascending=[True, True],
                    kind="stable",
                ).drop(columns=["profile_order"])

            return df.reset_index(drop=True)

        # Fallback to legacy single-active-experiment evaluation view
        overall = evaluation_metrics.get("overall", {}) if isinstance(evaluation_metrics, dict) else {}
        eval_cfg = evaluation_metrics.get("evaluation_config", {}) if isinstance(evaluation_metrics, dict) else {}
        model_meta = evaluation_metrics.get("model_metadata", {}) if isinstance(evaluation_metrics, dict) else {}

        rows = [{
            "experiment_name": "current_active_experiment",
            "model_version": model_meta.get("model_version", "unknown"),
            "feature_profile": None,
            "schema_version": model_meta.get("schema_version", "unknown"),
            "calibration_version": model_meta.get("calibration_version", "unknown"),
            "calibration_method": None,
            "top_k_positive": eval_cfg.get("top_k_positive"),
            "label_scheme": eval_cfg.get("label_scheme"),
            "min_score_norm": eval_cfg.get("min_score_norm"),
            "temporal_feature_count": None,
            "train_rows": None,
            "val_rows": None,
            "test_rows": None,
            "val_log_loss": None,
            "val_roc_auc": None,
            "val_brier_raw": None,
            "val_brier_calibrated": None,
            "val_log_loss_calibrated": None,
            "participants_evaluated": overall.get("participants_evaluated"),
            "mean_hybrid_precision@1": overall.get("mean_hybrid_precision@1"),
            "mean_hybrid_recall@1": overall.get("mean_hybrid_recall@1"),
            "mean_hybrid_ndcg@1": overall.get("mean_hybrid_ndcg@1"),
            "mean_hybrid_precision@3": overall.get("mean_hybrid_precision@3"),
            "mean_hybrid_recall@3": overall.get("mean_hybrid_recall@3"),
            "mean_hybrid_ndcg@3": overall.get("mean_hybrid_ndcg@3"),
            "mean_hybrid_precision@5": overall.get("mean_hybrid_precision@5"),
            "mean_hybrid_recall@5": overall.get("mean_hybrid_recall@5"),
            "mean_hybrid_ndcg@5": overall.get("mean_hybrid_ndcg@5"),
            "mean_brier_ml": overall.get("mean_brier_ml"),
            "mean_top1_changed": overall.get("mean_top1_changed"),
            "mean_top3_changed": overall.get("mean_top3_changed"),
            "mean_any_rank_changed": overall.get("mean_any_rank_changed"),
            "mean_mean_absolute_rank_shift": overall.get("mean_mean_absolute_rank_shift"),
            "mean_promoted_count": overall.get("mean_promoted_count"),
            "mean_demoted_count": overall.get("mean_demoted_count"),
        }]

        return pd.DataFrame(rows)
        
        
    @staticmethod
    def _build_followup_window_distribution_df(bundles_df: pd.DataFrame) -> pd.DataFrame:
        if "followup_window_label" not in bundles_df.columns:
            return pd.DataFrame(columns=["followup_window_label", "count", "pct"])

        vc = bundles_df["followup_window_label"].fillna("unknown").value_counts(dropna=False)
        total = len(bundles_df)
        return pd.DataFrame({
            "followup_window_label": vc.index.astype(str),
            "count": vc.values,
            "pct": [round((v / total) * 100, 4) if total else 0.0 for v in vc.values],
        })
