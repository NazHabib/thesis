from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Export thesis-ready summary tables from reranking, bundling, evaluation, explainability, and pipeline manifests."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reranked",
            type=str,
            default="apps/ml_layer/output/reranked_recommendations.jsonl",
        )
        parser.add_argument(
            "--manifests-dir",
            type=str,
            default="apps/ml_layer/artifacts/manifests",
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
        parser.add_argument(
            "--manifest-path",
            type=str,
            default="apps/ml_layer/output/full_pipeline_manifest.json",
        )
        parser.add_argument(
            "--gated",
            type=str,
            default="apps/ml_layer/output/gated_candidates_after_coldstart.jsonl",
        )
        parser.add_argument(
            "--ranked",
            type=str,
            default="apps/ml_layer/output/ranked_before_ml.jsonl",
        )
        parser.add_argument(
            "--explained",
            type=str,
            default="apps/ml_layer/output/explained_recommendations.jsonl",
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)
        out_dir = base_dir / options["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir = base_dir / options["manifests_dir"]

        reranked_path = base_dir / options["reranked"]
        bundles_path = base_dir / options["bundles"]
        evaluation_json_path = base_dir / options["evaluation_json"]
        shap_dir = base_dir / options["shap_dir"]
        manifest_path = base_dir / options["manifest_path"]
        gated_path = base_dir / options["gated"]
        ranked_path = base_dir / options["ranked"]
        explained_path = base_dir / options["explained"]

        manifest_paths = [
            (base_dir / p) if not Path(p).is_absolute() else Path(p)
            for p in options["manifest_paths"]
        ]
        
        ablation_df = self._build_ablation_summary_df(manifest_paths)
        ablation_df.to_csv(out_dir / "ablation_summary.csv", index=False)

        manifest = {}
        if manifest_path.exists():
            self.stdout.write("Loading full pipeline manifest...")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        self.stdout.write("Loading gated candidates...")
        gated_df = pd.read_json(gated_path, lines=True)

        self.stdout.write("Loading ranked candidates...")
        ranked_df = pd.read_json(ranked_path, lines=True)

        self.stdout.write("Loading reranked recommendations...")
        reranked_df = pd.read_json(reranked_path, lines=True)

        self.stdout.write("Loading bundled recommendations...")
        bundles_df = pd.read_json(bundles_path, lines=True)

        self.stdout.write("Loading explained recommendations...")
        explained_df = pd.read_json(explained_path, lines=True)

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
        fusion_strategy_distribution_df = self._build_fusion_strategy_distribution_df(reranked_df)
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
            manifests_dir=manifests_dir,
            evaluation_metrics=evaluation_metrics,
        )

        pipeline_stage_counts_df = self._build_pipeline_stage_counts_df(
            manifest=manifest,
            gated_df=gated_df,
            ranked_df=ranked_df,
            reranked_df=reranked_df,
            bundles_df=bundles_df,
            explained_df=explained_df,
        )

        cold_start_summary_df = self._build_manifest_section_df(
            section_name="cold_start",
            payload=manifest.get("cold_start", {}),
        )

        ranking_manifest_summary_df = self._build_manifest_section_df(
            section_name="ranking",
            payload=manifest.get("ranking", {}),
        )

        ml_reranking_manifest_summary_df = self._build_manifest_section_df(
            section_name="ml_reranking",
            payload=manifest.get("ml_reranking", {}),
        )

        bundling_manifest_summary_df = self._build_manifest_section_df(
            section_name="bundling",
            payload=manifest.get("bundling", {}),
        )

        explainability_manifest_summary_df = self._build_manifest_section_df(
            section_name="explainability",
            payload=manifest.get("explainability", {}),
        )

        xai_status_distribution_df = self._build_xai_status_distribution_df(explained_df)

        summary_metrics_df.to_csv(out_dir / "summary_metrics.csv", index=False)
        fusion_strategy_distribution_df.to_csv(
            out_dir / "fusion_strategy_distribution.csv",
            index=False
        )
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

        pipeline_stage_counts_df.to_csv(out_dir / "pipeline_stage_counts.csv", index=False)
        cold_start_summary_df.to_csv(out_dir / "cold_start_summary.csv", index=False)
        ranking_manifest_summary_df.to_csv(out_dir / "ranking_manifest_summary.csv", index=False)
        ml_reranking_manifest_summary_df.to_csv(out_dir / "ml_reranking_manifest_summary.csv", index=False)
        bundling_manifest_summary_df.to_csv(out_dir / "bundling_manifest_summary.csv", index=False)
        explainability_manifest_summary_df.to_csv(out_dir / "explainability_manifest_summary.csv", index=False)
        xai_status_distribution_df.to_csv(out_dir / "xai_status_distribution.csv", index=False)

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
    def _build_fusion_strategy_distribution_df(reranked_df: pd.DataFrame) -> pd.DataFrame:
        if "fusion_strategy" not in reranked_df.columns:
            return pd.DataFrame(columns=["fusion_strategy", "count", "pct"])

        vc = reranked_df["fusion_strategy"].fillna("unknown").value_counts(dropna=False)
        total = len(reranked_df)

        return pd.DataFrame({
            "fusion_strategy": vc.index.astype(str),
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
        manifests_dir: Path,
        evaluation_metrics: dict,
    ) -> pd.DataFrame:
        rows = []

        if manifests_dir.exists():
            for path in sorted(manifests_dir.glob("manifest_*.json")):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except Exception:
                    continue

                training = manifest.get("training", {})
                metrics = manifest.get("metrics", {})

                rows.append({
                    "experiment_name": manifest.get("model_version", path.stem.replace("manifest_", "")),
                    "model_version": manifest.get("model_version", "unknown"),
                    "patient_profile_feature_count": training.get("patient_profile_feature_count"),
                    "feature_profile": training.get("feature_profile", "unknown"),
                    "schema_version": manifest.get("feature_schema_version", "unknown"),
                    "calibration_version": manifest.get("calibration_version", "unknown"),
                    "calibration_method": manifest.get("calibration_method", "unknown"),
                    "top_k_positive": training.get("top_k_positive"),
                    "label_scheme": training.get("label_scheme"),
                    "min_score_norm": training.get("min_score_norm"),
                    "temporal_feature_count": training.get("temporal_feature_count"),
                    "train_rows": training.get("train_rows"),
                    "val_rows": training.get("val_rows"),
                    "test_rows": training.get("test_rows"),
                    "val_log_loss": metrics.get("val_log_loss"),
                    "val_roc_auc": metrics.get("val_roc_auc"),
                    "val_brier_raw": metrics.get("val_brier_raw"),
                    "val_brier_calibrated": metrics.get("val_brier_calibrated"),
                    "val_log_loss_calibrated": metrics.get("val_log_loss_calibrated"),
                })

        overall = evaluation_metrics.get("overall", {}) if isinstance(evaluation_metrics, dict) else {}
        model_meta = evaluation_metrics.get("model_metadata", {}) if isinstance(evaluation_metrics, dict) else {}
        eval_cfg = evaluation_metrics.get("evaluation_config", {}) if isinstance(evaluation_metrics, dict) else {}

        if rows:
            for row in rows:
                if row["model_version"] == model_meta.get("model_version"):
                    row["participants_evaluated"] = overall.get("participants_evaluated")
                    row["mean_hybrid_precision@1"] = overall.get("mean_hybrid_precision@1")
                    row["mean_hybrid_recall@1"] = overall.get("mean_hybrid_recall@1")
                    row["mean_hybrid_ndcg@1"] = overall.get("mean_hybrid_ndcg@1")
                    row["mean_hybrid_precision@3"] = overall.get("mean_hybrid_precision@3")
                    row["mean_hybrid_recall@3"] = overall.get("mean_hybrid_recall@3")
                    row["mean_hybrid_ndcg@3"] = overall.get("mean_hybrid_ndcg@3")
                    row["mean_hybrid_precision@5"] = overall.get("mean_hybrid_precision@5")
                    row["mean_hybrid_recall@5"] = overall.get("mean_hybrid_recall@5")
                    row["mean_hybrid_ndcg@5"] = overall.get("mean_hybrid_ndcg@5")
                    row["mean_brier_ml"] = overall.get("mean_brier_ml")
                    row["mean_top1_changed"] = overall.get("mean_top1_changed")
                    row["mean_top3_changed"] = overall.get("mean_top3_changed")
                    row["mean_any_rank_changed"] = overall.get("mean_any_rank_changed")
                    row["mean_mean_absolute_rank_shift"] = overall.get("mean_mean_absolute_rank_shift")
                    row["mean_promoted_count"] = overall.get("mean_promoted_count")
                    row["mean_demoted_count"] = overall.get("mean_demoted_count")

            return pd.DataFrame(rows)

        return pd.DataFrame([{
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
        }])

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

    @staticmethod
    def _build_pipeline_stage_counts_df(
        manifest: dict,
        gated_df: pd.DataFrame,
        ranked_df: pd.DataFrame,
        reranked_df: pd.DataFrame,
        bundles_df: pd.DataFrame,
        explained_df: pd.DataFrame,
    ) -> pd.DataFrame:
        run_scope = manifest.get("run_scope", {}) if isinstance(manifest, dict) else {}

        participants = 0
        if "ParticipantKey" in gated_df.columns and not gated_df.empty:
            participants = int(gated_df["ParticipantKey"].nunique())

        return pd.DataFrame([
            {
                "participant_key": run_scope.get("participant_key"),
                "limit": run_scope.get("limit"),
                "participants": participants,
                "gated_candidates": int(len(gated_df)),
                "ranked_candidates": int(len(ranked_df)),
                "reranked_candidates": int(len(reranked_df)),
                "final_bundles": int(len(bundles_df)),
                "explained_recommendations": int(len(explained_df)),
            }
        ])

    @staticmethod
    def _flatten_dict(prefix: str, payload: dict) -> list[dict]:
        rows = []

        for key, value in payload.items():
            metric = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, dict):
                rows.extend(Command._flatten_dict(metric, value))
            else:
                rows.append({
                    "metric": metric,
                    "value": value,
                })

        return rows

    @staticmethod
    def _build_manifest_section_df(section_name: str, payload: dict) -> pd.DataFrame:
        if not isinstance(payload, dict) or not payload:
            return pd.DataFrame(columns=["section", "metric", "value"])

        rows = Command._flatten_dict("", payload)
        for row in rows:
            row["section"] = section_name

        return pd.DataFrame(rows)[["section", "metric", "value"]]

    @staticmethod
    def _build_xai_status_distribution_df(explained_df: pd.DataFrame) -> pd.DataFrame:
        if "ml_xai_payload" not in explained_df.columns:
            return pd.DataFrame(columns=["xai_status", "count", "pct"])

        statuses = explained_df["ml_xai_payload"].apply(
            lambda x: x.get("xai_status", "unknown") if isinstance(x, dict) else "unknown"
        )
        vc = statuses.value_counts(dropna=False)
        total = len(explained_df)

        return pd.DataFrame({
            "xai_status": vc.index.astype(str),
            "count": vc.values,
            "pct": [round((v / total) * 100, 4) if total else 0.0 for v in vc.values],
        })
        
    @staticmethod
    def _build_ablation_summary_df(manifest_paths: list[Path]) -> pd.DataFrame:
        rows = []

        for path in manifest_paths:
            if not path.exists():
                continue

            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            training = manifest.get("training", {})
            metrics = manifest.get("metrics", {})

            rows.append({
            	"patient_profile_feature_count": training.get("patient_profile_feature_count", 0),
                "model_version": manifest.get("model_version", path.stem),
                "feature_profile": training.get("feature_profile"),
                "collaborative_feature_count": training.get("collaborative_feature_count", 0),
                "temporal_feature_count": training.get("temporal_feature_count", 0),
                "max_ml_contrib": manifest.get("max_ml_contrib"),
                "val_log_loss": metrics.get("val_log_loss"),
                "val_roc_auc": metrics.get("val_roc_auc"),
                "val_brier_calibrated": metrics.get("val_brier_calibrated"),
                "val_log_loss_calibrated": metrics.get("val_log_loss_calibrated"),
            })

        return pd.DataFrame(rows)
