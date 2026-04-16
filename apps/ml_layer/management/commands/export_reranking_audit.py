import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Export reranking audit tables."

    def add_arguments(self, parser):
        parser.add_argument("--reranked", type=str, default="apps/ml_layer/output/reranked_recommendations.jsonl")
        parser.add_argument("--evaluation-json", type=str, default="apps/ml_layer/output/evaluation_metrics.json")
        parser.add_argument("--evaluation-csv", type=str, default="apps/ml_layer/output/evaluation_per_participant.csv")
        parser.add_argument("--output-dir", type=str, default="apps/ml_layer/output/audit")

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)
        out_dir = base_dir / options["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        reranked_df = pd.read_json(base_dir / options["reranked"], lines=True)

        evaluation_json_path = base_dir / options["evaluation_json"]
        evaluation_csv_path = base_dir / options["evaluation_csv"]

        evaluation_metrics = {}
        if evaluation_json_path.exists():
            with open(evaluation_json_path, "r", encoding="utf-8") as f:
                evaluation_metrics = json.load(f)

        per_participant_df = pd.DataFrame()
        if evaluation_csv_path.exists():
            per_participant_df = pd.read_csv(evaluation_csv_path)

        total = len(reranked_df)
        ml_rows = int((reranked_df["rerank_status"] == "ml_reranked").sum())
        fallback_rows = int((reranked_df["rerank_status"] == "rules_only_fallback").sum())

        summary_df = pd.DataFrame([
            {"metric": "rows_total", "value": total},
            {"metric": "rows_ml_reranked", "value": ml_rows},
            {"metric": "rows_rules_only_fallback", "value": fallback_rows},
            {"metric": "pct_ml_reranked", "value": round((ml_rows / total) * 100, 4) if total else 0.0},
            {"metric": "pct_rules_only_fallback", "value": round((fallback_rows / total) * 100, 4) if total else 0.0},
            {"metric": "mean_ml_contrib", "value": round(float(reranked_df["ml_contrib"].mean()), 6)},
            {"metric": "median_ml_contrib", "value": round(float(reranked_df["ml_contrib"].median()), 6)},
            {"metric": "mean_lambda_u", "value": round(float(reranked_df["lambda_u"].mean()), 6)},
        ])

        fallback_df = (
            reranked_df["fallback_reason"]
            .fillna("none")
            .value_counts(dropna=False)
            .rename_axis("fallback_reason")
            .reset_index(name="count")
        )

        changed_participants_df = (
            reranked_df.assign(score_delta=(reranked_df["final_score"] - reranked_df["total_score"]).abs())
            .groupby("ParticipantKey", as_index=False)["score_delta"]
            .sum()
            .sort_values("score_delta", ascending=False, kind="stable")
            .head(100)
        )

        ranking_impact_summary_df = self._build_ranking_impact_summary_df(evaluation_metrics)
        top_rank_shift_participants_df = self._build_top_rank_shift_participants_df(per_participant_df)

        summary_df.to_csv(out_dir / "reranking_summary.csv", index=False)
        fallback_df.to_csv(out_dir / "fallback_breakdown.csv", index=False)
        changed_participants_df.to_csv(out_dir / "top_changed_participants.csv", index=False)
        ranking_impact_summary_df.to_csv(out_dir / "ranking_impact_summary.csv", index=False)
        top_rank_shift_participants_df.to_csv(out_dir / "top_rank_shift_participants.csv", index=False)

        self.stdout.write(self.style.SUCCESS("Reranking audit exported."))
        self.stdout.write(f"Saved to: {out_dir}")

    @staticmethod
    def _build_ranking_impact_summary_df(evaluation_metrics: dict) -> pd.DataFrame:
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
    def _build_top_rank_shift_participants_df(per_participant_df: pd.DataFrame) -> pd.DataFrame:
        if per_participant_df.empty or "mean_absolute_rank_shift" not in per_participant_df.columns:
            return pd.DataFrame(columns=["ParticipantKey", "mean_absolute_rank_shift", "max_absolute_rank_shift", "top1_changed", "top3_changed"])

        cols = [
            c for c in [
                "ParticipantKey",
                "mean_absolute_rank_shift",
                "max_absolute_rank_shift",
                "top1_changed",
                "top3_changed",
                "promoted_count",
                "demoted_count",
            ]
            if c in per_participant_df.columns
        ]

        return (
            per_participant_df[cols]
            .sort_values(
                by=[c for c in ["mean_absolute_rank_shift", "max_absolute_rank_shift"] if c in cols],
                ascending=False,
                kind="stable",
            )
            .head(100)
            .reset_index(drop=True)
        )
