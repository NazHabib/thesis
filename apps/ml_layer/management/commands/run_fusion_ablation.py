from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run ablation comparing rules-only baseline and fusion strategies."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=200)
        parser.add_argument(
            "--output-dir",
            type=str,
            default="apps/ml_layer/output/fusion_ablation",
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)
        out_dir = base_dir / options["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        limit = int(options["limit"])

        strategies = [
            "minmax_additive_v1",
            "weighted_average_v1",
        ]

        rows = []

        for strategy in strategies:
            self.stdout.write(f"Running fusion strategy: {strategy}")

            call_command(
                "run_full_hybrid_pipeline",
                limit=limit,
                save_intermediate=True,
                fusion_strategy=strategy,
            )

            reranked = pd.read_json(
                base_dir / "apps/ml_layer/output/reranked_recommendations.jsonl",
                lines=True,
            )

            bundles = pd.read_json(
                base_dir / "apps/ml_layer/output/final_recommendations_after_ml.jsonl",
                lines=True,
            )

            strategy_dir = out_dir / strategy
            strategy_dir.mkdir(parents=True, exist_ok=True)

            reranked.to_json(
                strategy_dir / "reranked_recommendations.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )
            bundles.to_json(
                strategy_dir / "final_recommendations_after_ml.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )

            rows.append(self._summarise_run(strategy, reranked, bundles))

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(out_dir / "fusion_ablation_summary.csv", index=False)

        self.stdout.write(self.style.SUCCESS("Fusion ablation complete."))
        self.stdout.write(f"Saved to: {out_dir}")
        self.stdout.write(str(summary_df))

    @staticmethod
    def _summarise_run(
        strategy: str,
        reranked: pd.DataFrame,
        bundles: pd.DataFrame,
    ) -> dict:
        total_rows = len(reranked)

        ml_reranked_rows = int(
            (reranked["rerank_status"] == "ml_reranked").sum()
        ) if "rerank_status" in reranked.columns else 0

        fallback_rows = int(
            (reranked["rerank_status"] == "rules_only_fallback").sum()
        ) if "rerank_status" in reranked.columns else 0

        fusion_counts = (
            reranked["fusion_strategy"].fillna("unknown").value_counts().to_dict()
            if "fusion_strategy" in reranked.columns else {}
        )

        return {
            "strategy": strategy,
            "reranked_rows": total_rows,
            "final_bundles": len(bundles),
            "ml_reranked_rows": ml_reranked_rows,
            "rules_only_fallback_rows": fallback_rows,
            "ml_reranked_pct": round((ml_reranked_rows / total_rows) * 100, 4)
            if total_rows else 0.0,
            "fallback_pct": round((fallback_rows / total_rows) * 100, 4)
            if total_rows else 0.0,
            "mean_final_score": round(float(reranked["final_score"].mean()), 6)
            if "final_score" in reranked.columns else 0.0,
            "median_final_score": round(float(reranked["final_score"].median()), 6)
            if "final_score" in reranked.columns else 0.0,
            "mean_ml_score": round(float(reranked["ml_score"].mean()), 6)
            if "ml_score" in reranked.columns else 0.0,
            "mean_ml_contrib": round(float(reranked["ml_contrib"].mean()), 6)
            if "ml_contrib" in reranked.columns else 0.0,
            "mean_fusion_rule_component": round(float(reranked["fusion_rule_component"].mean()), 6)
            if "fusion_rule_component" in reranked.columns else 0.0,
            "mean_fusion_ml_component": round(float(reranked["fusion_ml_component"].mean()), 6)
            if "fusion_ml_component" in reranked.columns else 0.0,
            "top_strategy_count": json.dumps(fusion_counts, ensure_ascii=False),
        }
