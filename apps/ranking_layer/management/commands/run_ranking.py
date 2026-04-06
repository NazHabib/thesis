import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from apps.ranking_layer.services.ranking.service import run_ranking_layer


class Command(BaseCommand):
    help = "Run ranking layer on cold-start outputs."

    def add_arguments(self, parser):
        parser.add_argument("--safety", type=str, required=True, help="Path to safety_context.jsonl")
        parser.add_argument("--candidates", type=str, required=True, help="Path to gated_candidates.jsonl")
        parser.add_argument(
            "--outdir",
            type=str,
            default="apps/ranking_layer/output",
            help="Output directory",
        )

    def handle(self, *args, **options):
        safety_path = options["safety"]
        candidates_path = options["candidates"]
        outdir = Path(options["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Loading safety_context...")
        safety_context_df = pd.read_json(safety_path, orient="records", lines=True)

        self.stdout.write("Loading gated_candidates...")
        gated_candidates_df = pd.read_json(candidates_path, orient="records", lines=True)

        self.stdout.write("Running ranking layer...")
        result = run_ranking_layer(safety_context_df, gated_candidates_df)

        ranked_path = outdir / "ranked_recommendations.jsonl"
        manifest_path = outdir / "manifest.json"

        result["ranked_recommendations"].to_json(
            ranked_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        manifest = {
            **result["manifest"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "input_files": {
                "safety_context": safety_path,
                "gated_candidates": candidates_path,
            },
            "output_files": {
                "ranked_recommendations": str(ranked_path),
            },
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self.stdout.write(self.style.SUCCESS("Ranking layer executed successfully."))
        self.stdout.write(f"Saved: {ranked_path}")
        self.stdout.write(f"Saved: {manifest_path}")
