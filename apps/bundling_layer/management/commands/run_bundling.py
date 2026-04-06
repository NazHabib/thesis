import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from apps.bundling_layer.services.bundling.service import run_bundling_layer


class Command(BaseCommand):
    help = "Run bundling layer on safety_context + ranked_recommendations."

    def add_arguments(self, parser):
        parser.add_argument("--safety", type=str, required=True, help="Path to safety_context.jsonl")
        parser.add_argument("--ranked", type=str, required=True, help="Path to ranked_recommendations.jsonl")
        parser.add_argument(
            "--outdir",
            type=str,
            default="apps/bundling_layer/output",
            help="Output directory",
        )

    def handle(self, *args, **options):
        safety_path = options["safety"]
        ranked_path = options["ranked"]
        outdir = Path(options["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Loading safety_context...")
        safety_context_df = pd.read_json(safety_path, orient="records", lines=True)

        self.stdout.write("Loading ranked recommendations...")
        ranked_df = pd.read_json(ranked_path, orient="records", lines=True)

        self.stdout.write("Running bundling layer...")
        result = run_bundling_layer(safety_context_df, ranked_df)

        final_path = outdir / "final_recommendations.jsonl"
        manifest_path = outdir / "manifest.json"

        result["final_recommendations"].to_json(
            final_path,
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
                "ranked_recommendations": ranked_path,
            },
            "output_files": {
                "final_recommendations": str(final_path),
            },
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self.stdout.write(self.style.SUCCESS("Bundling layer executed successfully."))
        self.stdout.write(f"Saved: {final_path}")
        self.stdout.write(f"Saved: {manifest_path}")
