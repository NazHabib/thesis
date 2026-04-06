import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.explainability_layer.services.explainability.service import run_explainability_layer


class Command(BaseCommand):
    help = "Run explainability & communication layer on final bundled recommendations"

    def add_arguments(self, parser):
        parser.add_argument(
            "--safety-context",
            type=str,
            default="apps/recs_csl/output/safety_context.jsonl",
        )
        parser.add_argument(
            "--final-recommendations",
            type=str,
            default="apps/bundling_layer/output/final_recommendations.jsonl",
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)

        safety_path = base_dir / options["safety_context"]
        final_path = base_dir / options["final_recommendations"]

        self.stdout.write("Loading safety context...")
        safety_context_df = pd.read_json(safety_path, lines=True)

        self.stdout.write("Loading final recommendations...")
        final_recommendations_df = pd.read_json(final_path, lines=True)

        self.stdout.write("Running explainability layer...")
        out = run_explainability_layer(
            safety_context_df=safety_context_df,
            final_recommendations_df=final_recommendations_df,
        )

        explained_df = out["explained_recommendations"]
        manifest = out["manifest"]

        output_dir = base_dir / "apps" / "explainability_layer" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        explained_path = output_dir / "explained_recommendations.jsonl"
        manifest_path = output_dir / "manifest.json"

        explained_df.to_json(
            explained_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS("Explainability layer complete."))
        self.stdout.write(f"Output rows: {len(explained_df)}")
        self.stdout.write(f"Saved explained recommendations to: {explained_path}")
        self.stdout.write(f"Saved manifest to: {manifest_path}")
