import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.ml_layer.services.ml_reranking.service import run_ml_reranking_layer


class Command(BaseCommand):
    help = "Run the ML reranking layer on current ranking outputs."

    def add_arguments(self, parser):
        parser.add_argument(
            "--participant-features",
            type=str,
            default="nhanes_step1_out/NHANES_Master_engineered.parquet",
            help="Path to participant feature parquet.",
        )
        parser.add_argument(
            "--safety-context",
            type=str,
            default="apps/recs_csl/output/safety_context.jsonl",
            help="Path to safety_context jsonl.",
        )
        parser.add_argument(
            "--ranked",
            type=str,
            default="apps/ranking_layer/output/ranked_recommendations.jsonl",
            help="Path to ranked recommendations jsonl.",
        )
        parser.add_argument(
            "--out-jsonl",
            type=str,
            default="apps/ml_layer/output/reranked_recommendations.jsonl",
            help="Output jsonl path.",
        )
        parser.add_argument(
            "--out-manifest",
            type=str,
            default="apps/ml_layer/output/manifest.json",
            help="Output manifest path.",
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)

        participant_features_path = base_dir / options["participant_features"]
        safety_context_path = base_dir / options["safety_context"]
        ranked_path = base_dir / options["ranked"]
        out_jsonl_path = base_dir / options["out_jsonl"]
        out_manifest_path = base_dir / options["out_manifest"]

        self.stdout.write("Loading inputs...")

        safety_context_df = pd.read_json(safety_context_path, lines=True)
        ranked_df = pd.read_json(ranked_path, lines=True)

        participant_features_df = None
        if participant_features_path.exists():
            participant_features_df = pd.read_parquet(participant_features_path)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Loaded participant features: {participant_features_df.shape}"
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"Participant feature file not found: {participant_features_path}. Running without participant features."
                )
            )

        result = run_ml_reranking_layer(
            safety_context_df=safety_context_df,
            ranked_df=ranked_df,
            participant_features_df=participant_features_df,
        )

        reranked_df = result["reranked_recommendations"]
        manifest = result["manifest"]

        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        out_manifest_path.parent.mkdir(parents=True, exist_ok=True)

        reranked_df.to_json(out_jsonl_path, orient="records", lines=True, force_ascii=False)

        with open(out_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        self.stdout.write(
            self.style.SUCCESS(
                f"ML reranking complete. Output rows: {len(reranked_df)}"
            )
        )
        self.stdout.write(f"Saved reranked recommendations to: {out_jsonl_path}")
        self.stdout.write(f"Saved manifest to: {out_manifest_path}")
