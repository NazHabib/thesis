from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.explainability_layer.services.explainability.shap_service import (
    run_participant_shap_explanation,
    save_participant_shap_outputs,
)


class Command(BaseCommand):
    help = "Generate SHAP explanations for one participant from the saved ML reranking artifacts."

    def add_arguments(self, parser):
        parser.add_argument(
            "--participant-key",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--safety-context",
            type=str,
            default="apps/recs_csl/output/safety_context.jsonl",
        )
        parser.add_argument(
            "--reranked",
            type=str,
            default="apps/ml_layer/output/reranked_recommendations.jsonl",
        )
        parser.add_argument(
            "--participant-features",
            type=str,
            default="nhanes_step1_out/NHANES_Master_engineered.parquet",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="apps/explainability_layer/output/shap",
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)

        participant_key = options["participant_key"]
        safety_path = base_dir / options["safety_context"]
        reranked_path = base_dir / options["reranked"]
        participant_features_path = base_dir / options["participant_features"]
        output_dir = base_dir / options["output_dir"]

        self.stdout.write("Loading safety context...")
        safety_context_df = pd.read_json(safety_path, lines=True)

        self.stdout.write("Loading reranked recommendations...")
        reranked_df = pd.read_json(reranked_path, lines=True)

        self.stdout.write("Loading participant features...")
        participant_features_df = pd.read_parquet(participant_features_path)

        self.stdout.write(f"Running SHAP for participant: {participant_key}")
        result = run_participant_shap_explanation(
            participant_key=participant_key,
            safety_context_df=safety_context_df,
            reranked_df=reranked_df,
            participant_features_df=participant_features_df,
        )

        saved = save_participant_shap_outputs(result=result, output_dir=output_dir)

        self.stdout.write(self.style.SUCCESS("Participant SHAP explanation complete."))
        self.stdout.write(f"Saved JSON report to: {saved['json_path']}")
        self.stdout.write(f"Saved per-action CSV to: {saved['csv_path']}")
        self.stdout.write(f"Saved global-importance CSV to: {saved['global_csv_path']}")
        self.stdout.write(f"Saved clinical-only global-importance CSV to: {saved['global_clinical_csv_path']}")
