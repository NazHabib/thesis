from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.patient_profile_layer.services.profile_builder import build_profiles_from_path
from apps.patient_profile_layer.services.profile_schema import PROFILE_VERSION_DEFAULT


class Command(BaseCommand):
    help = "Build cached patient profiles (JSONL + Parquet snapshots)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--participant-features",
            type=str,
            default="nhanes_step1_out/NHANES_Master_engineered.parquet",
        )
        parser.add_argument(
            "--output-jsonl",
            type=str,
            default="apps/patient_profile_layer/output/patient_profiles.jsonl",
        )
        parser.add_argument(
            "--output-parquet",
            type=str,
            default="apps/patient_profile_layer/output/patient_profiles.parquet",
        )
        parser.add_argument(
            "--output-dataset-root",
            type=str,
            default="apps/patient_profile_layer/output/patient_profiles_dataset",
        )
        parser.add_argument(
            "--profile-version",
            type=str,
            default=PROFILE_VERSION_DEFAULT,
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--incremental",
            action="store_true",
            help="Build only profiles missing from the existing Parquet snapshot.",
        )
        parser.add_argument(
            "--skip-if-exists",
            action="store_true",
            help="Return immediately if the target Parquet snapshot already exists.",
        )
        parser.add_argument(
            "--no-partitioned-dataset",
            action="store_true",
            help="Do not write the partitioned Parquet dataset root.",
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)

        participant_features_path = base_dir / options["participant_features"]
        output_jsonl_path = base_dir / options["output_jsonl"]
        output_parquet_path = base_dir / options["output_parquet"]
        output_dataset_root = base_dir / options["output_dataset_root"]

        if options["skip_if_exists"] and output_parquet_path.exists():
            self.stdout.write("Patient profile snapshot already exists; reusing.")
            self.stdout.write(f"Parquet: {output_parquet_path}")
            return

        self.stdout.write("Loading participant features...")
        self.stdout.write("Building patient profiles...")

        profile_df = build_profiles_from_path(
            participant_features_path=participant_features_path,
            output_jsonl_path=output_jsonl_path,
            output_parquet_path=output_parquet_path,
            output_dataset_root=output_dataset_root,
            profile_version=options["profile_version"],
            limit=options["limit"],
            incremental=bool(options["incremental"]),
            write_partitioned_dataset=not bool(options["no_partitioned_dataset"]),
        )

        self.stdout.write(self.style.SUCCESS("Patient profile snapshot complete."))
        self.stdout.write(f"Rows: {len(profile_df)}")
        self.stdout.write(f"JSONL: {output_jsonl_path}")
        self.stdout.write(f"Parquet: {output_parquet_path}")
        if not bool(options["no_partitioned_dataset"]):
            self.stdout.write(f"Dataset root: {output_dataset_root}")
