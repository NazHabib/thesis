from __future__ import annotations

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.collaborative_layer.services.resolver import CollaborativeResolver
from apps.collaborative_layer.services.snapshot import save_collaborative_snapshot


class Command(BaseCommand):
    help = "Build cached collaborative features (parquet snapshot)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--ranked",
            type=str,
            default="apps/ranking_layer/output/ranked_recommendations.jsonl",
        )
        parser.add_argument(
            "--participant-features",
            type=str,
            default="nhanes_step1_out/NHANES_Master_engineered.parquet",
        )
        parser.add_argument(
            "--safety-context",
            type=str,
            default="apps/recs_csl/output/safety_context.jsonl",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="apps/collaborative_layer/output/collaborative_features_snapshot.parquet",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)

        ranked_path = base_dir / options["ranked"]
        participant_features_path = base_dir / options["participant_features"]
        safety_context_path = base_dir / options["safety_context"]
        output_path = base_dir / options["output"]

        self.stdout.write("Loading ranked candidates...")
        ranked_df = pd.read_json(ranked_path, lines=True)

        if options["limit"] is not None:
            keys = (
                ranked_df["ParticipantKey"]
                .astype(str)
                .drop_duplicates()
                .head(int(options["limit"]))
                .tolist()
            )
            ranked_df = ranked_df[
                ranked_df["ParticipantKey"].astype(str).isin(keys)
            ].copy()

        self.stdout.write("Loading participant features...")
        participant_features_df = pd.read_parquet(participant_features_path)

        keys = ranked_df["ParticipantKey"].astype(str).unique().tolist()
        participant_features_df = participant_features_df[
            participant_features_df["ParticipantKey"].astype(str).isin(keys)
        ].copy()

        self.stdout.write("Loading safety context...")
        safety_context_df = pd.read_json(safety_context_path, lines=True)
        safety_context_df = safety_context_df[
            safety_context_df["ParticipantKey"].astype(str).isin(keys)
        ].copy()

        self.stdout.write("Resolving collaborative features...")
        resolver = CollaborativeResolver()

        collab_df = resolver.resolve_frame(
            ranked_candidates_df=ranked_df[
                ["ParticipantKey", "action_code", "bundle_family"]
            ].copy(),
            participant_features_df=participant_features_df,
            safety_context_df=safety_context_df,
            reference_ranked_df=ranked_df,
            use_online_if_missing=True,
        )

        save_collaborative_snapshot(collab_df, output_path)

        self.stdout.write(self.style.SUCCESS("Collaborative snapshot complete."))
        self.stdout.write(f"Rows: {len(collab_df)}")
        self.stdout.write(f"Saved to: {output_path}")
