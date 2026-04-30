from pathlib import Path

import pandas as pd
from django.conf import settings

from .aggregation import build_collaborative_features
from .constants import (
    COLLAB_DEFAULTS,
    COLLAB_FEATURE_COLUMNS,
    K_NEIGHBORS_DEFAULT,
    MIN_NEIGHBORS_DEFAULT,
    SIMILARITY_THRESHOLD_DEFAULT,
)
from .neighbor_index import compute_neighbors
from .preprocessing import build_similarity_preprocessor


class CollaborativeResolver:
    """
    Runtime/offline resolver for collaborative features.

    Version 1:
    - uses current participant feature table as reference population
    - uses ranked candidates as reference action memory
    - computes exact cosine neighbors
    - returns safe neutral defaults when unavailable
    """

    def resolve_frame(
        self,
        ranked_candidates_df: pd.DataFrame,
        participant_features_df: pd.DataFrame | None,
        safety_context_df: pd.DataFrame | None,
        snapshot_id=None,
        use_online_if_missing: bool = True,
        reference_ranked_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        base = ranked_candidates_df[
            ["ParticipantKey", "action_code", "bundle_family"]
        ].drop_duplicates().copy()

        if base.empty:
            return pd.DataFrame(columns=COLLAB_FEATURE_COLUMNS)

        fallback = self._fallback_frame(base)

        if participant_features_df is None or participant_features_df.empty:
            return fallback

        if reference_ranked_df is None:
            reference_ranked_df = ranked_candidates_df.copy()

        profile_df = self._build_profile_df(
            participant_features_df=participant_features_df,
            safety_context_df=safety_context_df,
        )

        if profile_df.empty or profile_df["ParticipantKey"].nunique() < 2:
            return fallback

        try:
            preprocessor = build_similarity_preprocessor(profile_df)
            participant_keys = profile_df["ParticipantKey"].astype(str).tolist()
            matrix = preprocessor.fit_transform(profile_df)

            neighbors_df = compute_neighbors(
                participant_keys=participant_keys,
                matrix=matrix,
                k=K_NEIGHBORS_DEFAULT,
                similarity_threshold=SIMILARITY_THRESHOLD_DEFAULT,
            )

            features_df = build_collaborative_features(
                ranked_candidates_df=base,
                neighbors_df=neighbors_df,
                reference_ranked_df=reference_ranked_df,
                min_neighbors=MIN_NEIGHBORS_DEFAULT,
            )

            return self._ensure_schema(features_df)

        except Exception as exc:
            raise RuntimeError(
                f"CollaborativeResolver failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _build_profile_df(
        self,
        participant_features_df: pd.DataFrame,
        safety_context_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        profile = participant_features_df.copy()

        if "ParticipantKey" not in profile.columns:
            return pd.DataFrame(columns=["ParticipantKey"])

        if safety_context_df is not None and not safety_context_df.empty:
            ctx_cols = [
                c for c in [
                    "ParticipantKey",
                    "feature_tier",
                    "data_completeness_score",
                    "followup_urgency",
                    "followup_horizon_days",
                    "monitoring_need_level",
                    "temporal_risk_tier",
                    "persistence_risk_proxy",
                    "repeat_measurement_available",
                    "temporal_data_quality",
                    "longitudinal_ready_flag",
                ]
                if c in safety_context_df.columns and c not in profile.columns
            ]

            if "ParticipantKey" not in ctx_cols:
                ctx_cols = ["ParticipantKey"] + ctx_cols

            context = safety_context_df[ctx_cols].drop_duplicates("ParticipantKey")
            profile = profile.merge(
                context,
                on="ParticipantKey",
                how="left",
                validate="m:1",
            )

        return profile.drop_duplicates("ParticipantKey").copy()

    def _fallback_frame(self, base: pd.DataFrame) -> pd.DataFrame:
        out = base.copy()

        for col, default in COLLAB_DEFAULTS.items():
            out[col] = default

        return out[COLLAB_FEATURE_COLUMNS].copy()

    def _ensure_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for col, default in COLLAB_DEFAULTS.items():
            if col not in out.columns:
                out[col] = default
            else:
                out[col] = out[col].fillna(default)

        for col in ["ParticipantKey", "action_code", "bundle_family"]:
            if col not in out.columns:
                out[col] = None

        return out[COLLAB_FEATURE_COLUMNS].copy()
