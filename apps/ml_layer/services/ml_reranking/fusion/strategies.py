from __future__ import annotations

import pandas as pd

from .contracts import FusionArtifacts, FusionContext


def participant_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    s_min = float(s.min())
    s_max = float(s.max())
    score_range = s_max - s_min

    if score_range <= 1e-8:
        return pd.Series(0.0, index=s.index)

    return (s - s_min) / score_range


class MinMaxAdditiveFusion:
    name = "minmax_additive_v1"
    version = "v1"

    def fuse(
        self,
        df: pd.DataFrame,
        ctx: FusionContext,
        artifacts: FusionArtifacts | None = None,
    ) -> pd.DataFrame:
        out = df.copy()

        out["rule_score_norm_participant"] = participant_minmax(out["total_score"])
        out["fusion_rule_component"] = out["rule_score_norm_participant"]

        out["fusion_ml_component"] = (
            ctx.lambda_u * pd.to_numeric(out["ml_score"], errors="coerce").fillna(0.0)
        ).clip(lower=0.0, upper=ctx.max_ml_contrib)

        out["ml_contrib"] = out["fusion_ml_component"]
        out["final_score"] = out["fusion_rule_component"] + out["fusion_ml_component"]

        out["fusion_strategy"] = self.name
        out["fusion_version"] = self.version
        out["fusion_status"] = "fused"
        out["fusion_artifact_version"] = None

        return out


class WeightedAverageFusion:
    name = "weighted_average_v1"
    version = "v1"

    def fuse(
        self,
        df: pd.DataFrame,
        ctx: FusionContext,
        artifacts: FusionArtifacts | None = None,
    ) -> pd.DataFrame:
        out = df.copy()

        alpha = max(0.0, min(1.0, float(ctx.lambda_u)))

        out["rule_score_norm_participant"] = participant_minmax(out["total_score"])
        out["fusion_rule_component"] = (1.0 - alpha) * out["rule_score_norm_participant"]
        out["fusion_ml_component"] = alpha * pd.to_numeric(out["ml_score"], errors="coerce").fillna(0.0)

        out["ml_contrib"] = out["fusion_ml_component"]
        out["final_score"] = out["fusion_rule_component"] + out["fusion_ml_component"]

        out["fusion_strategy"] = self.name
        out["fusion_version"] = self.version
        out["fusion_status"] = "fused"
        out["fusion_artifact_version"] = None

        return out
