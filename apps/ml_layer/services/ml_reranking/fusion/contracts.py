from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True, slots=True)
class FusionContext:
    participant_key: str
    strategy_name: str
    strategy_version: str
    lambda_u: float
    max_ml_contrib: float
    feature_tier: str
    triage_mode: bool
    model_version: str
    schema_version: str
    calibration_version: str


@dataclass(frozen=True, slots=True)
class FusionArtifacts:
    combiner: Any | None = None
    artifact_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FusionStrategy(Protocol):
    name: str
    version: str

    def fuse(
        self,
        df: pd.DataFrame,
        ctx: FusionContext,
        artifacts: FusionArtifacts | None = None,
    ) -> pd.DataFrame:
        ...
