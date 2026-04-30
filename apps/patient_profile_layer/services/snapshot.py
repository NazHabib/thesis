from __future__ import annotations

from pathlib import Path

from django.conf import settings
import pandas as pd


DEFAULT_PROFILE_PARQUET_PATH = (
    Path(settings.BASE_DIR)
    / "apps"
    / "patient_profile_layer"
    / "output"
    / "patient_profiles.parquet"
)


def load_patient_profiles_snapshot(
    path: Path | None = None,
    *,
    columns: list[str] | None = None,
    filters: list | None = None,
) -> pd.DataFrame:
    src = path or DEFAULT_PROFILE_PARQUET_PATH
    if not src.exists():
        return pd.DataFrame()

    kwargs: dict = {}
    if columns is not None:
        kwargs["columns"] = columns
    if filters is not None:
        kwargs["filters"] = filters

    return pd.read_parquet(src, **kwargs)
