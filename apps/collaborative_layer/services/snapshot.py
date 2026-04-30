from __future__ import annotations

from pathlib import Path
import pandas as pd
from django.conf import settings


DEFAULT_SNAPSHOT_PATH = (
    Path(settings.BASE_DIR)
    / "apps"
    / "collaborative_layer"
    / "output"
    / "collaborative_features_snapshot.parquet"
)


def save_collaborative_snapshot(df: pd.DataFrame, path: Path | None = None) -> Path:
    out_path = path or DEFAULT_SNAPSHOT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def load_collaborative_snapshot(path: Path | None = None) -> pd.DataFrame:
    in_path = path or DEFAULT_SNAPSHOT_PATH
    if not in_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(in_path)
