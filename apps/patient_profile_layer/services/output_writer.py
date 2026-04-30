from __future__ import annotations

from pathlib import Path
import shutil

from django.conf import settings
import pandas as pd

from .profile_schema import validate_profile_df


DEFAULT_JSONL_PATH = (
    Path(settings.BASE_DIR)
    / "apps"
    / "patient_profile_layer"
    / "output"
    / "patient_profiles.jsonl"
)

DEFAULT_PARQUET_PATH = (
    Path(settings.BASE_DIR)
    / "apps"
    / "patient_profile_layer"
    / "output"
    / "patient_profiles.parquet"
)

DEFAULT_DATASET_ROOT = (
    Path(settings.BASE_DIR)
    / "apps"
    / "patient_profile_layer"
    / "output"
    / "patient_profiles_dataset"
)


def _derive_survey_cycle(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.split("_", n=1)
        .str[0]
        .fillna("unknown")
        .astype("string")
    )


def write_profile_outputs(
    df: pd.DataFrame,
    *,
    jsonl_path: Path | None = None,
    parquet_path: Path | None = None,
    dataset_root: Path | None = None,
    write_partitioned_dataset: bool = True,
) -> dict[str, Path]:
    validated = validate_profile_df(df)

    jsonl_path = jsonl_path or DEFAULT_JSONL_PATH
    parquet_path = parquet_path or DEFAULT_PARQUET_PATH
    dataset_root = dataset_root or DEFAULT_DATASET_ROOT

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    validated.to_json(
        jsonl_path,
        orient="records",
        lines=True,
        force_ascii=False,
        date_format="iso",
    )

    validated.to_parquet(
        parquet_path,
        engine="pyarrow",
        index=False,
        compression="snappy",
    )

    if write_partitioned_dataset:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        dataset_root.mkdir(parents=True, exist_ok=True)

        dataset_df = validated.copy()
        dataset_df["survey_cycle"] = _derive_survey_cycle(dataset_df["participant_key"])

        table = pa.Table.from_pandas(dataset_df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(dataset_root),
            partition_cols=["survey_cycle", "profile_version"],
            compression="snappy",
        )

    return {
        "jsonl_path": jsonl_path,
        "parquet_path": parquet_path,
        "dataset_root": dataset_root,
    }
