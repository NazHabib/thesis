from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ParticipantSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_keys: list[str]
    val_keys: list[str]
    test_keys: list[str]


def make_participant_level_split(
    df: pd.DataFrame,
    participant_col: str = "ParticipantKey",
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> ParticipantSplit:
    """
    Strict participant-level split.

    Prevents leakage by ensuring all candidate rows from a participant stay
    in exactly one split.
    """
    _validate_split_inputs(df, participant_col, train_frac, val_frac, test_frac)

    participant_keys = (
        df[participant_col]
        .dropna()
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    rng = np.random.default_rng(seed)
    shuffled = np.array(participant_keys, dtype=object)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))

    # ensure totals reconcile cleanly
    if n_train >= n_total:
        n_train = max(1, n_total - 2)
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)

    train_keys = shuffled[:n_train].tolist()
    val_keys = shuffled[n_train:n_train + n_val].tolist()
    test_keys = shuffled[n_train + n_val:].tolist()

    train_df = df.loc[df[participant_col].astype(str).isin(train_keys)].copy()
    val_df = df.loc[df[participant_col].astype(str).isin(val_keys)].copy()
    test_df = df.loc[df[participant_col].astype(str).isin(test_keys)].copy()

    _assert_no_participant_overlap(train_df, val_df, test_df, participant_col)

    return ParticipantSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_keys=train_keys,
        val_keys=val_keys,
        test_keys=test_keys,
    )


def make_stratified_participant_level_split(
    df: pd.DataFrame,
    stratify_col: str = "feature_tier",
    participant_col: str = "ParticipantKey",
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> ParticipantSplit:
    """
    Approximate participant-level stratified split using one record per participant.

    Uses the first observed value of stratify_col per participant.
    This is appropriate here because feature_tier is participant-level in your pipeline.
    """
    _validate_split_inputs(df, participant_col, train_frac, val_frac, test_frac)

    if stratify_col not in df.columns:
        return make_participant_level_split(
            df=df,
            participant_col=participant_col,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )

    participant_meta = (
        df[[participant_col, stratify_col]]
        .drop_duplicates(subset=[participant_col])
        .copy()
    )

    rng = np.random.default_rng(seed)

    train_keys: list[str] = []
    val_keys: list[str] = []
    test_keys: list[str] = []

    for _, group in participant_meta.groupby(stratify_col, dropna=False):
        keys = group[participant_col].astype(str).tolist()
        arr = np.array(keys, dtype=object)
        rng.shuffle(arr)

        n_total = len(arr)
        if n_total == 1:
            train_keys.extend(arr.tolist())
            continue

        n_train = int(round(n_total * train_frac))
        n_val = int(round(n_total * val_frac))

        if n_train >= n_total:
            n_train = max(1, n_total - 2) if n_total >= 3 else 1
        if n_train + n_val >= n_total:
            n_val = max(0, n_total - n_train - 1)

        train_keys.extend(arr[:n_train].tolist())
        val_keys.extend(arr[n_train:n_train + n_val].tolist())
        test_keys.extend(arr[n_train + n_val:].tolist())

    # Deduplicate defensively, preserve order
    train_keys = _unique_preserve_order(train_keys)
    val_keys = [k for k in _unique_preserve_order(val_keys) if k not in set(train_keys)]
    test_keys = [
        k for k in _unique_preserve_order(test_keys)
        if k not in set(train_keys) and k not in set(val_keys)
    ]

    assigned = set(train_keys) | set(val_keys) | set(test_keys)
    all_keys = set(df[participant_col].astype(str).dropna().tolist())
    unassigned = [k for k in df[participant_col].astype(str).dropna().tolist() if k in (all_keys - assigned)]
    for k in _unique_preserve_order(unassigned):
        test_keys.append(k)

    train_df = df.loc[df[participant_col].astype(str).isin(train_keys)].copy()
    val_df = df.loc[df[participant_col].astype(str).isin(val_keys)].copy()
    test_df = df.loc[df[participant_col].astype(str).isin(test_keys)].copy()

    _assert_no_participant_overlap(train_df, val_df, test_df, participant_col)

    return ParticipantSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_keys=train_keys,
        val_keys=val_keys,
        test_keys=test_keys,
    )


def _validate_split_inputs(
    df: pd.DataFrame,
    participant_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> None:
    if participant_col not in df.columns:
        raise ValueError(f"Missing participant column: {participant_col}")

    total = float(train_frac + val_frac + test_frac)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0, got {total}"
        )

    if df.empty:
        raise ValueError("Cannot split empty dataframe")


def _assert_no_participant_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    participant_col: str,
) -> None:
    train_keys = set(train_df[participant_col].astype(str).unique().tolist())
    val_keys = set(val_df[participant_col].astype(str).unique().tolist())
    test_keys = set(test_df[participant_col].astype(str).unique().tolist())

    if train_keys & val_keys:
        raise ValueError("Participant leakage detected between train and val splits")
    if train_keys & test_keys:
        raise ValueError("Participant leakage detected between train and test splits")
    if val_keys & test_keys:
        raise ValueError("Participant leakage detected between val and test splits")


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
