from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


@dataclass
class EvaluationBundle:
    overall: dict[str, Any]
    subgroup_results: dict[str, dict[str, Any]]
    per_participant_metrics: pd.DataFrame


def evaluate_reranker_predictions(
    df: pd.DataFrame,
    participant_col: str = "ParticipantKey",
    label_col: str = "label",
    rule_score_col: str = "total_score",
    ml_score_col: str = "ml_score",
    final_score_col: str = "final_score",
    ks: tuple[int, ...] = (1, 3, 5),
) -> EvaluationBundle:
    """
    Evaluate rules-only, ML-only, and hybrid ranking views on the same dataframe.
    """
    required = [participant_col, label_col, rule_score_col, ml_score_col, final_score_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required evaluation columns: {missing}")

    per_participant_metrics = _compute_per_participant_metrics(
        df=df,
        participant_col=participant_col,
        label_col=label_col,
        rule_score_col=rule_score_col,
        ml_score_col=ml_score_col,
        final_score_col=final_score_col,
        ks=ks,
    )

    overall = _aggregate_metrics(per_participant_metrics)

    subgroup_results = {}
    for subgroup_col in ["feature_tier", "rerank_status", "Severity_band"]:
        if subgroup_col in df.columns:
            subgroup_results[subgroup_col] = _evaluate_subgroups(
                df=df,
                subgroup_col=subgroup_col,
                participant_col=participant_col,
                label_col=label_col,
                rule_score_col=rule_score_col,
                ml_score_col=ml_score_col,
                final_score_col=final_score_col,
                ks=ks,
            )

    return EvaluationBundle(
        overall=overall,
        subgroup_results=subgroup_results,
        per_participant_metrics=per_participant_metrics,
    )


def _compute_per_participant_metrics(
    df: pd.DataFrame,
    participant_col: str,
    label_col: str,
    rule_score_col: str,
    ml_score_col: str,
    final_score_col: str,
    ks: tuple[int, ...],
) -> pd.DataFrame:
    metric_rows: list[dict[str, Any]] = []

    # Avoid annoying pandas arrow/string edge cases in repeated grouping
    working = df.copy()
    working[participant_col] = working[participant_col].astype(str)

    for participant_key, group in working.groupby(participant_col, sort=False):
        g = group.copy()

        labels = g[label_col].fillna(0).astype(int).to_numpy()
        if len(labels) == 0:
            continue

        row: dict[str, Any] = {
            participant_col: participant_key,
            "n_candidates": int(len(g)),
            "n_positives": int(labels.sum()),
        }

        # rules-only ranking
        g_rule = g.sort_values(
            by=[rule_score_col, "confidence_score", "priority"],
            ascending=[False, False, False],
            kind="stable",
        )
        labels_rule = g_rule[label_col].fillna(0).astype(int).to_numpy()
        row.update(_ranking_metrics_dict(labels_rule, prefix="rule", ks=ks))

        # ml-only ranking
        g_ml = g.sort_values(
            by=[ml_score_col, rule_score_col, "confidence_score"],
            ascending=[False, False, False],
            kind="stable",
        )
        labels_ml = g_ml[label_col].fillna(0).astype(int).to_numpy()
        row.update(_ranking_metrics_dict(labels_ml, prefix="ml", ks=ks))

        # hybrid ranking
        g_hybrid = g.sort_values(
            by=[final_score_col, rule_score_col, "confidence_score"],
            ascending=[False, False, False],
            kind="stable",
        )
        labels_hybrid = g_hybrid[label_col].fillna(0).astype(int).to_numpy()
        row.update(_ranking_metrics_dict(labels_hybrid, prefix="hybrid", ks=ks))

        # probability quality
        try:
            row["brier_ml"] = float(
                brier_score_loss(
                    g[label_col].fillna(0).astype(int).to_numpy(),
                    np.clip(g[ml_score_col].fillna(0).astype(float).to_numpy(), 1e-8, 1 - 1e-8),
                )
            )
        except Exception:
            row["brier_ml"] = None

        metric_rows.append(row)

    return pd.DataFrame(metric_rows)


def _ranking_metrics_dict(labels_ranked: np.ndarray, prefix: str, ks: tuple[int, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"{prefix}_precision@{k}"] = _precision_at_k(labels_ranked, k)
        out[f"{prefix}_recall@{k}"] = _recall_at_k(labels_ranked, k)
        out[f"{prefix}_ndcg@{k}"] = _ndcg_at_k(labels_ranked, k)
    return out


def _precision_at_k(labels_ranked: np.ndarray, k: int) -> float:
    if len(labels_ranked) == 0:
        return 0.0
    topk = labels_ranked[: min(k, len(labels_ranked))]
    if len(topk) == 0:
        return 0.0
    return float(np.mean(topk))


def _recall_at_k(labels_ranked: np.ndarray, k: int) -> float:
    total_pos = int(np.sum(labels_ranked))
    if total_pos == 0:
        return 0.0
    topk = labels_ranked[: min(k, len(labels_ranked))]
    return float(np.sum(topk) / total_pos)


def _dcg_at_k(labels_ranked: np.ndarray, k: int) -> float:
    topk = labels_ranked[: min(k, len(labels_ranked))]
    if len(topk) == 0:
        return 0.0
    gains = (2 ** topk - 1) / np.log2(np.arange(2, len(topk) + 2))
    return float(np.sum(gains))


def _ndcg_at_k(labels_ranked: np.ndarray, k: int) -> float:
    dcg = _dcg_at_k(labels_ranked, k)
    ideal = np.sort(labels_ranked)[::-1]
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def _aggregate_metrics(per_participant_metrics: pd.DataFrame) -> dict[str, Any]:
    if per_participant_metrics.empty:
        return {}

    numeric_cols = [
        c for c in per_participant_metrics.columns
        if c != "ParticipantKey" and pd.api.types.is_numeric_dtype(per_participant_metrics[c])
    ]

    overall = {
        f"mean_{col}": float(per_participant_metrics[col].dropna().mean())
        for col in numeric_cols
    }
    overall["participants_evaluated"] = int(per_participant_metrics["ParticipantKey"].nunique())
    overall["rows_evaluated"] = int(len(per_participant_metrics))
    return overall


def _evaluate_subgroups(
    df: pd.DataFrame,
    subgroup_col: str,
    participant_col: str,
    label_col: str,
    rule_score_col: str,
    ml_score_col: str,
    final_score_col: str,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    working = df.copy()
    working[participant_col] = working[participant_col].astype(str)
    working[subgroup_col] = working[subgroup_col].astype(str)

    for subgroup_value, subgroup_df in working.groupby(subgroup_col, dropna=False):
        subgroup_metrics = _compute_per_participant_metrics(
            df=subgroup_df,
            participant_col=participant_col,
            label_col=label_col,
            rule_score_col=rule_score_col,
            ml_score_col=ml_score_col,
            final_score_col=final_score_col,
            ks=ks,
        )
        out[str(subgroup_value)] = _aggregate_metrics(subgroup_metrics)

    return out
