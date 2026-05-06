from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score


@dataclass
class LGBMTrainingResult:
    model: LGBMClassifier
    train_pred_proba: np.ndarray
    val_pred_proba: np.ndarray
    metrics: dict


def train_lgbm_binary_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    random_state: int = 42,
) -> LGBMTrainingResult:
    """
    Conservative LightGBM configuration for clinical tabular reranking.
    """
    scale_pos_weight = _compute_scale_pos_weight(y_train)

    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_split_gain=0.01,
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
    )

    train_pred_proba = model.predict_proba(X_train)[:, 1]
    val_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "train_log_loss": float(log_loss(y_train, train_pred_proba)),
        "val_log_loss": float(log_loss(y_val, val_pred_proba)),
        "train_roc_auc": _safe_roc_auc(y_train, train_pred_proba),
        "val_roc_auc": _safe_roc_auc(y_val, val_pred_proba),
        "scale_pos_weight": float(scale_pos_weight),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
    }

    return LGBMTrainingResult(
        model=model,
        train_pred_proba=train_pred_proba,
        val_pred_proba=val_pred_proba,
        metrics=metrics,
    )


def _compute_scale_pos_weight(y) -> float:
    y_arr = np.asarray(y)
    pos = max(1, int((y_arr == 1).sum()))
    neg = max(1, int((y_arr == 0).sum()))
    return float(neg / pos)


def _safe_roc_auc(y_true, y_score) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None
