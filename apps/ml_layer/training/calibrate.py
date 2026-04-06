from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class CalibrationResult:
    calibrator: object
    method: str
    val_pred_raw: np.ndarray
    val_pred_calibrated: np.ndarray
    metrics: dict


class PlattCalibrator:
    """
    Lightweight Platt scaling wrapper using logistic regression on raw scores/probabilities.
    """

    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        x = np.asarray(p_raw).reshape(-1, 1)
        y = np.asarray(y_true).astype(int)
        self.model.fit(x, y)
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        x = np.asarray(p_raw).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    """
    Lightweight isotonic calibration wrapper.
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        x = np.asarray(p_raw)
        y = np.asarray(y_true).astype(int)
        self.model.fit(x, y)
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        x = np.asarray(p_raw)
        return np.asarray(self.model.predict(x))


def calibrate_probabilities(
    y_val,
    val_pred_raw,
    method: str = "isotonic",
) -> CalibrationResult:
    y_val_arr = np.asarray(y_val).astype(int)
    val_pred_raw_arr = np.asarray(val_pred_raw).astype(float)

    method = str(method).strip().lower()

    if method == "isotonic":
        calibrator = IsotonicCalibrator().fit(val_pred_raw_arr, y_val_arr)
    elif method == "platt":
        calibrator = PlattCalibrator().fit(val_pred_raw_arr, y_val_arr)
    else:
        raise ValueError(f"Unsupported calibration method: {method}")

    val_pred_calibrated = np.clip(calibrator.predict(val_pred_raw_arr), 1e-8, 1 - 1e-8)

    metrics = {
        "calibration_method": method,
        "val_brier_raw": float(brier_score_loss(y_val_arr, val_pred_raw_arr)),
        "val_brier_calibrated": float(brier_score_loss(y_val_arr, val_pred_calibrated)),
        "val_log_loss_raw": float(log_loss(y_val_arr, np.clip(val_pred_raw_arr, 1e-8, 1 - 1e-8))),
        "val_log_loss_calibrated": float(log_loss(y_val_arr, val_pred_calibrated)),
        "val_pred_raw_min": float(np.min(val_pred_raw_arr)),
        "val_pred_raw_max": float(np.max(val_pred_raw_arr)),
        "val_pred_calibrated_min": float(np.min(val_pred_calibrated)),
        "val_pred_calibrated_max": float(np.max(val_pred_calibrated)),
    }

    return CalibrationResult(
        calibrator=calibrator,
        method=method,
        val_pred_raw=val_pred_raw_arr,
        val_pred_calibrated=val_pred_calibrated,
        metrics=metrics,
    )
