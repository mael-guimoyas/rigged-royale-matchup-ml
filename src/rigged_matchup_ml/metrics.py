from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def calibration_table(targets: np.ndarray, probabilities: np.ndarray, bins: int) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not np.any(mask):
            continue
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_prediction": float(probabilities[mask].mean()),
                "observed_win_rate": float(targets[mask].mean()),
                "absolute_error": float(
                    abs(probabilities[mask].mean() - targets[mask].mean())
                ),
            }
        )
    return rows


def binary_metrics(targets: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> dict[str, Any]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    calibration = calibration_table(targets, clipped, bins)
    ece = sum(row["count"] * row["absolute_error"] for row in calibration) / len(targets)
    return {
        "rows": int(len(targets)),
        "win_rate": float(targets.mean()),
        "mean_prediction": float(clipped.mean()),
        "accuracy": float(accuracy_score(targets, clipped >= 0.5)),
        "auc": float(roc_auc_score(targets, clipped)) if len(np.unique(targets)) > 1 else None,
        "log_loss": float(log_loss(targets, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(targets, clipped)),
        "expected_calibration_error": float(ece),
        "calibration": calibration,
    }


def binary_metrics_by_group(
    targets: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    bins: int = 15,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in sorted({str(value) for value in groups.tolist()}):
        mask = groups.astype(str) == group
        if not np.any(mask):
            continue
        result[group] = binary_metrics(targets[mask], probabilities[mask], bins)
    return result
