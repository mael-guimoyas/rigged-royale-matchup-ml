from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
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


def quantile_calibration_table(
    targets: np.ndarray, probabilities: np.ndarray, bins: int
) -> list[dict[str, Any]]:
    """Equal-count (quantile) bins. Robust when predictions cluster near 0.5."""
    if len(targets) == 0:
        return []
    order = np.argsort(probabilities, kind="stable")
    rows: list[dict[str, Any]] = []
    for group in np.array_split(order, min(bins, len(order))):
        if group.size == 0:
            continue
        group_predictions = probabilities[group]
        group_targets = targets[group]
        rows.append(
            {
                "count": int(group.size),
                "mean_prediction": float(group_predictions.mean()),
                "observed_win_rate": float(group_targets.mean()),
                "absolute_error": float(
                    abs(group_predictions.mean() - group_targets.mean())
                ),
            }
        )
    return rows


def _expected_calibration_error(table: list[dict[str, Any]], total: int) -> float:
    if total == 0:
        return 0.0
    return sum(row["count"] * row["absolute_error"] for row in table) / total


def calibration_slope_intercept(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[float | None, float | None]:
    """Logistic recalibration fit: target ~ logit(p).

    A perfectly calibrated model gives slope 1 and intercept 0. Slope < 1 means
    the model is over-confident; slope > 1 means under-confident.
    """
    if len(np.unique(targets)) < 2:
        return None, None
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logits, targets.astype(int))
    except (ValueError, np.linalg.LinAlgError):
        return None, None
    return float(model.coef_[0][0]), float(model.intercept_[0])


def _scalar_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    multiclass = len(np.unique(targets)) > 1
    return {
        "accuracy": float(accuracy_score(targets, clipped >= 0.5)),
        "auc": float(roc_auc_score(targets, clipped)) if multiclass else None,
        "log_loss": float(log_loss(targets, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(targets, clipped)),
    }


def bootstrap_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    samples: int,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap confidence intervals for the scalar metrics.

    Essential on small splits (e.g. unseen two-new-deck matchups) where a point
    estimate of AUC or Brier can be indistinguishable from noise.
    """
    if samples <= 0 or len(targets) == 0:
        return {}
    rng = np.random.default_rng(seed)
    size = len(targets)
    collected: dict[str, list[float]] = {
        "accuracy": [],
        "auc": [],
        "log_loss": [],
        "brier_score": [],
    }
    for _ in range(samples):
        index = rng.integers(0, size, size)
        sample_metrics = _scalar_metrics(targets[index], probabilities[index])
        for key, value in sample_metrics.items():
            if value is not None:
                collected[key].append(value)
    lower_q = (1 - confidence) / 2 * 100
    upper_q = (1 + confidence) / 2 * 100
    intervals: dict[str, dict[str, float]] = {}
    for key, values in collected.items():
        if not values:
            continue
        array = np.asarray(values)
        intervals[key] = {
            "low": float(np.percentile(array, lower_q)),
            "high": float(np.percentile(array, upper_q)),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        }
    return intervals


def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    calibration = calibration_table(targets, clipped, bins)
    quantile_calibration = quantile_calibration_table(targets, clipped, bins)
    ece = _expected_calibration_error(calibration, len(targets))
    quantile_ece = _expected_calibration_error(quantile_calibration, len(targets))
    slope, intercept = calibration_slope_intercept(targets, clipped)
    scalars = _scalar_metrics(targets, clipped)
    metrics: dict[str, Any] = {
        "rows": int(len(targets)),
        "win_rate": float(targets.mean()) if len(targets) else 0.0,
        "mean_prediction": float(clipped.mean()) if len(clipped) else 0.0,
        "accuracy": scalars["accuracy"],
        "auc": scalars["auc"],
        "log_loss": scalars["log_loss"],
        "brier_score": scalars["brier_score"],
        "expected_calibration_error": float(ece),
        "expected_calibration_error_quantile": float(quantile_ece),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "calibration": calibration,
        "calibration_quantile": quantile_calibration,
    }
    if bootstrap_samples > 0:
        metrics["confidence_intervals"] = bootstrap_intervals(
            targets, clipped, bootstrap_samples, bootstrap_seed
        )
    return metrics


def binary_metrics_by_group(
    targets: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    bins: int = 15,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in sorted({str(value) for value in groups.tolist()}):
        mask = groups.astype(str) == group
        if not np.any(mask):
            continue
        result[group] = binary_metrics(
            targets[mask],
            probabilities[mask],
            bins,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    return result
