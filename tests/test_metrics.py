import numpy as np

from rigged_matchup_ml.metrics import (
    binary_metrics,
    bootstrap_intervals,
    calibration_slope_intercept,
    quantile_calibration_table,
)


def test_quantile_calibration_table_uses_equal_counts() -> None:
    targets = np.array([0, 1] * 50)
    probabilities = np.linspace(0.0, 1.0, 100)

    table = quantile_calibration_table(targets, probabilities, bins=10)

    assert len(table) == 10
    assert sum(row["count"] for row in table) == 100
    assert all(row["count"] == 10 for row in table)


def test_calibration_slope_is_none_for_single_class() -> None:
    slope, intercept = calibration_slope_intercept(
        np.zeros(10), np.full(10, 0.3)
    )

    assert slope is None
    assert intercept is None


def test_calibration_slope_below_one_for_overconfident_model() -> None:
    rng = np.random.default_rng(0)
    # Labels follow a true log-odds z; the model reports a doubled log-odds, i.e.
    # it is over-confident. A logistic recalibration must then recover slope < 1.
    z = rng.normal(0.0, 1.5, size=20000)
    true_probability = 1.0 / (1.0 + np.exp(-z))
    targets = (rng.random(20000) < true_probability).astype(float)
    probabilities = 1.0 / (1.0 + np.exp(-2.0 * z))

    slope, _ = calibration_slope_intercept(targets, probabilities)

    assert slope is not None
    assert slope < 1.0


def test_bootstrap_intervals_are_ordered() -> None:
    targets = np.array([0] * 50 + [1] * 50)
    probabilities = np.concatenate([np.full(50, 0.2), np.full(50, 0.8)])

    intervals = bootstrap_intervals(targets, probabilities, samples=100, seed=0)

    for metric in ("auc", "brier_score", "log_loss", "accuracy"):
        assert intervals[metric]["low"] <= intervals[metric]["high"]


def test_binary_metrics_exposes_new_fields() -> None:
    targets = np.array([0, 1] * 50)
    probabilities = np.linspace(0.05, 0.95, 100)

    metrics = binary_metrics(targets, probabilities, bins=5, bootstrap_samples=20)

    assert "expected_calibration_error_quantile" in metrics
    assert "calibration_slope" in metrics
    assert "calibration_quantile" in metrics
    assert "confidence_intervals" in metrics
