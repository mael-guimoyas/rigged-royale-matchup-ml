import numpy as np

from rigged_matchup_ml.trainer import fit_logit_calibration, fit_segment_temperatures


def test_fit_segment_temperatures_requires_support() -> None:
    logits = np.array([-2.0, -1.0, 1.0, 2.0, -2.0, 2.0], dtype=np.float32)
    targets = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    segments = np.array(["large", "large", "large", "large", "small", "small"])

    temperatures = fit_segment_temperatures(logits, targets, segments, min_rows=4)

    assert set(temperatures) == {"large"}
    assert 0.05 <= temperatures["large"] <= 10.0


def test_fit_logit_calibration_returns_temperature_and_bias() -> None:
    logits = np.zeros(100, dtype=np.float32)
    targets = np.asarray([1.0] * 70 + [0.0] * 30, dtype=np.float32)

    calibration = fit_logit_calibration(logits, targets)

    assert 0.05 <= calibration["temperature"] <= 10.0
    assert calibration["bias"] > 0.0
