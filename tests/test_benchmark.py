import numpy as np

from rigged_matchup_ml.benchmark import _temperature_bias_vectors, compute_noise_floor


def _reference_temperature_bias(segments: list[str], payload: dict) -> tuple[list[float], list[float]]:
    """The exact per-row mapping the old _calibrated_probabilities used."""
    temperature = float(payload["temperature"])
    segment_temperatures = payload.get("segment_temperatures") or {}
    calibration = payload.get("calibration") or {}
    global_calibration = calibration.get("global", {})
    global_temperature = float(global_calibration.get("temperature", temperature))
    global_bias = float(global_calibration.get("bias", 0.0))
    segment_calibrations = calibration.get("segments") or {}
    temps: list[float] = []
    biases: list[float] = []
    for segment in segments:
        if segment_calibrations:
            sc = segment_calibrations.get(segment, {})
            temps.append(float(sc.get("temperature", global_temperature)))
            biases.append(float(sc.get("bias", global_bias)))
        else:
            temps.append(float(segment_temperatures.get(segment, temperature)))
            biases.append(0.0)
    return temps, biases


def _assert_temperature_bias_matches(segments: list[str], payload: dict) -> None:
    temps, biases = _temperature_bias_vectors(segments, payload)
    ref_temps, ref_biases = _reference_temperature_bias(segments, payload)
    assert np.allclose(temps, ref_temps)
    assert np.allclose(biases, ref_biases)


def test_temperature_bias_global_only() -> None:
    payload = {"temperature": 1.3}
    _assert_temperature_bias_matches(
        ["ladder:5000-6999", "ranked:league-1", "ladder:5000-6999"], payload
    )


def test_temperature_bias_segment_temperatures_branch() -> None:
    payload = {
        "temperature": 1.1,
        "segment_temperatures": {"ladder:5000-6999": 0.9, "ranked:league-1": 1.4},
    }
    _assert_temperature_bias_matches(
        ["ladder:5000-6999", "ranked:league-1", "unknown-segment"], payload
    )


def test_temperature_bias_full_calibration_branch() -> None:
    payload = {
        "temperature": 1.0,
        # segment_temperatures present but must be ignored once calibration.segments exists.
        "segment_temperatures": {"ladder:5000-6999": 99.0},
        "calibration": {
            "global": {"temperature": 1.2, "bias": -0.05},
            "segments": {
                "ladder:5000-6999": {"temperature": 0.8, "bias": 0.1},
                "ranked:league-1": {"temperature": 1.5},  # bias falls back to global
            },
        },
    }
    _assert_temperature_bias_matches(["ladder:5000-6999", "ranked:league-1", "missing"], payload)


def test_noise_floor_irreducible_brier() -> None:
    # Matchup A wins 75% of its 4 games, matchup B wins 25% of its 4 games.
    targets = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    keys = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    noise_floor = compute_noise_floor(
        targets, keys, {"model": np.full(8, 0.5)}, min_support=4
    )

    assert noise_floor["supported_matchups"] == 2
    assert noise_floor["supported_rows"] == 8
    # Both matchups have rate p with p*(1-p) = 0.75 * 0.25 = 0.1875.
    assert abs(noise_floor["irreducible_brier"] - 0.1875) < 1e-9
    assert "oracle_in_sample" in noise_floor
    assert "model_on_supported" in noise_floor
    assert "model_brier_gap_to_floor" in noise_floor


def test_noise_floor_oracle_beats_constant() -> None:
    targets = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    keys = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    noise_floor = compute_noise_floor(
        targets, keys, {"model": np.full(8, 0.5)}, min_support=4
    )

    # Predicting the observed per-matchup rate must beat the uninformed 0.5.
    assert noise_floor["oracle_in_sample"]["brier_score"] < 0.25
    assert noise_floor["model_on_supported"]["brier_score"] >= noise_floor["irreducible_brier"]


def test_noise_floor_warns_without_support() -> None:
    targets = np.array([1, 0, 1, 0])
    keys = np.array(["A", "A", "B", "B"])

    noise_floor = compute_noise_floor(
        targets, keys, {"model": np.full(4, 0.5)}, min_support=5
    )

    assert noise_floor["supported_rows"] == 0
    assert "warning" in noise_floor
    assert "irreducible_brier" not in noise_floor
