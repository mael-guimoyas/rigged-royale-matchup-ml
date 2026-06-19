import numpy as np

from rigged_matchup_ml.benchmark import compute_noise_floor


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
