import pytest

from scripts.optimizer_variance_benchmark import (
    DeckStat,
    matchup_sigma,
    orientation_balanced_shrunk_mean,
)


def deck_stat() -> DeckStat:
    return DeckStat(
        segment="ladder:9000-11999",
        key="1,2,3,4,5,6,7,8",
        cards=(1, 2, 3, 4, 5, 6, 7, 8),
        train_n=100,
        validation_n=100,
        validation_wins=30,
        validation_team_n=20,
        validation_team_wins=16,
        validation_opponent_n=80,
        validation_opponent_wins=14,
        test_n=100,
        test_wins=50,
        test_team_n=50,
        test_team_wins=25,
        test_opponent_n=50,
        test_opponent_wins=25,
        train_rank=1,
    )


def test_orientation_balanced_mean_is_invariant_when_sides_are_swapped() -> None:
    direct = orientation_balanced_shrunk_mean(16, 20, 14, 80, 0.5, 25)
    swapped = orientation_balanced_shrunk_mean(14, 80, 16, 20, 0.5, 25)
    assert direct == swapped


def test_observed_rate_does_not_pool_an_imbalanced_orientation_mix() -> None:
    stat = deck_stat()
    pooled = (stat.validation_wins + 0.5 * 25) / (stat.validation_n + 25)
    assert stat.observed("validation", 25) != pooled
    assert stat.observed("validation", 25) > pooled


def test_missing_orientation_shrinks_to_the_neutral_prior() -> None:
    value = orientation_balanced_shrunk_mean(8, 10, 0, 0, 0.5, 20)
    assert value == 0.575


def test_corrected_panel_score_balances_residuals_by_orientation() -> None:
    stat = deck_stat()
    stat.model_score = 0.6
    stat.validation_team_residual_sum = 2.0
    stat.validation_team_residual_n = 10
    stat.validation_opponent_residual_sum = -1.0
    stat.validation_opponent_residual_n = 40

    expected_residual = orientation_balanced_shrunk_mean(2.0, 10, -1.0, 40, 0.0, 20)
    assert stat.corrected_panel_score("validation", 20) == 0.6 + expected_residual


def test_matchup_sigma_uses_the_weighted_second_moment() -> None:
    assert matchup_sigma(0.5, 0.29) == pytest.approx(0.2)
    assert matchup_sigma(0.5, 0.25 - 1e-15) == 0.0
