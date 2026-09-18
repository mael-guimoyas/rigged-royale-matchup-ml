from __future__ import annotations

import pytest

from rigged_matchup_ml.training_weights import (
    effective_sample_size,
    segment_weights,
    weighting_report,
)


def _weighted_mean(counts: dict[str, int], weights: dict[str, float]) -> float:
    total = sum(counts.values())
    return sum(counts[segment] * weights[segment] for segment in counts) / total


def test_weights_keep_a_mean_of_one() -> None:
    """The loss scale -- and with it the effective learning rate -- must not move."""
    counts = {"ladder:5000-6999": 2_000, "ladder:12000-13999": 400_000, "ranked": 50_000}
    weights = segment_weights(counts)
    assert _weighted_mean(counts, weights) == pytest.approx(1.0)


def test_cap_bounds_the_weight_ratio_exactly() -> None:
    counts = {"starved": 10, "saturated": 1_000_000}
    weights = segment_weights(counts, power=0.3, cap=3.0)
    ratio = max(weights.values()) / min(weights.values())
    assert ratio == pytest.approx(3.0)
    # Uncapped, this pair would sit at (1e6/10)^0.3 ~ 31.6x.
    assert segment_weights(counts, power=0.3, cap=100.0)["starved"] / segment_weights(
        counts, power=0.3, cap=100.0
    )["saturated"] == pytest.approx(31.62, rel=1e-3)


def test_rarer_segment_gets_the_larger_weight() -> None:
    counts = {"a": 100, "b": 1_000, "c": 10_000}
    weights = segment_weights(counts)
    assert weights["a"] > weights["b"] > weights["c"]


def test_uniform_corpus_is_left_alone() -> None:
    counts = {"a": 500, "b": 500, "c": 500}
    weights = segment_weights(counts)
    assert all(weight == pytest.approx(1.0) for weight in weights.values())


def test_power_zero_disables_the_correction() -> None:
    counts = {"a": 10, "b": 1_000_000}
    weights = segment_weights(counts, power=0.0)
    assert all(weight == pytest.approx(1.0) for weight in weights.values())


def test_empty_and_zero_counts() -> None:
    assert segment_weights({}) == {}
    assert segment_weights({"a": 0}) == {}
    # A segment with no train rows is absent, not zero-weighted: the caller decides
    # what an unknown segment is worth (the trainer gives it 1).
    assert "a" not in segment_weights({"a": 0, "b": 10})


def test_effective_sample_size_is_the_row_count_when_weights_are_flat() -> None:
    counts = {"a": 300, "b": 700}
    assert effective_sample_size(counts, {"a": 1.0, "b": 1.0}) == pytest.approx(1000.0)


def test_effective_sample_size_falls_when_weights_spread() -> None:
    counts = {"a": 100, "b": 900}
    weights = segment_weights(counts, power=1.0, cap=50.0)
    assert effective_sample_size(counts, weights) < sum(counts.values())


def test_report_preserves_the_row_mass() -> None:
    counts = {"a": 1_000, "b": 100}
    wins = {"a": 500, "b": 50}
    report = weighting_report(counts, wins, segment_weights(counts))
    # Mean weight 1 means the weighted corpus is the same size as the real one.
    assert report["rows"] == 1_100
    assert sum(entry["weighted_share"] for entry in report["segments"]) == pytest.approx(1.0)


def test_report_lifts_the_starved_segment_share() -> None:
    counts = {"saturated": 900_000, "starved": 1_000}
    wins = {"saturated": 450_000, "starved": 500}
    report = weighting_report(counts, wins, segment_weights(counts))
    starved = next(entry for entry in report["segments"] if entry["segment"] == "starved")
    assert starved["weighted_share"] > starved["share"]
    assert report["warnings"] == []


def test_report_flags_a_collapsed_effective_sample_size() -> None:
    counts = {"saturated": 100_000, "starved": 100}
    wins = {"saturated": 50_000, "starved": 50}
    weights = segment_weights(counts, power=1.0, cap=100.0)
    report = weighting_report(counts, wins, weights)
    codes = {warning["code"] for warning in report["warnings"]}
    assert "effective_sample_size" in codes
    assert report["effective_sample_fraction"] < 0.5


def test_report_flags_a_win_rate_shift() -> None:
    """Segments disagreeing on the base rate is a collection problem, not a weighting one."""
    counts = {"a": 1_000, "b": 100}
    wins = {"a": 500, "b": 80}
    report = weighting_report(counts, wins, segment_weights(counts))
    codes = {warning["code"] for warning in report["warnings"]}
    assert "win_rate_shift" in codes
    assert report["win_rate"] == pytest.approx(580 / 1100)
    assert report["weighted_win_rate"] > report["win_rate"]


def test_even_segments_shift_nothing() -> None:
    counts = {"a": 1_000, "b": 100}
    wins = {"a": 500, "b": 50}
    report = weighting_report(counts, wins, segment_weights(counts))
    assert report["win_rate_shift"] == pytest.approx(0.0, abs=1e-9)
    assert report["warnings"] == []


def test_weights_cannot_see_the_labels() -> None:
    """Structural guarantee: identical counts give identical weights whatever the wins."""
    counts = {"a": 1_000, "b": 100}
    weights = segment_weights(counts)
    honest = weighting_report(counts, {"a": 500, "b": 50}, weights)
    skewed = weighting_report(counts, {"a": 900, "b": 10}, weights)
    assert [entry["weight"] for entry in honest["segments"]] == [
        entry["weight"] for entry in skewed["segments"]
    ]
    assert honest["effective_sample_size"] == skewed["effective_sample_size"]
