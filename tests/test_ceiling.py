import numpy as np

from rigged_matchup_ml.ceiling import (
    _capture_fraction,
    _per_matchup_rates,
    derive_diagnosis,
    format_ceiling_report,
)


def test_capture_fraction_reaches_one_at_ceiling() -> None:
    # baseline 0.25 (constant), ceiling 0.18 (floor); model AT the ceiling -> 1.0.
    assert _capture_fraction(0.18, 0.25, 0.18) == 1.0
    # model exactly at the baseline captures nothing.
    assert _capture_fraction(0.25, 0.25, 0.18) == 0.0
    # halfway between baseline and ceiling.
    assert abs(_capture_fraction(0.215, 0.25, 0.18) - 0.5) < 1e-9
    # degenerate interval -> None instead of dividing by zero.
    assert _capture_fraction(0.2, 0.2, 0.2) is None


def test_per_matchup_rates_marks_support() -> None:
    targets = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    # Dense integer ids: matchup A = 0, matchup B = 1.
    matchup_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    row_rate, supported_row, supported, observed = _per_matchup_rates(
        targets, matchup_ids, num_matchups=2, min_support=4
    )

    assert observed == 2
    assert supported == 2
    assert supported_row.all()
    # Matchup A wins 3/4, matchup B wins 1/4.
    assert np.allclose(row_rate[:4], 0.75)
    assert np.allclose(row_rate[4:], 0.25)


def test_diagnosis_flags_low_discrimination_and_coverage() -> None:
    weaknesses, recommendations = derive_diagnosis(
        {
            "auc_capture": 0.20,
            "model_auc": 0.56,
            "oracle_auc": 0.80,
            "gap_to_floor": 0.02,
            "irreducible_brier": 0.18,
            "brier_capture_vs_prior": 0.3,
            "calibration_slope": 1.0,
            "ece_quantile": 0.005,
            "coverage": 0.30,
            "supported_matchups": 10,
            "by_segment": {},
        }
    )

    areas = {weakness["area"] for weakness in weaknesses}
    assert "discrimination" in areas
    assert "coverage" in areas
    # Both high-severity gaps must produce ranked, contiguous recommendations.
    assert recommendations
    assert [r["priority"] for r in recommendations] == list(range(1, len(recommendations) + 1))
    assert any(r["area"] == "coverage" for r in recommendations)


def test_diagnosis_reports_good_when_near_ceiling() -> None:
    weaknesses, recommendations = derive_diagnosis(
        {
            "auc_capture": 0.90,
            "model_auc": 0.62,
            "oracle_auc": 0.635,
            "gap_to_floor": 0.001,
            "irreducible_brier": 0.18,
            "brier_capture_vs_prior": 0.95,
            "calibration_slope": 1.02,
            "ece_quantile": 0.004,
            "coverage": 0.92,
            "supported_matchups": 5000,
            "by_segment": {},
        }
    )

    severities = {weakness["area"]: weakness["severity"] for weakness in weaknesses}
    assert severities["discrimination"] == "good"
    assert severities["brier_gap"] == "good"
    assert severities["coverage"] == "good"
    # Near the ceiling, no architecture/data action should be pushed.
    assert recommendations == []


def test_format_report_handles_no_support_warning() -> None:
    report = {
        "split": "test",
        "rows": 10,
        "win_rate": 0.5,
        "min_support": 100,
        "warning": "No matchup reached min_support; lower --min-support to estimate a ceiling.",
    }
    text = format_ceiling_report(report)
    assert "PLAFOND THEORIQUE" in text
    assert "min_support" in text
