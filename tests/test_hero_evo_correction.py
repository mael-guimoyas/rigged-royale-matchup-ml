from __future__ import annotations

import pytest

from rigged_matchup_ml.hero_evo_correction import (
    CORRECTION_KEY,
    correction_profile,
    hero_evo_logit_adjustment,
)


def bundle() -> dict:
    return {
        CORRECTION_KEY: {
            "schema_version": 1,
            "application": "post_calibration_antisymmetric_logit",
            "shrinkage": 0.75,
            "coefficients_by_hero_card_id": {
                "26000014": 0.2,
                "26000000": -0.1,
            },
        }
    }


def row() -> dict:
    return {
        "team_card_ids": [26000014, 26000001],
        "team_evolution_levels": [0, 1],
        "team_hero_levels": [1, 0],
        "opponent_card_ids": [26000000, 26000002],
        "opponent_evolution_levels": [0, 0],
        "opponent_hero_levels": [1, 0],
    }


def test_correction_requires_hero_and_evo_on_the_same_side() -> None:
    assert hero_evo_logit_adjustment(bundle(), row()) == pytest.approx(0.15)

    no_team_evo = row()
    no_team_evo["team_evolution_levels"] = [0, 0]
    assert hero_evo_logit_adjustment(bundle(), no_team_evo) == 0.0


def test_correction_is_antisymmetric_and_accepts_legacy_hero_bits() -> None:
    original = row()
    original["opponent_evolution_levels"] = [2, 1]
    original["opponent_hero_levels"] = [0, 0]
    adjustment = hero_evo_logit_adjustment(bundle(), original)

    swapped = {
        "team_card_ids": original["opponent_card_ids"],
        "team_evolution_levels": original["opponent_evolution_levels"],
        "team_hero_levels": original["opponent_hero_levels"],
        "opponent_card_ids": original["team_card_ids"],
        "opponent_evolution_levels": original["team_evolution_levels"],
        "opponent_hero_levels": original["team_hero_levels"],
    }
    assert hero_evo_logit_adjustment(bundle(), swapped) == pytest.approx(-adjustment)


def test_legacy_checkpoint_without_correction_is_a_no_op() -> None:
    assert correction_profile({}) is None
    assert hero_evo_logit_adjustment({}, row()) == 0.0


def test_invalid_correction_is_rejected() -> None:
    invalid = bundle()
    invalid[CORRECTION_KEY]["shrinkage"] = 1.1
    with pytest.raises(ValueError, match="between 0 and 1"):
        correction_profile(invalid)
