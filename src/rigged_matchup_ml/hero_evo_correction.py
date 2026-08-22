from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

CORRECTION_KEY = "hero_evo_statistical_correction"

# Estimated on 100 counterfactual validation contexts per Hero and checked on
# 100 later contexts per Hero. These are the unshrunk logit offsets that centre
# each Hero's mean Evo interaction. Runtime applies only 75%: that held-out
# choice removed 77% of the systematic interaction while costing less than
# 0.001 AUC on 50,000 chronological factual battles.
V5_A9D92A10_SOURCE_SHA256 = (
    "a9d92a103d6f6cdf69ea08d9a6fda4bbe11802a609bca9221a765fe6e75004cc"
)
V5_A9D92A10_HERO_COEFFICIENTS: dict[str, float] = {
    "26000000": -0.12270391401388153,  # Knight
    "26000002": 0.121914931615236,  # Goblins
    "26000003": 0.003359766481694337,  # Giant
    "26000006": 0.06707349868822299,  # Balloon
    "26000011": 0.030388915875134596,  # Valkyrie
    "26000014": 0.049239217900792075,  # Musketeer
    "26000017": 0.01533883061650021,  # Wizard
    "26000018": 0.01688316802458945,  # Mini P.E.K.K.A
    "26000027": 0.01715341610725983,  # Dark Prince
    "26000034": 0.04233730216738984,  # Bowler
    "26000038": 0.2778820937775123,  # Ice Golem
    "26000039": 0.03646475300508411,  # Mega Minion
    "26000062": 0.10449587071166974,  # Magic Archer
    "26000102": 0.017602322437152963,  # Berserker
    "27000009": 0.002313526811369077,  # Tombstone
    "28000015": 0.19497032892987579,  # Barbarian Barrel
}


def v5_a9d92a10_profile(shrinkage: float = 0.75) -> dict[str, Any]:
    """Correction profile measured for the v5-a9d92a103d6f checkpoint only."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("Hero/Evo correction shrinkage must be between 0 and 1")
    return {
        "schema_version": 1,
        "application": "post_calibration_antisymmetric_logit",
        "source_checkpoint_sha256": V5_A9D92A10_SOURCE_SHA256,
        "shrinkage": float(shrinkage),
        "coefficients_by_hero_card_id": dict(V5_A9D92A10_HERO_COEFFICIENTS),
        "fit": {
            "validation_contexts_per_hero": 100,
            "chronological_test_contexts_per_hero": 100,
            "factual_test_rows": 50_000,
        },
    }


def attach_v5_a9d92a10_correction(
    source: Path,
    destination: Path,
    shrinkage: float = 0.75,
) -> Path:
    """Copy the measured checkpoint and embed its post-calibration correction."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Source and destination checkpoints must be different")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != V5_A9D92A10_SOURCE_SHA256:
        raise ValueError(
            "Correction profile is checkpoint-specific: expected SHA256 "
            f"{V5_A9D92A10_SOURCE_SHA256}, got {digest}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload[CORRECTION_KEY] = v5_a9d92a10_profile(shrinkage)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def correction_profile(bundle: dict[str, Any]) -> tuple[float, dict[str, float]] | None:
    """Validate and normalize an optional correction embedded in a checkpoint."""
    raw = bundle.get(CORRECTION_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported Hero/Evo statistical correction schema")
    if raw.get("application") != "post_calibration_antisymmetric_logit":
        raise ValueError("Unsupported Hero/Evo statistical correction application stage")
    shrinkage = float(raw.get("shrinkage", 0.0))
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("Hero/Evo correction shrinkage must be between 0 and 1")
    coefficients = raw.get("coefficients_by_hero_card_id")
    if not isinstance(coefficients, dict):
        raise TypeError("Hero/Evo correction coefficients are missing")
    return shrinkage, {str(card_id): float(value) for card_id, value in coefficients.items()}


def _side_correction(
    card_ids: list[int],
    evolution_levels: list[int],
    hero_levels: list[int],
    coefficients: dict[str, float],
) -> float:
    # Existing shards and requests may encode Hero in evolutionLevel bit 2 or in
    # the separate Hero array. Evolution is bit 1. Apply a Hero coefficient only
    # when an Evo exists elsewhere (or, defensively, anywhere) on the same side.
    has_evolution = any(max(0, int(level)) & 1 for level in evolution_levels[:8])
    if not has_evolution:
        return 0.0
    total = 0.0
    for index, card_id in enumerate(card_ids[:8]):
        raw_evolution = (
            max(0, int(evolution_levels[index])) if index < len(evolution_levels) else 0
        )
        explicit_hero = (
            max(0, int(hero_levels[index])) if index < len(hero_levels) else 0
        )
        if explicit_hero > 0 or raw_evolution & 2:
            total += coefficients.get(str(int(card_id)), 0.0)
    return total


def hero_evo_logit_adjustment(bundle: dict[str, Any], row: dict[str, Any]) -> float:
    """Return team-minus-opponent post-calibration logit adjustment."""
    profile = correction_profile(bundle)
    if profile is None:
        return 0.0
    shrinkage, coefficients = profile
    team = _side_correction(
        list(row["team_card_ids"]),
        list(row["team_evolution_levels"]),
        list(row["team_hero_levels"]),
        coefficients,
    )
    opponent = _side_correction(
        list(row["opponent_card_ids"]),
        list(row["opponent_evolution_levels"]),
        list(row["opponent_hero_levels"]),
        coefficients,
    )
    return shrinkage * (team - opponent)
