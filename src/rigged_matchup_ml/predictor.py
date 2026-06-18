from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .dataset import encode_row
from .model import SymmetricMatchupModel


def _batchify(row: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in row.items()}


def matchup_label(probability: float) -> str:
    if probability < 0.40:
        return "very_bad"
    if probability < 0.45:
        return "bad"
    if probability <= 0.55:
        return "neutral"
    if probability <= 0.60:
        return "good"
    return "very_good"


def predict_payload(checkpoint_path: Path, input_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SymmetricMatchupModel(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    request = json.loads(input_path.read_text(encoding="utf-8"))
    request.setdefault("matrix_prior", 0.5)
    request.setdefault("win", False)
    encoded = _batchify(encode_row(request, payload["vocabulary"]))
    with torch.no_grad():
        probability = float(
            model.probability(encoded, temperature=float(payload["temperature"])).item()
        )
    reverse = {**request}
    reverse["team_card_ids"], reverse["opponent_card_ids"] = (
        request["opponent_card_ids"],
        request["team_card_ids"],
    )
    reverse["team_evolution_levels"], reverse["opponent_evolution_levels"] = (
        request["opponent_evolution_levels"],
        request["team_evolution_levels"],
    )
    reverse["team_hero_levels"], reverse["opponent_hero_levels"] = (
        request["opponent_hero_levels"],
        request["team_hero_levels"],
    )
    reverse["team_card_roles"], reverse["opponent_card_roles"] = (
        request["opponent_card_roles"],
        request["team_card_roles"],
    )
    reverse["team_tower_troop_id"], reverse["opponent_tower_troop_id"] = (
        request["opponent_tower_troop_id"],
        request["team_tower_troop_id"],
    )
    reverse["matrix_prior"] = 1.0 - request["matrix_prior"]
    reverse_encoded = _batchify(encode_row(reverse, payload["vocabulary"]))
    with torch.no_grad():
        reverse_probability = float(
            model.probability(
                reverse_encoded, temperature=float(payload["temperature"])
            ).item()
        )
    return {
        "team_win_probability": probability,
        "opponent_win_probability": reverse_probability,
        "matchup_label": matchup_label(probability),
        "symmetry_error": abs((probability + reverse_probability) - 1.0),
        "segment": request["segment"],
        "patch": request["patch"],
    }
