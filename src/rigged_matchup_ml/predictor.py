from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .card_stats import metadata_for
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


def load_bundle(checkpoint_path: Path) -> dict[str, Any]:
    """Load a checkpoint once and attach an eval-ready model under ``model``.

    The returned bundle is the raw checkpoint payload (vocabulary, calibration,
    temperatures, model_config, ...) plus a ready ``SymmetricMatchupModel``. Pass
    it to :func:`predict_from_row` for each request so the model is loaded once.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SymmetricMatchupModel(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    payload["model"] = model
    return payload


def _calibration_for_segment(bundle: dict[str, Any], segment: Any) -> tuple[float, float, float]:
    """Resolve the (temperature, bias) to apply for ``segment`` plus the global temperature.

    Mirrors the per-segment logit calibration fitted in training: prefer a
    segment-specific (temperature, bias), then a segment temperature, then the
    global calibration.
    """
    global_temperature_raw = float(bundle["temperature"])
    segment_temperatures = bundle.get("segment_temperatures") or {}
    calibration = bundle.get("calibration") or {}
    global_calibration = calibration.get("global", {})
    segment_calibrations = calibration.get("segments") or {}
    global_temperature = float(global_calibration.get("temperature", global_temperature_raw))
    global_bias = float(global_calibration.get("bias", 0.0))
    segment_calibration = segment_calibrations.get(str(segment), {})
    calibrated_temperature = float(
        segment_calibration.get(
            "temperature",
            segment_temperatures.get(str(segment), global_temperature),
        )
    )
    calibrated_bias = float(segment_calibration.get("bias", global_bias))
    return calibrated_temperature, calibrated_bias, global_temperature_raw


def _repeat_batch(
    batch: dict[str, torch.Tensor], count: int
) -> dict[str, torch.Tensor]:
    return {
        key: value.expand(count, *value.shape[1:])
        for key, value in batch.items()
    }


def _ablation_contributions(
    model: SymmetricMatchupModel,
    batch: dict[str, torch.Tensor],
    baseline_logit: float,
    mask_key: str,
    pair_count: int,
    pair_indices: list[int],
) -> dict[int, float]:
    """Signed team-logit delta from removing one interaction pair at a time."""
    if not pair_indices:
        return {}
    repeated = _repeat_batch(batch, len(pair_indices))
    device = next(iter(batch.values())).device
    keep = torch.ones((len(pair_indices), pair_count), dtype=torch.bool, device=device)
    keep[torch.arange(len(pair_indices), device=device), pair_indices] = False
    repeated[mask_key] = keep
    with torch.no_grad():
        ablated = model(repeated)
    return {
        pair_index: float(baseline_logit - ablated[row].item())
        for row, pair_index in enumerate(pair_indices)
    }


def _cross_pairs(
    model: SymmetricMatchupModel,
    batch: dict[str, torch.Tensor],
    baseline_logit: float,
    mask_key: str,
    weights: torch.Tensor,
    source_ids: list[int],
    target_ids: list[int],
    source_valid: list[bool],
    target_valid: list[bool],
    top_k: int,
    direction: int,
) -> list[dict[str, Any]]:
    """Rank cross-deck pairs by signed ablation, preserving target win conditions.

    ``direction`` is +1 for the team's advantage channel and -1 for the opponent's
    advantage channel. Every target-side win condition receives one output slot;
    remaining slots require a contribution with the expected sign.
    """
    rows = min(len(source_ids), weights.shape[0])
    cols = min(len(target_ids), weights.shape[1])
    pairs: list[dict[str, Any]] = []
    for i in range(rows):
        if not source_valid[i]:
            continue
        for j in range(cols):
            if not target_valid[j]:
                continue
            pairs.append(
                {
                    "attention": float(weights[i, j].item()),
                    "flat_index": i * weights.shape[1] + j,
                    "source": int(source_ids[i]),
                    "target": int(target_ids[j]),
                    "target_position": j,
                }
            )
    if not pairs:
        return []

    required_targets = {
        position
        for position, card_id in enumerate(target_ids[:cols])
        if target_valid[position] and metadata_for(card_id)["role"] == "win_condition"
    }
    ranked_attention = sorted(pairs, key=lambda pair: pair["attention"], reverse=True)
    candidate_limit = max(12, top_k * 4)
    candidates = ranked_attention[:candidate_limit]
    candidates.extend(
        pair for pair in pairs if pair["target_position"] in required_targets
    )
    by_index = {pair["flat_index"]: pair for pair in candidates}
    candidates = list(by_index.values())
    contributions = _ablation_contributions(
        model,
        batch,
        baseline_logit,
        mask_key,
        int(weights.numel()),
        list(by_index),
    )
    for pair in candidates:
        pair["contribution"] = contributions[pair["flat_index"]]
        pair["directional_effect"] = direction * pair["contribution"]

    selected: list[dict[str, Any]] = []
    for target_position in sorted(required_targets):
        options = [
            pair for pair in candidates if pair["target_position"] == target_position
        ]
        if options:
            selected.append(
                max(
                    options,
                    key=lambda pair: (pair["directional_effect"], pair["attention"]),
                )
            )

    selected_indices = {pair["flat_index"] for pair in selected}
    positive = sorted(
        (
            pair
            for pair in candidates
            if pair["flat_index"] not in selected_indices
            and pair["directional_effect"] > 0
        ),
        key=lambda pair: (pair["directional_effect"], pair["attention"]),
        reverse=True,
    )
    desired = max(top_k, len(selected))
    selected.extend(positive[: max(0, desired - len(selected))])

    peak_attention = max(pair["attention"] for pair in pairs) or 1.0
    return [
        {
            "source_card_id": pair["source"],
            "target_card_id": pair["target"],
            "weight": round(pair["attention"] / peak_attention, 6),
            "contribution": round(pair["contribution"], 6),
        }
        for pair in selected
    ]


def _synergy_pairs(
    model: SymmetricMatchupModel,
    batch: dict[str, torch.Tensor],
    baseline_logit: float,
    weights: torch.Tensor,
    pair_positions: torch.Tensor,
    card_ids: list[int],
    valid: list[bool],
    top_k: int,
) -> list[dict[str, Any]]:
    """Positive intra-deck effects, ranked by signed leave-one-pair-out ablation."""
    first_positions = pair_positions[0]
    second_positions = pair_positions[1]
    pairs: list[dict[str, Any]] = []
    for k in range(weights.shape[0]):
        i = int(first_positions[k].item())
        j = int(second_positions[k].item())
        if i >= len(card_ids) or j >= len(card_ids) or not valid[i] or not valid[j]:
            continue
        pairs.append(
            {
                "attention": float(weights[k].item()),
                "flat_index": k,
                "source": int(card_ids[i]),
                "target": int(card_ids[j]),
            }
        )
    if not pairs:
        return []
    ranked_attention = sorted(pairs, key=lambda pair: pair["attention"], reverse=True)
    candidates = ranked_attention[: max(12, top_k * 4)]
    contributions = _ablation_contributions(
        model,
        batch,
        baseline_logit,
        "team_synergy_pair_keep",
        int(weights.numel()),
        [pair["flat_index"] for pair in candidates],
    )
    for pair in candidates:
        pair["contribution"] = contributions[pair["flat_index"]]
    selected = sorted(
        (pair for pair in candidates if pair["contribution"] > 0),
        key=lambda pair: (pair["contribution"], pair["attention"]),
        reverse=True,
    )[:top_k]
    peak_attention = max(pair["attention"] for pair in pairs) or 1.0
    return [
        {
            "source_card_id": pair["source"],
            "target_card_id": pair["target"],
            "weight": round(pair["attention"] / peak_attention, 6),
            "contribution": round(pair["contribution"], 6),
        }
        for pair in selected
    ]


def _interactions_for_row(
    bundle: dict[str, Any],
    forward: dict[str, torch.Tensor],
    baseline_logit: float,
    team_card_ids: list[int],
    opponent_card_ids: list[int],
    top_k: int,
) -> dict[str, list[dict[str, Any]]] | None:
    """Build role-aware interaction attributions with signed ablation effects."""
    model = bundle["model"]
    explain = getattr(model, "explain", None)
    if explain is None:
        return None
    maps = explain(forward)
    if not maps:
        return None

    team_present = forward.get("team_card_present", forward["team_cards"].ne(0))
    opponent_present = forward.get(
        "opponent_card_present", forward["opponent_cards"].ne(0)
    )
    team_valid = [bool(value) for value in team_present[0].tolist()]
    opponent_valid = [bool(value) for value in opponent_present[0].tolist()]

    answers: list[dict[str, Any]] = []
    threats: list[dict[str, Any]] = []
    if "cross_team_to_opponent" in maps:
        # team→opponent feeds the team's advantage: your card answers theirs.
        answers = _cross_pairs(
            model,
            forward,
            baseline_logit,
            "team_cross_pair_keep",
            maps["cross_team_to_opponent"][0],
            team_card_ids,
            opponent_card_ids,
            team_valid,
            opponent_valid,
            top_k,
            direction=1,
        )
        # opponent→team feeds the opponent's advantage: their card threatens yours.
        threats = _cross_pairs(
            model,
            forward,
            baseline_logit,
            "opponent_cross_pair_keep",
            maps["cross_opponent_to_team"][0],
            opponent_card_ids,
            team_card_ids,
            opponent_valid,
            team_valid,
            top_k,
            direction=-1,
        )

    synergies: list[dict[str, Any]] = []
    if "team_synergy" in maps and "synergy_pairs" in maps:
        synergies = _synergy_pairs(
            model,
            forward,
            baseline_logit,
            maps["team_synergy"][0],
            maps["synergy_pairs"],
            team_card_ids,
            team_valid,
            top_k,
        )

    if not answers and not threats and not synergies:
        return None
    return {"answers": answers, "threats": threats, "synergies": synergies}


def predict_from_row(
    bundle: dict[str, Any],
    row: dict[str, Any],
    include_interactions: bool = False,
    interactions_top_k: int = 3,
) -> dict[str, Any]:
    """Predict a single matchup from a fully-formed model row.

    ``row`` must carry every field :func:`encode_row` reads (the 8-length card /
    evolution / hero / role arrays, tower troop ids, segment, patch). ``matrix_prior``
    and ``win`` default sensibly. The reverse (opponent-as-team) pass reuses
    ``encode_row(..., swapped=True)`` so the antisymmetry guarantee holds.

    With ``include_interactions`` the result also carries an ``interactions`` block
    (``answers`` / ``threats`` / ``synergies``) selected by signed pair ablation,
    with attention retained as a secondary salience signal. Target-side win
    conditions receive a reserved explanation slot.
    """
    request = {**row}
    request.setdefault("matrix_prior", 0.5)
    request.setdefault("win", False)

    model = bundle["model"]
    vocabulary = bundle["vocabulary"]
    calibrated_temperature, calibrated_bias, global_temperature = _calibration_for_segment(
        bundle, request["segment"]
    )

    forward = _batchify(encode_row(request, vocabulary))
    reverse = _batchify(encode_row(request, vocabulary, swapped=True))
    with torch.no_grad():
        logit = model(forward)
        reverse_logit = model(reverse)
        probability = float(
            torch.sigmoid(logit / max(calibrated_temperature, 1e-4) + calibrated_bias).item()
        )
        reverse_probability = float(
            torch.sigmoid(
                reverse_logit / max(calibrated_temperature, 1e-4) + calibrated_bias
            ).item()
        )
        raw_probability = float(torch.sigmoid(logit).item())
        raw_reverse_probability = float(torch.sigmoid(reverse_logit).item())

    result: dict[str, Any] = {
        "team_win_probability": probability,
        "opponent_win_probability": reverse_probability,
        "raw_team_win_probability": raw_probability,
        "raw_opponent_win_probability": raw_reverse_probability,
        "matchup_label": matchup_label(probability),
        "symmetry_error": abs((probability + reverse_probability) - 1.0),
        "raw_symmetry_error": abs((raw_probability + raw_reverse_probability) - 1.0),
        "segment": request["segment"],
        "patch": request["patch"],
        "temperature": calibrated_temperature,
        "bias": calibrated_bias,
        "global_temperature": global_temperature,
    }

    if include_interactions:
        interactions = _interactions_for_row(
            bundle,
            forward,
            float(logit.item()),
            list(request["team_card_ids"][:8]),
            list(request["opponent_card_ids"][:8]),
            interactions_top_k,
        )
        if interactions is not None:
            result["interactions"] = interactions

    return result


def predict_payload(checkpoint_path: Path, input_path: Path) -> dict[str, Any]:
    bundle = load_bundle(checkpoint_path)
    request = json.loads(input_path.read_text(encoding="utf-8"))
    return predict_from_row(bundle, request)
