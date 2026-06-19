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


def predict_from_row(bundle: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Predict a single matchup from a fully-formed model row.

    ``row`` must carry every field :func:`encode_row` reads (the 8-length card /
    evolution / hero / role arrays, tower troop ids, segment, patch). ``matrix_prior``
    and ``win`` default sensibly. The reverse (opponent-as-team) pass reuses
    ``encode_row(..., swapped=True)`` so the antisymmetry guarantee holds.
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

    return {
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


def predict_payload(checkpoint_path: Path, input_path: Path) -> dict[str, Any]:
    bundle = load_bundle(checkpoint_path)
    request = json.loads(input_path.read_text(encoding="utf-8"))
    return predict_from_row(bundle, request)
