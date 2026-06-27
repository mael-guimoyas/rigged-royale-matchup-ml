from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import torch
from torch.utils.data import default_collate

from .config import AppConfig
from .dataset import (
    FEATURE_COLUMNS,
    _assemble_batch,
    _decode_batch,
    _EncodeContext,
    encode_row,
)
from .metrics import _scalar_metrics, binary_metrics, bootstrap_intervals
from .model import SymmetricMatchupModel
from .unseen_evaluation import matchup_key


BENCHMARK_COLUMNS = FEATURE_COLUMNS + ["team_deck_key", "opponent_deck_key"]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def compute_noise_floor(
    targets: np.ndarray,
    matchup_keys: np.ndarray,
    probability_sets: dict[str, np.ndarray],
    min_support: int,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Irreducible-error floor from observed per-matchup win-rates.

    With binary win/loss labels and a true matchup win-rate p, the lowest
    attainable per-game Brier is p*(1-p). Estimating p from each matchup's
    observed rate (support >= min_support) gives a floor the model cannot beat,
    plus an in-sample oracle that predicts that rate directly.
    """
    targets = targets.astype(np.float64)
    rows_total = int(targets.shape[0])
    unique_keys, inverse = np.unique(matchup_keys, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique_keys.shape[0])
    win_sums = np.bincount(inverse, weights=targets, minlength=unique_keys.shape[0])
    observed_rates = np.divide(
        win_sums, counts, out=np.full_like(win_sums, 0.5), where=counts > 0
    )
    supported_matchup = counts >= int(min_support)
    supported_row = supported_matchup[inverse]
    supported_rows = int(supported_row.sum())

    noise_floor: dict[str, Any] = {
        "min_support": int(min_support),
        "observed_matchups": int(unique_keys.shape[0]),
        "supported_matchups": int(supported_matchup.sum()),
        "supported_rows": supported_rows,
        "coverage": supported_rows / rows_total if rows_total else 0.0,
    }
    if supported_rows == 0:
        noise_floor["warning"] = (
            "No matchup reached min_support; lower min_support to estimate a noise floor."
        )
        return noise_floor

    row_observed_rate = observed_rates[inverse][supported_row]
    supported_targets = targets[supported_row]
    irreducible_brier = float(np.mean(row_observed_rate * (1.0 - row_observed_rate)))
    noise_floor["irreducible_brier"] = irreducible_brier
    noise_floor["oracle_in_sample"] = _scalar_metrics(supported_targets, row_observed_rate)

    for name, probabilities in probability_sets.items():
        supported_probabilities = probabilities[supported_row]
        entry = _scalar_metrics(supported_targets, supported_probabilities)
        if bootstrap_samples > 0:
            entry["confidence_intervals"] = bootstrap_intervals(
                supported_targets, supported_probabilities, bootstrap_samples, bootstrap_seed
            )
        noise_floor[f"{name}_on_supported"] = entry

    model_entry = noise_floor.get("model_on_supported")
    prior_entry = noise_floor.get("matrix_prior_on_supported")
    if model_entry is not None:
        noise_floor["model_brier_gap_to_floor"] = (
            float(model_entry["brier_score"]) - irreducible_brier
        )
        if prior_entry is not None:
            headroom = float(prior_entry["brier_score"]) - irreducible_brier
            captured = float(prior_entry["brier_score"]) - float(model_entry["brier_score"])
            noise_floor["brier_headroom_captured_vs_prior"] = (
                captured / headroom if headroom > 0 else None
            )
    return noise_floor


def _calibrated_probabilities(
    model: SymmetricMatchupModel,
    rows: list[dict],
    payload: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Apply the same per-segment temperature/bias calibration as evaluation."""
    temperature = float(payload["temperature"])
    segment_temperatures = payload.get("segment_temperatures") or {}
    calibration = payload.get("calibration") or {}
    global_calibration = calibration.get("global", {})
    global_temperature = float(global_calibration.get("temperature", temperature))
    global_bias = float(global_calibration.get("bias", 0.0))
    segment_calibrations = calibration.get("segments") or {}

    encoded = default_collate([encode_row(row, payload["vocabulary"]) for row in rows])
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(encoded)
    temperatures: list[float] = []
    biases: list[float] = []
    for row in rows:
        segment = str(row["segment"])
        if segment_calibrations:
            segment_calibration = segment_calibrations.get(segment, {})
            temperatures.append(float(segment_calibration.get("temperature", global_temperature)))
            biases.append(float(segment_calibration.get("bias", global_bias)))
        else:
            temperatures.append(float(segment_temperatures.get(segment, temperature)))
            biases.append(0.0)
    batch_temperatures = torch.tensor(temperatures, dtype=torch.float32, device=device).clamp_min(
        1e-4
    )
    batch_biases = torch.tensor(biases, dtype=torch.float32, device=device)
    return torch.sigmoid(logits / batch_temperatures + batch_biases).cpu().numpy()


def _temperature_bias_vectors(
    segments: list[str], payload: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row temperature/bias arrays, identical mapping to _calibrated_probabilities.

    Mapped per *unique* segment string (a handful of segments) then expanded, so it
    stays vectorised instead of doing a dict lookup per row.
    """
    temperature = float(payload["temperature"])
    segment_temperatures = payload.get("segment_temperatures") or {}
    calibration = payload.get("calibration") or {}
    global_calibration = calibration.get("global", {})
    global_temperature = float(global_calibration.get("temperature", temperature))
    global_bias = float(global_calibration.get("bias", 0.0))
    segment_calibrations = calibration.get("segments") or {}

    resolved: dict[str, tuple[float, float]] = {}
    for segment in set(segments):
        if segment_calibrations:
            segment_calibration = segment_calibrations.get(segment, {})
            resolved[segment] = (
                float(segment_calibration.get("temperature", global_temperature)),
                float(segment_calibration.get("bias", global_bias)),
            )
        else:
            resolved[segment] = (float(segment_temperatures.get(segment, temperature)), 0.0)
    count = len(segments)
    temperatures = np.fromiter((resolved[s][0] for s in segments), dtype=np.float32, count=count)
    biases = np.fromiter((resolved[s][1] for s in segments), dtype=np.float32, count=count)
    return temperatures, biases


@torch.no_grad()
def _calibrated_probabilities_batch(
    model: SymmetricMatchupModel,
    record_batch,
    payload: dict[str, Any],
    device: torch.device,
    context: _EncodeContext,
    sub_batch_size: int = 16_384,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Vectorised calibrated probabilities for a whole pyarrow RecordBatch.

    Replaces the per-row ``encode_row`` + ``default_collate`` path (single-thread
    Python, GPU-starving) with the training hot-path numpy encoders so the GPU is
    actually fed. Returns the probabilities, the decoded arrays (``win`` and
    ``matrix_prior`` are reused by the caller, so they need not be re-read) and the
    per-row segment strings. No swap augmentation is applied, so ``decoded['win']``
    is the row's true target.
    """
    decoded = _decode_batch(record_batch, context)
    count = int(decoded["win"].shape[0])
    segments = record_batch.column("segment").to_pylist()
    if count == 0:
        return np.empty(0, dtype=np.float32), decoded, segments
    no_swap = np.zeros(count, dtype=bool)
    assembled = _assemble_batch(decoded, no_swap, 0, count)
    temperatures, biases = _temperature_bias_vectors(segments, payload)

    probabilities = np.empty(count, dtype=np.float32)
    for start in range(0, count, sub_batch_size):
        stop = min(start + sub_batch_size, count)
        sub_batch = {key: value[start:stop].to(device) for key, value in assembled.items()}
        logits = model(sub_batch)
        temperature = (
            torch.from_numpy(temperatures[start:stop]).to(device).clamp_min(1e-4)
        )
        bias = torch.from_numpy(biases[start:stop]).to(device)
        probabilities[start:stop] = (
            torch.sigmoid(logits / temperature + bias).cpu().numpy().astype(np.float32)
        )
    return probabilities, decoded, segments


@torch.no_grad()
def benchmark_model(
    config: AppConfig,
    checkpoint_path: Path,
    split: str = "test",
    min_support: int = 100,
    batch_size: int = 16_384,
) -> dict[str, Any]:
    """Compare the model against meaningful baselines on the same split.

    Produces a single report so the model's numbers are anchored:
      - constant 0.5             (a model that knows nothing)
      - matrix_prior alone       (the empirical matrix baseline)
      - the trained model        (calibrated)
      - the irreducible noise floor: with binary win/loss labels and a true
        matchup win-rate p, the lowest attainable Brier per game is p*(1-p).
        We estimate p from each matchup's observed rate (support >= min_support)
        and report both the floor and an in-sample oracle that predicts it.

    If the model's Brier on supported matchups is already close to the floor,
    there is little discrimination left to extract and effort should move to
    coverage or data rather than architecture.
    """
    prepared_dir = config.resolve(config.data["prepared_dir"])
    artifact_dir = config.resolve(config.training["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_dir = prepared_dir / split
    if not split_dir.exists() or not list(split_dir.glob("*.parquet")):
        raise RuntimeError(f"No prepared Parquet files found in {split_dir}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = _device(str(config.training["device"]))
    model = SymmetricMatchupModel(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()

    model_probabilities: list[np.ndarray] = []
    prior_probabilities: list[float] = []
    targets: list[int] = []
    matchup_keys: list[str] = []

    dataset = pads.dataset(split_dir, format="parquet")
    scanner = dataset.scanner(columns=BENCHMARK_COLUMNS, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        rows = record_batch.to_pylist()
        if not rows:
            continue
        model_probabilities.append(_calibrated_probabilities(model, rows, payload, device))
        for row in rows:
            prior_probabilities.append(float(row["matrix_prior"]))
            targets.append(int(bool(row["win"])))
            matchup_keys.append(
                matchup_key(row["team_deck_key"], row["opponent_deck_key"])
            )

    target_array = np.asarray(targets, dtype=np.int64)
    model_array = np.concatenate(model_probabilities)
    prior_array = np.asarray(prior_probabilities, dtype=np.float64)
    constant_array = np.full_like(prior_array, 0.5)
    rows_total = int(target_array.shape[0])

    bootstrap_samples = int(config.evaluation.get("bootstrap_samples", 0))
    bootstrap_seed = int(config.training.get("seed", 0))
    bins = int(config.evaluation["calibration_bins"])

    report: dict[str, Any] = {
        "split": split,
        "rows": rows_total,
        "win_rate": float(target_array.mean()) if rows_total else 0.0,
        "calibrated": True,
        "models": {
            "model": binary_metrics(
                target_array,
                model_array,
                bins,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "matrix_prior": _scalar_metrics(target_array, prior_array),
            "constant_0.5": _scalar_metrics(target_array, constant_array),
        },
    }

    report["noise_floor"] = compute_noise_floor(
        target_array,
        np.asarray(matchup_keys),
        {"model": model_array, "matrix_prior": prior_array},
        int(min_support),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    report["min_support"] = int(min_support)

    output = artifact_dir / f"benchmark-{split}-report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
