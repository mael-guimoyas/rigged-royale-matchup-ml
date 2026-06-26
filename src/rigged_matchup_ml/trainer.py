from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .card2vec import load_card2vec
from .card_stats import CARD_METADATA_VECTOR_SIZE
from .config import AppConfig
from .dataset import load_vocabulary, matchup_dataloader
from .metrics import binary_metrics, binary_metrics_by_group
from .model import SymmetricMatchupModel


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _estimate_optimizer_steps(
    prepared_dir: Path, config: AppConfig, accumulation_steps: int
) -> int:
    """Best-effort total optimizer-step count for the cosine schedule.

    Reads the prepared manifest written by `prepare`. Falls back to a single
    epoch's worth of guessed steps if the manifest is unavailable.
    """
    batch_size = max(1, int(config.training["batch_size"]))
    epochs = max(1, int(config.training["epochs"]))
    manifest_path = prepared_dir / "manifest.json"
    train_rows = 0
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_rows = int(manifest.get("counts", {}).get("train", 0))
    if train_rows <= 0:
        train_rows = batch_size * 1000
    micro_batches = math.ceil(train_rows / batch_size)
    optimizer_steps_per_epoch = math.ceil(micro_batches / max(1, accumulation_steps))
    # Partial micro-batches at each Parquet record-batch boundary make the real
    # step count slightly higher than this estimate. A small safety margin keeps
    # the cosine schedule from decaying to exactly 0 before the last epoch ends.
    return max(1, int(optimizer_steps_per_epoch * epochs * 1.1))


def _build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * max(0.0, warmup_fraction)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _maybe_init_card2vec(
    model: SymmetricMatchupModel,
    prepared_dir: Path,
    device: torch.device,
) -> str:
    """Warm-start the card embedding from self-supervised deck-co-occurrence vectors."""
    weight = model.deck_encoder.card_embedding.weight
    vectors = load_card2vec(prepared_dir, tuple(weight.shape))
    if vectors is None:
        return "skipped (missing or shape mismatch)"
    with torch.no_grad():
        weight.copy_(torch.tensor(vectors, dtype=weight.dtype, device=device))
    return f"loaded {list(vectors.shape)}"


def _card_weight_tensor(
    prepared_dir: Path,
    vocabulary: dict[str, dict[str, int]],
    power: float,
    cap: float,
    device: torch.device,
) -> torch.Tensor | None:
    """Per-vocab-index inverse-frequency weight, normalised so the frequency-weighted
    mean weight is 1 (keeps the loss scale and learning-rate behaviour unchanged)."""
    path = prepared_dir / "card_frequencies.json"
    if not path.exists():
        return None
    counts = json.loads(path.read_text(encoding="utf-8"))
    card_vocabulary = vocabulary["cards"]
    size = len(card_vocabulary) + 1
    weights = np.zeros(size, dtype=np.float64)
    count_vector = np.zeros(size, dtype=np.float64)
    for card_id, index in card_vocabulary.items():
        count = float(counts.get(str(card_id), 0))
        count_vector[index] = count
        weights[index] = (1.0 / count) ** power if count > 0 else 0.0
    weighted_total = float((count_vector * weights).sum())
    if weighted_total > 0:
        weights *= float(count_vector.sum()) / weighted_total
    weights = np.clip(weights, 0.0, cap)
    weights[0] = 0.0  # padding index carries no weight
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _weighted_bce(
    loss_none: nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
    card_weight: torch.Tensor,
    team_cards: torch.Tensor,
    opponent_cards: torch.Tensor,
) -> torch.Tensor:
    """BCE re-weighted per sample by the rarity of the 16 cards in the matchup."""
    pair_weights = torch.cat(
        [card_weight[team_cards], card_weight[opponent_cards]], dim=1
    )
    valid = pair_weights > 0
    sample_weight = pair_weights.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return (loss_none(logits, target) * sample_weight).mean()


def _model_config(config: AppConfig, vocabulary: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "card_count": len(vocabulary["cards"]) + 1,
        "tower_count": len(vocabulary["towers"]) + 1,
        "segment_count": len(vocabulary["segments"]) + 1,
        "patch_count": len(vocabulary["patches"]) + 1,
        "card_metadata_dim": int(
            config.model.get("card_metadata_dim", CARD_METADATA_VECTOR_SIZE)
        ),
        **config.model,
    }


def _loader(
    prepared_dir: Path,
    split: str,
    vocabulary: dict[str, dict[str, int]],
    config: AppConfig,
    training: bool,
) -> DataLoader:
    batch_size = int(config.training["batch_size"])
    if not training:
        batch_size = int(config.training.get("evaluation_batch_size", batch_size))
    return matchup_dataloader(
        prepared_dir / split,
        vocabulary,
        shuffle=training,
        augment_swap=training,
        seed=int(config.training["seed"]),
        batch_size=batch_size,
        num_workers=int(config.training["num_workers"]),
    )


def _segment_temperature_tensor(
    segment_ids: torch.Tensor,
    vocabulary: dict[str, dict[str, int]],
    fallback_temperature: float,
    segment_temperatures: dict[str, float] | None,
    device: torch.device,
) -> torch.Tensor:
    if not segment_temperatures:
        return torch.full_like(segment_ids, float(fallback_temperature), dtype=torch.float32)
    segment_names = {value: key for key, value in vocabulary["segments"].items()}
    values = [
        float(segment_temperatures.get(segment_names.get(int(segment_id), ""), fallback_temperature))
        for segment_id in segment_ids.detach().cpu().tolist()
    ]
    return torch.tensor(values, dtype=torch.float32, device=device).clamp_min(1e-4)


def _calibration_tensors(
    segment_ids: torch.Tensor,
    vocabulary: dict[str, dict[str, int]],
    calibration: dict[str, Any] | None,
    fallback_temperature: float,
    segment_temperatures: dict[str, float] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if calibration:
        global_calibration = calibration.get("global", {})
        global_temperature = float(
            global_calibration.get("temperature", fallback_temperature)
        )
        global_bias = float(global_calibration.get("bias", 0.0))
        segment_calibrations = calibration.get("segments") or {}
        segment_names = {value: key for key, value in vocabulary["segments"].items()}
        temperatures: list[float] = []
        biases: list[float] = []
        for segment_id in segment_ids.detach().cpu().tolist():
            segment = segment_names.get(int(segment_id), "")
            segment_calibration = segment_calibrations.get(segment, {})
            temperatures.append(
                float(segment_calibration.get("temperature", global_temperature))
            )
            biases.append(float(segment_calibration.get("bias", global_bias)))
        return (
            torch.tensor(temperatures, dtype=torch.float32, device=device).clamp_min(1e-4),
            torch.tensor(biases, dtype=torch.float32, device=device),
        )

    temperatures = _segment_temperature_tensor(
        segment_ids,
        vocabulary,
        fallback_temperature,
        segment_temperatures,
        device,
    )
    return temperatures, torch.zeros_like(temperatures)


@torch.no_grad()
def collect_predictions(
    model: SymmetricMatchupModel,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in tqdm(loader, desc="Evaluate", leave=False):
        batch = _to_device(batch, device)
        output = model(batch)
        logits.append(output.cpu().numpy())
        probabilities.append(torch.sigmoid(output / temperature).cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(probabilities), np.concatenate(targets)


@torch.no_grad()
def collect_predictions_with_segments(
    model: SymmetricMatchupModel,
    loader: DataLoader,
    device: torch.device,
    vocabulary: dict[str, dict[str, int]],
    temperature: float = 1.0,
    segment_temperatures: dict[str, float] | None = None,
    calibration: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    segments: list[np.ndarray] = []
    segment_names = {value: key for key, value in vocabulary["segments"].items()}
    for batch in tqdm(loader, desc="Evaluate", leave=False):
        batch = _to_device(batch, device)
        output = model(batch)
        logits.append(output.cpu().numpy())
        batch_temperatures, batch_biases = _calibration_tensors(
            batch["segment"],
            vocabulary,
            calibration,
            temperature,
            segment_temperatures,
            device,
        )
        probabilities.append(
            torch.sigmoid(output / batch_temperatures + batch_biases).cpu().numpy()
        )
        targets.append(batch["target"].cpu().numpy())
        segment_ids = batch["segment"].cpu().numpy()
        segments.append(
            np.asarray([segment_names.get(int(value), "<unknown>") for value in segment_ids])
        )
    return (
        np.concatenate(logits),
        np.concatenate(probabilities),
        np.concatenate(targets),
        np.concatenate(segments),
    )


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(logits_tensor / log_temperature.exp(), targets_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 10.0).item())


def fit_logit_calibration(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    log_temperature = nn.Parameter(torch.zeros(()))
    bias = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature, bias], lr=0.05, max_iter=100)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        calibrated = logits_tensor / log_temperature.exp() + bias
        loss = loss_fn(calibrated, targets_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "temperature": float(log_temperature.exp().clamp(0.05, 10.0).item()),
        "bias": float(bias.clamp(-5.0, 5.0).item()),
    }


def fit_segment_temperatures(
    logits: np.ndarray,
    targets: np.ndarray,
    segments: np.ndarray,
    min_rows: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    segment_values = segments.astype(str)
    for segment in sorted(set(segment_values.tolist())):
        mask = segment_values == segment
        if int(mask.sum()) < min_rows or len(np.unique(targets[mask])) < 2:
            continue
        result[segment] = fit_temperature(logits[mask], targets[mask])
    return result


def fit_segment_calibrations(
    logits: np.ndarray,
    targets: np.ndarray,
    segments: np.ndarray,
    min_rows: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    segment_values = segments.astype(str)
    for segment in sorted(set(segment_values.tolist())):
        mask = segment_values == segment
        if int(mask.sum()) < min_rows or len(np.unique(targets[mask])) < 2:
            continue
        result[segment] = fit_logit_calibration(logits[mask], targets[mask])
    return result


def train_model(config: AppConfig) -> Path:
    seed = int(config.training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    prepared_dir = config.resolve(config.data["prepared_dir"])
    vocabulary = load_vocabulary(prepared_dir)
    model_config = _model_config(config, vocabulary)
    device = _device(str(config.training["device"]))
    model = SymmetricMatchupModel(**model_config).to(device)
    if bool(config.training.get("card2vec_init", False)):
        card2vec_dir = config.training.get("card2vec_path")
        card2vec_dir = config.resolve(card2vec_dir) if card2vec_dir else prepared_dir
        status = _maybe_init_card2vec(model, card2vec_dir, device)
        print(json.dumps({"card2vec_init": status}))
    train_loader = _loader(prepared_dir, "train", vocabulary, config, training=True)
    validation_loader = _loader(prepared_dir, "validation", vocabulary, config, training=False)
    optimizer_options = {
        "lr": float(config.training["learning_rate"]),
        "weight_decay": float(config.training["weight_decay"]),
    }
    if device.type == "cuda":
        optimizer_options["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
    except (TypeError, AttributeError, RuntimeError):
        # Some torch builds reject `fused` (TypeError) or break importing the
        # fused kernel's lazy torch._dynamo dependency (AttributeError on e.g.
        # Kaggle images). Fall back to the plain optimizer, which is unaffected.
        optimizer_options.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_none = nn.BCEWithLogitsLoss(reduction="none")
    card_weight: torch.Tensor | None = None
    if bool(config.training.get("loss_card_frequency_weighting", False)):
        frequencies_dir = config.training.get("card_frequencies_path")
        frequencies_dir = config.resolve(frequencies_dir) if frequencies_dir else prepared_dir
        card_weight = _card_weight_tensor(
            frequencies_dir,
            vocabulary,
            float(config.training.get("loss_frequency_power", 0.5)),
            float(config.training.get("loss_frequency_cap", 5.0)),
            device,
        )
        print(json.dumps({"loss_card_frequency_weighting": card_weight is not None}))
    label_smoothing = float(config.training.get("label_smoothing", 0.0))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation_steps = max(1, int(config.training.get("gradient_accumulation_steps", 1)))
    total_optimizer_steps = _estimate_optimizer_steps(prepared_dir, config, accumulation_steps)
    scheduler = _build_scheduler(
        optimizer,
        total_optimizer_steps,
        float(config.training.get("warmup_fraction", 0.05)),
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0

    def optimizer_step() -> None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.parameters(), float(config.training["gradient_clip_norm"])
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(config.training["epochs"])):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Train {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        pending_gradients = 0
        for batch in progress:
            batch = _to_device(batch, device)
            target = batch["target"]
            if label_smoothing > 0.0:
                target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch)
                if card_weight is not None:
                    loss = _weighted_bce(
                        loss_none,
                        logits,
                        target,
                        card_weight,
                        batch["team_cards"],
                        batch["opponent_cards"],
                    )
                else:
                    loss = loss_fn(logits, target)
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            pending_gradients += 1
            if pending_gradients >= accumulation_steps:
                optimizer_step()
                pending_gradients = 0
            losses.append(float(loss.item()))
            progress.set_postfix(
                loss=f"{np.mean(losses[-100:]):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        if pending_gradients:
            optimizer_step()

        val_logits, val_probabilities, val_targets = collect_predictions(
            model, validation_loader, device
        )
        val_metrics = binary_metrics(
            val_targets, val_probabilities, int(config.evaluation["calibration_bins"])
        )
        print(json.dumps({"epoch": epoch + 1, "validation": val_metrics}, indent=2))
        if val_metrics["log_loss"] < best_loss:
            best_loss = val_metrics["log_loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(config.training["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    val_logits, _, val_targets, val_segments = collect_predictions_with_segments(
        model, validation_loader, device, vocabulary, temperature=1.0
    )
    temperature = fit_temperature(val_logits, val_targets)
    segment_temperatures = fit_segment_temperatures(
        val_logits,
        val_targets,
        val_segments,
        int(config.training.get("segment_calibration_min_rows", 2000)),
    )
    calibration = {
        "global": fit_logit_calibration(val_logits, val_targets),
        "segments": fit_segment_calibrations(
            val_logits,
            val_targets,
            val_segments,
            int(config.training.get("segment_calibration_min_rows", 2000)),
        ),
        "mode": "logit_temperature_bias_by_segment",
    }

    artifact_dir = config.resolve(config.training["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / "matchup-model.pt"
    torch.save(
        {
            "model_state": best_state,
            "model_config": model_config,
            "vocabulary": vocabulary,
            "temperature": temperature,
            "segment_temperatures": segment_temperatures,
            "calibration": calibration,
            "segment_calibration_min_rows": int(
                config.training.get("segment_calibration_min_rows", 2000)
            ),
            # Bucket edges so the inference server can resolve a request's segment
            # (domain.segment_for) without re-reading the training config.
            "data_config": {
                "trophy_buckets": list(config.data["trophy_buckets"]),
                "top_ladder_buckets": list(config.data["top_ladder_buckets"]),
            },
            "feature_version": 5,
        },
        checkpoint,
    )
    return checkpoint


def evaluate_checkpoint(config: AppConfig, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = _device(str(config.training["device"]))
    model = SymmetricMatchupModel(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    prepared_dir = config.resolve(config.data["prepared_dir"])
    test_loader = _loader(prepared_dir, "test", payload["vocabulary"], config, training=False)
    _, probabilities, targets, segments = collect_predictions_with_segments(
        model,
        test_loader,
        device,
        payload["vocabulary"],
        float(payload["temperature"]),
        payload.get("segment_temperatures"),
        payload.get("calibration"),
    )
    bootstrap_samples = int(config.evaluation.get("bootstrap_samples", 0))
    bootstrap_seed = int(config.training.get("seed", 0))
    # Bootstrap CIs only matter on small splits; on the full test set (millions
    # of rows) the point estimate is already razor-sharp and 300 resamples x AUC
    # sort over the whole array is minutes of pointless CPU work. Auto-skip above
    # the threshold (0 disables the guard, keeping the old behaviour).
    bootstrap_max_rows = int(config.evaluation.get("bootstrap_max_rows", 250_000))
    overall_bootstrap = bootstrap_samples
    if bootstrap_max_rows and len(targets) > bootstrap_max_rows:
        overall_bootstrap = 0
    metrics = binary_metrics(
        targets,
        probabilities,
        int(config.evaluation["calibration_bins"]),
        bootstrap_samples=overall_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    metrics["by_segment"] = binary_metrics_by_group(
        targets,
        probabilities,
        segments,
        int(config.evaluation["calibration_bins"]),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_max_rows=bootstrap_max_rows,
    )
    output = checkpoint_path.with_name("test-metrics.json")
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
