from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AppConfig
from .dataset import MatchupIterableDataset, load_vocabulary
from .metrics import binary_metrics, binary_metrics_by_group
from .model import SymmetricMatchupModel


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _model_config(config: AppConfig, vocabulary: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "card_count": len(vocabulary["cards"]) + 1,
        "tower_count": len(vocabulary["towers"]) + 1,
        "segment_count": len(vocabulary["segments"]) + 1,
        "patch_count": len(vocabulary["patches"]) + 1,
        **config.model,
    }


def _loader(
    prepared_dir: Path,
    split: str,
    vocabulary: dict[str, dict[str, int]],
    config: AppConfig,
    training: bool,
) -> DataLoader:
    dataset = MatchupIterableDataset(
        prepared_dir / split,
        vocabulary,
        shuffle=training,
        augment_swap=training,
        seed=int(config.training["seed"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(config.training["batch_size"]),
        num_workers=int(config.training["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


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
        probabilities.append(torch.sigmoid(output / temperature).cpu().numpy())
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


def train_model(config: AppConfig) -> Path:
    seed = int(config.training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prepared_dir = config.resolve(config.data["prepared_dir"])
    vocabulary = load_vocabulary(prepared_dir)
    model_config = _model_config(config, vocabulary)
    device = _device(str(config.training["device"]))
    model = SymmetricMatchupModel(**model_config).to(device)
    train_loader = _loader(prepared_dir, "train", vocabulary, config, training=True)
    validation_loader = _loader(prepared_dir, "validation", vocabulary, config, training=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0

    for epoch in range(int(config.training["epochs"])):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Train {epoch + 1}")
        for batch in progress:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch)
                loss = loss_fn(logits, batch["target"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), float(config.training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{np.mean(losses[-100:]):.4f}")

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
    val_logits, _, val_targets = collect_predictions(model, validation_loader, device)
    temperature = fit_temperature(val_logits, val_targets)

    artifact_dir = config.resolve(config.training["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / "matchup-model.pt"
    torch.save(
        {
            "model_state": best_state,
            "model_config": model_config,
            "vocabulary": vocabulary,
            "temperature": temperature,
            "feature_version": 1,
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
    )
    metrics = binary_metrics(
        targets, probabilities, int(config.evaluation["calibration_bins"])
    )
    metrics["by_segment"] = binary_metrics_by_group(
        targets,
        probabilities,
        segments,
        int(config.evaluation["calibration_bins"]),
    )
    output = checkpoint_path.with_name("test-metrics.json")
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
