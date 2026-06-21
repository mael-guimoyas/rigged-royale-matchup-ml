"""Sweep (torch threads, dataloader workers) to find max train it/s on this CPU.

Replicates the real train step (forward/backward/AdamW) so numbers transfer to
`rigged-matchup train`. Runtime torch.set_num_threads avoids per-config process
restarts; a fresh DataLoader per config respawns workers.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from rigged_matchup_ml.config import load_config
from rigged_matchup_ml.dataset import load_vocabulary, matchup_dataloader
from rigged_matchup_ml.model import SymmetricMatchupModel
from rigged_matchup_ml.trainer import _model_config, _to_device

CONFIGS = [(12, 2), (10, 2), (10, 4), (8, 4), (6, 6)]
WARMUP = 6
MEASURE = 24
BATCH = 2048


def bench(threads: int, workers: int, prepared_dir, vocab, model_cfg, device) -> float:
    torch.set_num_threads(threads)
    model = SymmetricMatchupModel(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = matchup_dataloader(
        prepared_dir / "train", vocab, shuffle=True, augment_swap=True,
        seed=42, batch_size=BATCH, num_workers=workers,
    )
    model.train()
    it = iter(loader)
    t0 = None
    done = 0
    for step in range(WARMUP + MEASURE):
        batch = _to_device(next(it), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch["target"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step == WARMUP:
            t0 = time.perf_counter()
        elif step >= WARMUP:
            done += 1
    elapsed = time.perf_counter() - t0
    del it, loader
    its = done / elapsed
    return its


def main() -> None:
    config = load_config("config/default.yaml")
    prepared_dir = config.resolve(config.data["prepared_dir"])
    vocab = load_vocabulary(prepared_dir)
    model_cfg = _model_config(config, vocab)
    device = torch.device("cpu")
    print(f"cpu_count torch_max={torch.get_num_threads()}  batch={BATCH}")
    results = []
    for threads, workers in CONFIGS:
        its = bench(threads, workers, prepared_dir, vocab, model_cfg, device)
        rows = its * BATCH
        results.append((threads, workers, its, rows))
        print(f"threads={threads:2d} workers={workers:2d} -> {its:5.3f} it/s  {rows:7.0f} rows/s  ({1/its:4.2f} s/it)")
    best = max(results, key=lambda r: r[2])
    print(f"\nBEST: threads={best[0]} workers={best[1]}  {best[2]:.3f} it/s  {best[3]:.0f} rows/s")


if __name__ == "__main__":
    main()
