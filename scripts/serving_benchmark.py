"""Measure the throughput of the batch prediction path, per device and batch size.

Answers the only question that decides whether renting a GPU is worth it: how
many matchup rows per second `/predict/batch` can turn over, and how that total
splits between the CPU-side row encoder and the model's forward pass. The
encoder never moves to the GPU, so it is what caps the achievable speedup.

    python scripts/serving_benchmark.py --device cpu
    python scripts/serving_benchmark.py --device cuda --sizes 512,1024,2048,4096
    python scripts/serving_benchmark.py --device cuda --json artifacts/serving-benchmark.json

`--compare-legacy` additionally times the pre-vectorisation encoder (a Python
loop per row, run once per direction) so the gain from the current one is
measured rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rigged_matchup_ml.dataset import (
    encode_rows,
    encode_rows_both_directions,
)
from rigged_matchup_ml.predictor import (
    load_bundle,
    predict_from_rows,
    resolve_device,
)
from rigged_matchup_ml.serve import MatchupRequest, request_to_row

DEFAULT_SIZES = (128, 256, 512, 1024, 2048, 4096)
TOWER_PRINCESS_ID = 159000000


def _synthetic_rows(bundle: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    """Build `count` distinct site-shaped rows from the checkpoint's own vocabulary.

    Distinct decks matter: repeating one row would let caches and branch
    prediction flatter the encoder in a way a real meta sweep never does.
    """
    vocabulary = bundle["vocabulary"]
    cards = sorted(int(card) for card in vocabulary["cards"])
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(count):
        picked = rng.sample(cards, 16)
        request = MatchupRequest(
            team_card_ids=picked[:8],
            opponent_card_ids=picked[8:],
            mode_key="ladder",
            team_tower_troop_id=TOWER_PRINCESS_ID,
            opponent_tower_troop_id=TOWER_PRINCESS_ID,
            team_trophies=rng.choice([5200, 9000, 12500, 14500]),
            team_evolution_card_ids=picked[: rng.choice([0, 1, 2])],
            team_hero_card_ids=picked[2:3] if rng.random() < 0.3 else [],
        )
        row = request_to_row(request, bundle)
        row["win"] = False
        rows.append(row)
    return rows


def _timed(function, repeats: int, device: torch.device) -> float:
    """Median wall time of `function` over `repeats` runs, in seconds.

    The median rather than the mean because a single OS scheduling hiccup
    otherwise dominates a short measurement. CUDA work is queued
    asynchronously, so the device is synchronised before the clock is read or
    the numbers are meaningless.
    """
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    timings.sort()
    return timings[len(timings) // 2]


def _measure(
    bundle: dict[str, Any],
    rows: list[dict[str, Any]],
    device: torch.device,
    repeats: int,
    compare_legacy: bool,
) -> dict[str, Any]:
    vocabulary = bundle["vocabulary"]
    context = bundle.get("encode_context")
    count = len(rows)

    total = _timed(lambda: predict_from_rows(bundle, rows), repeats, device)
    encode = _timed(
        lambda: encode_rows_both_directions(rows, vocabulary, context), repeats, device
    )
    result = {
        "rows": count,
        "total_ms": round(total * 1000, 2),
        "encode_ms": round(encode * 1000, 2),
        "model_and_output_ms": round(max(0.0, total - encode) * 1000, 2),
        "encode_share": round(encode / total, 4) if total > 0 else None,
        "rows_per_second": round(count / total) if total > 0 else None,
    }
    if compare_legacy:
        def legacy() -> None:
            encode_rows(rows, vocabulary)
            encode_rows(rows, vocabulary, swapped=[True] * count)

        legacy_encode = _timed(legacy, repeats, device)
        result["legacy_encode_ms"] = round(legacy_encode * 1000, 2)
        result["encode_speedup"] = round(legacy_encode / encode, 2) if encode > 0 else None
    return result


def _run_sweep(bundle: dict[str, Any], device: torch.device, arguments) -> None:
    """Time a full sweep of `--sweep` matchups, in `--chunk`-sized batches.

    This is the question that actually matters for coverage: not how fast one
    batch is, but how long a panel of N opponents takes end to end. Rows are
    generated per chunk rather than all at once, both to keep host memory bounded
    at large N and because that is how the site produces them.
    """
    total_rows = arguments.sweep
    chunk = max(1, arguments.chunk)
    chunks = (total_rows + chunk - 1) // chunk

    print(f"\nsweep: {total_rows} matchups in {chunks} batches of up to {chunk}")
    generation = 0.0
    inference = 0.0
    done = 0
    started = time.perf_counter()
    for index in range(chunks):
        size = min(chunk, total_rows - done)
        mark = time.perf_counter()
        rows = _synthetic_rows(bundle, size, arguments.seed + index)
        generation += time.perf_counter() - mark

        mark = time.perf_counter()
        results = predict_from_rows(bundle, rows)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference += time.perf_counter() - mark
        assert len(results) == size
        done += size
        if chunks > 4 and (index + 1) % max(1, chunks // 4) == 0:
            elapsed = time.perf_counter() - started
            print(f"  {done:>7}/{total_rows}  {elapsed:6.1f} s  ({done / elapsed:,.0f} rows/s)")

    wall = time.perf_counter() - started
    print()
    print(f"  inference        {inference:8.2f} s  ({done / inference:,.0f} rows/s)")
    print(f"  row generation   {generation:8.2f} s  (benchmark-only, not a serving cost)")
    print(f"  wall clock       {wall:8.2f} s")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"  peak GPU memory  {peak:8.0f} MiB at chunk size {chunk}")

    if arguments.json:
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "device": bundle["device"],
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "sweep_rows": done,
            "chunk": chunk,
            "inference_seconds": round(inference, 3),
            "wall_seconds": round(wall, 3),
            "rows_per_second": round(done / inference) if inference > 0 else None,
        }
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {arguments.json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/matchup-model.pt"))
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, cuda:N, or auto. Defaults to MODEL_DEVICE / auto.",
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in DEFAULT_SIZES),
        help="Comma-separated batch sizes to sweep.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Also time the per-row encoder, to measure the vectorisation gain.",
    )
    parser.add_argument(
        "--sweep",
        type=int,
        default=None,
        help=(
            "Instead of the size table, time a whole sweep of this many matchups, "
            "processed in chunks of --chunk. This is the shape of a real meta panel."
        ),
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=2048,
        help="Rows per batch when running --sweep.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the results here.")
    arguments = parser.parse_args()

    device = resolve_device(arguments.device)
    bundle = load_bundle(arguments.checkpoint, device=device)
    sizes = [int(size) for size in arguments.sizes.split(",") if size.strip()]

    print(f"device={bundle['device']}  torch={torch.__version__}  threads={torch.get_num_threads()}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"checkpoint={arguments.checkpoint}  model_version={bundle['resolved_model_version']}")

    # One untimed pass so CUDA context creation and kernel loading are not
    # charged to the first measured batch.
    predict_from_rows(bundle, _synthetic_rows(bundle, 8, arguments.seed))

    if arguments.sweep:
        _run_sweep(bundle, device, arguments)
        return

    results = []
    header = f"{'rows':>6}  {'total ms':>9}  {'encode ms':>10}  {'model ms':>9}  {'rows/s':>8}  {'encode%':>8}"
    print(header)
    print("-" * len(header))
    for size in sizes:
        rows = _synthetic_rows(bundle, size, arguments.seed + size)
        measurement = _measure(bundle, rows, device, arguments.repeats, arguments.compare_legacy)
        results.append(measurement)
        share = measurement["encode_share"]
        print(
            f"{measurement['rows']:>6}  {measurement['total_ms']:>9}  "
            f"{measurement['encode_ms']:>10}  {measurement['model_and_output_ms']:>9}  "
            f"{measurement['rows_per_second']:>8}  {share * 100:>7.1f}%"
        )
        if arguments.compare_legacy:
            print(
                f"        legacy encoder {measurement['legacy_encode_ms']} ms "
                f"-> {measurement['encode_speedup']}x faster now"
            )

    if arguments.json:
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "device": bundle["device"],
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "model_version": bundle["resolved_model_version"],
            "repeats": arguments.repeats,
            "measurements": results,
        }
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {arguments.json}")


if __name__ == "__main__":
    main()
