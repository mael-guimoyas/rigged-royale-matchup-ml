"""Break down the time from process start to first served prediction.

This is the part of a RunPod cold start that the image controls. FlashBoot can
remove the scheduling and image-pull time around it, but it cannot remove the
seconds this script measures — importing torch, creating the CUDA context,
loading the checkpoint and warming the kernels all happen inside the container
every time a worker starts.

Run it as its own process; timings are meaningless in a warm interpreter:

    python scripts/startup_profile.py --device cuda
    python scripts/startup_profile.py --device cuda --json artifacts/startup-profile.json

The interpreter-start figure is taken from the process creation time, so it
includes what Python spent before this file's first line ran.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Clock the imports themselves, so `import torch` is inside the measurement
# rather than already paid for by the time main() runs.
_SCRIPT_ENTERED = time.perf_counter()

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_IMPORT_STARTED = time.perf_counter()
import torch

_TORCH_IMPORTED = time.perf_counter()

from rigged_matchup_ml.predictor import (
    load_bundle,
    predict_from_rows,
    resolve_device,
    warm_up_bundle,
)
from rigged_matchup_ml.serve import MatchupRequest, request_to_row

_PACKAGE_IMPORTED = time.perf_counter()


def _process_start_seconds() -> float | None:
    """Seconds between process creation and this module's first statement.

    Uses psutil when available; otherwise the number is simply not reported
    rather than guessed, since a wrong figure here would misattribute the whole
    interpreter startup.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        return time.time() - psutil.Process(os.getpid()).create_time()
    except Exception:  # noqa: BLE001 - diagnostics only, never worth failing on
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/matchup-model.pt"))
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:N, or auto.")
    parser.add_argument(
        "--batch",
        type=int,
        default=512,
        help="Size of the first served batch, timed separately from the warm-up.",
    )
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    interpreter_start = _process_start_seconds()
    marks: list[tuple[str, float]] = [
        ("import torch", _TORCH_IMPORTED - _IMPORT_STARTED),
        ("import rigged_matchup_ml", _PACKAGE_IMPORTED - _TORCH_IMPORTED),
    ]

    start = time.perf_counter()
    device = resolve_device(arguments.device)
    if device.type == "cuda":
        torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
    marks.append(("create CUDA context", time.perf_counter() - start))

    start = time.perf_counter()
    bundle = load_bundle(arguments.checkpoint, device=device)
    marks.append(("load checkpoint", time.perf_counter() - start))

    start = time.perf_counter()
    warm_up_bundle(bundle)
    marks.append(("warm up", time.perf_counter() - start))

    # The first real batch after the warm-up: what the first request to reach a
    # freshly started worker would actually wait for.
    rows = []
    cards = sorted(int(card) for card in bundle["vocabulary"]["cards"])
    for index in range(arguments.batch):
        offset = index % max(1, len(cards) - 16)
        picked = cards[offset : offset + 16]
        if len(picked) < 16:
            picked = cards[:16]
        request = MatchupRequest(
            team_card_ids=picked[:8],
            opponent_card_ids=picked[8:16],
            mode_key="ladder",
            team_trophies=9000,
        )
        rows.append(dict(request_to_row(request, bundle), win=False))
    start = time.perf_counter()
    predict_from_rows(bundle, rows)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    marks.append((f"first batch of {arguments.batch}", time.perf_counter() - start))

    total = sum(duration for _, duration in marks)
    if interpreter_start is not None:
        before_script = max(0.0, interpreter_start - (time.perf_counter() - _SCRIPT_ENTERED))
        marks.insert(0, ("interpreter start", before_script))
        total += before_script

    print(f"device={bundle['device']}  torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print()
    width = max(len(label) for label, _ in marks)
    for label, duration in marks:
        print(f"{label:<{width}}  {duration * 1000:8.0f} ms  {duration / total * 100:5.1f}%")
    print(f"{'TOTAL to first prediction':<{width}}  {total * 1000:8.0f} ms")
    if interpreter_start is None:
        print("\n(interpreter start not measured: pip install psutil to include it)")

    if arguments.json:
        payload: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "device": bundle["device"],
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "batch": arguments.batch,
            "stages_ms": {label: round(duration * 1000, 1) for label, duration in marks},
            "total_ms": round(total * 1000, 1),
        }
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {arguments.json}")


if __name__ == "__main__":
    main()
