from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

from rigged_matchup_ml.predictor import load_bundle, predict_from_rows

ROOT = Path(__file__).resolve().parents[1]


POOLED_SEGMENT_SQL = """
case
  when segment like 'ranked:league-%' then
    case
      when cast(split_part(segment, '-', 2) as integer) between 1 and 2
        then 'ranked:league-1-2'
      when cast(split_part(segment, '-', 2) as integer) between 3 and 4
        then 'ranked:league-3-4'
      when cast(split_part(segment, '-', 2) as integer) between 5 and 7
        then 'ranked:league-5-7'
      else segment
    end
  else segment
end
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare biased and exactly antisymmetric calibrations."
    )
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "matchup-model.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "symmetric-calibration-benchmark.json",
    )
    parser.add_argument("--per-segment", type=int, default=62_500)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def raw_glob(raw_dir: Path) -> str:
    return (raw_dir.resolve() / "*.parquet").as_posix().replace("'", "''")


def metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    target = targets.astype(np.float64)
    return {
        "brier": float(np.mean((clipped - target) ** 2)),
        "log_loss": float(
            np.mean(-(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))
        ),
        "mean_probability": float(np.mean(clipped)),
        "observed_rate": float(np.mean(target)),
        "bias": float(np.mean(clipped - target)),
    }


def temperature_for(bundle: dict[str, Any], segment: str) -> float:
    return max(
        1e-4,
        float(
            (bundle.get("segment_temperatures") or {}).get(segment, bundle.get("temperature", 1.0))
        ),
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    parquet = raw_glob(args.raw)
    connection = duckdb.connect()
    connection.execute(f"set threads={max(1, args.threads)}")
    validation_cutoff = float(
        connection.execute(
            f"select quantile_cont(epoch(battle_time), 0.85) from read_parquet('{parquet}')"
        ).fetchone()[0]
    )
    query = f"""
      with test as (
        select *, {POOLED_SEGMENT_SQL} as pooled_segment
        from read_parquet('{parquet}')
        where epoch(battle_time) > {validation_cutoff}
          and segment not in ('ladder:top-100', 'ladder:top-1000')
      ), sampled as (
        select *
        from test
        qualify row_number() over (
          partition by pooled_segment order by hash(game_id)
        ) <= {max(1, args.per_segment)}
      )
      select win, pooled_segment as segment, patch,
             team_card_ids, opponent_card_ids,
             team_evolution_levels, opponent_evolution_levels,
             team_hero_levels, opponent_hero_levels,
             team_card_roles, opponent_card_roles,
             team_tower_troop_id, opponent_tower_troop_id,
             matrix_prior
      from sampled
      order by pooled_segment, game_id
    """
    cursor = connection.execute(query)
    columns = [description[0] for description in cursor.description]
    bundle = load_bundle(args.checkpoint.resolve())
    bundle["model"].to(device)
    values: dict[str, list[float]] = defaultdict(list)
    targets: list[int] = []
    segments: list[str] = []
    completed = 0
    while batch := cursor.fetchmany(args.batch_size):
        rows = [dict(zip(columns, row, strict=True)) for row in batch]
        predictions = predict_from_rows(bundle, rows)
        for row, prediction in zip(rows, predictions, strict=True):
            raw = min(1.0 - 1e-7, max(1e-7, float(prediction["raw_team_win_probability"])))
            raw_logit = math.log(raw / (1.0 - raw))
            symmetric = 1.0 / (
                1.0 + math.exp(-raw_logit / temperature_for(bundle, str(row["segment"])))
            )
            projected = 0.5 * (
                float(prediction["pre_projection_team_win_probability"])
                + 1.0
                - float(prediction["pre_projection_opponent_win_probability"])
            )
            values["biased"].append(float(prediction["pre_projection_team_win_probability"]))
            values["projected"].append(projected)
            values["temperature_only"].append(symmetric)
            values["raw"].append(raw)
            targets.append(int(bool(row["win"])))
            segments.append(str(row["segment"]))
        completed += len(batch)
        if completed == len(batch) or completed % (args.batch_size * 20) == 0:
            print(f"predictions={completed:,}", flush=True)

    target_array = np.asarray(targets, dtype=np.int8)
    segment_array = np.asarray(segments)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint.resolve()),
        "rows": len(targets),
        "validation_cutoff_epoch": validation_cutoff,
        "per_segment_limit": args.per_segment,
        "device": str(device),
        "overall": {},
        "by_segment": {},
    }
    arrays = {name: np.asarray(items) for name, items in values.items()}
    for name, probabilities in arrays.items():
        report["overall"][name] = metrics(target_array, probabilities)
    for segment in sorted(set(segments)):
        mask = segment_array == segment
        report["by_segment"][segment] = {
            name: metrics(target_array[mask], probabilities[mask])
            for name, probabilities in arrays.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
