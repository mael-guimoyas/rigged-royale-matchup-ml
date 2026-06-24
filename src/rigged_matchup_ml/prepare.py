from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .config import AppConfig


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def prepare_splits(config: AppConfig, overwrite: bool = False) -> dict[str, Any]:
    raw_dir = config.resolve(config.data["raw_dir"])
    prepared_dir = config.resolve(config.data["prepared_dir"])
    if not list(raw_dir.glob("*.parquet")):
        raise RuntimeError(f"No extracted Parquet files found in {raw_dir}")
    if prepared_dir.exists() and overwrite:
        shutil.rmtree(prepared_dir)
    if prepared_dir.exists() and any(prepared_dir.iterdir()):
        raise RuntimeError(f"{prepared_dir} is not empty. Pass --overwrite to rebuild it.")
    prepared_dir.mkdir(parents=True, exist_ok=True)

    raw_glob = _quoted(raw_dir / "*.parquet")
    train_fraction = float(config.data["train_fraction"])
    validation_fraction = float(config.data["validation_fraction"])
    validation_boundary = train_fraction + validation_fraction
    connection = duckdb.connect()
    connection.execute("set preserve_insertion_order=false")
    _log("prepare: computing chronological train/validation cutoffs")
    quantiles = connection.execute(
        f"""
        select quantile_cont(epoch(battle_time), [{train_fraction}, {validation_boundary}])
        from read_parquet('{raw_glob}')
        """
    ).fetchone()[0]
    train_cutoff, validation_cutoff = quantiles

    split_conditions = {
        "train": f"epoch(battle_time) <= {train_cutoff}",
        "validation": (
            f"epoch(battle_time) > {train_cutoff} and epoch(battle_time) <= {validation_cutoff}"
        ),
        "test": f"epoch(battle_time) > {validation_cutoff}",
    }
    counts: dict[str, int] = {}
    for split, condition in split_conditions.items():
        destination = prepared_dir / split
        destination.mkdir(parents=True, exist_ok=True)
        output = _quoted(destination / "data.parquet")
        _log(f"prepare: writing {split} split")
        connection.execute(
            f"""
            copy (
              select * from read_parquet('{raw_glob}') where {condition}
            ) to '{output}' (format parquet, compression zstd, row_group_size 100000)
            """
        )
        counts[split] = connection.execute(
            f"select count(*) from read_parquet('{output}')"
        ).fetchone()[0]
        _log(f"prepare: {split} rows={counts[split]:,}")

    train_file = _quoted(prepared_dir / "train" / "*.parquet")
    _log("prepare: building vocabularies from train split")
    card_ids = [
        row[0]
        for row in connection.execute(
            f"""
            select distinct card_id from (
              select unnest(team_card_ids) card_id from read_parquet('{train_file}')
              union all
              select unnest(opponent_card_ids) card_id from read_parquet('{train_file}')
            ) order by card_id
            """
        ).fetchall()
    ]
    tower_ids = [
        row[0]
        for row in connection.execute(
            f"""
            select distinct tower_id from (
              select team_tower_troop_id tower_id from read_parquet('{train_file}')
              union all
              select opponent_tower_troop_id tower_id from read_parquet('{train_file}')
            ) order by tower_id
            """
        ).fetchall()
    ]
    segments = [
        row[0]
        for row in connection.execute(
            f"select distinct segment from read_parquet('{train_file}') order by segment"
        ).fetchall()
    ]
    patches = [
        row[0]
        for row in connection.execute(
            f"select distinct patch from read_parquet('{train_file}') order by patch"
        ).fetchall()
    ]
    vocabulary = {
        "cards": {str(value): index + 1 for index, value in enumerate(card_ids)},
        "towers": {str(value): index + 1 for index, value in enumerate(tower_ids)},
        "segments": {str(value): index + 1 for index, value in enumerate(segments)},
        "patches": {str(value): index + 1 for index, value in enumerate(patches)},
    }
    (prepared_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Per-card train frequency, for inverse-frequency loss weighting: rare cards
    # are under-sampled, so without this the model just learns the popular meta.
    _log("prepare: counting per-card train frequencies")
    card_counts = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            f"""
            select card_id, count(*) from (
              select unnest(team_card_ids) card_id from read_parquet('{train_file}')
              union all
              select unnest(opponent_card_ids) card_id from read_parquet('{train_file}')
            ) group by card_id
            """
        ).fetchall()
    }
    (prepared_dir / "card_frequencies.json").write_text(
        json.dumps(card_counts, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "counts": counts,
        "train_cutoff_epoch": train_cutoff,
        "validation_cutoff_epoch": validation_cutoff,
        "vocabulary_sizes": {key: len(value) + 1 for key, value in vocabulary.items()},
        "split_policy": "chronological 70/15/15 by battle_time",
    }
    (prepared_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    connection.close()
    _log("prepare: done")
    return manifest
