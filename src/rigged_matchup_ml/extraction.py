from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from tqdm import tqdm

from .config import AppConfig
from .domain import parse_battle_row
from .prior import load_prior_provider


SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("source_fingerprint", pa.string()),
        ("battle_time", pa.timestamp("us", tz="UTC")),
        ("inserted_at", pa.timestamp("us", tz="UTC")),
        ("mode_key", pa.string()),
        ("segment", pa.string()),
        ("patch", pa.string()),
        ("team_card_ids", pa.list_(pa.int64())),
        ("opponent_card_ids", pa.list_(pa.int64())),
        ("team_evolution_levels", pa.list_(pa.int8())),
        ("opponent_evolution_levels", pa.list_(pa.int8())),
        ("team_hero_levels", pa.list_(pa.int8())),
        ("opponent_hero_levels", pa.list_(pa.int8())),
        ("team_card_roles", pa.list_(pa.int8())),
        ("opponent_card_roles", pa.list_(pa.int8())),
        ("team_tower_troop_id", pa.int64()),
        ("opponent_tower_troop_id", pa.int64()),
        ("team_deck_key", pa.string()),
        ("opponent_deck_key", pa.string()),
        ("matrix_prior", pa.float32()),
        ("win", pa.bool_()),
    ]
)


class Deduplicator:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("pragma journal_mode=WAL")
        self.connection.execute("pragma synchronous=NORMAL")
        self.connection.execute("create table if not exists seen (game_id text primary key)")

    def keep_new(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        with self.connection:
            for record in records:
                cursor = self.connection.execute(
                    "insert or ignore into seen(game_id) values (?)", (record["game_id"],)
                )
                if cursor.rowcount == 1:
                    kept.append(record)
        return kept

    def close(self) -> None:
        self.connection.close()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "last_inserted_at": "1970-01-01T00:00:00+00:00",
            "last_fingerprint": "",
            "next_part": 0,
            "accepted": 0,
            "scanned": 0,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    # A fingerprint-only state safely rescans; SQLite removes duplicates.
    state.setdefault("last_inserted_at", "1970-01-01T00:00:00+00:00")
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def extract_from_supabase(config: AppConfig, max_rows: int | None = None) -> dict[str, int]:
    load_dotenv()
    database_url = os.getenv("SUPABASE_DB_URL")
    if not database_url:
        raise RuntimeError("SUPABASE_DB_URL is missing. Copy .env.example to .env.")

    raw_dir = config.resolve(config.data["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_path = config.resolve(config.data["state_file"])
    state = _load_state(state_path)
    deduplicator = Deduplicator(config.resolve(config.data["dedup_db"]))
    batch_size = int(config.database["batch_size"])
    timeout = int(config.database["statement_timeout_ms"])
    prior_provider = load_prior_provider(config.data.get("matrix_prior_provider"))
    remaining = max_rows
    progress = tqdm(desc="Supabase rows", initial=state["scanned"], unit="row")

    query = """
        select fingerprint, battle_time, inserted_at, mode_key, raw
        from public.battles
        where (inserted_at, fingerprint) > (%s::timestamptz, %s)
        order by inserted_at, fingerprint
        limit %s
    """
    try:
        with psycopg.connect(database_url, row_factory=dict_row, autocommit=True) as connection:
            connection.execute(f"set statement_timeout = {timeout}")
            connection.execute("set default_transaction_read_only = on")
            while remaining is None or remaining > 0:
                limit = batch_size if remaining is None else min(batch_size, remaining)
                rows = connection.execute(
                    query,
                    (state["last_inserted_at"], state["last_fingerprint"], limit),
                ).fetchall()
                if not rows:
                    break
                parsed = [parse_battle_row(row, config.data) for row in rows]
                for record in parsed:
                    if record is not None:
                        record["matrix_prior"] = min(
                            0.999, max(0.001, float(prior_provider(record)))
                        )
                accepted = deduplicator.keep_new([record for record in parsed if record is not None])
                if accepted:
                    part_path = raw_dir / f"part-{state['next_part']:06d}.parquet"
                    table = pa.Table.from_pylist(accepted, schema=SCHEMA)
                    pq.write_table(table, part_path, compression="zstd", row_group_size=50_000)
                    state["next_part"] += 1
                    state["accepted"] += len(accepted)
                state["last_fingerprint"] = rows[-1]["fingerprint"]
                state["last_inserted_at"] = rows[-1]["inserted_at"].isoformat()
                state["scanned"] += len(rows)
                _save_state(state_path, state)
                progress.update(len(rows))
                if remaining is not None:
                    remaining -= len(rows)
    finally:
        progress.close()
        deduplicator.close()
    return {"scanned": int(state["scanned"]), "accepted": int(state["accepted"])}
