from __future__ import annotations

import json
import os
import sqlite3
import time
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


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[offset : offset + size] for offset in range(0, len(values), size)]


class Deduplicator:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30.0)
        self.connection.execute("pragma journal_mode=WAL")
        self.connection.execute("pragma synchronous=NORMAL")
        # Two processes (e.g. drain-db + collect-api) may write this dedup DB at
        # once. WAL allows that, but a second writer must wait for the lock
        # instead of raising "database is locked"; busy_timeout makes it wait.
        self.connection.execute("pragma busy_timeout=30000")
        self.connection.execute("pragma temp_store=MEMORY")
        self.connection.execute("pragma cache_size=-200000")
        self.connection.execute(
            "create table if not exists seen (game_id text primary key) without rowid"
        )

    def filter_new(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return records whose game_id is unseen, WITHOUT persisting them.

        Split from `commit` so a caller can durably archive the kept rows
        (write Parquet, upload) before marking them seen. If the process dies
        in between, the rows are simply re-offered next run (at worst a
        duplicate shard) rather than being marked seen but never archived.
        """
        if not records:
            return []

        unique_records: list[dict[str, Any]] = []
        batch_seen: set[str] = set()
        for record in records:
            game_id = str(record["game_id"])
            if game_id in batch_seen:
                continue
            batch_seen.add(game_id)
            unique_records.append(record)

        existing: set[str] = set()
        for chunk in _chunks(sorted(batch_seen), 900):
            placeholders = ", ".join("?" for _ in chunk)
            existing.update(
                row[0]
                for row in self.connection.execute(
                    f"select game_id from seen where game_id in ({placeholders})",
                    chunk,
                ).fetchall()
            )
        return [
            record for record in unique_records if str(record["game_id"]) not in existing
        ]

    def commit(self, records: list[dict[str, Any]]) -> None:
        """Persist game_ids as seen. Call only after the rows are durably archived."""
        if not records:
            return
        with self.connection:
            self.connection.executemany(
                "insert or ignore into seen(game_id) values (?)",
                [(record["game_id"],) for record in records],
            )

    def keep_new(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = self.filter_new(records)
        self.commit(kept)
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


def _ranked_league_sql(
    raw_alias: str = "raw",
    player_league_col: str | None = None,
    player_profile_col: str | None = None,
) -> str:
    """Resolve the ranked league number for a battle.

    The Clash Royale battlelog `raw` JSON almost never embeds the player's
    Path of Legends league. The authoritative value lives in `public.players`
    (`league_number` column, fed from `currentPathOfLegendSeasonResult`). So we
    probe the battle JSON first (most battle-accurate when present) and fall back
    to the joined player's stored league, which covers the vast majority of rows.
    """
    text_candidates = [
        f"{raw_alias} ->> 'leagueNumber'",
        f"{raw_alias} ->> 'league_number'",
        f"{raw_alias} #>> '{{currentPathOfLegendSeasonResult,leagueNumber}}'",
        f"{raw_alias} #>> '{{team,0,leagueNumber}}'",
        f"{raw_alias} #>> '{{opponent,0,leagueNumber}}'",
        f"{raw_alias} #>> '{{team,0,currentPathOfLegendSeasonResult,leagueNumber}}'",
        f"{raw_alias} #>> '{{opponent,0,currentPathOfLegendSeasonResult,leagueNumber}}'",
        f"{raw_alias} #>> '{{team,0,pathOfLegendSeasonResult,leagueNumber}}'",
        f"{raw_alias} #>> '{{opponent,0,pathOfLegendSeasonResult,leagueNumber}}'",
    ]
    if player_profile_col is not None:
        text_candidates.append(
            f"{player_profile_col} #>> '{{currentPathOfLegendSeasonResult,leagueNumber}}'"
        )
    cases = [
        f"case when {candidate} ~ '^[0-9]+$' then ({candidate})::integer end"
        for candidate in text_candidates
    ]
    if player_league_col is not None:
        cases.append(f"case when {player_league_col} > 0 then {player_league_col} end")
    return f"coalesce({', '.join(cases)})"


def _fetch_ranked_segments(
    connection: Any,
    fingerprints: list[str],
) -> tuple[dict[str, str], int]:
    if not fingerprints:
        return {}, 0
    league_sql = _ranked_league_sql("b.raw", "p.league_number", "p.profile")
    query = f"""
        select b.fingerprint, {league_sql} league_number
        from public.battles b
        left join public.players p on p.tag = b.player_tag
        where b.fingerprint = any(%s)
    """
    rows = connection.execute(query, (fingerprints,)).fetchall()
    mapping: dict[str, str] = {}
    for row in rows:
        league = row["league_number"]
        mapping[str(row["fingerprint"])] = (
            f"ranked:league-{league}"
            if league is not None and int(league) > 0
            else "ranked:unknown"
        )
    return mapping, len(rows)


def _replace_string_column(table: pa.Table, name: str, values: list[str]) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index == -1:
        return table.append_column(name, pa.array(values, type=pa.string()))
    return table.set_column(index, name, pa.array(values, type=pa.string()))


def _write_replacement(path: Path, table: pa.Table) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=50_000)
    temporary.replace(path)


def backfill_ranked_segments(config: AppConfig, batch_size: int = 10_000) -> dict[str, int]:
    """Rewrite old raw Parquet extracts so ranked rows are split by leagueNumber."""
    load_dotenv()
    database_url = os.getenv("SUPABASE_DB_URL")
    if not database_url:
        raise RuntimeError("SUPABASE_DB_URL is missing. Copy .env.example to .env.")

    raw_dir = config.resolve(config.data["raw_dir"])
    timeout = int(config.database["statement_timeout_ms"])
    report = {
        "files_scanned": 0,
        "files_changed": 0,
        "rows_scanned": 0,
        "ranked_rows": 0,
        "updated_rows": 0,
        "league_rows": 0,
        "unknown_rows": 0,
        "db_rows_found": 0,
    }

    with psycopg.connect(database_url, row_factory=dict_row, autocommit=True) as connection:
        connection.execute(f"set statement_timeout = {timeout}")
        connection.execute("set default_transaction_read_only = on")
        for parquet_path in sorted(raw_dir.glob("*.parquet")):
            report["files_scanned"] += 1
            table = pq.read_table(parquet_path)
            report["rows_scanned"] += table.num_rows
            rows = table.select(["source_fingerprint", "mode_key", "segment"]).to_pylist()
            ranked_fingerprints = sorted(
                {
                    str(row["source_fingerprint"])
                    for row in rows
                    if str(row["mode_key"]).lower() == "ranked"
                    and not str(row["segment"]).startswith("ranked:league-")
                }
            )
            if not ranked_fingerprints:
                continue

            segment_by_fingerprint: dict[str, str] = {}
            for offset in range(0, len(ranked_fingerprints), batch_size):
                fetched, found = _fetch_ranked_segments(
                    connection,
                    ranked_fingerprints[offset : offset + batch_size],
                )
                segment_by_fingerprint.update(fetched)
                report["db_rows_found"] += found

            changed = False
            new_segments: list[str] = []
            for row in rows:
                current_segment = str(row["segment"])
                if str(row["mode_key"]).lower() != "ranked":
                    new_segments.append(current_segment)
                    continue
                report["ranked_rows"] += 1
                fingerprint = str(row["source_fingerprint"])
                new_segment = segment_by_fingerprint.get(fingerprint, "ranked:unknown")
                new_segments.append(new_segment)
                if new_segment != current_segment:
                    changed = True
                    report["updated_rows"] += 1
                if new_segment.startswith("ranked:league-"):
                    report["league_rows"] += 1
                else:
                    report["unknown_rows"] += 1

            if changed:
                _write_replacement(
                    parquet_path,
                    _replace_string_column(table, "segment", new_segments),
                )
                report["files_changed"] += 1

    return report


def extract_from_supabase(
    config: AppConfig,
    max_rows: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    load_dotenv()
    database_url = os.getenv("SUPABASE_DB_URL")
    if not database_url:
        raise RuntimeError("SUPABASE_DB_URL is missing. Copy .env.example to .env.")

    raw_dir = config.resolve(config.data["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_path = config.resolve(config.data["state_file"])
    state = _load_state(state_path)
    initial_scanned = int(state["scanned"])
    initial_accepted = int(state["accepted"])
    deduplicator = Deduplicator(config.resolve(config.data["dedup_db"]))
    batch_size = int(batch_size or config.database["batch_size"])
    timeout = int(config.database["statement_timeout_ms"])
    prior_provider_path = config.data.get("matrix_prior_provider")
    prior_provider = load_prior_provider(prior_provider_path) if prior_provider_path else None
    remaining = max_rows
    started_at = time.perf_counter()
    progress = tqdm(desc="Supabase rows", initial=state["scanned"], unit="row")

    allowed_modes = [str(mode) for mode in config.data.get("allowed_modes") or []]
    mode_filter = ""
    if allowed_modes:
        placeholders = ", ".join(["%s"] * len(allowed_modes))
        mode_filter = f"and b.mode_key in ({placeholders})"

    league_sql = _ranked_league_sql("b.raw", "p.league_number", "p.profile")
    query = f"""
        select b.fingerprint, b.battle_time, b.inserted_at, b.mode_key,
               case when lower(b.mode_key) = 'ranked' then {league_sql} end league_number,
               b.raw
        from public.battles b
        left join public.players p on p.tag = b.player_tag
        where (b.inserted_at, b.fingerprint) > (%s::timestamptz, %s)
          {mode_filter}
        order by b.inserted_at, b.fingerprint
        limit %s
    """
    try:
        with psycopg.connect(database_url, row_factory=dict_row, autocommit=True) as connection:
            connection.execute(f"set statement_timeout = {timeout}")
            connection.execute("set default_transaction_read_only = on")
            while remaining is None or remaining > 0:
                limit = batch_size if remaining is None else min(batch_size, remaining)
                parameters: list[Any] = [
                    state["last_inserted_at"],
                    state["last_fingerprint"],
                    *allowed_modes,
                    limit,
                ]
                rows = connection.execute(query, parameters).fetchall()
                if not rows:
                    break
                parsed = [parse_battle_row(row, config.data) for row in rows]
                if prior_provider is not None:
                    for record in parsed:
                        if record is None:
                            continue
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
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                run_scanned = int(state["scanned"]) - initial_scanned
                progress.set_postfix(
                    accepted=state["accepted"],
                    rate=f"{(run_scanned / elapsed):.0f}/s",
                )
                if remaining is not None:
                    remaining -= len(rows)
    finally:
        progress.close()
        deduplicator.close()
    elapsed = max(time.perf_counter() - started_at, 1e-6)
    run_scanned = int(state["scanned"]) - initial_scanned
    run_accepted = int(state["accepted"]) - initial_accepted
    return {
        "scanned": int(state["scanned"]),
        "accepted": int(state["accepted"]),
        "run_scanned": run_scanned,
        "run_accepted": run_accepted,
        "next_part": int(state["next_part"]),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "scanned_rows_per_second": int(run_scanned / elapsed),
        "accepted_rows_per_second": int(run_accepted / elapsed),
    }


def _next_drain_index(raw_dir: Path) -> int:
    indices = [
        int(path.stem.split("-")[-1])
        for path in raw_dir.glob("drain-part-*.parquet")
        if path.stem.split("-")[-1].isdigit()
    ]
    return max(indices, default=-1) + 1


def drain_from_supabase(
    config: AppConfig,
    batch_size: int | None = None,
    max_rows: int | None = None,
    delete: bool = False,
    upload: bool = False,
    bucket: str = "training-battles",
    prefix: str = "battles",
) -> dict[str, Any]:
    """Archive `public.battles` to Parquet, then (optionally) DELETE the rows.

    Designed to run *alongside* `collect-api`: it writes `drain-part-*.parquet`
    shards (disjoint from `part-*`/`api-part-*`), shares the same dedup DB
    (WAL + busy_timeout makes concurrent writes safe), and only ever touches the
    `battles` table on the database side (collect-api only reads `players`).

    Per batch the order is strict: SELECT -> parse -> write Parquet (+ upload)
    -> mark dedup-seen -> DELETE. So a row is removed from Postgres only after it
    is durably archived. `delete=False` is a safe dry run (archives, deletes
    nothing). `delete=True` requires a read-WRITE SUPABASE_DB_URL user.
    """
    load_dotenv()
    database_url = os.getenv("SUPABASE_DB_URL")
    if not database_url:
        raise RuntimeError("SUPABASE_DB_URL is missing. Copy .env.example to .env.")

    raw_dir = config.resolve(config.data["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    deduplicator = Deduplicator(config.resolve(config.data["dedup_db"]))
    batch_size = int(batch_size or config.database["batch_size"])
    timeout = int(config.database["statement_timeout_ms"])
    prior_provider_path = config.data.get("matrix_prior_provider")
    prior_provider = load_prior_provider(prior_provider_path) if prior_provider_path else None

    uploader = None
    if upload:
        from .api_collect import StorageClient  # lazy: avoids an import cycle

        uploader = StorageClient(bucket, prefix, create=True)

    allowed_modes = [str(mode) for mode in config.data.get("allowed_modes") or []]
    mode_filter = ""
    if allowed_modes:
        placeholders = ", ".join(["%s"] * len(allowed_modes))
        mode_filter = f"and b.mode_key in ({placeholders})"

    league_sql = _ranked_league_sql("b.raw", "p.league_number", "p.profile")
    query = f"""
        select b.fingerprint, b.battle_time, b.inserted_at, b.mode_key,
               case when lower(b.mode_key) = 'ranked' then {league_sql} end league_number,
               b.raw
        from public.battles b
        left join public.players p on p.tag = b.player_tag
        where b.fingerprint > %s
          {mode_filter}
        order by b.fingerprint
        limit %s
    """
    shard_index = _next_drain_index(raw_dir)
    last_fingerprint = ""
    scanned = archived = deleted = 0
    started_at = time.perf_counter()
    progress = tqdm(desc="Drain battles", unit="row")

    try:
        with psycopg.connect(database_url, row_factory=dict_row, autocommit=True) as connection:
            connection.execute(f"set statement_timeout = {timeout}")
            if not delete:
                connection.execute("set default_transaction_read_only = on")
            while max_rows is None or scanned < max_rows:
                limit = batch_size if max_rows is None else min(batch_size, max_rows - scanned)
                rows = connection.execute(
                    query, [last_fingerprint, *allowed_modes, limit]
                ).fetchall()
                if not rows:
                    break
                fingerprints = [row["fingerprint"] for row in rows]
                parsed = [parse_battle_row(row, config.data) for row in rows]
                parsed = [record for record in parsed if record is not None]
                if prior_provider is not None:
                    for record in parsed:
                        record["matrix_prior"] = min(
                            0.999, max(0.001, float(prior_provider(record)))
                        )
                kept = deduplicator.filter_new(parsed)
                if kept:
                    name = f"drain-part-{shard_index:06d}.parquet"
                    path = raw_dir / name
                    temporary = path.with_suffix(".parquet.tmp")
                    table = pa.Table.from_pylist(kept, schema=SCHEMA)
                    pq.write_table(
                        table, temporary, compression="zstd", row_group_size=50_000
                    )
                    temporary.replace(path)
                    if uploader is not None:
                        uploader.upload(name, path.read_bytes())
                    deduplicator.commit(kept)  # seen only after durable archive
                    shard_index += 1
                    archived += len(kept)
                if delete:
                    cursor = connection.execute(
                        "delete from public.battles where fingerprint = any(%s)",
                        (fingerprints,),
                    )
                    deleted += cursor.rowcount or 0
                last_fingerprint = rows[-1]["fingerprint"]
                scanned += len(rows)
                progress.update(len(rows))
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                progress.set_postfix(
                    archived=archived,
                    deleted=deleted,
                    rate=f"{(scanned / elapsed):.0f}/s",
                )
    finally:
        progress.close()
        deduplicator.close()

    elapsed = max(time.perf_counter() - started_at, 1e-6)
    return {
        "scanned": scanned,
        "archived": archived,
        "deleted": deleted,
        "delete_enabled": delete,
        "uploaded_to_storage": bool(uploader),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "rows_per_second": int(scanned / elapsed),
        "raw_dir": str(raw_dir),
        "bucket": bucket if uploader else None,
    }
