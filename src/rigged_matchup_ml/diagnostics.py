from __future__ import annotations

import os
from typing import Any

import duckdb
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from .config import AppConfig
from .extraction import _fetch_ranked_segments


def _quoted(value: object) -> str:
    return str(value).replace("'", "''")


def diagnose_ranked_segments(
    config: AppConfig,
    sample_size: int = 100,
) -> dict[str, Any]:
    raw_dir = config.resolve(config.data["raw_dir"])
    raw_glob = _quoted(raw_dir / "*.parquet")
    connection = duckdb.connect()
    try:
        segment_rows = connection.execute(
            f"""
            select segment, count(*) as row_count
            from read_parquet('{raw_glob}')
            where lower(mode_key) = 'ranked'
            group by segment
            order by row_count desc
            """
        ).fetchall()
        total_ranked = sum(int(row[1]) for row in segment_rows)
        unknown_rows = sum(
            int(row[1]) for row in segment_rows if str(row[0]) == "ranked:unknown"
        )
        sample_fingerprints = [
            row[0]
            for row in connection.execute(
                f"""
                select source_fingerprint
                from read_parquet('{raw_glob}')
                where lower(mode_key) = 'ranked' and segment = 'ranked:unknown'
                limit {int(sample_size)}
                """
            ).fetchall()
        ]
    finally:
        connection.close()

    report: dict[str, Any] = {
        "total_ranked_rows": total_ranked,
        "unknown_rows": unknown_rows,
        "unknown_share": unknown_rows / total_ranked if total_ranked else 0.0,
        "segments": [
            {"segment": str(segment), "rows": int(rows)} for segment, rows in segment_rows
        ],
        "sample_size": len(sample_fingerprints),
        "database_match_count": None,
        "database_note": None,
    }

    load_dotenv(config.resolve(".env"))
    database_url = os.getenv("SUPABASE_DB_URL")
    if not database_url or not sample_fingerprints:
        report["database_note"] = "No database URL or no ranked:unknown sample available."
        return report

    with psycopg.connect(database_url, row_factory=dict_row, autocommit=True) as db:
        db.execute("set default_transaction_read_only = on")
        match_count = db.execute(
            "select count(*) n from public.battles where fingerprint = any(%s)",
            (sample_fingerprints,),
        ).fetchone()["n"]
        # Re-resolve the sample with the players join so we can see how many of the
        # currently-unknown rows would recover a league after backfill/re-extract.
        resolved, _ = _fetch_ranked_segments(db, sample_fingerprints)
    report["database_match_count"] = int(match_count)
    recoverable = sum(
        1 for value in resolved.values() if value.startswith("ranked:league-")
    )
    report["recoverable_with_players_join"] = recoverable
    report["recoverable_share"] = (
        recoverable / len(sample_fingerprints) if sample_fingerprints else 0.0
    )
    if int(match_count) == 0:
        report["database_note"] = (
            "None of the sampled local source_fingerprint values exist in the current "
            "SUPABASE_DB_URL database, so backfill cannot repair these local rows."
        )
    elif recoverable > 0:
        report["database_note"] = (
            f"{recoverable}/{len(sample_fingerprints)} sampled unknown rows now resolve a "
            "league via the public.players join. Run `backfill-ranked-segments` to repair "
            "existing Parquet, then rebuild splits."
        )
    elif int(match_count) < len(sample_fingerprints):
        report["database_note"] = (
            "Only part of the sampled local fingerprints exist in the current database, "
            "and none resolved a league even with the players join."
        )
    else:
        report["database_note"] = (
            "Sampled fingerprints exist but no league was found in the battle JSON nor in "
            "public.players. leagueNumber is genuinely absent for these players."
        )
    return report
