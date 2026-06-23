"""One-off: build the covering index that the Supabase seed query needs.

The seed query `select tag from public.players order by last_analyzed_at desc
nulls last limit N` has no `where tracked` filter, so it can't use the partial
players_tracked_refresh index and falls back to a full parallel seq scan + top-N
sort (~20s on throttled compute). This covering index turns it into a bounded
index-only scan.

Run with the project venv:  .venv\\Scripts\\python.exe scripts\\create_seed_index.py
Idempotent. CREATE INDEX CONCURRENTLY needs autocommit and no statement_timeout,
which is why this runs locally instead of through the MCP SQL tool (whose client
timeout aborts the build and leaves an invalid index behind).
"""

from __future__ import annotations

import os
import sys

import psycopg
from dotenv import load_dotenv

INDEX = "players_last_analyzed_seed_idx"


def main() -> int:
    load_dotenv()
    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        print("SUPABASE_DB_URL missing from environment/.env", file=sys.stderr)
        return 1

    # autocommit is mandatory: CREATE/DROP INDEX CONCURRENTLY cannot run inside a
    # transaction block.
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("set statement_timeout = 0")  # no client-side cap on the build

        # Clear any invalid leftover from a previously aborted concurrent build.
        valid = conn.execute(
            "select indisvalid from pg_index where indexrelid = "
            "to_regclass(%s)",
            (f"public.{INDEX}",),
        ).fetchone()
        if valid is not None and valid[0] is False:
            print(f"dropping invalid leftover index {INDEX}")
            conn.execute(f"drop index if exists public.{INDEX}")

        print(f"building {INDEX} (concurrently, no lock on writers)...")
        conn.execute(
            f"create index concurrently if not exists {INDEX} "
            "on public.players (last_analyzed_at desc nulls last) include (tag)"
        )

        row = conn.execute(
            "select indisvalid, pg_size_pretty(pg_relation_size(indexrelid)) "
            "from pg_index where indexrelid = to_regclass(%s)",
            (f"public.{INDEX}",),
        ).fetchone()

    if row and row[0]:
        print(f"done: {INDEX} valid, size {row[1]}")
        return 0
    print(f"FAILED: {INDEX} not valid (state={row})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
