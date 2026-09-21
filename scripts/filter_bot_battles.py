"""Write a copy of a raw corpus without the battles that involve a bot.

Shards carry no player tag, so bots are found by their loadout: the exact set of cards
with their Evo and Hero levels plus the tower troop, hashed. A real meta loadout is
shared by thousands of players and lands near 50%; high-volume loadouts split into two
clear modes, one around 55-60% (players) and one around 15-35% (bots: constant
trophies while losing ~90%, thousands of games a day). A loadout seen at least
--min-appearances times, both sides counted, with a win rate at or below --max-win-rate
is a bot; a battle is dropped when either side plays one.

Validated on season-87-balanced with a 5% A/B mini-training: without bots the model
stops dragging popular decks down (-3 pts -> ~0 on player-vs-player test battles) and
the segment weighting no longer trips prepare's win_rate_shift guard.

Stats are computed on the whole source corpus, so the threshold decides who is a bot,
not the sample. The manifest records the rule and the per-segment drop.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Directory of raw Parquet shards")
    parser.add_argument("destination", type=Path, help="Output root (raw/ and manifest go here)")
    parser.add_argument("--min-appearances", type=int, default=300)
    parser.add_argument("--max-win-rate", type=float, default=0.40)
    parser.add_argument("--shard-size", default="40MB",
                        help="Target shard size; Storage uploads each file in one request")
    parser.add_argument("--memory-limit", default=None, help="DuckDB memory_limit, e.g. 2GB")
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def loadout(side: str) -> str:
    return (
        f"hash(list_sort(list_transform({side}_card_ids, (c, i) -> c * 100 + "
        f"{side}_evolution_levels[i] * 10 + {side}_hero_levels[i])), "
        f"coalesce({side}_tower_troop_id, 0))"
    )


def main() -> None:
    args = parse_args()
    source = (args.source.resolve() / "*.parquet").as_posix().replace("'", "''")
    raw_out = args.destination.resolve() / "raw"
    if raw_out.exists() and any(raw_out.glob("*.parquet")):
        raise SystemExit(f"{raw_out} already holds shards; remove them first.")
    raw_out.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute("set preserve_insertion_order=false")
    if args.memory_limit:
        connection.execute(f"set memory_limit='{args.memory_limit}'")
    if args.threads:
        connection.execute(f"set threads={int(args.threads)}")

    # Hashed keys keep the GROUP BY spillable on a workstation; string keys do not.
    connection.execute(
        f"""
        create temp table bots as
        with sides as (
          select {loadout('team')} k, win::int g from read_parquet('{source}')
          union all
          select {loadout('opponent')} k, 1 - win::int g from read_parquet('{source}')
        )
        select k, count(*) appearances, avg(g) win_rate from sides group by k
        having count(*) >= {int(args.min_appearances)} and avg(g) <= {float(args.max_win_rate)}
        """
    )
    bot_loadouts = connection.execute("select count(*) from bots").fetchone()[0]
    connection.execute(
        f"""
        copy (
          select * from read_parquet('{source}')
          where {loadout('team')} not in (select k from bots)
            and {loadout('opponent')} not in (select k from bots)
        ) to '{raw_out.as_posix()}'
          (format parquet, compression zstd, row_group_size 100000,
           file_size_bytes '{args.shard_size}', filename_pattern 'nobots-part-{{i}}')
        """
    )
    connection.execute(
        f"copy bots to '{(args.destination.resolve() / 'bot_loadouts.parquet').as_posix()}' (format parquet)"
    )

    kept_glob = (raw_out / "*.parquet").as_posix()
    rows = connection.execute(
        f"""
        select s.segment, s.n, coalesce(k.n, 0)
        from (select segment, count(*) n from read_parquet('{source}') group by 1) s
        left join (select segment, count(*) n from read_parquet('{kept_glob}') group by 1) k
        using (segment) order by 1
        """
    ).fetchall()
    source_rows = sum(row[1] for row in rows)
    kept_rows = sum(row[2] for row in rows)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(args.source),
        "rule": {
            "loadout": "hash(sorted card_id*100 + evo*10 + hero, tower_troop_id)",
            "min_appearances": args.min_appearances,
            "max_win_rate": args.max_win_rate,
            "drop_battle_if": "either side plays a bot loadout",
        },
        "bot_loadouts": bot_loadouts,
        "source_rows": source_rows,
        "kept_rows": kept_rows,
        "dropped_rows": source_rows - kept_rows,
        "by_segment": [
            {"segment": segment, "source": n, "kept": kept, "dropped_share": 1 - kept / n}
            for segment, n, kept in rows
        ],
    }
    (args.destination / "filter-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "by_segment"}, indent=2))


if __name__ == "__main__":
    main()
