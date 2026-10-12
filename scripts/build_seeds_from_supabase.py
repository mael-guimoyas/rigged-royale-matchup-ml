"""Build a seed list with an equal quota per trophy band from Supabase.

Why
---
The collector snowballs from its seeds and discovery stays inside a band (a
6000-trophy player only meets 6000-trophy players), so the bands the seeds
reach are the bands the crawl can ever open. ``seeds_balanced.txt`` was built by
walking battlelogs from the site's visitors, which costs API requests and still
came out top-heavy. Supabase already knows thousands of players per band:
every row of ``public.battles`` names both players with the trophies they
started that battle with. This reads them back instead of re-crawling.

The local shards cannot do this: they keep a hashed ``game_id`` and the
trophies, but not the player tags.

How
---
Each tag is placed by its most recent observation (``public.battles``, both
sides, plus ``public.players``), keeping only those seen since ``--since`` so
inactive accounts do not take a seed slot. Ladder tags go to their 1,000-trophy
band from ``trophy_buckets``; tags last seen in Path of Legends go to
``ranked``. Each band then gets the same random draw of ``--per-band`` tags (all
of them when the band holds fewer).

Seeds stay out of the corpus (``--min-hop 1``), so the visitors' own skew does
not reach the training data; the seeds only have to open every band.

Usage
-----
    python scripts/build_seeds_from_supabase.py --per-band 1500 --out seeds_season88.txt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rigged_matchup_ml.api_collect import normalize_tag  # noqa: E402
from rigged_matchup_ml.config import load_config  # noqa: E402

# Most recent observation per tag. Both sides of a battle are read: the
# opponent half is what reaches players who never visited the site.
QUERY = """
with sides as (
    select x->>'tag' as tag,
           (x->>'startingTrophies')::int as trophies,
           b.mode_key,
           b.battle_time as seen_at
    from public.battles b,
         lateral (
             select jsonb_array_elements(b.raw->'team')
             union all
             select jsonb_array_elements(b.raw->'opponent')
         ) side(x)
    where b.mode_key in ('ladder', 'ranked')
      and b.battle_time >= %(since)s
    union all
    select p.tag, p.trophies, 'ladder', p.last_analyzed_at
    from public.players p
    where p.last_analyzed_at >= %(since)s
      and p.trophies is not null
)
select distinct on (upper(replace(tag, '#', '')))
       tag, trophies, mode_key
from sides
where tag is not null
order by upper(replace(tag, '#', '')), seen_at desc
"""


def band_of(mode_key: str, trophies: int | None, buckets: list[int], minimum: int) -> str | None:
    if mode_key == "ranked":
        return "ranked"
    if trophies is None or trophies < minimum:
        return None
    for lower, upper in zip(buckets, buckets[1:]):
        if lower <= trophies < upper:
            return f"ladder:{lower}-{upper - 1}"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--per-band", type=int, default=1500, help="Seeds drawn per band.")
    parser.add_argument(
        "--since-days",
        type=float,
        default=60.0,
        help="Only tags observed in the last N days (active players).",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for a reproducible draw.")
    parser.add_argument("--out", default="seeds_season88.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    load_dotenv(config.resolve(".env"))
    database_url = os.getenv("SUPABASE_DB_URL")
    if not database_url:
        raise SystemExit("SUPABASE_DB_URL is missing.")

    buckets = [int(edge) for edge in config.data["trophy_buckets"]]
    minimum = int(config.data.get("collect_min_trophies", 5000))
    bands = [
        f"ladder:{lower}-{upper - 1}"
        for lower, upper in zip(buckets, buckets[1:])
        if lower >= minimum
    ] + ["ranked"]
    since = datetime.now(timezone.utc) - timedelta(days=args.since_days)

    import psycopg
    from psycopg.rows import tuple_row

    with psycopg.connect(database_url, row_factory=tuple_row) as connection:
        connection.execute("set default_transaction_read_only = on")
        connection.execute("set statement_timeout = '300s'")
        rows = connection.execute(QUERY, {"since": since}).fetchall()

    by_band: dict[str, list[str]] = defaultdict(list)
    for tag, trophies, mode_key in rows:
        normalized = normalize_tag(tag or "")
        band = band_of(mode_key, trophies, buckets, minimum)
        if normalized and band is not None:
            by_band[band].append(normalized)

    rng = random.Random(args.seed)
    seeds: list[str] = []
    print(f"{len(rows):,} tags vus depuis {since:%Y-%m-%d}", file=sys.stderr)
    print(f"{'bande':<24}{'dispo':>8}{'seeds':>8}", file=sys.stderr)
    short: list[str] = []
    for band in bands:
        tags = sorted(set(by_band[band]))
        drawn = rng.sample(tags, min(args.per_band, len(tags)))
        seeds.extend(drawn)
        if len(drawn) < args.per_band:
            short.append(band)
        print(f"{band:<24}{len(tags):>8,}{len(drawn):>8,}", file=sys.stderr)

    rng.shuffle(seeds)
    Path(args.out).write_text("\n".join(seeds) + "\n", encoding="utf-8")
    print(f"\n{len(seeds):,} seeds ecrits dans {args.out}", file=sys.stderr)
    if short:
        print(
            "Bandes sous le quota (toutes leurs seeds sont prises) : " + ", ".join(short),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
