"""Build a seed list that covers the whole ladder instead of only its top.

Why
---
A battle only enters the corpus if one of its two players was fetched, and the
fetched player is always the ``team`` side. So the fetched population decides
what the corpus measures, and today it is almost entirely endgame accounts: of
150 tags sampled from ``seeds.txt``, 58% sit exactly at the 14000 trophy
ceiling and 86% are above 12000. Matchmaking pairs like with like, so the
snowball started from those never descends -- a control collection landed 92.2%
of its battles in ladder:14000-999998.

The fix is upstream of the crawl: give it a seed list spread across the ladder.
Seeds do not have to be unbiased themselves, because the collector drops their
own battles (``--min-hop``); they only have to reach every band, so that the
opponents discovered from them -- the ones the matchmaker picked, which is as
close to a random active player as this API offers -- exist at every level.

How
---
Bootstrap from tags that already span the ladder (``public.players``, the
site's own visitors, thin but spread: 37 in 5000-6999 up to 323 at 14000+),
then walk battlelogs and keep the opponents each one reveals, filling a quota
per trophy band. Discovery stays inside a band by construction, which is
exactly what makes the quota reachable: a 6000-trophy seed only ever yields
6000-trophy players.

Usage
-----
    python scripts/build_balanced_seeds.py --per-band 2000 --out seeds_balanced.txt
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rigged_matchup_ml.api_collect import (  # noqa: E402
    ClashRoyaleClient,
    _fetch_player,
    _ladder_bucket_label,
    _seed_tags,
)
from rigged_matchup_ml.config import load_config  # noqa: E402


def band_of(mode_key: str, trophies: int | None, buckets: list[int]) -> str | None:
    """Which band a discovered candidate belongs to.

    `_ladder_bucket_label` returns None above the top edge because the collector
    treats that as out of range; here the top band is open-ended and wanted, so
    it is named explicitly.
    """
    if mode_key == "ranked":
        return "ranked"
    if trophies is None:
        return None
    if trophies >= buckets[-1]:
        return f"ladder:{buckets[-1]}-999998"
    return _ladder_bucket_label(trophies, buckets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--tags-file",
        default="",
        help="Extra bootstrap tags. Combined with --from-supabase.",
    )
    parser.add_argument(
        "--from-supabase",
        type=int,
        default=5000,
        help="Bootstrap tags pulled from public.players (needs SUPABASE_DB_URL).",
    )
    parser.add_argument(
        "--per-band",
        type=int,
        default=2000,
        help="Target seed count per trophy band.",
    )
    parser.add_argument(
        "--max-fetches",
        type=int,
        default=4000,
        help="Battlelog reads to spend discovering. This is the cost knob.",
    )
    parser.add_argument("--requests-per-second", type=float, default=25.0)
    parser.add_argument("--out", default="seeds_balanced.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    load_dotenv(config.resolve(".env"))

    # trophy_buckets already closes with a sentinel edge, so the pairwise walk
    # produces the open-ended top band itself; appending one more would add a
    # band nothing can ever fall into and stall the "every band full" exit.
    buckets = list(config.data["trophy_buckets"])
    bands = [
        f"ladder:{lower}-{upper - 1}" for lower, upper in zip(buckets, buckets[1:])
    ] + ["ranked"]

    bootstrap = _seed_tags(
        config,
        Path(args.tags_file) if args.tags_file else None,
        args.from_supabase,
    )
    if not bootstrap:
        raise SystemExit("No bootstrap tags: pass --tags-file or --from-supabase.")
    random.shuffle(bootstrap)
    print(f"bootstrap: {len(bootstrap):,} tags", file=sys.stderr)

    client = ClashRoyaleClient(token_mode="1", requests_per_second=args.requests_per_second)
    allowed_modes = set(config.data.get("allowed_modes") or [])

    found: dict[str, set[str]] = defaultdict(set)
    # Bootstrap tags are never written out: they are the visitors whose bias this
    # exists to escape. They are only walked to reveal their opponents.
    pending = list(bootstrap)
    walked: set[str] = set()
    fetches = 0
    skipped = Counter()

    while pending and fetches < args.max_fetches:
        if all(len(found[band]) >= args.per_band for band in bands):
            break
        tag = pending.pop()
        if tag in walked:
            continue
        walked.add(tag)
        fetches += 1
        try:
            _records, candidates, _seen, _stale = _fetch_player(
                client,
                tag,
                config.data,
                allowed_modes,
                True,
                0,
                None,
            )
        except Exception as error:  # noqa: BLE001 - discovery is best effort
            skipped[type(error).__name__] += 1
            continue
        for candidate, mode_key, trophies in candidates:
            band = band_of(mode_key, trophies, buckets)
            if band is None or len(found[band]) >= args.per_band:
                continue
            if candidate in walked or candidate in bootstrap:
                continue
            found[band].add(candidate)
            # Keep walking inside bands that are still short, so a thin band
            # deepens instead of stalling on its handful of bootstrap tags.
            if len(found[band]) < args.per_band:
                pending.append(candidate)
        if fetches % 250 == 0:
            filled = {band: len(found[band]) for band in bands if found[band]}
            print(f"  {fetches:,} lectures, {filled}", file=sys.stderr)

    seeds: list[str] = []
    print(f"\n{fetches:,} battlelogs lus, echecs {dict(skipped)}", file=sys.stderr)
    print(f"{'bande':<24}{'seeds':>8}", file=sys.stderr)
    for band in bands:
        tags = sorted(found[band])
        print(f"{band:<24}{len(tags):>8,}", file=sys.stderr)
        seeds.extend(tags)

    unique = list(dict.fromkeys(seeds))
    random.shuffle(unique)
    Path(args.out).write_text("\n".join(unique) + "\n", encoding="utf-8")
    print(f"\n{len(unique):,} seeds ecrits dans {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
