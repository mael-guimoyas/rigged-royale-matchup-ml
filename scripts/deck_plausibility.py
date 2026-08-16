"""Does this deck look like something a human would sleeve together?

The composition rules in `deck_composition_rules.py` ask questions about cards
one at a time: does the deck carry a win condition, a spell, enough anti-air.
The decks the optimizer emits pass every one of them, because each card it
picks is individually reasonable — only the *combination* is one nobody plays.
The admission benchmark shows the consequence: the tight envelope never changed
the emitted deck in any measured search.

So the statistic has to look at combinations. This one is the cheapest thing
that does: pointwise mutual information over card pairs, counted on the whole
corpus. For each of a deck's 28 pairs it asks whether those two cards appear
together more or less often than their individual popularity predicts, and
averages the answer. An archetype scores high because its cards are chosen for
each other; eight individually-popular cards assembled by a search score low.

Popularity is divided out on purpose. Counting raw pair frequency instead would
just re-rank decks by how mainstream their cards are, which is the popularity
bias the support-penalty benchmark already rejected.

The script measures three populations on the same scale:

* the real population, weighted by battles, which sets the thresholds;
* the optimizer's own trajectories, read from `plausibility-envelope.json`,
  which walk from a real deck at round 0 to the emitted deck at the last round;
* any deck passed on the command line, for spot checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

POOLED_SEGMENT_SQL = """case
  when mode_key = 'ranked'
    and try_cast(regexp_extract(segment, '^ranked:league-([0-9]+)$', 1) as integer)
      between 1 and 2 then 'ranked:league-1-2'
  when mode_key = 'ranked'
    and try_cast(regexp_extract(segment, '^ranked:league-([0-9]+)$', 1) as integer)
      between 3 and 4 then 'ranked:league-3-4'
  when mode_key = 'ranked'
    and try_cast(regexp_extract(segment, '^ranked:league-([0-9]+)$', 1) as integer)
      between 5 and 7 then 'ranked:league-5-7'
  else segment
end"""

# Splitting the key into eight integer columns inside DuckDB keeps 8.9 million
# decks out of Python strings entirely; parsing them here costs minutes and a
# gigabyte, and the engine does it while it writes the cache.
DECK_SQL = """
        with base as (
          select team_deck_key, opponent_deck_key
          from read_parquet('{parquet}')
          where segment not in ('ladder:top-100', 'ladder:top-1000')
        ), sides as (
          select team_deck_key as deck_key from base
          union all
          select opponent_deck_key from base
        ), counted as (
          select deck_key, count(*) as plays
          from sides
          group by deck_key
        ), split as (
          select str_split(deck_key, ',') as cards, plays from counted
        )
        select
          cast(cards[1] as bigint) as card1,
          cast(cards[2] as bigint) as card2,
          cast(cards[3] as bigint) as card3,
          cast(cards[4] as bigint) as card4,
          cast(cards[5] as bigint) as card5,
          cast(cards[6] as bigint) as card6,
          cast(cards[7] as bigint) as card7,
          cast(cards[8] as bigint) as card8,
          plays
        from split
        where len(cards) = 8
"""


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / "data" / "raw")
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "artifacts" / "deck-plays-all.parquet",
        help="Cached deck play counts with no minimum, rebuilt when stale.",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--envelope",
        type=Path,
        default=root / "artifacts" / "plausibility-envelope.json",
        help="Search trajectories to place against the real distribution.",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "deck-plausibility.json"
    )
    parser.add_argument(
        "--reference-min-plays",
        type=int,
        default=30,
        help="Battles a deck needs to join the reference population.",
    )
    parser.add_argument(
        "--deck",
        action="append",
        default=[],
        help="Comma-separated card ids to score against the reference, repeatable.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="4GB")
    return parser.parse_args()


def raw_glob(raw_dir: Path) -> str:
    return (raw_dir.resolve() / "*.parquet").as_posix().replace("'", "''")


def deck_plays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every distinct deck in the corpus with its battle count.

    No minimum applies here: the pair statistics want the whole population of
    decks people build, not only the ones that reached a play threshold.
    """
    cache = args.cache
    sidecar = cache.with_suffix(".meta.json")
    signature = {"raw": str(args.raw.resolve()), "layout": "card-columns"}
    fresh = (
        not args.rebuild
        and cache.exists()
        and sidecar.exists()
        and json.loads(sidecar.read_text(encoding="utf-8")) == signature
    )
    connection = duckdb.connect()
    connection.execute(f"pragma threads={args.threads}")
    connection.execute(f"pragma memory_limit='{args.memory_limit}'")
    if fresh:
        log(f"reading cached deck plays from {cache}")
    else:
        log("counting every distinct deck in the corpus")
        cache.parent.mkdir(parents=True, exist_ok=True)
        target = cache.resolve().as_posix().replace("'", "''")
        query = DECK_SQL.format(parquet=raw_glob(args.raw))
        connection.execute(f"copy ({query}) to '{target}' (format parquet)")
        sidecar.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        log(f"cached deck plays to {cache}")

    cached = cache.resolve().as_posix().replace("'", "''")
    columns = connection.execute(
        f"select * from read_parquet('{cached}')"
    ).fetchnumpy()
    connection.close()

    decks = np.column_stack(
        [np.asarray(columns[f"card{slot}"], dtype=np.int64) for slot in range(1, 9)]
    )
    plays = np.asarray(columns["plays"], dtype=np.float64)
    log(f"{len(decks):,} distinct decks, {plays.sum():,.0f} deck-slots")
    return decks, plays, np.unique(decks)


PAIRS = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)], dtype=np.int64)


class PairModel:
    """Play-weighted card and card-pair frequencies, and the PMI they imply."""

    def __init__(self, decks: np.ndarray, weights: np.ndarray, cards: np.ndarray) -> None:
        self.cards = cards
        self.card_index = {int(card): index for index, card in enumerate(cards)}
        size = len(cards)
        lookup = np.full(int(cards.max()) + 1, -1, dtype=np.int64)
        lookup[cards] = np.arange(size)
        indexed = lookup[decks]
        total = float(weights.sum())

        card_counts = np.zeros(size, dtype=np.float64)
        for slot in range(8):
            card_counts += np.bincount(indexed[:, slot], weights=weights, minlength=size)

        pair_counts = np.zeros(size * size, dtype=np.float64)
        for first, second in PAIRS:
            left = indexed[:, first]
            right = indexed[:, second]
            low = np.minimum(left, right)
            high = np.maximum(left, right)
            pair_counts += np.bincount(low * size + high, weights=weights, minlength=size * size)
        pair_counts = pair_counts.reshape(size, size)
        pair_counts = pair_counts + pair_counts.T - np.diag(np.diag(pair_counts))

        # Probabilities are per deck, so a card present in a deck counts once.
        self.card_probability = card_counts / total
        self.pair_probability = pair_counts / total
        # Laplace floor: an unobserved pair must score badly, not undefined.
        floor = 1.0 / max(total, 1.0)
        expected = np.outer(self.card_probability, self.card_probability)
        self.pmi = np.log(np.maximum(self.pair_probability, floor) / np.maximum(expected, 1e-12))
        self.indexed = indexed
        self.weights = weights
        self.total = total

    def deck_pmi(self, indexed: np.ndarray) -> dict[str, np.ndarray]:
        """Mean, minimum, and unseen-pair count over each deck's 28 pairs."""
        rows = indexed.shape[0]
        values = np.empty((rows, len(PAIRS)), dtype=np.float64)
        unseen = np.zeros(rows, dtype=np.int64)
        for index, (first, second) in enumerate(PAIRS):
            left = indexed[:, first]
            right = indexed[:, second]
            values[:, index] = self.pmi[left, right]
            unseen += (self.pair_probability[left, right] <= 0.0).astype(np.int64)
        return {"mean": values.mean(axis=1), "min": values.min(axis=1), "unseen_pairs": unseen}

    def score_cards(self, decks: np.ndarray) -> dict[str, np.ndarray]:
        size = len(self.cards)
        lookup = np.full(int(self.cards.max()) + 1, -1, dtype=np.int64)
        lookup[self.cards] = np.arange(size)
        clipped = np.clip(decks, 0, len(lookup) - 1)
        indexed = lookup[clipped]
        known = (indexed >= 0).all(axis=1)
        indexed = np.where(indexed >= 0, indexed, 0)
        result = self.deck_pmi(indexed)
        result["known"] = known
        return result


def percentiles(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    if values.size == 0:
        return {}
    if weights is None:
        array = np.sort(values)
        quantiles = np.arange(1, array.size + 1) / array.size
    else:
        order = np.argsort(values)
        array = values[order]
        quantiles = np.cumsum(weights[order]) / weights.sum()
    picks = {
        "p001": 0.001,
        "p01": 0.01,
        "p05": 0.05,
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p99": 0.99,
    }
    output = {
        name: float(array[min(int(np.searchsorted(quantiles, share)), array.size - 1)])
        for name, share in picks.items()
    }
    output["min"] = float(array[0])
    output["max"] = float(array[-1])
    output["mean"] = float(values.mean())
    return output


def rank_of(value: float, reference: np.ndarray) -> float:
    """Share of the reference population scoring below `value`."""
    return float((reference < value).mean())


def main() -> None:
    args = parse_args()
    decks, plays, cards = deck_plays(args)
    log(f"building pair model over {len(cards)} cards")
    model = PairModel(decks, plays, cards)

    statistics = model.deck_pmi(model.indexed)
    reference_mask = plays >= args.reference_min_plays
    reference = statistics["mean"][reference_mask]
    reference_weights = plays[reference_mask]
    log(
        f"reference population: {reference_mask.sum():,} decks with "
        f"{args.reference_min_plays}+ battles"
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "distinct_decks": int(len(decks)),
        "reference_min_plays": args.reference_min_plays,
        "reference_decks": int(reference_mask.sum()),
        "all_decks_mean_pmi": percentiles(statistics["mean"]),
        "reference_mean_pmi": percentiles(reference),
        "reference_mean_pmi_play_weighted": percentiles(reference, reference_weights),
    }

    # How the statistic separates by how often a deck is actually played.
    tiers = [(1, 2), (2, 10), (10, 30), (30, 100), (100, 1000), (1000, 10**12)]
    report["by_play_tier"] = [
        {
            "plays": [low, high],
            "decks": int(mask.sum()),
            "mean_pmi": float(statistics["mean"][mask].mean()) if mask.any() else None,
            "median_pmi": float(np.median(statistics["mean"][mask])) if mask.any() else None,
        }
        for low, high in tiers
        for mask in [(plays >= low) & (plays < high)]
    ]

    # The optimizer's own trajectories, round by round.
    if args.envelope.exists():
        envelope = json.loads(args.envelope.read_text(encoding="utf-8"))
        rounds: dict[int, list[float]] = {}
        trajectories = []
        for segment, entry in envelope.get("segments", {}).items():
            for index, search in enumerate(entry.get("searches") or []):
                for name, steps in (
                    ("unconstrained", search["trajectory"]),
                    (
                        "constrained",
                        (search.get("constrained") or {}).get("trajectory") or [],
                    ),
                ):
                    if not steps:
                        continue
                    deck_array = np.asarray([step["cards"] for step in steps], dtype=np.int64)
                    scored = model.score_cards(deck_array)
                    walk = []
                    for position, step in enumerate(steps):
                        value = float(scored["mean"][position])
                        walk.append(
                            {
                                "round": step["round"],
                                "model_score": step["profile"]["score"],
                                "mean_pmi": value,
                                "min_pmi": float(scored["min"][position]),
                                "unseen_pairs": int(scored["unseen_pairs"][position]),
                                "reference_rank": rank_of(value, reference),
                            }
                        )
                        rounds.setdefault(step["round"], []).append(value)
                    trajectories.append(
                        {
                            "segment": segment,
                            "start": index,
                            "variant": name,
                            "walk": walk,
                        }
                    )
        report["trajectories"] = trajectories
        report["by_round"] = [
            {
                "round": key,
                "decks": len(values),
                "mean_pmi": float(np.mean(values)),
                "median_reference_rank": float(
                    np.median([rank_of(value, reference) for value in values])
                ),
            }
            for key, values in sorted(rounds.items())
        ]

    if args.deck:
        spot = np.asarray(
            [[int(card) for card in deck.split(",")] for deck in args.deck], dtype=np.int64
        )
        scored = model.score_cards(spot)
        report["decks"] = [
            {
                "cards": [int(card) for card in spot[index]],
                "mean_pmi": float(scored["mean"][index]),
                "min_pmi": float(scored["min"][index]),
                "unseen_pairs": int(scored["unseen_pairs"][index]),
                "reference_rank": rank_of(float(scored["mean"][index]), reference),
            }
            for index in range(spot.shape[0])
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done — {args.output}")


if __name__ == "__main__":
    main()
