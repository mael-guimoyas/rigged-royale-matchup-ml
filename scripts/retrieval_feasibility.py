"""Can we recommend a deck people actually play, instead of inventing one?

The optimizer's failure is a selection artefact: it maximises a model score over
thousands of decks that have never been played, so it finds the decks the model
is most wrong about rather than the decks that are best. Every attempt to detect
those decks after the fact has failed, because the error is a property of the
deck and nothing observable separates it from a good one.

Retrieval sidesteps the whole problem. If the candidate set only contains decks
with real battles behind them, the model is scoring in-distribution, its error
is 2.7x smaller, and the winner it picks can be shown with an observed win rate
instead of a projection.

The question is whether the real population is dense enough to answer the
product's constraints. This script measures that: for real visitor-shaped decks,
how many genuinely played decks sit within one, two, three or four card changes,
per segment. A mode whose constraint leaves thousands of real neighbours can be
served by retrieval; one that leaves none has to keep synthesising, and should
say so.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        type=Path,
        default=root / "artifacts" / "played-deck-population.parquet",
        help="Per-segment play counts written by plausibility_envelope.py.",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "retrieval-feasibility.json"
    )
    parser.add_argument(
        "--seed-plays",
        type=int,
        nargs=2,
        default=(30, 300),
        metavar=("MIN", "MAX"),
        help="Play band the sampled visitor decks are drawn from.",
    )
    parser.add_argument("--seeds-per-segment", type=int, default=200)
    parser.add_argument(
        "--catalog-min-plays",
        type=int,
        nargs="+",
        default=(30, 100, 300),
        help="Play floors the retrieval catalog is tested at.",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def load_population(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per segment: the played decks as card ids, and their play counts."""
    connection = duckdb.connect()
    cached = path.resolve().as_posix().replace("'", "''")
    rows = connection.execute(
        f"select seg, deck_key, plays from read_parquet('{cached}') order by seg, plays desc"
    ).fetchall()
    connection.close()

    grouped: dict[str, list[tuple[list[int], int]]] = defaultdict(list)
    for segment, deck_key, plays in rows:
        cards = [int(value) for value in str(deck_key).split(",")]
        if len(cards) == 8:
            grouped[str(segment)].append((cards, int(plays)))
    return {
        segment: (
            np.asarray([cards for cards, _ in entries], dtype=np.int64),
            np.asarray([plays for _, plays in entries], dtype=np.int64),
        )
        for segment, entries in grouped.items()
    }


def membership_matrix(decks: np.ndarray, cards: np.ndarray) -> np.ndarray:
    """One row per deck, one column per card, 1 where the deck holds the card.

    Overlap between two decks is then a dot product, so the whole catalog can be
    compared to a seed in a single matrix-vector product rather than 44,000 set
    intersections.
    """
    lookup = np.full(int(cards.max()) + 1, -1, dtype=np.int64)
    lookup[cards] = np.arange(len(cards))
    matrix = np.zeros((decks.shape[0], len(cards)), dtype=np.float32)
    rows = np.repeat(np.arange(decks.shape[0]), 8)
    matrix[rows, lookup[decks].ravel()] = 1.0
    return matrix


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    population = load_population(args.population)
    log(f"loaded {len(population)} segments")

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed_plays": list(args.seed_plays),
        "seeds_per_segment": args.seeds_per_segment,
        "segments": {},
    }

    low, high = args.seed_plays
    for segment, (decks, plays) in sorted(population.items()):
        cards = np.unique(decks)
        matrix = membership_matrix(decks, cards)
        band = np.flatnonzero((plays >= low) & (plays <= high))
        if band.size == 0:
            continue
        picks = rng.sample(list(band), min(args.seeds_per_segment, band.size))

        entry: dict[str, Any] = {
            "catalog_decks": int(decks.shape[0]),
            "seeds": len(picks),
            "floors": {},
        }
        for floor in args.catalog_min_plays:
            eligible = plays >= floor
            counts: dict[int, list[int]] = {changes: [] for changes in (1, 2, 3, 4)}
            for index in picks:
                overlap = matrix[eligible] @ matrix[index]
                for changes in counts:
                    # A deck `changes` cards away shares 8 - changes cards.
                    counts[changes].append(int((overlap >= 8 - changes).sum()) - 1)
            entry["floors"][str(floor)] = {
                "catalog_decks": int(eligible.sum()),
                "neighbours": {
                    str(changes): {
                        "median": float(np.median(values)),
                        "p10": float(np.percentile(values, 10)),
                        "p90": float(np.percentile(values, 90)),
                        "share_with_none": float(np.mean(np.asarray(values) <= 0)),
                    }
                    for changes, values in counts.items()
                },
            }
        report["segments"][segment] = entry
        log(f"{segment}: {decks.shape[0]:,} catalog decks, {len(picks)} seeds measured")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done — {args.output}")


if __name__ == "__main__":
    main()
