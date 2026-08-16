"""How much does picking the best-scoring deck inflate its score? Measured.

Every previous attempt to size the optimizer's error hit the same wall: the deck
it emits has no battles, so its number cannot be checked. Retrieval removes the
wall. When the candidate set is real decks, the winner of the search has battles
behind it, and the gap between what the model promised and what the deck did is
observable.

This reads `retrieval-benchmark.json`, which recorded the whole scored
neighbourhood of each seed, and asks one question at several candidate-set
sizes: if the search had only been allowed to look at the first N candidates,
how would the winner's predicted score compare to its observed win rate, and how
does that gap grow with N?

The gap has two parts and only one of them is the curse:

* a constant offset, because the model score is measured against the segment's
  top-100 meta panel while the observed rate comes from whoever the deck's
  pilots actually met. That offset applies to every candidate equally.
* a part that grows with N, which is selection: the argmax of N noisy estimates
  climbs even when no candidate is better than another.

Subtracting the neighbourhood's own mean gap isolates the second. Extrapolating
it to the candidate counts the product's own search reaches (`group` scores up
to OPTIMIZE_COMBO_CAP = 4500) is the closest thing to a measurement of the
inflation the site currently displays.
"""

from __future__ import annotations

import argparse
import json
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
        "--benchmark", type=Path, default=root / "artifacts" / "retrieval-benchmark.json"
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=root / "artifacts" / "oriented-deck-results.parquet",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "selection-curse.json"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=(1, 2, 5, 10, 25, 50, 100, 200, 400),
        help="Candidate-set sizes the selection is replayed at.",
    )
    parser.add_argument(
        "--min-battles",
        type=int,
        default=30,
        help="Battles a candidate needs before its observed rate is used.",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=200,
        help="Random candidate sets drawn per seed and per size.",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def load_catalogue(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Segment catalogues in the same order retrieval_benchmark.py indexed them."""
    connection = duckdb.connect()
    cached = path.resolve().as_posix().replace("'", "''")
    rows = connection.execute(
        f"select * from read_parquet('{cached}') order by seg, team_n + opponent_n desc"
    ).fetchall()
    connection.close()

    grouped: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for segment, deck_key, team_n, team_wins, opponent_n, opponent_wins in rows:
        if len(str(deck_key).split(",")) == 8:
            grouped[str(segment)].append(
                (int(team_n), int(team_wins), int(opponent_n), int(opponent_wins))
            )
    catalogue: dict[str, dict[str, np.ndarray]] = {}
    for segment, entries in grouped.items():
        array = np.asarray(entries, dtype=np.float64)
        team_n, team_wins, opponent_n, opponent_wins = array.T
        with np.errstate(invalid="ignore", divide="ignore"):
            team_rate = np.where(team_n > 0, team_wins / team_n, np.nan)
            opponent_rate = np.where(opponent_n > 0, opponent_wins / opponent_n, np.nan)
        catalogue[segment] = {
            "plays": team_n + opponent_n,
            # Equal weight per orientation: the logged player's side wins 0.5299
            # of battles, and a deck does not appear equally often on each side.
            "balanced_rate": 0.5 * (team_rate + opponent_rate),
        }
    return catalogue


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    catalogue = load_catalogue(args.catalogue)
    log(f"{len(benchmark['comparisons'])} comparisons, {len(catalogue)} segments")

    rows: dict[int, list[dict[str, float]]] = {size: [] for size in args.sizes}
    invented_gap: list[float] = []

    for comparison in benchmark["comparisons"]:
        segment = comparison["segment"]
        entry = catalogue.get(segment)
        if entry is None:
            continue
        candidates = comparison["candidates"]
        index = np.asarray(candidates["index"], dtype=np.int64)
        scores = np.asarray(candidates["model_score"], dtype=np.float64)
        observed = entry["balanced_rate"][index]
        plays = entry["plays"][index]

        usable = np.isfinite(observed) & (plays >= args.min_battles)
        # Every size has to be measured on the same seeds. A seed whose
        # neighbourhood cannot supply the largest candidate set would otherwise
        # drop out of the large sizes only, and the curve would then mix the
        # effect of N with a change in which seeds are being averaged — which is
        # what made the first run of this non-monotone.
        if usable.sum() < max(args.sizes):
            continue
        # The neighbourhood's own average gap is the panel-versus-real-opponents
        # offset. What is left after removing it is selection.
        baseline = float(np.mean(scores[usable] - observed[usable]))

        # Candidate sets have to be drawn at random. The recorded neighbourhood
        # is already sorted by model score, so taking its first N would hand
        # every size the same winner and flatten the curve to a constant.
        pool = np.flatnonzero(usable)
        generator = np.random.default_rng(args.seed)
        for size in args.sizes:
            if pool.size < size:
                continue
            for _ in range(args.draws):
                drawn = generator.choice(pool, size=size, replace=False)
                winner = drawn[int(np.argmax(scores[drawn]))]
                rows[size].append(
                    {
                        "predicted": float(scores[winner]),
                        "observed": float(observed[winner]),
                        "gap": float(scores[winner] - observed[winner]),
                        "excess": float(scores[winner] - observed[winner] - baseline),
                        "plays": float(plays[winner]),
                    }
                )

        invented_gap.append(
            float(comparison["invented"]["model_score"] - comparison["retrieved"]["model_score"])
        )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "min_battles": args.min_battles,
        "by_candidate_count": [],
    }
    for size in args.sizes:
        values = rows[size]
        if not values:
            continue
        gap = np.asarray([row["gap"] for row in values])
        excess = np.asarray([row["excess"] for row in values])
        report["by_candidate_count"].append(
            {
                "candidates": size,
                "n": len(values),
                "mean_predicted": float(np.mean([row["predicted"] for row in values])),
                "mean_observed": float(np.mean([row["observed"] for row in values])),
                "mean_gap": float(gap.mean()),
                "mean_excess_over_neighbourhood": float(excess.mean()),
                "excess_ci95": [
                    float(excess.mean() - 1.96 * excess.std(ddof=1) / np.sqrt(excess.size)),
                    float(excess.mean() + 1.96 * excess.std(ddof=1) / np.sqrt(excess.size)),
                ],
            }
        )

    if invented_gap:
        array = np.asarray(invented_gap)
        report["invented_minus_retrieved_model_score"] = {
            "n": int(array.size),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "share_invented_higher": float((array > 0).mean()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done — {args.output}")


if __name__ == "__main__":
    main()
