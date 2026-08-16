"""Recommend a played deck, or invent one? Same seeds, same budget, both answers.

`retrieval_feasibility.py` shows the real population is dense enough to answer a
three- or four-card change. This script asks whether it is any *good*: for the
same visitor decks and the same change budget, it puts the greedy beam's invented
deck next to the best genuinely-played deck the same budget can reach.

The comparison is deliberately unfair to retrieval in one way and fair in
another. Unfair: the beam may reach any legal deck, retrieval only decks with
battles behind them, so the beam should win on model score by construction.
Fair: the retrieved deck's number can be *checked*. It has real battles, so the
script reports its observed win rate next to its predicted one, and the gap
between them is the only honest error measurement available anywhere in this
comparison — the invented deck has no battles and its number can never be
falsified.

Win rates are computed per orientation and averaged with equal weight. The
corpus wins 0.5299 of battles on the logged player's side, and a given deck does
not appear equally often on both sides, so pooling the two orientations would
score a deck by where it tends to sit in the log rather than by how it plays.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plausibility_envelope import (  # noqa: E402
    POOLED_SEGMENT_SQL,
    DeckScorer,
    SegmentPanel,
    beam_search,
    log,
    raw_glob,
    resolve_device,
    score_decks,
)
from rigged_matchup_ml.predictor import load_bundle  # noqa: E402

# The two orientations are aggregated separately and joined. Unioning them into
# one relation with a side marker doubles the group-by input to 56 million rows
# carrying a string column, which exhausts a 4 GB budget; two grouped halves of
# 28 million integers each do not.
ORIENTED_SQL = """
        with base as (
          select {pooled} as seg, team_deck_key, opponent_deck_key, win
          from read_parquet('{parquet}')
          where segment not in ('ladder:top-100', 'ladder:top-1000')
        ), team as (
          select seg, team_deck_key as deck_key,
                 count(*) as n, count(*) filter (where win) as wins
          from base group by seg, team_deck_key
        ), opponent as (
          select seg, opponent_deck_key as deck_key,
                 count(*) as n, count(*) filter (where not win) as wins
          from base group by seg, opponent_deck_key
        )
        select
          coalesce(team.seg, opponent.seg) as seg,
          coalesce(team.deck_key, opponent.deck_key) as deck_key,
          coalesce(team.n, 0) as team_n,
          coalesce(team.wins, 0) as team_wins,
          coalesce(opponent.n, 0) as opponent_n,
          coalesce(opponent.wins, 0) as opponent_wins
        from team full outer join opponent
          on team.seg = opponent.seg and team.deck_key = opponent.deck_key
        where coalesce(team.n, 0) + coalesce(opponent.n, 0) >= {min_plays}
"""


class Catalogue:
    """One segment's played decks, with orientation-split results."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.cards = np.asarray([row[0] for row in rows], dtype=np.int64)
        self.team_n = np.asarray([row[1] for row in rows], dtype=np.float64)
        self.team_wins = np.asarray([row[2] for row in rows], dtype=np.float64)
        self.opponent_n = np.asarray([row[3] for row in rows], dtype=np.float64)
        self.opponent_wins = np.asarray([row[4] for row in rows], dtype=np.float64)
        self.plays = self.team_n + self.opponent_n
        self.vocabulary = np.unique(self.cards)
        lookup = np.full(int(self.vocabulary.max()) + 1, -1, dtype=np.int64)
        lookup[self.vocabulary] = np.arange(len(self.vocabulary))
        self.membership = np.zeros(
            (self.cards.shape[0], len(self.vocabulary)), dtype=np.float32
        )
        rows_index = np.repeat(np.arange(self.cards.shape[0]), 8)
        self.membership[rows_index, lookup[self.cards].ravel()] = 1.0

    def balanced_win_rate(self, index: int) -> float | None:
        """Mean of the two orientation win rates, or None if a side is empty."""
        if self.team_n[index] <= 0 or self.opponent_n[index] <= 0:
            return None
        team = self.team_wins[index] / self.team_n[index]
        opponent = self.opponent_wins[index] / self.opponent_n[index]
        return float(0.5 * (team + opponent))

    def neighbours(self, index: int, changes: int) -> np.ndarray:
        overlap = self.membership @ self.membership[index]
        found = np.flatnonzero(overlap >= 8 - changes)
        return found[found != index]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / "data" / "raw")
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "artifacts" / "matchup-model.pt"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "artifacts" / "oriented-deck-results.parquet",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "retrieval-benchmark.json"
    )
    parser.add_argument("--segments", nargs="*", default=None)
    parser.add_argument("--min-plays", type=int, default=30)
    parser.add_argument("--panel-size", type=int, default=100)
    parser.add_argument("--seeds-per-segment", type=int, default=25)
    parser.add_argument(
        "--seed-plays", type=int, nargs=2, default=(30, 300), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--changes",
        type=int,
        default=4,
        help="Card changes the beam is allowed, and the retrieval radius.",
    )
    parser.add_argument(
        "--retrieval-cap",
        type=int,
        default=2000,
        help="Most-played neighbours scored per seed, to bound the sweep.",
    )
    parser.add_argument(
        "--keep-candidates",
        type=int,
        default=400,
        help="Scored neighbours recorded per seed, best model score first.",
    )
    parser.add_argument("--search-pool", type=int, default=40)
    parser.add_argument("--search-beam", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def oriented_rows(args: argparse.Namespace) -> dict[str, Catalogue]:
    cache = args.cache
    sidecar = cache.with_suffix(".meta.json")
    signature = {"raw": str(args.raw.resolve()), "min_plays": args.min_plays}
    connection = duckdb.connect()
    connection.execute(f"pragma threads={args.threads}")
    connection.execute(f"pragma memory_limit='{args.memory_limit}'")
    # Row order is irrelevant here and preserving it holds whole partitions in
    # memory during the aggregation.
    connection.execute("set preserve_insertion_order=false")
    fresh = (
        not args.rebuild
        and cache.exists()
        and sidecar.exists()
        and json.loads(sidecar.read_text(encoding="utf-8")) == signature
    )
    if fresh:
        log(f"reading cached oriented results from {cache}")
    else:
        log("aggregating orientation-split results per deck")
        cache.parent.mkdir(parents=True, exist_ok=True)
        query = ORIENTED_SQL.format(
            pooled=POOLED_SEGMENT_SQL,
            parquet=raw_glob(args.raw),
            min_plays=args.min_plays,
        )
        target = cache.resolve().as_posix().replace("'", "''")
        connection.execute(f"copy ({query}) to '{target}' (format parquet)")
        sidecar.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        log(f"cached oriented results to {cache}")

    cached = cache.resolve().as_posix().replace("'", "''")
    rows = connection.execute(
        f"select * from read_parquet('{cached}') order by seg, team_n + opponent_n desc"
    ).fetchall()
    connection.close()

    grouped: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for segment, deck_key, team_n, team_wins, opponent_n, opponent_wins in rows:
        cards = [int(value) for value in str(deck_key).split(",")]
        if len(cards) == 8:
            grouped[str(segment)].append(
                (cards, int(team_n), int(team_wins), int(opponent_n), int(opponent_wins))
            )
    return {segment: Catalogue(entries) for segment, entries in grouped.items()}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    device = torch.device(resolve_device(args.device))
    bundle = load_bundle(args.checkpoint)
    bundle["model"].to(device)
    patch = max(str(value) for value in bundle["vocabulary"]["patches"])
    vocabulary_cards = {int(card) for card in bundle["vocabulary"]["cards"]}
    log(
        f"loaded model {bundle['resolved_model_version']} on "
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}"
    )

    catalogues = oriented_rows(args)
    segments = args.segments or sorted(catalogues)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "changes": args.changes,
        "min_plays": args.min_plays,
        "seed_plays": list(args.seed_plays),
        "comparisons": [],
    }

    low, high = args.seed_plays
    for segment in segments:
        catalogue = catalogues.get(segment)
        if catalogue is None:
            continue
        order = np.argsort(-catalogue.plays)
        top = order[: args.panel_size]
        total = float(catalogue.plays[top].sum())
        panel = SegmentPanel(
            segment=segment,
            decks=[tuple(int(card) for card in catalogue.cards[index]) for index in top],
            weights=[float(catalogue.plays[index] / total) for index in top],
        )
        scorer = DeckScorer(bundle, segment, patch, device)

        usage: dict[int, int] = defaultdict(int)
        for index in order[:400]:
            for card in catalogue.cards[index]:
                if int(card) in vocabulary_cards:
                    usage[int(card)] += int(catalogue.plays[index])
        pool = [card for card, _ in sorted(usage.items(), key=lambda item: (-item[1], item[0]))][
            : args.search_pool
        ]

        band = np.flatnonzero(
            (catalogue.plays >= low) & (catalogue.plays <= high)
        )
        if band.size == 0:
            continue
        picks = rng.sample(list(band), min(args.seeds_per_segment, band.size))

        for pick in picks:
            seed_cards = tuple(int(card) for card in catalogue.cards[pick])
            found = catalogue.neighbours(pick, args.changes)
            if found.size == 0:
                continue
            ranked = found[np.argsort(-catalogue.plays[found])][: args.retrieval_cap]
            neighbours = [
                tuple(int(card) for card in catalogue.cards[index]) for index in ranked
            ]
            profiles = score_decks(
                scorer,
                [seed_cards, *neighbours],
                panel.decks,
                panel.weights,
                args.batch_size,
                f"retrieval {len(neighbours):,}",
            )
            seed_profile = profiles[0]
            neighbour_scores = np.asarray([profile.score for profile in profiles[1:]])
            best_local = int(np.argmax(neighbour_scores))
            best_index = int(ranked[best_local])
            best_profile = profiles[1 + best_local]
            keep = np.argsort(-neighbour_scores)[: args.keep_candidates].tolist()

            trajectory = beam_search(
                scorer,
                seed_cards,
                pool,
                panel,
                args.changes,
                args.search_beam,
                args.batch_size,
            )
            invented = trajectory[-1]

            report["comparisons"].append(
                {
                    "segment": segment,
                    "seed": {
                        "cards": list(seed_cards),
                        "model_score": seed_profile.score,
                        "plays": int(catalogue.plays[pick]),
                        "observed_win_rate": catalogue.balanced_win_rate(pick),
                    },
                    "retrieved": {
                        "cards": [int(card) for card in catalogue.cards[best_index]],
                        "model_score": best_profile.score,
                        "unfavorable_share": best_profile.unfavorable_share,
                        "plays": int(catalogue.plays[best_index]),
                        "observed_win_rate": catalogue.balanced_win_rate(best_index),
                        "candidates": len(neighbours),
                    },
                    "invented": {
                        "cards": invented["cards"],
                        "model_score": invented["profile"]["score"],
                        "unfavorable_share": invented["profile"]["unfavorable_share"],
                        "candidates_considered": sum(
                            step.get("candidates_considered", 0) for step in trajectory[1:]
                        ),
                    },
                    # The scored neighbourhood, kept so alternative ranking rules
                    # can be compared offline without re-running the sweep.
                    # `index` addresses the segment catalogue as loaded here:
                    # oriented-deck-results.parquet, ordered by total plays.
                    "candidates": {
                        "index": [int(ranked[position]) for position in keep],
                        "model_score": [
                            round(profiles[1 + position].score, 6) for position in keep
                        ],
                        "unfavorable_share": [
                            round(profiles[1 + position].unfavorable_share, 6)
                            for position in keep
                        ],
                    },
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"{segment}: {len(report['comparisons'])} comparisons written")

    log(f"done — {args.output}")


if __name__ == "__main__":
    main()
