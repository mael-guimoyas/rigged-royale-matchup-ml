"""What does a second checkpoint buy? Three questions, answered without battles.

Run this as soon as two ensemble members exist. It decides whether to train the
rest, and it does so on the decks that actually matter — the ones the optimizer
emits, which have no battles and whose error is therefore unmeasurable by any
other means.

1. **Spread.** How far apart do members land on an emitted deck, compared with a
   real one? A large gap means the single checkpoint's confidence there is an
   illusion, and averaging will move the number. A small gap means the members
   share their blind spot and no number of extra seeds will help.

2. **Does the spread predict error?** On real decks the observed win rate is
   known, so the spread can be checked against the error it claims to measure.
   MC-dropout failed exactly here — correlation +0.065 with absolute error, flat
   across quintiles — and it is the reason that path was closed. An ensemble
   measures a different thing (disagreement between functions, not between
   activations of one function), but it has to pass the same test before being
   trusted.

3. **What sample splitting changes.** Selecting the winner with member A and
   reporting member B's score for it is unbiased by construction, because B's
   error is independent of A's choice. This replays that on real candidates and
   reports what the displayed number would have become.

The input is `retrieval-benchmark.json`, which already holds real neighbourhoods
with observed win rates plus the deck the beam invented from the same seed.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plausibility_envelope import (  # noqa: E402
    DeckScorer,
    log,
    resolve_device,
    score_decks,
)
from rigged_matchup_ml.predictor import load_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="Ensemble member checkpoints. Order fixes which member reports.",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=root / "artifacts" / "retrieval-benchmark.json"
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=root / "artifacts" / "oriented-deck-results.parquet",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "ensemble-disagreement.json"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=200,
        help="Recorded neighbours rescored per seed, best model score first.",
    )
    parser.add_argument("--min-battles", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_catalogue(path: Path) -> dict[str, dict[str, np.ndarray]]:
    connection = duckdb.connect()
    cached = path.resolve().as_posix().replace("'", "''")
    rows = connection.execute(
        f"select * from read_parquet('{cached}') order by seg, team_n + opponent_n desc"
    ).fetchall()
    connection.close()
    grouped: dict[str, list[Any]] = defaultdict(list)
    for segment, deck_key, team_n, team_wins, opponent_n, opponent_wins in rows:
        cards = [int(value) for value in str(deck_key).split(",")]
        if len(cards) == 8:
            grouped[str(segment)].append(
                (cards, team_n, team_wins, opponent_n, opponent_wins)
            )
    catalogue = {}
    for segment, entries in grouped.items():
        cards = np.asarray([entry[0] for entry in entries], dtype=np.int64)
        stats = np.asarray([entry[1:] for entry in entries], dtype=np.float64)
        team_n, team_wins, opponent_n, opponent_wins = stats.T
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = 0.5 * (
                np.where(team_n > 0, team_wins / team_n, np.nan)
                + np.where(opponent_n > 0, opponent_wins / opponent_n, np.nan)
            )
        catalogue[segment] = {"cards": cards, "plays": team_n + opponent_n, "rate": rate}
    return catalogue


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def main() -> None:
    args = parse_args()
    device = torch.device(resolve_device(args.device))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    catalogue = load_catalogue(args.catalogue)

    bundles = []
    for path in args.checkpoints:
        bundle = load_bundle(path)
        bundle["model"].to(device)
        bundles.append(bundle)
        log(f"loaded {bundle['resolved_model_version']} from {path}")
    patch = max(str(value) for value in bundles[0]["vocabulary"]["patches"])

    # Panels are rebuilt per segment exactly as retrieval_benchmark.py built
    # them, so a member's score here is comparable to the score recorded there.
    panels: dict[str, tuple[list[tuple[int, ...]], list[float]]] = {}
    for segment, entry in catalogue.items():
        order = np.argsort(-entry["plays"])[:100]
        total = float(entry["plays"][order].sum())
        panels[segment] = (
            [tuple(int(card) for card in entry["cards"][index]) for index in order],
            [float(entry["plays"][index] / total) for index in order],
        )

    emitted_spread: list[float] = []
    real_spread: list[float] = []
    real_error: list[float] = []
    real_spread_for_error: list[float] = []
    splitting: list[dict[str, float]] = []

    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for comparison in benchmark["comparisons"]:
        by_segment[comparison["segment"]].append(comparison)

    for segment, comparisons in sorted(by_segment.items()):
        panel_decks, panel_weights = panels[segment]
        entry = catalogue[segment]
        scorers = [DeckScorer(bundle, segment, patch, device) for bundle in bundles]

        for comparison in comparisons:
            index = np.asarray(comparison["candidates"]["index"], dtype=np.int64)[
                : args.candidates
            ]
            observed = entry["rate"][index]
            plays = entry["plays"][index]
            usable = np.isfinite(observed) & (plays >= args.min_battles)
            if usable.sum() < 10:
                continue

            decks = [tuple(int(card) for card in entry["cards"][row]) for row in index]
            invented = tuple(int(card) for card in comparison["invented"]["cards"])

            member_scores = []
            member_invented = []
            for scorer in scorers:
                profiles = score_decks(
                    scorer,
                    [*decks, invented],
                    panel_decks,
                    panel_weights,
                    args.batch_size,
                    "ensemble",
                )
                member_scores.append([profile.score for profile in profiles[:-1]])
                member_invented.append(profiles[-1].score)

            matrix = np.asarray(member_scores)
            emitted_spread.append(float(np.std(member_invented, ddof=1)))
            spread = matrix.std(axis=0, ddof=1)
            real_spread.extend(spread[usable].tolist())

            # Question 2: the spread has to predict the error it claims to
            # measure, on the decks where the error is knowable.
            mean_score = matrix.mean(axis=0)
            error = np.abs(mean_score - observed)
            real_error.extend(error[usable].tolist())
            real_spread_for_error.extend(spread[usable].tolist())

            # Question 3: select with member 0, report with member 1.
            pool = np.flatnonzero(usable)
            winner = pool[int(np.argmax(matrix[0][pool]))]
            splitting.append(
                {
                    "selected_by_a": float(matrix[0][winner]),
                    "reported_by_b": float(matrix[1][winner]),
                    "ensemble_mean": float(mean_score[winner]),
                    "observed": float(observed[winner]),
                }
            )
        log(f"{segment}: {len(splitting)} selections replayed")

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "members": [str(path) for path in args.checkpoints],
        "spread": {
            "emitted_decks": {
                "n": len(emitted_spread),
                "mean": float(np.mean(emitted_spread)) if emitted_spread else None,
                "p90": float(np.percentile(emitted_spread, 90)) if emitted_spread else None,
            },
            "real_decks": {
                "n": len(real_spread),
                "mean": float(np.mean(real_spread)) if real_spread else None,
                "p90": float(np.percentile(real_spread, 90)) if real_spread else None,
            },
        },
        "spread_predicts_error": {
            "correlation": correlation(
                np.asarray(real_spread_for_error), np.asarray(real_error)
            ),
            "note": "MC-dropout scored +0.065 here and was rejected.",
        },
    }

    if splitting:
        a = np.asarray([row["selected_by_a"] for row in splitting])
        b = np.asarray([row["reported_by_b"] for row in splitting])
        mean = np.asarray([row["ensemble_mean"] for row in splitting])
        observed = np.asarray([row["observed"] for row in splitting])
        report["sample_splitting"] = {
            "n": len(splitting),
            "mean_selected_by_a": float(a.mean()),
            "mean_reported_by_b": float(b.mean()),
            "mean_ensemble": float(mean.mean()),
            "mean_observed": float(observed.mean()),
            "gap_single_model": float((a - observed).mean()),
            "gap_held_out_member": float((b - observed).mean()),
            "gap_removed": float((a - observed).mean() - (b - observed).mean()),
        }

    # Quintiles of spread against error: a usable uncertainty has to be monotone
    # here, not merely correlated. This is the table MC-dropout came out flat on.
    if real_error:
        spread_array = np.asarray(real_spread_for_error)
        error_array = np.asarray(real_error)
        edges = np.percentile(spread_array, [0, 20, 40, 60, 80, 100])
        report["error_by_spread_quintile"] = [
            {
                "quintile": position + 1,
                "mean_spread": float(spread_array[mask].mean()),
                "mean_absolute_error": float(error_array[mask].mean()),
                "n": int(mask.sum()),
            }
            for position in range(5)
            for mask in [
                (spread_array >= edges[position])
                & (
                    spread_array <= edges[position + 1]
                    if position == 4
                    else spread_array < edges[position + 1]
                )
            ]
            if mask.any()
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done — {args.output}")


if __name__ == "__main__":
    main()
