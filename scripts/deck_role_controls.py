"""Is the functional-profile detector real, or is it reading the search's pool?

`deck_role_profile.py` separates emitted decks from real ones at AUC 0.917 on a
vector of roles, card kinds and the elixir curve. Before that number is used for
anything, it has to survive the confound that would explain it away for free.

The beam only ever swaps in the segment's forty most-played cards, while the
reference population draws on the whole catalogue. A classifier could reach a
high AUC by learning "built from popular cards only" — which would show up as
exactly these role and elixir shifts, and would say nothing about deck quality.

The control restricts the reference to real decks whose eight cards all sit
inside their own segment's top-forty pool, so both sides face the same card
restriction and only the way the cards are combined can separate them. Three
ablations run alongside, because a detector resting entirely on one feature is a
detector waiting to be gamed.

Everything is matched per segment: an emitted deck is only ever compared with
real decks from the segment it was built for.
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
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deck_role_profile import Catalog, auc  # noqa: E402


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
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root.parent / "riggedroyale" / "data" / "cards.json",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=root / "artifacts" / "retrieval-benchmark.json",
    )
    parser.add_argument(
        "--envelopes",
        type=Path,
        nargs="*",
        default=None,
        help="Extra envelope files; their searches carry a segment, so they match.",
    )
    parser.add_argument(
        "--emitted",
        type=Path,
        default=None,
        help="Optional JSON list of {segment, cards} produced by the live site.",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "deck-role-controls.json"
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=root / "artifacts" / "oriented-deck-results.parquet",
        help="Segment catalogue in the order retrieval_benchmark.py indexed it.",
    )
    parser.add_argument("--search-pool", type=int, default=40)
    parser.add_argument("--reference-per-segment", type=int, default=6000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def load_population(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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


def segment_pool(decks: np.ndarray, plays: np.ndarray, size: int) -> set[int]:
    """The search's own shortlist: most-played cards over the segment's top decks."""
    usage: dict[int, int] = defaultdict(int)
    for row in range(min(400, decks.shape[0])):
        for card in decks[row]:
            usage[int(card)] += int(plays[row])
    ranked = sorted(usage.items(), key=lambda item: (-item[1], item[0]))
    return {card for card, _ in ranked[:size]}


def name_lookup(path: Path) -> dict[str, int]:
    """Resolve a card by slug or display name.

    The live route answers with snapshots whose numeric id is only present on
    the slots the search left alone; the cards it swapped in carry a name, as a
    slug for the ones the optimizer chose and as a display name for the rest.
    Both spellings are indexed so a production deck resolves whole.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, int] = {}
    for card in data["cards"]:
        if card.get("id") is None:
            continue
        for spelling in (card.get("key"), card.get("name")):
            if spelling:
                lookup[str(spelling).strip().lower().replace(" ", "-")] = int(card["id"])
    return lookup


def emitted_by_segment(args: argparse.Namespace) -> dict[str, list[list[int]]]:
    found: dict[str, list[list[int]]] = defaultdict(list)
    if args.benchmark.exists():
        payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
        for row in payload["comparisons"]:
            found[row["segment"]].append(row["invented"]["cards"])
    for path in args.envelopes or []:
        if not Path(path).exists():
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for segment, entry in payload.get("segments", {}).items():
            for search in entry.get("searches") or []:
                found[segment].append(search["trajectory"][-1]["cards"])
    if args.emitted and args.emitted.exists():
        lookup = name_lookup(args.catalog)
        skipped = 0
        for row in json.loads(args.emitted.read_text(encoding="utf-8")):
            cards: list[int] = []
            for position, card in enumerate(row["cards"]):
                if card is not None:
                    cards.append(int(card))
                    continue
                spelling = (row.get("card_names") or [None] * 8)[position]
                resolved = lookup.get(
                    str(spelling).strip().lower().replace(" ", "-")
                ) if spelling else None
                if resolved is None:
                    break
                cards.append(resolved)
            if len(cards) == 8:
                found[row["segment"]].append(cards)
            else:
                skipped += 1
        if skipped:
            log(f"{skipped} emitted decks skipped: a card could not be resolved")
    return dict(found)


def cross_validated_auc(
    positive: np.ndarray, negative: np.ndarray, folds: int, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
    """Out-of-fold AUC, holding out emitted decks fold by fold.

    The negative class dwarfs the positive one, so the folds are cut on the
    positives; the negatives are split too, otherwise the same real decks would
    appear in training and in evaluation.
    """
    if positive.shape[0] < folds * 2 or negative.shape[0] < folds * 2:
        return float("nan"), np.zeros(positive.shape[1])
    positive_order = rng.permutation(positive.shape[0])
    negative_order = rng.permutation(negative.shape[0])
    scores: list[float] = []
    labels: list[int] = []
    weights = np.zeros(positive.shape[1])
    for fold in range(folds):
        positive_test = positive_order[fold::folds]
        negative_test = negative_order[fold::folds]
        positive_train = np.setdiff1d(positive_order, positive_test)
        negative_train = np.setdiff1d(negative_order, negative_test)
        features = np.vstack([negative[negative_train], positive[positive_train]])
        target = np.concatenate(
            [np.zeros(negative_train.size), np.ones(positive_train.size)]
        )
        model = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            features, target
        )
        weights += model.coef_[0] / folds
        scores.extend(model.decision_function(positive[positive_test]).tolist())
        labels.extend([1] * positive_test.size)
        scores.extend(model.decision_function(negative[negative_test]).tolist())
        labels.extend([0] * negative_test.size)
    score_array = np.asarray(scores)
    label_array = np.asarray(labels)
    return auc(score_array[label_array == 1], score_array[label_array == 0]), weights


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    catalog = Catalog(args.catalog)
    population = load_population(args.population)
    emitted = emitted_by_segment(args)
    log(f"{len(population)} segments, emitted decks in {len(emitted)} of them")

    names = catalog.feature_names
    elixir_columns = [
        position for position, name in enumerate(names) if name.startswith("elixir")
    ]
    role_columns = [
        position for position, name in enumerate(names) if name.startswith("role:")
    ]

    open_positive: list[np.ndarray] = []
    open_negative: list[np.ndarray] = []
    pooled_positive: list[np.ndarray] = []
    pooled_negative: list[np.ndarray] = []
    coverage: list[dict[str, Any]] = []

    for segment, decks_list in sorted(emitted.items()):
        if segment not in population:
            continue
        decks, plays = population[segment]
        pool = segment_pool(decks, plays, args.search_pool)
        inside = np.asarray(
            [all(int(card) in pool for card in row) for row in decks], dtype=bool
        )
        band = np.flatnonzero((plays >= 30) & (plays <= 300))
        restricted = np.asarray([row for row in band if inside[row]], dtype=np.int64)

        chosen = rng.choice(
            band, min(args.reference_per_segment, band.size), replace=False
        )
        chosen_restricted = (
            rng.choice(
                restricted, min(args.reference_per_segment, restricted.size), replace=False
            )
            if restricted.size
            else np.empty(0, dtype=np.int64)
        )

        emitted_array = np.asarray(decks_list, dtype=np.int64)
        emitted_inside = float(
            np.mean([all(int(card) in pool for card in row) for row in emitted_array])
        )
        coverage.append(
            {
                "segment": segment,
                "emitted": int(emitted_array.shape[0]),
                "emitted_inside_pool": emitted_inside,
                "real_in_band": int(band.size),
                "real_inside_pool": int(restricted.size),
                "real_inside_pool_share": float(restricted.size / band.size)
                if band.size
                else 0.0,
            }
        )

        open_positive.append(catalog.profile(emitted_array))
        open_negative.append(catalog.profile(decks[chosen]))
        if chosen_restricted.size >= 50:
            pooled_positive.append(catalog.profile(emitted_array))
            pooled_negative.append(catalog.profile(decks[chosen_restricted]))

    def stack(blocks: list[np.ndarray]) -> np.ndarray:
        return np.vstack(blocks) if blocks else np.empty((0, len(names)), dtype=np.float32)

    # The decisive control: compare each emitted deck against the *real* decks
    # the same search could have returned — same seed, same four-card radius,
    # same segment. Both sides then share seed, distance and card pool, and the
    # only thing left to separate them is whether a human ever assembled it.
    matched_positive: list[np.ndarray] = []
    matched_negative: list[np.ndarray] = []
    if args.benchmark.exists() and args.catalogue.exists():
        connection = duckdb.connect()
        cached = args.catalogue.resolve().as_posix().replace("'", "''")
        rows = connection.execute(
            f"select seg, deck_key, team_n + opponent_n as plays "
            f"from read_parquet('{cached}') order by seg, plays desc"
        ).fetchall()
        connection.close()
        catalogue_by_segment: dict[str, list[list[int]]] = defaultdict(list)
        for segment, deck_key, _ in rows:
            cards = [int(value) for value in str(deck_key).split(",")]
            if len(cards) == 8:
                catalogue_by_segment[str(segment)].append(cards)

        payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
        for row in payload["comparisons"]:
            entries = catalogue_by_segment.get(row["segment"])
            if not entries:
                continue
            index = row["candidates"]["index"]
            neighbours = [entries[position] for position in index if position < len(entries)]
            if len(neighbours) < 20:
                continue
            matched_positive.append(
                catalog.profile(np.asarray([row["invented"]["cards"]], dtype=np.int64))
            )
            matched_negative.append(catalog.profile(np.asarray(neighbours, dtype=np.int64)))
        log(
            f"matched control: {len(matched_positive)} emitted against "
            f"{sum(block.shape[0] for block in matched_negative):,} real neighbours"
        )

    configurations: dict[str, tuple[np.ndarray, np.ndarray, list[int] | None]] = {
        "baseline_open_reference": (stack(open_positive), stack(open_negative), None),
        "control_matched_neighbourhood": (
            stack(matched_positive),
            stack(matched_negative),
            None,
        ),
        "control_matched_roles_only": (
            stack(matched_positive),
            stack(matched_negative),
            role_columns,
        ),
        "control_reference_inside_search_pool": (
            stack(pooled_positive),
            stack(pooled_negative),
            None,
        ),
        "ablation_no_elixir": (
            stack(pooled_positive),
            stack(pooled_negative),
            [position for position in range(len(names)) if position not in elixir_columns],
        ),
        "ablation_roles_only": (
            stack(pooled_positive),
            stack(pooled_negative),
            role_columns,
        ),
        "ablation_elixir_only": (
            stack(pooled_positive),
            stack(pooled_negative),
            elixir_columns,
        ),
    }

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "search_pool": args.search_pool,
        "coverage": coverage,
        "configurations": {},
    }

    for name, (positive, negative, columns) in configurations.items():
        if positive.shape[0] == 0 or negative.shape[0] == 0:
            continue
        selected = columns if columns is not None else list(range(len(names)))
        centre = negative[:, selected].mean(axis=0)
        scale = negative[:, selected].std(axis=0)
        scale[scale == 0] = 1.0
        value, weights = cross_validated_auc(
            (positive[:, selected] - centre) / scale,
            (negative[:, selected] - centre) / scale,
            args.folds,
            np.random.default_rng(args.seed),
        )
        report["configurations"][name] = {
            "auc_out_of_fold": value,
            "emitted": int(positive.shape[0]),
            "real": int(negative.shape[0]),
            "features": len(selected),
            "weights": [
                {"feature": names[selected[position]], "weight": float(weights[position])}
                for position in np.argsort(-np.abs(weights))[:10]
            ],
        }
        log(f"{name}: AUC {value:.3f} on {positive.shape[0]} emitted")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done — {args.output}")


if __name__ == "__main__":
    main()
