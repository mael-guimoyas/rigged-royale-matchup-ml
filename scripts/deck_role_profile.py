"""Does a deck's functional shape give it away, where its card identities do not?

Every detector tried so far worked on card identities — pairwise PMI, attested
three- and four-card groups — or on single role floors taken one at a time, of
the "at least one win condition, at least two anti-air" kind. The decks the
optimizer emits pass all of them, and the reason is visible once stated: they
are built from popular cards in combinations that each clear every marginal
threshold while sitting nowhere near the joint distribution.

A player does not read a deck that way. They read its shape all at once: how
many win conditions, whether the ranged answers are affordable, whether anything
handles swarm, whether the elixir curve has a middle. Five ground melee troops
with one spell and no building is instantly wrong to them, even though each card
is popular and every floor is satisfied.

This turns that reading into a vector — role counts, card kinds, what can hit
air, and the elixir curve rather than its mean — and asks whether real decks and
emitted decks occupy different regions of it. The test is the same one that
closed the earlier pistes: reference and evaluation populations are disjoint,
and the answer is an AUC, not an anecdote.
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


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decks",
        type=Path,
        default=root / "artifacts" / "deck-plays-all.parquet",
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
        help="Source of the emitted decks, read from the `invented` entries.",
    )
    parser.add_argument(
        "--envelopes",
        type=Path,
        nargs="*",
        default=None,
        help="Extra plausibility-envelope files whose final search decks count as emitted.",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "deck-role-profile.json"
    )
    parser.add_argument("--reference-min-plays", type=int, default=30)
    parser.add_argument("--reference-max-plays", type=int, default=300)
    parser.add_argument("--reference-sample", type=int, default=30000)
    parser.add_argument("--holdout-sample", type=int, default=5000)
    parser.add_argument("--neighbours", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


class Catalog:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.by_id: dict[int, dict[str, Any]] = {}
        roles: set[str] = set()
        kinds: set[str] = set()
        for card in data["cards"]:
            if card.get("id") is None:
                continue
            self.by_id[int(card["id"])] = card
            roles.update(card.get("roles") or [])
            if card.get("kind"):
                kinds.add(str(card["kind"]))
        self.roles = sorted(roles)
        self.kinds = sorted(kinds)
        # Elixir bands rather than the mean: a deck of four 1-cost and four
        # 6-cost cards averages the same as a curve, and reads completely
        # differently to a player.
        self.bands = [(0, 2), (3, 3), (4, 4), (5, 5), (6, 99)]

    @property
    def feature_names(self) -> list[str]:
        return (
            [f"role:{role}" for role in self.roles]
            + [f"kind:{kind}" for kind in self.kinds]
            + ["hits_air", "air_unit", "elixir_mean", "elixir_std"]
            + [f"elixir:{low}-{high}" for low, high in self.bands]
        )

    def profile(self, decks: np.ndarray) -> np.ndarray:
        width = len(self.feature_names)
        output = np.zeros((decks.shape[0], width), dtype=np.float32)
        role_index = {role: position for position, role in enumerate(self.roles)}
        kind_offset = len(self.roles)
        kind_index = {kind: kind_offset + position for position, kind in enumerate(self.kinds)}
        extra = kind_offset + len(self.kinds)

        cache: dict[int, np.ndarray] = {}
        for card_id, card in self.by_id.items():
            vector = np.zeros(width, dtype=np.float32)
            for role in card.get("roles") or []:
                vector[role_index[role]] = 1.0
            if card.get("kind"):
                vector[kind_index[str(card["kind"])]] = 1.0
            targets = str(card.get("targets") or "")
            if "air" in targets.lower():
                vector[extra] = 1.0
            if str(card.get("transport") or "").lower() == "air":
                vector[extra + 1] = 1.0
            cache[card_id] = vector

        unknown = np.zeros(width, dtype=np.float32)
        for row in range(decks.shape[0]):
            elixir = []
            for card_id in decks[row]:
                output[row] += cache.get(int(card_id), unknown)
                card = self.by_id.get(int(card_id))
                if card and card.get("elixir"):
                    elixir.append(int(card["elixir"]))
            if elixir:
                values = np.asarray(elixir, dtype=np.float32)
                output[row, extra + 2] = values.mean()
                output[row, extra + 3] = values.std()
                for position, (low, high) in enumerate(self.bands):
                    output[row, extra + 4 + position] = int(
                        ((values >= low) & (values <= high)).sum()
                    )
        return output


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return float(
        (ranks[: positive.size].sum() - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def load_decks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    connection = duckdb.connect()
    cached = path.resolve().as_posix().replace("'", "''")
    columns = connection.execute(f"select * from read_parquet('{cached}')").fetchnumpy()
    connection.close()
    decks = np.column_stack(
        [np.asarray(columns[f"card{slot}"], dtype=np.int64) for slot in range(1, 9)]
    )
    return decks, np.asarray(columns["plays"], dtype=np.float64)


def emitted_decks(args: argparse.Namespace) -> np.ndarray:
    found: list[list[int]] = []
    if args.benchmark.exists():
        payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
        found.extend(row["invented"]["cards"] for row in payload["comparisons"])
    for path in args.envelopes or []:
        if not Path(path).exists():
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for entry in payload.get("segments", {}).values():
            for search in entry.get("searches") or []:
                found.append(search["trajectory"][-1]["cards"])
    return np.asarray(found, dtype=np.int64)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    catalog = Catalog(args.catalog)
    decks, plays = load_decks(args.decks)
    log(f"{len(decks):,} decks, {len(catalog.feature_names)} profile features")

    band = np.flatnonzero(
        (plays >= args.reference_min_plays) & (plays <= args.reference_max_plays)
    )
    picked = rng.choice(
        band, min(args.reference_sample + args.holdout_sample, band.size), replace=False
    )
    reference_rows = picked[: args.reference_sample]
    holdout_rows = picked[args.reference_sample :]

    emitted = emitted_decks(args)
    log(
        f"reference {reference_rows.size:,} | holdout {holdout_rows.size:,} | "
        f"emitted {emitted.shape[0]:,}"
    )

    reference = catalog.profile(decks[reference_rows])
    holdout = catalog.profile(decks[holdout_rows])
    generated = catalog.profile(emitted)

    centre = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale[scale == 0] = 1.0

    def standardise(values: np.ndarray) -> np.ndarray:
        return (values - centre) / scale

    reference_z = standardise(reference)
    holdout_z = standardise(holdout)
    generated_z = standardise(generated)

    def novelty(query: np.ndarray) -> np.ndarray:
        """Mean distance to the k nearest real profiles, in chunks."""
        output = np.empty(query.shape[0], dtype=np.float64)
        for start in range(0, query.shape[0], 512):
            block = query[start : start + 512]
            distances = np.linalg.norm(
                block[:, None, :] - reference_z[None, :, :], axis=2
            )
            nearest = np.partition(distances, args.neighbours, axis=1)[:, : args.neighbours]
            output[start : start + block.shape[0]] = nearest.mean(axis=1)
        return output

    log("scoring holdout")
    holdout_score = novelty(holdout_z)
    log("scoring emitted")
    generated_score = novelty(generated_z)

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": catalog.feature_names,
        "reference_decks": int(reference_rows.size),
        "holdout_decks": int(holdout_rows.size),
        "emitted_decks": int(emitted.shape[0]),
        "novelty": {
            "holdout_mean": float(holdout_score.mean()),
            "emitted_mean": float(generated_score.mean()),
            "auc_emitted_vs_real": auc(generated_score, holdout_score),
            "emitted_above_real_p95": float(
                (generated_score > np.percentile(holdout_score, 95)).mean()
            ),
            "emitted_above_real_p99": float(
                (generated_score > np.percentile(holdout_score, 99)).mean()
            ),
        },
    }

    # Which parts of the shape differ, in standard deviations of the real
    # population. This is the half a player could actually name.
    difference = (generated.mean(axis=0) - reference.mean(axis=0)) / scale
    order = np.argsort(-np.abs(difference))
    report["feature_shifts"] = [
        {
            "feature": catalog.feature_names[position],
            "real_mean": float(reference[:, position].mean()),
            "emitted_mean": float(generated[:, position].mean()),
            "shift_in_sd": float(difference[position]),
            "auc_single_feature": auc(
                generated[:, position], holdout[:, position]
            ),
        }
        for position in order[:15]
    ]

    # A distance in 33 standardised dimensions treats every feature as equally
    # informative, and most of them are noise here, so the nearest-neighbour
    # measure above concentrates and reads as chance. Learning the directions
    # that matter is what a player does implicitly. Folds hold out emitted decks
    # so the reported AUC is out-of-sample on the small side of the problem.
    log("fitting the discriminator")
    from sklearn.linear_model import LogisticRegression

    folds = 5
    order = rng.permutation(generated_z.shape[0])
    scores: list[float] = []
    labels: list[int] = []
    coefficients = np.zeros(reference_z.shape[1])
    for fold in range(folds):
        test_rows = order[fold::folds]
        train_rows = np.setdiff1d(order, test_rows)
        features = np.vstack([reference_z, generated_z[train_rows]])
        target = np.concatenate(
            [np.zeros(reference_z.shape[0]), np.ones(train_rows.size)]
        )
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0
        ).fit(features, target)
        coefficients += model.coef_[0] / folds
        scores.extend(model.decision_function(generated_z[test_rows]).tolist())
        labels.extend([1] * test_rows.size)
        scores.extend(model.decision_function(holdout_z).tolist())
        labels.extend([0] * holdout_z.shape[0])

    score_array = np.asarray(scores)
    label_array = np.asarray(labels)
    report["discriminator"] = {
        "auc_out_of_fold": auc(
            score_array[label_array == 1], score_array[label_array == 0]
        ),
        "folds": folds,
        "weights": [
            {"feature": catalog.feature_names[position], "weight": float(coefficients[position])}
            for position in np.argsort(-np.abs(coefficients))[:12]
        ],
    }
    # What a threshold would cost: the share of real decks it rejects to catch a
    # given share of emitted ones. A guard rail is only usable if that trade is
    # cheap on the left of this curve.
    emitted_scores = score_array[label_array == 1]
    real_scores = score_array[label_array == 0]
    report["discriminator"]["operating_points"] = [
        {
            "real_decks_rejected": share,
            "emitted_decks_caught": float(
                (emitted_scores > np.percentile(real_scores, 100 * (1 - share))).mean()
            ),
        }
        for share in (0.01, 0.05, 0.10, 0.20)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done — {args.output}")


if __name__ == "__main__":
    main()
