"""Freeze the functional-profile detector into coefficients the site can apply.

The detector separates decks the optimizer emitted from decks players actually
build, at AUC 0.898 out of fold on live-route output. This fits the shipping
version and writes it as plain numbers: one weight per feature on the raw
(unstandardised) profile, one intercept, and a threshold.

Two decisions are made here rather than left to the caller.

Fitted on production output only. The Python reimplementation of the beam emits
decks that are *more* expensive than real ones while the live route emits
*cheaper* ones, so a model fitted on the reimplementation would carry the wrong
sign. The transferable quantity is the AUC, not the coefficients.

Kept small and heavily regularised. Forty positives over thirty-odd correlated
features identifies the direction but not the individual coefficients, so the
sweep below reports out-of-fold AUC per regularisation strength and the chosen
one errs toward the flat end.
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

from deck_role_controls import name_lookup  # noqa: E402
from deck_role_profile import Catalog, auc  # noqa: E402


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emitted",
        type=Path,
        default=root / "artifacts" / "production-optimizer-decks.json",
    )
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
        "--output", type=Path, default=root / "artifacts" / "deck-plausibility-model.json"
    )
    parser.add_argument("--reference-per-segment", type=int, default=6000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--flag-rate",
        type=float,
        default=0.05,
        help="Share of real decks the threshold is allowed to flag.",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def load_population(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    connection = duckdb.connect()
    cached = path.resolve().as_posix().replace("'", "''")
    rows = connection.execute(
        f"select seg, deck_key, plays from read_parquet('{cached}')"
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


def resolve_emitted(path: Path, catalog_path: Path) -> dict[str, list[list[int]]]:
    lookup = name_lookup(catalog_path)
    found: dict[str, list[list[int]]] = defaultdict(list)
    for row in json.loads(path.read_text(encoding="utf-8")):
        cards: list[int] = []
        for position, card in enumerate(row["cards"]):
            if card is not None:
                cards.append(int(card))
                continue
            spelling = (row.get("card_names") or [None] * 8)[position]
            resolved = (
                lookup.get(str(spelling).strip().lower().replace(" ", "-"))
                if spelling
                else None
            )
            if resolved is None:
                break
            cards.append(resolved)
        if len(cards) == 8:
            found[row["segment"]].append(cards)
    return dict(found)


def cross_validated(
    positive: np.ndarray, negative: np.ndarray, strength: float, folds: int, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    positive_order = rng.permutation(positive.shape[0])
    negative_order = rng.permutation(negative.shape[0])
    scores: list[float] = []
    labels: list[int] = []
    for fold in range(folds):
        positive_test = positive_order[fold::folds]
        negative_test = negative_order[fold::folds]
        positive_train = np.setdiff1d(positive_order, positive_test)
        negative_train = np.setdiff1d(negative_order, negative_test)
        model = LogisticRegression(
            max_iter=5000, class_weight="balanced", C=strength
        ).fit(
            np.vstack([negative[negative_train], positive[positive_train]]),
            np.concatenate([np.zeros(negative_train.size), np.ones(positive_train.size)]),
        )
        scores.extend(model.decision_function(positive[positive_test]).tolist())
        labels.extend([1] * positive_test.size)
        scores.extend(model.decision_function(negative[negative_test]).tolist())
        labels.extend([0] * negative_test.size)
    array = np.asarray(scores)
    mask = np.asarray(labels)
    return auc(array[mask == 1], array[mask == 0])


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    catalog = Catalog(args.catalog)
    population = load_population(args.population)
    emitted = resolve_emitted(args.emitted, args.catalog)

    positive_blocks: list[np.ndarray] = []
    negative_blocks: list[np.ndarray] = []
    for segment, decks_list in sorted(emitted.items()):
        if segment not in population:
            continue
        decks, plays = population[segment]
        band = np.flatnonzero((plays >= 30) & (plays <= 300))
        chosen = rng.choice(band, min(args.reference_per_segment, band.size), replace=False)
        positive_blocks.append(catalog.profile(np.asarray(decks_list, dtype=np.int64)))
        negative_blocks.append(catalog.profile(decks[chosen]))

    positive = np.vstack(positive_blocks)
    negative = np.vstack(negative_blocks)
    log(f"{positive.shape[0]} emitted decks, {negative.shape[0]:,} real decks")

    centre = negative.mean(axis=0)
    scale = negative.std(axis=0)
    scale[scale == 0] = 1.0
    positive_z = (positive - centre) / scale
    negative_z = (negative - centre) / scale

    sweep = []
    for strength in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        value = cross_validated(
            positive_z, negative_z, strength, args.folds, args.seed
        )
        sweep.append({"C": strength, "auc_out_of_fold": value})
        log(f"C={strength}: AUC {value:.3f}")

    best = max(sweep, key=lambda row: row["auc_out_of_fold"])
    model = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=best["C"]
    ).fit(
        np.vstack([negative_z, positive_z]),
        np.concatenate([np.zeros(negative_z.shape[0]), np.ones(positive_z.shape[0])]),
    )

    # Fold the standardisation into the coefficients so the site applies one
    # dot product to the raw profile and needs no fitted statistics of its own.
    raw_weights = model.coef_[0] / scale
    raw_intercept = float(model.intercept_[0] - np.sum(model.coef_[0] * centre / scale))

    real_scores = negative @ raw_weights + raw_intercept
    emitted_scores = positive @ raw_weights + raw_intercept
    threshold = float(np.percentile(real_scores, 100 * (1 - args.flag_rate)))

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fitted_on": str(args.emitted),
        "emitted_decks": int(positive.shape[0]),
        "real_decks": int(negative.shape[0]),
        "regularisation_sweep": sweep,
        "chosen_C": best["C"],
        "auc_out_of_fold": best["auc_out_of_fold"],
        "threshold": threshold,
        "flag_rate_real": float((real_scores > threshold).mean()),
        "flag_rate_emitted": float((emitted_scores > threshold).mean()),
        "intercept": raw_intercept,
        "weights": {
            name: float(weight)
            for name, weight in zip(catalog.feature_names, raw_weights, strict=True)
        },
        # The site names the reason a deck was flagged by comparing it with the
        # real population feature by feature, so it needs every mean, not only
        # the ones that happen to lead the aggregate comparison below.
        "real_means": {
            name: float(negative[:, position].mean())
            for position, name in enumerate(catalog.feature_names)
        },
        "feature_order": catalog.feature_names,
        "roles": catalog.roles,
        "kinds": catalog.kinds,
        "elixir_bands": [list(band) for band in catalog.bands],
    }
    # The named reasons the site shows come from the biggest per-feature
    # contributions, so the copy can say what is off rather than print a score.
    contribution = (positive.mean(axis=0) - negative.mean(axis=0)) * raw_weights
    report["top_contributions"] = [
        {
            "feature": catalog.feature_names[position],
            "real_mean": float(negative[:, position].mean()),
            "emitted_mean": float(positive[:, position].mean()),
            "contribution": float(contribution[position]),
        }
        for position in np.argsort(-np.abs(contribution))[:10]
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(
        f"done — AUC {best['auc_out_of_fold']:.3f}, flags "
        f"{report['flag_rate_emitted']:.0%} of emitted and "
        f"{report['flag_rate_real']:.0%} of real decks"
    )


if __name__ == "__main__":
    main()
