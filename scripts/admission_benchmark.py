"""What does it cost to forbid the optimizer from leaving the real deck space?

The variance benchmark answers "which ranking picks the best deck". It cannot
answer the question that matters here, because every candidate it ever considers
already has >= 25 held-out battles on each side of the split. The decks the
optimizer actually emits have zero battles, so they are outside the benchmark by
construction and no penalty fitted inside it can be validated against them.

Admission control sidesteps that. Instead of re-ranking candidates, it removes
some of them from the search entirely — a deck may be recommended only if the
real population vouches for it. That restriction has a price, and unlike the
model's error on unplayed decks, the price *is* measurable: apply the rule inside
the benchmark's own pools and read the change in held-out quality.

Two rule families are swept:

* support floors — the exact deck must have been played at least N times during
  the training period;
* composition rules — the deck must look like the decks people build (measured
  by `deck_composition_rules.py`, thresholds passed in here).

For each rule the report gives the share of searches where the rule actually
binds (the unrestricted winner would have been rejected), the change in mean
held-out quality, and a paired bootstrap interval on that change. A rule that
binds often and costs nothing is a free guard rail; one that costs real quality
is a trade the product has to make deliberately.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import duckdb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deck_composition_rules import Catalog, features  # noqa: E402
from optimizer_variance_benchmark import (  # noqa: E402
    DeckStat,
    SearchPool,
    build_search_pools,
    chronological_cutoffs,
    load_deck_stats,
    log,
    raw_glob,
    score_candidates,
    score_heldout_residuals,
)
from rigged_matchup_ml.predictor import load_bundle  # noqa: E402


@dataclass
class Rule:
    label: str
    family: str
    admits: Callable[[DeckStat, dict[str, float] | None], bool]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / "data" / "raw")
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "artifacts" / "matchup-model.pt"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root.parent / "riggedroyale" / "data" / "cards.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "artifacts" / "admission-scored-candidates.pkl",
        help="Scored candidates are cached here; delete it to force a rescore.",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "admission-benchmark.json"
    )
    parser.add_argument("--panel-size", type=int, default=100)
    parser.add_argument("--min-eval-support", type=int, default=25)
    parser.add_argument("--prior-strength", type=float, default=25.0)
    parser.add_argument("--residual-samples", type=int, default=50)
    parser.add_argument("--starts-per-segment", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def scored_candidates(args: argparse.Namespace) -> dict[str, list[DeckStat]]:
    """Score every eligible deck once, then reuse it for all admission rules."""
    if args.cache.exists():
        log(f"loading cached candidates from {args.cache}")
        with args.cache.open("rb") as handle:
            return pickle.load(handle)

    torch.set_num_threads(max(1, args.threads))
    parquet = raw_glob(args.raw)
    connection = duckdb.connect()
    connection.execute(f"set threads={max(1, args.threads)}")
    train_cutoff, validation_cutoff = chronological_cutoffs(connection, parquet)
    by_segment = load_deck_stats(
        connection,
        parquet,
        train_cutoff,
        validation_cutoff,
        args.panel_size,
        args.min_eval_support,
    )
    bundle = load_bundle(args.checkpoint.resolve())
    log(f"loaded model {bundle['resolved_model_version']} on CPU")
    score_candidates(bundle, by_segment, args.panel_size, args.min_eval_support, args.batch_size)
    score_heldout_residuals(
        connection,
        parquet,
        bundle,
        by_segment,
        train_cutoff,
        validation_cutoff,
        args.min_eval_support,
        args.residual_samples,
        args.batch_size,
    )
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("wb") as handle:
        pickle.dump(by_segment, handle)
    log(f"cached candidates to {args.cache}")
    return by_segment


def build_rules(catalog: Catalog) -> list[Rule]:
    rules: list[Rule] = [
        Rule("unrestricted", "baseline", lambda stat, feature: True),
    ]
    for floor in (1, 5, 10, 30, 100, 300, 1000):
        rules.append(
            Rule(
                f"train support >= {floor}",
                "support",
                lambda stat, feature, floor=floor: stat.train_n >= floor,
            )
        )
    composition: dict[str, Callable[[dict[str, float]], bool]] = {
        "anti_air >= 2": lambda f: f["anti_air"] >= 2,
        "average_elixir <= 5.0": lambda f: f["average_elixir"] <= 5.0,
        "cheap_cards >= 2": lambda f: f["cheap_cards"] >= 2,
        "any_spells >= 1": lambda f: f["any_spells"] >= 1,
        "win_conditions >= 1": lambda f: f["win_conditions"] >= 1,
        "wide envelope (air, elixir, buildings, roles)": lambda f: (
            f["anti_air"] >= 2
            and f["average_elixir"] <= 5.0
            and f["buildings"] <= 2
            and f["distinct_roles"] >= 8
        ),
        "tight envelope (wide + cheap + spell)": lambda f: (
            f["anti_air"] >= 2
            and f["average_elixir"] <= 5.0
            and f["buildings"] <= 2
            and f["distinct_roles"] >= 8
            and f["cheap_cards"] >= 2
            and f["any_spells"] >= 1
        ),
    }
    for label, test in composition.items():
        rules.append(
            Rule(
                label,
                "composition",
                lambda stat, feature, test=test: feature is None or test(feature),
            )
        )
    return rules


def evaluate(
    pools: list[SearchPool],
    rules: list[Rule],
    feature_by_key: dict[str, dict[str, float] | None],
    prior_strength: float,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Select inside each pool under each rule and compare held-out quality.

    Falls back to the unrestricted winner when a rule empties a pool: a guard
    rail that leaves a search with nothing to return is not a product option, so
    the measured cost has to include that fallback rather than drop the search.
    """
    baseline_quality: list[float] = []
    baseline_keys: list[str] = []
    for pool in pools:
        winner = max(
            pool.candidates,
            key=lambda stat: (stat.model_score, stat.train_n, stat.key),
        )
        baseline_quality.append(winner.corrected_panel_score("test", prior_strength))
        baseline_keys.append(winner.key)
    baseline = np.asarray(baseline_quality)

    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for rule in rules:
        quality: list[float] = []
        bound = 0
        empty = 0
        supports: list[int] = []
        for index, pool in enumerate(pools):
            admitted = [
                stat
                for stat in pool.candidates
                if rule.admits(stat, feature_by_key.get(stat.key))
            ]
            if not admitted:
                empty += 1
                admitted = list(pool.candidates)
            winner = max(
                admitted, key=lambda stat: (stat.model_score, stat.train_n, stat.key)
            )
            if winner.key != baseline_keys[index]:
                bound += 1
            quality.append(winner.corrected_panel_score("test", prior_strength))
            supports.append(winner.train_n)
        values = np.asarray(quality)
        delta = values - baseline
        if bootstrap > 0 and len(delta) > 0:
            draws = rng.integers(0, len(delta), size=(bootstrap, len(delta)))
            means = delta[draws].mean(axis=1)
            interval = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        else:
            interval = [float("nan"), float("nan")]
        results.append(
            {
                "rule": rule.label,
                "family": rule.family,
                "searches": len(pools),
                "changed_selection_share": bound / len(pools) if pools else 0.0,
                "empty_pool_share": empty / len(pools) if pools else 0.0,
                "mean_test_quality": float(values.mean()),
                "delta_vs_unrestricted": float(delta.mean()),
                "delta_ci95": interval,
                "median_selected_support": float(np.median(supports)),
                "zero_support_selection_share": float(
                    np.mean([support == 0 for support in supports])
                ),
            }
        )
    return {"baseline_mean_test_quality": float(baseline.mean()), "rules": results}


def main() -> None:
    args = parse_args()
    catalog = Catalog(args.catalog)
    by_segment = scored_candidates(args)
    pools = build_search_pools(by_segment, args.min_eval_support, args.starts_per_segment)
    log(f"search pools: {len(pools):,}")

    feature_by_key: dict[str, dict[str, float] | None] = {}
    for stats in by_segment.values():
        for stat in stats:
            if stat.key in feature_by_key:
                continue
            feature_by_key[stat.key] = (
                features(catalog, stat.cards) if catalog.known(stat.cards) else None
            )
    unknown = sum(1 for value in feature_by_key.values() if value is None)
    log(f"decks with composition features: {len(feature_by_key) - unknown:,} (unknown {unknown:,})")

    rules = build_rules(catalog)
    report = evaluate(
        pools, rules, feature_by_key, args.prior_strength, args.bootstrap, args.seed
    )
    by_objective: dict[str, Any] = defaultdict(dict)
    for objective in ("single", "multi", "group", "complete"):
        subset = [pool for pool in pools if pool.objective == objective]
        if subset:
            by_objective[objective] = evaluate(
                subset, rules, feature_by_key, args.prior_strength, 0, args.seed
            )
    report["by_objective"] = dict(by_objective)
    report["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["pool_count"] = len(pools)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"\nbaseline held-out quality: {report['baseline_mean_test_quality']:.4f} "
        f"over {len(pools):,} searches\n"
    )
    header = f"{'rule':46} {'binds':>7} {'quality':>8} {'delta':>8} {'ci95':>20} {'sup p50':>8}"
    print(header)
    for row in report["rules"]:
        low, high = row["delta_ci95"]
        print(
            f"{row['rule']:46} {row['changed_selection_share']:7.3f} "
            f"{row['mean_test_quality']:8.4f} {row['delta_vs_unrestricted']:+8.4f} "
            f"[{low:+.4f}, {high:+.4f}] {row['median_selected_support']:8.0f}"
        )
    print(f"\noutput={args.output.resolve()}")


if __name__ == "__main__":
    main()
