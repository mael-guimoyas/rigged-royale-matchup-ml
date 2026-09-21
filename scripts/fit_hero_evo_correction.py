"""Fit and audit a checkpoint-specific Hero x Evolution correction.

The fit uses older controlled contexts and reports on later contexts. It also
checks a chronological factual slice so removing the counterfactual interaction
cannot quietly buy cleanliness by degrading ordinary battle predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from audit_hero_evo_correction import controlled_forms, factual_rows, real_hero_contexts
from sklearn.metrics import roc_auc_score

from rigged_matchup_ml.config import load_config
from rigged_matchup_ml.hero_evo_correction import CORRECTION_KEY, attach_checkpoint_correction
from rigged_matchup_ml.predictor import load_bundle, predict_from_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", type=Path, default=Path("artifacts/hero-evo-correction.json"))
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    # Defaults to data.raw_dir from the config, so the fit follows the corpus the
    # rest of the pipeline uses instead of a stale hard-coded directory.
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--fit-contexts-per-hero", type=int, default=100)
    parser.add_argument("--test-contexts-per-hero", type=int, default=100)
    parser.add_argument("--factual-rows", type=int, default=50_000)
    parser.add_argument("--shrinkage", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.raw_dir is None:
        config = load_config(args.config)
        args.raw_dir = config.resolve(config.data["raw_dir"])
    return args


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def predict_probabilities(
    bundle: dict[str, Any], rows: list[dict[str, Any]], batch_size: int
) -> np.ndarray:
    out: list[float] = []
    for start in range(0, len(rows), batch_size):
        out.extend(
            float(result["team_win_probability"])
            for result in predict_from_rows(bundle, rows[start : start + batch_size])
        )
    return np.asarray(out, dtype=np.float64)


def interactions(probabilities: np.ndarray) -> np.ndarray:
    values = logit(probabilities.reshape(-1, 4))
    return (values[:, 3] - values[:, 2]) - (values[:, 1] - values[:, 0])


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return {
        "auc": float(roc_auc_score(labels, clipped)),
        "brier": float(np.mean((clipped - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        ),
    }


def main() -> None:
    args = parse_args()
    if args.fit_contexts_per_hero <= 0 or args.test_contexts_per_hero <= 0:
        raise ValueError("Fit and test context counts must be positive")
    if not 0.0 <= args.shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1")

    per_hero = args.fit_contexts_per_hero + args.test_contexts_per_hero
    contexts = real_hero_contexts(args.raw_dir, per_hero)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        grouped[int(context["audit_hero_id"])].append(context)

    # real_hero_contexts is newest-first. Reserve the newest rows as a genuinely
    # later test; fit on the older slice immediately before it.
    test_contexts: list[dict[str, Any]] = []
    fit_contexts: list[dict[str, Any]] = []
    for hero_contexts in grouped.values():
        test_contexts.extend(hero_contexts[: args.test_contexts_per_hero])
        fit_contexts.extend(
            hero_contexts[
                args.test_contexts_per_hero : args.test_contexts_per_hero
                + args.fit_contexts_per_hero
            ]
        )

    source = load_bundle(args.source, args.device)
    fit_rows = [state for context in fit_contexts for state in controlled_forms(context)]
    fit_effects = interactions(predict_probabilities(source, fit_rows, args.batch_size))
    coefficients: dict[str, float] = {}
    for hero_id in sorted(grouped):
        mask = np.asarray(
            [int(context["audit_hero_id"]) == hero_id for context in fit_contexts]
        )
        coefficients[str(hero_id)] = -float(fit_effects[mask].mean())

    fit_receipt = {
        "fit_contexts_per_hero": args.fit_contexts_per_hero,
        "chronological_test_contexts_per_hero": args.test_contexts_per_hero,
        "factual_test_rows": args.factual_rows,
        "coefficient_target": "negative_mean_counterfactual_logit_interaction",
    }
    # Evaluate in memory first. A rejected fit must never leave behind a
    # deployable-looking checkpoint that someone could publish by accident.
    corrected = dict(source)
    corrected[CORRECTION_KEY] = {
        "schema_version": 1,
        "application": "post_calibration_antisymmetric_logit",
        "source_checkpoint_sha256": "pending-acceptance",
        "shrinkage": float(args.shrinkage),
        "coefficients_by_hero_card_id": coefficients,
        "fit": fit_receipt,
    }

    test_rows = [state for context in test_contexts for state in controlled_forms(context)]
    before_test = predict_probabilities(source, test_rows, args.batch_size)
    after_test = predict_probabilities(corrected, test_rows, args.batch_size)
    before_effects = interactions(before_test)
    after_effects = interactions(after_test)

    factual = factual_rows(args.raw_dir, args.factual_rows)
    factual_report: dict[str, Any] | None = None
    if factual:
        labels = np.asarray([bool(row.pop("win")) for row in factual], dtype=np.float64)
        before_factual = predict_probabilities(source, factual, args.batch_size)
        after_factual = predict_probabilities(corrected, factual, args.batch_size)
        before_metrics = metrics(labels, before_factual)
        after_metrics = metrics(labels, after_factual)
        factual_report = {
            "rows": len(factual),
            "before": before_metrics,
            "after": after_metrics,
            "delta": {
                key: after_metrics[key] - before_metrics[key] for key in before_metrics
            },
        }

    factual_deltas = factual_report["delta"] if factual_report else {}
    acceptance_checks = {
        "finite": bool(np.isfinite(after_effects).all()),
        "absolute_interaction_improved": bool(
            np.abs(after_effects).mean() < np.abs(before_effects).mean()
        ),
        "systematic_interaction_improved": bool(
            abs(after_effects.mean()) < abs(before_effects.mean())
        ),
        # These tolerances are deliberately tight, but non-zero: on a finite
        # chronological sample, changes below them are sampling noise rather
        # than evidence of a factual regression.
        "factual_auc_within_tolerance": factual_report is None
        or factual_deltas["auc"] >= -0.001,
        "factual_brier_within_tolerance": factual_report is None
        or factual_deltas["brier"] <= 0.00025,
        "factual_log_loss_within_tolerance": factual_report is None
        or factual_deltas["log_loss"] <= 0.0005,
    }
    accepted = all(acceptance_checks.values())
    if accepted:
        attach_checkpoint_correction(
            args.source,
            args.destination,
            coefficients,
            shrinkage=args.shrinkage,
            fit={**fit_receipt, "acceptance_checks": acceptance_checks},
        )

    report = {
        "source": str(args.source.resolve()),
        "destination": str(args.destination.resolve()) if accepted else None,
        "accepted": accepted,
        "acceptance_checks": acceptance_checks,
        "heroes_covered": len(coefficients),
        "shrinkage": args.shrinkage,
        "fit": fit_receipt,
        "counterfactual_test": {
            "contexts": len(test_contexts),
            "mean_interaction_logit_before": float(before_effects.mean()),
            "mean_interaction_logit_after": float(after_effects.mean()),
            "mean_absolute_interaction_logit_before": float(np.abs(before_effects).mean()),
            "mean_absolute_interaction_logit_after": float(np.abs(after_effects).mean()),
            "systematic_interaction_removed_share": float(
                1.0
                - abs(after_effects.mean())
                / max(abs(before_effects.mean()), np.finfo(np.float64).eps)
            ),
            "finite": bool(np.isfinite(after_effects).all()),
        },
        "coefficients_by_hero_card_id": coefficients,
        "factual": factual_report,
    }
    if not report["counterfactual_test"]["finite"] or not math.isfinite(
        report["counterfactual_test"]["systematic_interaction_removed_share"]
    ):
        raise RuntimeError("Correction audit produced non-finite values")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not accepted:
        raise RuntimeError("Correction rejected by the held-out acceptance gate")


if __name__ == "__main__":
    main()
