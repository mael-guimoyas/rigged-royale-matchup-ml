from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

OBJECTIVES = ("single", "multi", "group", "complete")
DISTRIBUTION_FEATURES = (
    "selected_matchup_sigma",
    "selected_unfavorable_share",
    "selected_hard_counter_share",
)


def fit_line(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    scores = np.asarray([row["selected_model_score"] for row in rows], dtype=np.float64)
    rates = np.asarray([row["selected_validation_quality"] for row in rows], dtype=np.float64)
    variance = float(np.var(scores))
    slope = 1.0 if variance == 0.0 else float(np.cov(scores, rates, ddof=0)[0, 1] / variance)
    return {
        "meanScore": float(np.mean(scores)),
        "meanRate": float(np.mean(rates)),
        "slope": slope,
        "n": len(rows),
    }


def evaluate(rows: list[dict[str, Any]], line: dict[str, float | int]) -> dict[str, float]:
    scores = np.asarray([row["selected_model_score"] for row in rows], dtype=np.float64)
    rates = np.asarray([row["selected_test_quality"] for row in rows], dtype=np.float64)
    fitted = float(line["meanRate"]) + float(line["slope"]) * (
        scores - float(line["meanScore"])
    )
    return {
        "rawMae": float(np.mean(np.abs(scores - rates))),
        "linearMae": float(np.mean(np.abs(fitted - rates))),
        "rawBias": float(np.mean(scores - rates)),
        "linearBias": float(np.mean(fitted - rates)),
    }


def distribution_signal_report(
    rows: list[dict[str, Any]], bootstrap_samples: int = 2_000
) -> dict[str, Any] | None:
    if not rows or any(any(row.get(feature) is None for feature in DISTRIBUTION_FEATURES) for row in rows):
        return None

    scores = np.asarray([row["selected_model_score"] for row in rows], dtype=np.float64)
    validation_residuals = np.asarray(
        [row["selected_validation_quality"] - row["selected_model_score"] for row in rows],
        dtype=np.float64,
    )
    test_rates = np.asarray([row["selected_test_quality"] for row in rows], dtype=np.float64)
    columns = [np.asarray([row[name] for row in rows], dtype=np.float64) for name in DISTRIBUTION_FEATURES]
    design = np.column_stack([np.ones(len(rows)), *columns])
    coefficients = np.linalg.lstsq(design, validation_residuals, rcond=None)[0]
    adjusted = scores + design @ coefficients
    raw_errors = np.abs(scores - test_rates)
    adjusted_errors = np.abs(adjusted - test_rates)
    improvements = raw_errors - adjusted_errors

    clusters: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[(str(row["segment"]), str(row["selected_key"]))].append(index)
    cluster_values = list(clusters.values())
    rng = np.random.default_rng(20260816)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    for sample in range(bootstrap_samples):
        chosen = rng.integers(0, len(cluster_values), len(cluster_values))
        indexes = np.concatenate([cluster_values[index] for index in chosen])
        bootstrap_means[sample] = float(np.mean(improvements[indexes]))
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

    zero_unfavorable = [row for row in rows if float(row["selected_unfavorable_share"]) == 0.0]
    zero_scores = np.asarray(
        [row["selected_model_score"] for row in zero_unfavorable], dtype=np.float64
    )
    zero_test = np.asarray(
        [row["selected_test_quality"] for row in zero_unfavorable], dtype=np.float64
    )
    return {
        "features": list(DISTRIBUTION_FEATURES),
        "coefficients": [float(value) for value in coefficients],
        "rawMae": float(np.mean(raw_errors)),
        "adjustedMae": float(np.mean(adjusted_errors)),
        "maeImprovement": float(np.mean(improvements)),
        "clusterBootstrap95": [float(lower), float(upper)],
        "clusters": len(cluster_values),
        "recommended": bool(lower > 0.0),
        "zeroUnfavorable": {
            "n": len(zero_unfavorable),
            "share": len(zero_unfavorable) / len(rows),
            "meanScore": float(np.mean(zero_scores)),
            "meanTestQuality": float(np.mean(zero_test)),
            "meanBias": float(np.mean(zero_scores - zero_test)),
        },
    }


def calibration_report(benchmark: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in benchmark["selections"] if float(row["k"]) == 0.0]
    overall_line = fit_line(rows)
    overall_test = evaluate(rows, overall_line)
    by_objective: dict[str, Any] = {}
    identity_wins_every_mode = True
    for objective in OBJECTIVES:
        objective_rows = [row for row in rows if row["objective"] == objective]
        line = fit_line(objective_rows)
        test = evaluate(objective_rows, line)
        identity_wins_every_mode &= test["rawMae"] <= test["linearMae"]
        by_objective[objective] = {"linear": line, "test": test}

    report = {
        "generatedFrom": str(benchmark.get("generated_at", "unknown")),
        "modelVersion": benchmark.get("model_version"),
        "target": "orientation-balanced corrected_panel_score",
        "targetExplanation": (
            "Common-panel deck quality: panel score plus the held-out model residual, "
            "balanced equally across original team and opponent orientations."
        ),
        "overall": {"linear": overall_line, "test": overall_test},
        "byObjective": by_objective,
        "recommendedMapping": "identity" if identity_wins_every_mode else "linear",
        "decision": (
            "Keep the raw panel score when identity wins test MAE in every objective; "
            "a regression against raw encountered win rate answers a different question."
        ),
    }
    distribution = distribution_signal_report(rows)
    if distribution is not None:
        report["distributionSignals"] = distribution
    return report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate the displayed deck panel score mapping.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=root / "artifacts" / "optimizer-variance-benchmark-orientation-balanced.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "displayed-score-calibration.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    report = calibration_report(benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    test = report["overall"]["test"]
    print(f"recommended_mapping={report['recommendedMapping']}")
    print(f"raw_mae={test['rawMae']:.6f}")
    print(f"linear_mae={test['linearMae']:.6f}")
    print(f"raw_bias={test['rawBias']:+.6f}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
