"""Theoretical ceiling analysis: how close the model is to the best achievable.

With binary win/loss labels and a true per-matchup win-rate ``p``, no model can
do better than the irreducible Brier ``p*(1-p)`` (the noise floor) nor than the
AUC obtained by predicting that true ``p`` (the discrimination ceiling). This
module estimates both ceilings from the observed per-matchup rates, measures how
much of the attainable signal the model already captures, then turns each gap
(discrimination, calibration, coverage, per-segment) into a prioritised,
actionable improvement list. It reuses the same calibrated inference and noise
floor as ``benchmark`` so the numbers are directly comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import torch

from .benchmark import BENCHMARK_COLUMNS, _calibrated_probabilities, _device
from .config import AppConfig
from .metrics import _scalar_metrics, binary_metrics
from .model import SymmetricMatchupModel
from .unseen_evaluation import matchup_key


# A segment needs at least this many supported rows before we trust its gap.
SEGMENT_MIN_ROWS = 500


def _per_matchup_rates(
    targets: np.ndarray, matchup_ids: np.ndarray, num_matchups: int, min_support: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return per-row observed matchup rate, support mask, and matchup counts.

    ``matchup_ids`` are dense integer ids in ``[0, num_matchups)`` (one per row).
    Encoding keys to ints during the scan avoids materialising millions of long
    matchup-key strings, which otherwise blows up memory on full splits.
    """
    counts = np.bincount(matchup_ids, minlength=num_matchups).astype(np.float64)
    win_sums = np.bincount(matchup_ids, weights=targets, minlength=num_matchups)
    rates = np.divide(win_sums, counts, out=np.full_like(win_sums, 0.5), where=counts > 0)
    supported_matchup = counts >= int(min_support)
    return (
        rates[matchup_ids],
        supported_matchup[matchup_ids],
        int(supported_matchup.sum()),
        int(num_matchups),
    )


def _capture_fraction(model_value: float, baseline: float, ceiling: float) -> float | None:
    """Fraction of the baseline->ceiling interval the model has closed.

    1.0 means the model reached the ceiling, 0.0 means it is no better than the
    baseline, negative means it is worse than the baseline.
    """
    span = baseline - ceiling
    if abs(span) <= 1e-9:
        return None
    return float((baseline - model_value) / span)


def _safe(value: Any) -> float | None:
    return None if value is None else float(value)


@torch.no_grad()
def analyze_ceiling(
    config: AppConfig,
    checkpoint_path: Path,
    split: str = "test",
    min_support: int = 100,
    batch_size: int = 16_384,
) -> dict[str, Any]:
    """Estimate the theoretical ceiling and how far the model is from it."""
    prepared_dir = config.resolve(config.data["prepared_dir"])
    artifact_dir = config.resolve(config.training["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_dir = prepared_dir / split
    if not split_dir.exists() or not list(split_dir.glob("*.parquet")):
        raise RuntimeError(f"No prepared Parquet files found in {split_dir}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = _device(str(config.training["device"]))
    model = SymmetricMatchupModel(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()

    # Keys/segments are integer-encoded during the scan so we never hold millions
    # of long matchup-key strings in memory (the OOM trap on full splits). Each
    # chunk stays a compact numpy array; only the unique sets live as dicts.
    model_chunks: list[np.ndarray] = []
    prior_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    key_id_chunks: list[np.ndarray] = []
    segment_id_chunks: list[np.ndarray] = []
    key_to_id: dict[str, int] = {}
    segment_to_id: dict[str, int] = {}

    dataset = pads.dataset(split_dir, format="parquet")
    scanner = dataset.scanner(columns=BENCHMARK_COLUMNS, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        rows = record_batch.to_pylist()
        if not rows:
            continue
        model_chunks.append(
            _calibrated_probabilities(model, rows, payload, device).astype(np.float32)
        )
        prior_chunks.append(
            np.fromiter((float(row["matrix_prior"]) for row in rows), dtype=np.float32, count=len(rows))
        )
        target_chunks.append(
            np.fromiter((int(bool(row["win"])) for row in rows), dtype=np.int8, count=len(rows))
        )
        key_id_chunks.append(
            np.fromiter(
                (
                    key_to_id.setdefault(
                        matchup_key(row["team_deck_key"], row["opponent_deck_key"]),
                        len(key_to_id),
                    )
                    for row in rows
                ),
                dtype=np.int32,
                count=len(rows),
            )
        )
        segment_id_chunks.append(
            np.fromiter(
                (segment_to_id.setdefault(str(row["segment"]), len(segment_to_id)) for row in rows),
                dtype=np.int32,
                count=len(rows),
            )
        )

    target_array = np.concatenate(target_chunks).astype(np.float64)
    model_array = np.concatenate(model_chunks).astype(np.float64)
    prior_array = np.concatenate(prior_chunks).astype(np.float64)
    matchup_ids = np.concatenate(key_id_chunks)
    segment_ids = np.concatenate(segment_id_chunks)
    segment_names = np.empty(len(segment_to_id), dtype=object)
    for name, index in segment_to_id.items():
        segment_names[index] = name
    rows_total = int(target_array.shape[0])

    row_rate, supported_row, supported_matchups, observed_matchups = _per_matchup_rates(
        target_array, matchup_ids, len(key_to_id), min_support
    )
    bins = int(config.evaluation["calibration_bins"])

    report: dict[str, Any] = {
        "split": split,
        "rows": rows_total,
        "min_support": int(min_support),
        "win_rate": float(target_array.mean()) if rows_total else 0.0,
    }

    supported_rows = int(supported_row.sum())
    if supported_rows == 0:
        report["warning"] = (
            "No matchup reached min_support; lower --min-support to estimate a ceiling."
        )
        report["weaknesses"] = [
            {
                "area": "coverage",
                "severity": "high",
                "summary": "Aucun matchup n'atteint le support minimum.",
                "detail": "Impossible d'estimer le plafond: collecte plus de combats ou baisse --min-support.",
            }
        ]
        report["recommendations"] = []
        output = artifact_dir / f"ceiling-{split}-report.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    sup_targets = target_array[supported_row]
    sup_rate = row_rate[supported_row]
    sup_model = model_array[supported_row]
    sup_prior = prior_array[supported_row]
    constant_brier = 0.25  # Brier of the uninformed constant-0.5 predictor.

    irreducible_brier = float(np.mean(sup_rate * (1.0 - sup_rate)))
    oracle = _scalar_metrics(sup_targets, sup_rate)
    model_supported = _scalar_metrics(sup_targets, sup_model)
    prior_supported = _scalar_metrics(sup_targets, sup_prior)

    ceiling = {
        "irreducible_brier": irreducible_brier,
        "oracle_brier": _safe(oracle["brier_score"]),
        "oracle_auc": _safe(oracle["auc"]),
        "observed_matchups": observed_matchups,
        "supported_matchups": supported_matchups,
        "supported_rows": supported_rows,
        "coverage": supported_rows / rows_total if rows_total else 0.0,
    }
    report["theoretical_ceiling"] = ceiling

    # Calibration is measured on every row, discrimination on supported rows only.
    calibration = binary_metrics(target_array, model_array, bins)
    model_auc = _safe(model_supported["auc"])
    oracle_auc = ceiling["oracle_auc"]
    model_brier = float(model_supported["brier_score"])
    prior_brier = float(prior_supported["brier_score"])

    auc_capture = (
        _capture_fraction(model_auc, 0.5, oracle_auc)
        if model_auc is not None and oracle_auc is not None
        else None
    )
    model_block = {
        "auc_supported": model_auc,
        "brier_supported": model_brier,
        "brier_all_rows": _safe(calibration["brier_score"]),
        "gap_to_floor": model_brier - irreducible_brier,
        "auc_capture_vs_oracle": auc_capture,
        "brier_capture_vs_prior": _capture_fraction(model_brier, prior_brier, irreducible_brier),
        "brier_capture_vs_constant": _capture_fraction(
            model_brier, constant_brier, irreducible_brier
        ),
        "calibration_slope": _safe(calibration["calibration_slope"]),
        "calibration_intercept": _safe(calibration["calibration_intercept"]),
        "expected_calibration_error_quantile": _safe(
            calibration["expected_calibration_error_quantile"]
        ),
        "mean_prediction": _safe(calibration["mean_prediction"]),
    }
    report["model"] = model_block
    report["baselines"] = {
        "matrix_prior_brier_supported": prior_brier,
        "constant_0.5_brier": constant_brier,
    }

    # Per-segment gap to the (segment-local) floor, worst first.
    global_gap = model_block["gap_to_floor"]
    by_segment: dict[str, dict[str, Any]] = {}
    for segment_id, segment in enumerate(segment_names):
        seg_mask = (segment_ids == segment_id) & supported_row
        seg_rows = int(seg_mask.sum())
        if seg_rows < SEGMENT_MIN_ROWS:
            continue
        seg_rate = row_rate[seg_mask]
        seg_targets = target_array[seg_mask]
        seg_model = model_array[seg_mask]
        seg_irreducible = float(np.mean(seg_rate * (1.0 - seg_rate)))
        seg_brier = float(_scalar_metrics(seg_targets, seg_model)["brier_score"])
        seg_gap = seg_brier - seg_irreducible
        by_segment[segment] = {
            "supported_rows": seg_rows,
            "win_rate": float(seg_targets.mean()),
            "irreducible_brier": seg_irreducible,
            "model_brier": seg_brier,
            "gap_to_floor": seg_gap,
            "gap_ratio_vs_global": (seg_gap / global_gap) if global_gap > 1e-9 else None,
        }
    report["by_segment"] = dict(
        sorted(by_segment.items(), key=lambda item: item[1]["gap_to_floor"], reverse=True)
    )

    weaknesses, recommendations = derive_diagnosis(
        {
            "auc_capture": auc_capture,
            "model_auc": model_auc,
            "oracle_auc": oracle_auc,
            "gap_to_floor": global_gap,
            "irreducible_brier": irreducible_brier,
            "brier_capture_vs_prior": model_block["brier_capture_vs_prior"],
            "calibration_slope": model_block["calibration_slope"],
            "ece_quantile": model_block["expected_calibration_error_quantile"],
            "coverage": ceiling["coverage"],
            "supported_matchups": supported_matchups,
            "by_segment": report["by_segment"],
        }
    )
    report["weaknesses"] = weaknesses
    report["recommendations"] = recommendations

    output = artifact_dir / f"ceiling-{split}-report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def derive_diagnosis(stats: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Map ceiling gaps to ranked weaknesses and concrete repo-grounded fixes.

    Pure function (no torch / no IO) so the diagnosis logic stays unit-testable.
    """
    weaknesses: list[dict] = []
    recommendations: list[dict] = []
    severity_rank = {"high": 0, "medium": 1, "low": 2, "good": 3}

    auc_capture = stats.get("auc_capture")
    gap = float(stats.get("gap_to_floor") or 0.0)
    slope = stats.get("calibration_slope")
    ece = stats.get("ece_quantile")
    coverage = float(stats.get("coverage") or 0.0)

    # 1. Discrimination: how much of the attainable AUC the model captures.
    if auc_capture is not None:
        if auc_capture < 0.40:
            severity = "high"
        elif auc_capture < 0.65:
            severity = "medium"
        else:
            severity = "good"
        weaknesses.append(
            {
                "area": "discrimination",
                "severity": severity,
                "metric": "auc_capture_vs_oracle",
                "value": round(auc_capture, 4),
                "summary": (
                    f"Le modele capte {auc_capture * 100:.0f}% de la discrimination "
                    f"atteignable (AUC modele vs AUC oracle {stats.get('model_auc'):.3f}"
                    f"->{stats.get('oracle_auc'):.3f})."
                ),
                "detail": (
                    "L'oracle predit le taux reel par matchup; l'ecart restant est du "
                    "signal que l'architecture/les features n'extraient pas encore."
                ),
            }
        )
        if severity != "good":
            recommendations.append(
                {
                    "priority": severity_rank[severity],
                    "area": "discrimination",
                    "lever": "model capacity + features",
                    "action": (
                        "Augmente la capacite: model.hidden_dim 192->256, embedding_dim "
                        "64->96, transformer_layers/cross_heads; garde card2vec_init et "
                        "use_card_importance actifs; entraine plus longtemps (training.epochs)."
                    ),
                    "why": "Capter une plus grande part de l'AUC oracle restante.",
                }
            )
            recommendations.append(
                {
                    "priority": severity_rank[severity] + 1,
                    "area": "discrimination",
                    "lever": "empirical prior residual",
                    "action": (
                        "Lance `attach-prior` puis mets model.matrix_prior_learnable: true "
                        "pour que le reseau apprenne un residu autour de la matrice."
                    ),
                    "why": "Donne un point de depart fort, le reseau n'a plus qu'a corriger.",
                }
            )

    # 2. Brier gap to the irreducible floor (data/discrimination headroom).
    if gap > 0.010:
        severity = "high"
    elif gap > 0.004:
        severity = "medium"
    else:
        severity = "good"
    weaknesses.append(
        {
            "area": "brier_gap",
            "severity": severity,
            "metric": "gap_to_floor",
            "value": round(gap, 5),
            "summary": (
                f"Ecart de Brier au plancher irreductible: {gap:.4f} "
                f"(plancher {stats.get('irreducible_brier'):.4f})."
            ),
            "detail": (
                "Proche de 0 = quasi au plafond: l'effort doit aller vers la couverture "
                "et les donnees plutot que l'architecture (cf. README benchmark)."
            ),
        }
    )

    # 3. Calibration: over/under-confidence on top of raw discrimination.
    slope_off = slope is not None and abs(slope - 1.0) > 0.15
    ece_off = ece is not None and ece > 0.02
    if slope_off or ece_off:
        hard = (slope is not None and abs(slope - 1.0) > 0.30) or (ece is not None and ece > 0.04)
        severity = "high" if hard else "medium"
        direction = ""
        if slope is not None:
            direction = " surconfiant" if slope < 1.0 else " sousconfiant"
        weaknesses.append(
            {
                "area": "calibration",
                "severity": severity,
                "metric": "calibration_slope / ece_quantile",
                "value": {"slope": _safe(slope), "ece_quantile": _safe(ece)},
                "summary": f"Calibration imparfaite{direction} (pente {slope}, ECE {ece}).",
                "detail": "Pente <1 = surconfiant, >1 = sousconfiant; vise pente ~1 et ECE bas.",
            }
        )
        if slope is not None and slope < 1.0:
            action = (
                "Modele surconfiant: augmente training.label_smoothing (0.02->0.04) et "
                "verifie la calibration par segment (training.segment_calibration_min_rows)."
            )
        else:
            action = (
                "Modele sousconfiant: baisse training.label_smoothing et resserre la "
                "calibration par segment (training.segment_calibration_min_rows)."
            )
        recommendations.append(
            {
                "priority": severity_rank[severity],
                "area": "calibration",
                "lever": "label smoothing + per-segment calibration",
                "action": action,
                "why": "Rapprocher la pente de 1 et reduire l'ECE sans toucher a la discrimination.",
            }
        )

    # 4. Coverage: share of the meta with enough games to trust a matchup rate.
    if coverage < 0.50:
        severity = "high"
    elif coverage < 0.75:
        severity = "medium"
    else:
        severity = "good"
    weaknesses.append(
        {
            "area": "coverage",
            "severity": severity,
            "metric": "coverage",
            "value": round(coverage, 4),
            "summary": (
                f"{coverage * 100:.0f}% des lignes tombent dans un matchup supporte "
                f"(>= min_support); {stats.get('supported_matchups')} matchups supportes."
            ),
            "detail": "Une faible couverture signifie que le plafond ne couvre qu'une part de la meta.",
        }
    )
    if severity != "good":
        recommendations.append(
            {
                "priority": severity_rank[severity],
                "area": "coverage",
                "lever": "data collection",
                "action": (
                    "Collecte plus de combats sur les zones creuses: "
                    "`collect-api --balance` (vise les bandes de trophees sous-representees), "
                    "puis `prepare`. Verifie la generalisation hors-meta avec `evaluate-unseen`."
                ),
                "why": "Plus de matchups franchissent le support: le plafond couvre plus de meta.",
            }
        )

    # 5. Per-segment weak spots: segments far above the global gap.
    by_segment = stats.get("by_segment") or {}
    worst = [
        (name, info)
        for name, info in by_segment.items()
        if info.get("gap_ratio_vs_global") is not None and info["gap_ratio_vs_global"] > 1.8
    ]
    worst.sort(key=lambda item: item[1]["gap_to_floor"], reverse=True)
    if worst:
        names = ", ".join(f"{name} (x{info['gap_ratio_vs_global']:.1f})" for name, info in worst[:5])
        weaknesses.append(
            {
                "area": "segment",
                "severity": "medium",
                "metric": "gap_ratio_vs_global",
                "value": names,
                "summary": f"Segments nettement au-dessus de l'ecart global: {names}.",
                "detail": "Ces contextes meta sont moins bien servis par le modele actuel.",
            }
        )
        recommendations.append(
            {
                "priority": severity_rank["medium"],
                "area": "segment",
                "lever": "segment adapters + targeted data",
                "action": (
                    "use_segment_adapters est deja actif: collecte plus de combats pour ces "
                    "segments (`collect-api --balance`) et assure une calibration par segment "
                    "(assez de lignes vs training.segment_calibration_min_rows)."
                ),
                "why": "Reduire l'ecart au plancher la ou il est le plus large.",
            }
        )

    weaknesses.sort(key=lambda item: severity_rank.get(item["severity"], 9))
    recommendations.sort(key=lambda item: item["priority"])
    for index, recommendation in enumerate(recommendations, start=1):
        recommendation["priority"] = index
    return weaknesses, recommendations


def format_ceiling_report(report: dict[str, Any]) -> str:
    """Human-readable terminal summary (printed on the Pod)."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  PLAFOND THEORIQUE DU MODELE  —  faiblesses & ameliorations")
    lines.append("=" * 72)
    lines.append(
        f"split={report['split']}  rows={report['rows']:,}  "
        f"win_rate={report['win_rate']:.4f}  min_support={report['min_support']}"
    )

    if report.get("warning"):
        lines.append("")
        lines.append(f"!! {report['warning']}")
        return "\n".join(lines)

    ceiling = report["theoretical_ceiling"]
    model = report["model"]
    lines.append("")
    lines.append("PLAFOND (meilleur atteignable, estime sur les matchups supportes)")
    lines.append(f"  Brier irreductible (plancher) : {ceiling['irreducible_brier']:.4f}")
    if ceiling.get("oracle_auc") is not None:
        lines.append(f"  AUC oracle (plafond AUC)      : {ceiling['oracle_auc']:.4f}")
    lines.append(
        f"  Couverture                    : {ceiling['coverage'] * 100:.1f}%  "
        f"({ceiling['supported_matchups']}/{ceiling['observed_matchups']} matchups)"
    )

    lines.append("")
    lines.append("MODELE vs PLAFOND")
    if model.get("auc_supported") is not None:
        lines.append(f"  AUC modele                    : {model['auc_supported']:.4f}")
    if model.get("auc_capture_vs_oracle") is not None:
        lines.append(
            f"  Discrimination captee         : {model['auc_capture_vs_oracle'] * 100:.0f}% "
            "de l'AUC oracle"
        )
    lines.append(f"  Brier modele (supporte)       : {model['brier_supported']:.4f}")
    lines.append(f"  Ecart au plancher             : {model['gap_to_floor']:.4f}")
    if model.get("brier_capture_vs_prior") is not None:
        lines.append(
            f"  Marge prior->plancher captee  : {model['brier_capture_vs_prior'] * 100:.0f}%"
        )
    lines.append(
        f"  Calibration: pente {model.get('calibration_slope')}  "
        f"ECE(q) {model.get('expected_calibration_error_quantile')}"
    )

    lines.append("")
    lines.append("FAIBLESSES (triees par severite)")
    icons = {"high": "[HIGH]", "medium": "[MED ]", "low": "[LOW ]", "good": "[ OK ]"}
    for weakness in report["weaknesses"]:
        lines.append(f"  {icons.get(weakness['severity'], '[????]')} {weakness['summary']}")
        if weakness.get("detail"):
            lines.append(f"          -> {weakness['detail']}")

    lines.append("")
    lines.append("AMELIORATIONS (pour s'approcher du plafond, par priorite)")
    if not report["recommendations"]:
        lines.append("  (aucune: le modele est deja proche du plafond sur cette mesure)")
    for recommendation in report["recommendations"]:
        lines.append(
            f"  {recommendation['priority']}. [{recommendation['area']}] "
            f"{recommendation['action']}"
        )
        lines.append(f"      pourquoi: {recommendation['why']}")

    by_segment = report.get("by_segment") or {}
    if by_segment:
        lines.append("")
        lines.append("PAR SEGMENT (ecart au plancher local, pire en premier)")
        lines.append(f"  {'segment':<22}{'rows':>10}{'brier':>9}{'floor':>9}{'gap':>9}{'xglob':>7}")
        for name, info in list(by_segment.items())[:12]:
            ratio = info.get("gap_ratio_vs_global")
            ratio_str = f"{ratio:.1f}" if ratio is not None else "  -"
            lines.append(
                f"  {name[:22]:<22}{info['supported_rows']:>10,}"
                f"{info['model_brier']:>9.4f}{info['irreducible_brier']:>9.4f}"
                f"{info['gap_to_floor']:>9.4f}{ratio_str:>7}"
            )

    lines.append("=" * 72)
    return "\n".join(lines)
