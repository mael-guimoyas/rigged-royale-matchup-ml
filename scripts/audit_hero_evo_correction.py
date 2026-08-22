"""Audit a corrected checkpoint on real battles and controlled Hero/Evo forms."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from sklearn.metrics import roc_auc_score

from rigged_matchup_ml.domain import ROLE_HERO, ROLE_NORMAL
from rigged_matchup_ml.hero_evo_correction import V5_A9D92A10_HERO_COEFFICIENTS
from rigged_matchup_ml.predictor import load_bundle, predict_from_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Uncorrected checkpoint")
    parser.add_argument("corrected", type=Path, help="Checkpoint with embedded correction")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--contexts-per-hero", type=int, default=100)
    parser.add_argument("--factual-rows", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def real_hero_contexts(raw_dir: Path, per_hero: int) -> list[dict[str, Any]]:
    hero_ids = ",".join(V5_A9D92A10_HERO_COEFFICIENTS)
    parquet_glob = (raw_dir.resolve() / "*.parquet").as_posix().replace("'", "''")
    candidate_limit = max(per_hero * 3, per_hero)
    query = f"""
        WITH candidates AS (
          SELECT game_id, battle_time, mode_key, segment, patch,
                 team_card_ids, opponent_card_ids,
                 team_evolution_levels, opponent_evolution_levels,
                 team_hero_levels, opponent_hero_levels,
                 team_card_roles, opponent_card_roles,
                 team_tower_troop_id, opponent_tower_troop_id,
                 matrix_prior, unnest(team_card_ids) AS hero_id
          FROM read_parquet('{parquet_glob}')
          UNION ALL
          SELECT game_id, battle_time, mode_key, segment, patch,
                 opponent_card_ids, team_card_ids,
                 opponent_evolution_levels, team_evolution_levels,
                 opponent_hero_levels, team_hero_levels,
                 opponent_card_roles, team_card_roles,
                 opponent_tower_troop_id, team_tower_troop_id,
                 1.0 - matrix_prior, unnest(opponent_card_ids) AS hero_id
          FROM read_parquet('{parquet_glob}')
        )
        SELECT * EXCLUDE (position)
        FROM (
          SELECT *, row_number() OVER (
            PARTITION BY hero_id ORDER BY battle_time DESC, game_id
          ) AS position
          FROM candidates
          WHERE hero_id IN ({hero_ids})
            AND list_has_any(team_evolution_levels, [1, 3])
        )
        WHERE position <= {candidate_limit}
        ORDER BY hero_id, battle_time DESC
    """
    candidates = duckdb.sql(query).to_arrow_table().to_pylist()
    counts: dict[int, int] = {}
    selected: list[dict[str, Any]] = []
    for row in candidates:
        hero_id = int(row.pop("hero_id"))
        if counts.get(hero_id, 0) >= per_hero:
            continue
        hero_index = list(row["team_card_ids"]).index(hero_id)
        evo_indices = [
            index
            for index, level in enumerate(row["team_evolution_levels"])
            if int(level) & 1 and index != hero_index
        ]
        if not evo_indices:
            continue
        row["audit_hero_id"] = hero_id
        row["audit_hero_index"] = hero_index
        row["audit_evo_index"] = evo_indices[0]
        selected.append(row)
        counts[hero_id] = counts.get(hero_id, 0) + 1
    missing = {
        hero_id: per_hero - counts.get(int(hero_id), 0)
        for hero_id in V5_A9D92A10_HERO_COEFFICIENTS
        if counts.get(int(hero_id), 0) < per_hero
    }
    if missing:
        raise RuntimeError(f"Not enough real contexts for all Heroes: {missing}")
    return selected


def controlled_forms(context: dict[str, Any]) -> list[dict[str, Any]]:
    hero_index = int(context["audit_hero_index"])
    evo_index = int(context["audit_evo_index"])
    base = {key: deepcopy(value) for key, value in context.items() if not key.startswith("audit_")}
    base["team_evolution_levels"] = [0] * 8
    base["team_hero_levels"] = [0] * 8
    base["opponent_evolution_levels"] = [0] * 8
    base["opponent_hero_levels"] = [0] * 8
    base["team_card_roles"] = [
        ROLE_NORMAL if int(role) == ROLE_HERO else int(role) for role in base["team_card_roles"]
    ]
    base["opponent_card_roles"] = [
        ROLE_NORMAL if int(role) == ROLE_HERO else int(role)
        for role in base["opponent_card_roles"]
    ]

    normal_no_evo = deepcopy(base)
    hero_no_evo = deepcopy(base)
    hero_no_evo["team_hero_levels"][hero_index] = 1
    hero_no_evo["team_card_roles"][hero_index] = ROLE_HERO
    normal_evo = deepcopy(base)
    normal_evo["team_evolution_levels"][evo_index] = 1
    hero_evo = deepcopy(hero_no_evo)
    hero_evo["team_evolution_levels"][evo_index] = 1
    return [normal_no_evo, hero_no_evo, normal_evo, hero_evo]


def factual_rows(raw_dir: Path, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    parquet_glob = (raw_dir.resolve() / "*.parquet").as_posix().replace("'", "''")
    return duckdb.sql(
        f"""
        SELECT team_card_ids, opponent_card_ids,
               team_evolution_levels, opponent_evolution_levels,
               team_hero_levels, opponent_hero_levels,
               team_card_roles, opponent_card_roles,
               team_tower_troop_id, opponent_tower_troop_id,
               segment, patch, matrix_prior, win
        FROM read_parquet('{parquet_glob}')
        ORDER BY battle_time DESC, game_id
        LIMIT {int(limit)}
        """
    ).to_arrow_table().to_pylist()


def probability_pairs(
    bundle: dict[str, Any], rows: list[dict[str, Any]], batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    before: list[float] = []
    after: list[float] = []
    for start in range(0, len(rows), batch_size):
        results = predict_from_rows(bundle, rows[start : start + batch_size])
        before.extend(float(result["pre_correction_team_win_probability"]) for result in results)
        after.extend(float(result["team_win_probability"]) for result in results)
    return np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)


def factual_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    return {
        "accuracy": float(np.mean((clipped >= 0.5) == labels)),
        "auc": float(roc_auc_score(labels, clipped)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "brier_score": float(np.mean((clipped - labels) ** 2)),
    }


def main() -> None:
    args = parse_args()
    source = load_bundle(args.source, args.device)
    corrected = load_bundle(args.corrected, args.device)
    contexts = real_hero_contexts(args.raw_dir, args.contexts_per_hero)
    controlled = [state for context in contexts for state in controlled_forms(context)]
    before_flat, after_flat = probability_pairs(corrected, controlled, args.batch_size)
    before = before_flat.reshape(-1, 4)
    after = after_flat.reshape(-1, 4)
    source_smoke = predict_from_rows(source, controlled[:64])
    source_matches_embedded_base = max(
        abs(
            float(result["team_win_probability"])
            - float(before_flat[index])
        )
        for index, result in enumerate(source_smoke)
    )
    before_interaction = (before[:, 3] - before[:, 2]) - (before[:, 1] - before[:, 0])
    after_interaction = (after[:, 3] - after[:, 2]) - (after[:, 1] - after[:, 0])
    unchanged_without_joint_form = float(np.max(np.abs(after[:, :3] - before[:, :3])))

    per_hero: dict[str, dict[str, float | int]] = {}
    for hero_id in V5_A9D92A10_HERO_COEFFICIENTS:
        mask = np.asarray([context["audit_hero_id"] == int(hero_id) for context in contexts])
        per_hero[hero_id] = {
            "contexts": int(mask.sum()),
            "before_interaction_pp": float(before_interaction[mask].mean() * 100),
            "after_interaction_pp": float(after_interaction[mask].mean() * 100),
        }

    factual = factual_rows(args.raw_dir, args.factual_rows)
    factual_report: dict[str, Any] | None = None
    if factual:
        labels = np.asarray([bool(row["win"]) for row in factual], dtype=np.float64)
        for row in factual:
            row.pop("win", None)
        factual_before_values, factual_after_values = probability_pairs(
            corrected, factual, args.batch_size
        )
        factual_before = factual_metrics(labels, factual_before_values)
        factual_after = factual_metrics(labels, factual_after_values)
        factual_report = {
            "rows": len(factual),
            "before": factual_before,
            "after": factual_after,
            "delta": {
                key: factual_after[key] - factual_before[key] for key in factual_before
            },
        }

    report = {
        "contexts": len(contexts),
        "heroes_covered": len(per_hero),
        "counterfactual": {
            "before_interaction_pp": float(before_interaction.mean() * 100),
            "after_interaction_pp": float(after_interaction.mean() * 100),
            "negative_share_before": float(np.mean(before_interaction < 0)),
            "negative_share_after": float(np.mean(after_interaction < 0)),
            "max_change_without_joint_hero_evo": unchanged_without_joint_form,
            "max_source_vs_embedded_base_difference": source_matches_embedded_base,
            "finite": bool(np.isfinite(after).all() and math.isfinite(after_interaction.mean())),
        },
        "per_hero": per_hero,
        "factual": factual_report,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
