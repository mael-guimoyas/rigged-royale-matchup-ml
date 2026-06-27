from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import torch

from .config import AppConfig
from .dataset import matchup_dataloader
from .metrics import binary_metrics, binary_metrics_by_group
from .model import SymmetricMatchupModel
from .trainer import collect_predictions_with_segments


DIFFICULTY_LEVELS = [
    {
        "name": "all_unseen_matchups",
        "difficulty": "overall",
        "description": "Exact unordered matchup absent from train.",
        "condition": "is_unseen_matchup",
    },
    {
        "name": "known_decks_new_matchup",
        "difficulty": "easy",
        "description": "Both decks were seen in train, but this deck-vs-deck matchup was not.",
        "condition": "is_unseen_matchup and team_deck_seen and opponent_deck_seen",
    },
    {
        "name": "one_new_deck",
        "difficulty": "medium",
        "description": "The matchup is unseen and exactly one full deck is absent from train.",
        "condition": "is_unseen_matchup and team_deck_seen != opponent_deck_seen",
    },
    {
        "name": "two_new_decks",
        "difficulty": "hard",
        "description": "The matchup is unseen and both full decks are absent from train.",
        "condition": "is_unseen_matchup and not team_deck_seen and not opponent_deck_seen",
    },
]


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def matchup_key(first_deck_key: str, second_deck_key: str) -> str:
    """Canonical unordered matchup key: A vs B and B vs A collapse together."""
    first, second = sorted((str(first_deck_key), str(second_deck_key)))
    return f"{first}::{second}"


def _matchup_key_sql() -> str:
    return (
        "case when team_deck_key <= opponent_deck_key "
        "then team_deck_key || '::' || opponent_deck_key "
        "else opponent_deck_key || '::' || team_deck_key end"
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _require_parquet_files(split_dir: Path) -> None:
    if not split_dir.exists() or not list(split_dir.glob("*.parquet")):
        raise RuntimeError(f"No prepared Parquet files found in {split_dir}")


def build_unseen_matchup_splits(
    prepared_dir: Path,
    output_dir: Path,
    split: str = "test",
    levels: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Create difficulty-based splits for matchups absent from prepared train.

    ``levels`` selects which difficulty levels to materialise (default: all).
    Passing only ``all_unseen_matchups`` skips the 3 stratified sub-levels, which
    means 1 Parquet copy + 1 inference pass instead of 4 -- much faster/cheaper
    when only the headline unseen metric is needed.
    """
    if split == "train":
        raise ValueError("Unseen-matchup evaluation must use validation or test, not train")
    selected_levels = list(levels) if levels else DIFFICULTY_LEVELS

    train_dir = prepared_dir / "train"
    evaluation_dir = prepared_dir / split
    _require_parquet_files(train_dir)
    _require_parquet_files(evaluation_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_glob = _quoted(train_dir / "*.parquet")
    evaluation_glob = _quoted(evaluation_dir / "*.parquet")
    key_sql = _matchup_key_sql()

    manifest: dict[str, Any]
    connection = duckdb.connect()
    try:
        connection.execute("set preserve_insertion_order=false")
        connection.execute(
            f"""
            create or replace temp view train_rows as
            select team_deck_key, opponent_deck_key, {key_sql} as matchup_key
            from read_parquet('{train_glob}')
            """
        )
        connection.execute(
            """
            create or replace temp view train_decks as
            select team_deck_key as deck_key from train_rows
            union
            select opponent_deck_key as deck_key from train_rows
            """
        )
        connection.execute(
            f"""
            create or replace temp view evaluation_rows as
            select *, {key_sql} as matchup_key
            from read_parquet('{evaluation_glob}')
            """
        )
        connection.execute(
            """
            create or replace temp view evaluation_enriched as
            with train_matchups as (
              select distinct matchup_key from train_rows
            )
            select
              e.*,
              t.matchup_key is null as is_unseen_matchup,
              team_decks.deck_key is not null as team_deck_seen,
              opponent_decks.deck_key is not null as opponent_deck_seen
            from evaluation_rows e
            left join train_matchups t using (matchup_key)
            left join train_decks team_decks on e.team_deck_key = team_decks.deck_key
            left join train_decks opponent_decks
              on e.opponent_deck_key = opponent_decks.deck_key
            """
        )
        base_row = connection.execute(
            """
            select
              (select count(*) from train_rows) train_rows,
              (select count(distinct matchup_key) from train_rows) train_matchups,
              (select count(distinct deck_key) from train_decks) train_decks,
              (select count(*) from evaluation_rows) original_rows,
              (select count(distinct matchup_key) from evaluation_rows) original_matchups,
              (
                select count(*)
                from evaluation_enriched
                where not is_unseen_matchup
              ) excluded_seen_rows
            """
        ).fetchone()
        base_columns = [description[0] for description in connection.description]
        manifest = dict(zip(base_columns, base_row, strict=True))
        manifest["split"] = split
        manifest["split_policy"] = (
            "Rows are grouped by whether their canonical unordered matchup_key and full "
            "deck keys appeared in prepared train"
        )
        manifest["levels"] = {}

        original_rows = int(manifest["original_rows"])
        for level in selected_levels:
            level_dir = output_dir / str(level["name"])
            level_dir.mkdir(parents=True, exist_ok=True)
            output_path = level_dir / "data.parquet"
            if output_path.exists():
                output_path.unlink()
            condition = str(level["condition"])
            stats_row = connection.execute(
                f"""
                select count(*) as row_count, count(distinct matchup_key) as matchup_count
                from evaluation_enriched
                where {condition}
                """
            ).fetchone()
            rows, matchups = int(stats_row[0]), int(stats_row[1])
            connection.execute(
                f"""
                copy (
                  select *
                  from evaluation_enriched
                  where {condition}
                ) to '{_quoted(output_path)}'
                (format parquet, compression zstd, row_group_size 100000)
                """
            )
            manifest["levels"][str(level["name"])] = {
                "difficulty": level["difficulty"],
                "description": level["description"],
                "rows": rows,
                "matchups": matchups,
                "row_share": rows / original_rows if original_rows else 0.0,
                "split_dir": str(level_dir),
            }
    finally:
        connection.close()

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def build_unseen_matchup_split(
    prepared_dir: Path,
    output_dir: Path,
    split: str = "test",
) -> dict[str, Any]:
    """Create a split containing only matchups absent from prepared train."""
    manifest = build_unseen_matchup_splits(prepared_dir, output_dir, split=split)
    strict_level = manifest["levels"]["all_unseen_matchups"]
    source_path = Path(strict_level["split_dir"]) / "data.parquet"
    destination_path = output_dir / "data.parquet"
    if destination_path.exists():
        destination_path.unlink()
    destination_path.write_bytes(source_path.read_bytes())
    compat_manifest = {
        key: value for key, value in manifest.items() if key != "levels"
    }
    compat_manifest.update(
        {
            "strict_rows": strict_level["rows"],
            "strict_matchups": strict_level["matchups"],
            "strict_row_share": strict_level["row_share"],
            "strict_split_dir": strict_level["split_dir"],
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(compat_manifest, indent=2), encoding="utf-8"
    )
    return compat_manifest


@torch.no_grad()
def _evaluate_level(
    model: SymmetricMatchupModel,
    checkpoint_payload: dict[str, Any],
    config: AppConfig,
    split_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    loader = matchup_dataloader(
        split_dir,
        checkpoint_payload["vocabulary"],
        shuffle=False,
        augment_swap=False,
        seed=int(config.training["seed"]),
        batch_size=int(
            config.training.get("evaluation_batch_size", config.training["batch_size"])
        ),
        num_workers=int(config.training["num_workers"]),
    )
    _, probabilities, targets, segments = collect_predictions_with_segments(
        model,
        loader,
        device,
        checkpoint_payload["vocabulary"],
        float(checkpoint_payload["temperature"]),
        checkpoint_payload.get("segment_temperatures"),
        checkpoint_payload.get("calibration"),
    )
    bootstrap_samples = int(config.evaluation.get("bootstrap_samples", 0))
    bootstrap_seed = int(config.training.get("seed", 0))
    # Same guard as evaluate_checkpoint: on million-row unseen levels the point
    # estimate is already razor-sharp and 300x AUC resamples is the slow part that
    # burns GPU-pod time for nothing. Skip the bootstrap above the threshold.
    bootstrap_max_rows = int(config.evaluation.get("bootstrap_max_rows", 250_000))
    overall_bootstrap = bootstrap_samples
    if bootstrap_max_rows and len(targets) > bootstrap_max_rows:
        overall_bootstrap = 0
    metrics = binary_metrics(
        targets,
        probabilities,
        int(config.evaluation["calibration_bins"]),
        bootstrap_samples=overall_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    metrics["by_segment"] = binary_metrics_by_group(
        targets,
        probabilities,
        segments,
        int(config.evaluation["calibration_bins"]),
        bootstrap_max_rows=bootstrap_max_rows,
    )
    return metrics


@torch.no_grad()
def evaluate_unseen_matchups(
    config: AppConfig,
    checkpoint_path: Path,
    split: str = "test",
    quick: bool = False,
) -> dict[str, Any]:
    prepared_dir = config.resolve(config.data["prepared_dir"])
    artifact_dir = config.resolve(config.training["artifact_dir"])
    strict_dir = artifact_dir / f"unseen-{split}-matchups"
    # quick: only the overall all_unseen_matchups level -> 1 copy + 1 inference
    # pass instead of 4, for a fast headline-only run.
    levels = [DIFFICULTY_LEVELS[0]] if quick else None
    split_manifest = build_unseen_matchup_splits(
        prepared_dir, strict_dir, split=split, levels=levels
    )

    report: dict[str, Any] = {
        "split": split,
        "strict_split_base_dir": str(strict_dir),
        "strict_split": split_manifest,
        "levels": {},
    }
    if int(split_manifest["levels"]["all_unseen_matchups"]["rows"]) == 0:
        for level_name, level_manifest in split_manifest["levels"].items():
            report["levels"][level_name] = {
                "difficulty": level_manifest["difficulty"],
                "description": level_manifest["description"],
                "split_dir": level_manifest["split_dir"],
                "metrics": None,
                "threshold_check": {"auc_gt_0_55": None, "brier_lt_0_25": None},
            }
        report["warning"] = "No unseen matchups remain after filtering against train."
        output = artifact_dir / f"unseen-{split}-matchup-metrics.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = _device(str(config.training["device"]))
    model = SymmetricMatchupModel(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.to(device)

    for level_name, level_manifest in split_manifest["levels"].items():
        level_report = {
            "difficulty": level_manifest["difficulty"],
            "description": level_manifest["description"],
            "rows": level_manifest["rows"],
            "matchups": level_manifest["matchups"],
            "row_share": level_manifest["row_share"],
            "split_dir": level_manifest["split_dir"],
            "metrics": None,
            "threshold_check": {"auc_gt_0_55": None, "brier_lt_0_25": None},
        }
        if int(level_manifest["rows"]) > 0:
            metrics = _evaluate_level(
                model,
                payload,
                config,
                Path(level_manifest["split_dir"]),
                device,
            )
            level_report["metrics"] = metrics
            level_report["threshold_check"] = {
                "auc_gt_0_55": metrics["auc"] is not None and metrics["auc"] > 0.55,
                "brier_lt_0_25": metrics["brier_score"] < 0.25,
            }
        report["levels"][level_name] = level_report

    report["metrics"] = report["levels"]["all_unseen_matchups"]["metrics"]
    report["threshold_check"] = report["levels"]["all_unseen_matchups"]["threshold_check"]
    output = artifact_dir / f"unseen-{split}-matchup-metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
