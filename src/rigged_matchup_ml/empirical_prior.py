from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from .config import AppConfig
from .metrics import binary_metrics, binary_metrics_by_group


PRIOR_MIN = 0.001
PRIOR_MAX = 0.999
COVERAGE_LEVELS = [
    "archGlobal",
    "archBucket",
    "planGlobal",
    "planBucket",
    "spellAnswerGlobal",
]
SCORE_COLUMNS = [
    "team_card_ids",
    "opponent_card_ids",
    "segment",
    "win",
]


def clamp_prior(value: float) -> float:
    if not np.isfinite(value):
        return 0.5
    return float(min(PRIOR_MAX, max(PRIOR_MIN, value)))


def reverse_prior(value: float) -> float:
    return clamp_prior(1.0 - value)


def fallback_prior() -> float:
    return 0.5


def _repo_root(config: AppConfig) -> Path:
    return config.source_path.parent.parent.resolve()


def _default_app_dir(config: AppConfig) -> Path:
    configured = config.data.get("empirical_app_dir", "../riggedroyale")
    path = Path(str(configured))
    if path.is_absolute():
        return path
    return (_repo_root(config) / path).resolve()


def _tsx_executable(app_dir: Path) -> Path:
    suffix = ".cmd" if os.name == "nt" else ""
    executable = app_dir / "node_modules" / ".bin" / f"tsx{suffix}"
    if not executable.exists():
        raise RuntimeError(
            f"tsx was not found at {executable}. Run npm install in {app_dir} first."
        )
    return executable


def _helper_path(config: AppConfig) -> Path:
    helper = _repo_root(config) / "tools" / "empirical-prior-helper.ts"
    if not helper.exists():
        raise RuntimeError(f"Empirical prior helper not found: {helper}")
    return helper


def _helper_command(config: AppConfig, app_dir: Path, *args: str) -> list[str]:
    return [str(_tsx_executable(app_dir)), str(_helper_path(config)), *args]


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n"


def _iter_train_battles(
    config: AppConfig,
    train_cutoff_epoch: float,
    fetch_size: int = 2_000,
    max_rows: int | None = None,
) -> Iterable[list[dict[str, Any]]]:
    load_dotenv(_repo_root(config) / ".env")
    database_url = os.getenv("SUPABASE_DB_URL")
    if not database_url:
        raise RuntimeError("SUPABASE_DB_URL is missing in .env")

    timeout = int(config.database["statement_timeout_ms"])
    allowed_modes = list(config.data.get("allowed_modes") or [])
    mode_filter = "and mode_key = any(%s)" if allowed_modes else ""
    limit_filter = "limit %s" if max_rows is not None else ""
    parameters: list[Any] = [train_cutoff_epoch]
    if allowed_modes:
        parameters.append(allowed_modes)
    if max_rows is not None:
        parameters.append(max_rows)

    query = f"""
        select player_tag, raw
        from public.battles
        where battle_time is not null
          and battle_time <= to_timestamp(%s)
          {mode_filter}
        {limit_filter}
    """

    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        connection.execute(f"set statement_timeout = {timeout}")
        connection.commit()
        connection.execute("start transaction read only")
        try:
            with connection.cursor(name="empirical_prior_train_rows") as cursor:
                cursor.execute(query, parameters)
                while True:
                    rows = cursor.fetchmany(fetch_size)
                    if not rows:
                        break
                    yield list(rows)
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _build_train_matrix(
    config: AppConfig,
    app_dir: Path,
    matrix_path: Path,
    max_matrix_rows: int | None,
) -> dict[str, Any]:
    prepared_train = config.resolve(config.data["prepared_dir"]) / "train"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    command = _helper_command(config, app_dir, "build-matrix-from-records")
    row_count = 0
    with matrix_path.open("w", encoding="utf-8") as stdout:
        process = subprocess.Popen(
            command,
            cwd=_repo_root(config),
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        assert process.stdin is not None
        try:
            for parquet_path in sorted(prepared_train.glob("*.parquet")):
                parquet = pq.ParquetFile(parquet_path)
                for batch in parquet.iter_batches(
                    batch_size=32_768, columns=SCORE_COLUMNS
                ):
                    for row in pa.Table.from_batches([batch]).to_pylist():
                        if max_matrix_rows is not None and row_count >= max_matrix_rows:
                            break
                        process.stdin.write(
                            _json_line(
                                {
                                    "team_card_ids": row["team_card_ids"],
                                    "opponent_card_ids": row["opponent_card_ids"],
                                    "segment": row["segment"],
                                    "win": bool(row["win"]),
                                }
                            )
                        )
                        row_count += 1
                    if max_matrix_rows is not None and row_count >= max_matrix_rows:
                        break
                if max_matrix_rows is not None and row_count >= max_matrix_rows:
                    break
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else ""
            return_code = process.wait()
        except Exception:
            process.kill()
            raise
    if return_code != 0:
        raise RuntimeError(f"Empirical matrix helper failed: {stderr.strip()}")

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    return {
        "matrix_source": str(prepared_train),
        "train_prepared_rows_scanned": row_count,
        "matrix_path": str(matrix_path),
        "matrix_total_games": int(matrix.get("totalGames", 0)),
        "matrix_edge_counts": {
            level: len(matrix.get(level, [])) for level in COVERAGE_LEVELS
        },
    }


def _score_rows(
    config: AppConfig,
    app_dir: Path,
    matrix_path: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    input_text = "".join(
        _json_line(
            {
                "team_card_ids": row["team_card_ids"],
                "opponent_card_ids": row["opponent_card_ids"],
                "segment": row["segment"],
            }
        )
        for row in rows
    )
    result = subprocess.run(
        _helper_command(config, app_dir, "score-records", "--matrix", str(matrix_path)),
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=_repo_root(config),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Empirical scoring helper failed: {result.stderr.strip()}")
    outputs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(outputs) != len(rows):
        raise RuntimeError(
            f"Empirical scoring helper returned {len(outputs)} rows for {len(rows)} inputs"
        )
    return outputs


def _empty_coverage() -> dict[str, int]:
    return {level: 0 for level in COVERAGE_LEVELS}


def _coverage_rates(counts: dict[str, int], total: int) -> dict[str, float]:
    if total == 0:
        return {level: 0.0 for level in COVERAGE_LEVELS}
    return {level: counts[level] / total for level in COVERAGE_LEVELS}


def _replace_matrix_prior(
    table: pa.Table,
    priors: list[float],
) -> pa.Table:
    prior_array = pa.array(priors, type=pa.float32())
    index = table.schema.get_field_index("matrix_prior")
    if index == -1:
        return table.append_column("matrix_prior", prior_array)
    return table.set_column(index, "matrix_prior", prior_array)


def _write_replacement(path: Path, table: pa.Table) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=100_000)
    temporary.replace(path)


def _score_split(
    config: AppConfig,
    app_dir: Path,
    matrix_path: Path,
    split_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    split_priors: list[float] = []
    split_targets: list[float] = []
    split_segments: list[str] = []
    coverage_counts = _empty_coverage()
    total_rows = 0

    for parquet_path in sorted(split_dir.glob("*.parquet")):
        table = pq.read_table(parquet_path)
        priors: list[float] = []
        for offset in range(0, table.num_rows, batch_size):
            rows = table.slice(offset, batch_size).select(SCORE_COLUMNS).to_pylist()
            scored = _score_rows(config, app_dir, matrix_path, rows)
            for row, output in zip(rows, scored, strict=True):
                prior = clamp_prior(float(output["prior"]))
                priors.append(prior)
                split_priors.append(prior)
                split_targets.append(float(bool(row["win"])))
                split_segments.append(str(row["segment"]))
                total_rows += 1
                coverage = output.get("coverage", {})
                for level in COVERAGE_LEVELS:
                    if bool(coverage.get(level, False)):
                        coverage_counts[level] += 1
        _write_replacement(parquet_path, _replace_matrix_prior(table, priors))

    summary: dict[str, Any] = {
        "rows": total_rows,
        "mean_prior": float(np.mean(split_priors)) if split_priors else 0.5,
        "coverage": _coverage_rates(coverage_counts, total_rows),
    }
    if split_priors and split_targets:
        targets = np.asarray(split_targets, dtype=np.float32)
        probabilities = np.asarray(split_priors, dtype=np.float32)
        summary["prior_metrics"] = binary_metrics(targets, probabilities)
        summary["prior_metrics_by_segment"] = binary_metrics_by_group(
            targets,
            probabilities,
            np.asarray(split_segments),
        )
    return summary


def attach_empirical_prior(
    config: AppConfig,
    app_dir: Path | None = None,
    max_matrix_rows: int | None = None,
    score_batch_size: int = 32_768,
) -> dict[str, Any]:
    prepared_dir = config.resolve(config.data["prepared_dir"])
    artifact_dir = config.resolve(config.training["artifact_dir"])
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_cutoff_epoch = float(manifest["train_cutoff_epoch"])
    validation_cutoff_epoch = float(manifest["validation_cutoff_epoch"])
    resolved_app_dir = (app_dir or _default_app_dir(config)).resolve()
    matrix_path = artifact_dir / "empirical-train-matrix.json"

    matrix_summary = _build_train_matrix(
        config,
        resolved_app_dir,
        matrix_path,
        max_matrix_rows,
    )
    split_summaries = {
        split: _score_split(
            config,
            resolved_app_dir,
            matrix_path,
            prepared_dir / split,
            score_batch_size,
        )
        for split in ("train", "validation", "test")
    }

    report = {
        "cutoff_policy": (
            "data/prepared/train only; split was built chronologically with "
            "battle_time <= train_cutoff_epoch"
        ),
        "train_cutoff_epoch": train_cutoff_epoch,
        "validation_cutoff_epoch": validation_cutoff_epoch,
        "app_dir": str(resolved_app_dir),
        **matrix_summary,
        "splits": split_summaries,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "prior-metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
