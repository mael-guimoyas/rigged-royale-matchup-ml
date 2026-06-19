from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.dataset as pads
import torch
from torch.utils.data import default_collate

from .dataset import FEATURE_COLUMNS, encode_row
from .model import SymmetricMatchupModel


META_COLUMNS = FEATURE_COLUMNS + ["game_id", "team_deck_key", "opponent_deck_key"]


def _class_sql(column: str, low: float, high: float) -> str:
    return f"case when {column} < {low} then 'bad' when {column} > {high} then 'good' else 'neutral' end"


@torch.no_grad()
def evaluate_meta(
    prepared_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    min_support: int = 100,
    low: float = 0.45,
    high: float = 0.55,
    batch_size: int = 16_384,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SymmetricMatchupModel(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    temperature = float(payload["temperature"])
    segment_temperatures = payload.get("segment_temperatures") or {}
    calibration = payload.get("calibration") or {}
    global_calibration = calibration.get("global", {})
    global_temperature = float(global_calibration.get("temperature", temperature))
    global_bias = float(global_calibration.get("bias", 0.0))
    segment_calibrations = calibration.get("segments") or {}
    dataset = pads.dataset(prepared_dir / "test", format="parquet")
    database_path = output_dir / "meta-evaluation.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute("drop table if exists predictions")
    connection.execute(
        """
        create table predictions (
          game_id varchar, segment varchar, team_deck_key varchar,
          opponent_deck_key varchar, target boolean, probability double
        )
        """
    )
    scanner = dataset.scanner(columns=META_COLUMNS, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        rows = record_batch.to_pylist()
        encoded = default_collate([encode_row(row, payload["vocabulary"]) for row in rows])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(encoded)
        temperatures: list[float] = []
        biases: list[float] = []
        for row in rows:
            segment = str(row["segment"])
            if segment_calibrations:
                segment_calibration = segment_calibrations.get(segment, {})
                temperatures.append(
                    float(segment_calibration.get("temperature", global_temperature))
                )
                biases.append(float(segment_calibration.get("bias", global_bias)))
            else:
                temperatures.append(float(segment_temperatures.get(segment, temperature)))
                biases.append(0.0)
        batch_temperatures = torch.tensor(
            temperatures, dtype=torch.float32, device=device
        ).clamp_min(1e-4)
        batch_biases = torch.tensor(biases, dtype=torch.float32, device=device)
        probabilities = torch.sigmoid(logits / batch_temperatures + batch_biases).cpu().tolist()
        table = pa.table(
            {
                "game_id": [row["game_id"] for row in rows],
                "segment": [row["segment"] for row in rows],
                "team_deck_key": [row["team_deck_key"] for row in rows],
                "opponent_deck_key": [row["opponent_deck_key"] for row in rows],
                "target": [row["win"] for row in rows],
                "probability": probabilities,
            }
        )
        connection.register("prediction_batch", table)
        connection.execute("insert into predictions select * from prediction_batch")
        connection.unregister("prediction_batch")

    predicted_class = _class_sql("predicted_rate", low, high)
    observed_class = _class_sql("observed_rate", low, high)
    report_row = connection.execute(
        f"""
        with pairs as (
          select segment, team_deck_key, opponent_deck_key,
                 count(*) n, avg(probability) predicted_rate,
                 avg(target::int) observed_rate
          from predictions
          group by all
        ), supported as (
          select *, {predicted_class} predicted_class, {observed_class} observed_class
          from pairs where n >= {int(min_support)}
        )
        select
          (select count(*) from predictions) test_games,
          (select count(*) from pairs) observed_exact_matchups,
          count(*) supported_matchups,
          coalesce(sum(n), 0) supported_games,
          coalesce(sum(n)::double / (select count(*) from predictions), 0) coverage,
          coalesce(sum(n * (predicted_class = observed_class)::int)::double / nullif(sum(n),0), 0)
            meta_weighted_class_accuracy,
          coalesce(avg(abs(predicted_rate - observed_rate)), 0) matchup_mae,
          coalesce(sum(n * abs(predicted_rate - observed_rate)) / nullif(sum(n),0), 0)
            matchup_mae_weighted
        from supported
        """
    ).fetchone()
    columns = [description[0] for description in connection.description]
    report = dict(zip(columns, report_row, strict=True))
    distribution = connection.execute(
        f"""
        select {_class_sql('probability', low, high)} matchup_class,
               count(*) games, count(*)::double / sum(count(*)) over() meta_share
        from predictions group by 1 order by 1
        """
    ).fetchall()
    report["predicted_meta_distribution"] = [
        {"class": row[0], "games": row[1], "share": row[2]} for row in distribution
    ]
    report["min_support"] = min_support
    report["neutral_interval"] = [low, high]
    (output_dir / "meta-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    connection.close()
    return report
