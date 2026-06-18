from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import load_config
from .empirical_prior import attach_empirical_prior
from .extraction import extract_from_supabase
from .meta_evaluation import evaluate_meta
from .predictor import predict_payload
from .prepare import prepare_splits
from .trainer import evaluate_checkpoint, train_model


app = typer.Typer(no_args_is_help=True, help="Rigged Royale matchup-quality ML pipeline")
DEFAULT_CONFIG = Path("config/default.yaml")


@app.command()
def extract(
    config: Path = DEFAULT_CONFIG,
    max_rows: int | None = typer.Option(None, help="Useful for a smoke test"),
) -> None:
    """Stream and deduplicate 1v1 battles from Supabase into Parquet shards."""
    result = extract_from_supabase(load_config(config), max_rows=max_rows)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def prepare(config: Path = DEFAULT_CONFIG, overwrite: bool = False) -> None:
    """Create leakage-safe chronological train/validation/test splits and vocabularies."""
    result = prepare_splits(load_config(config), overwrite=overwrite)
    typer.echo(json.dumps(result, indent=2))


@app.command("attach-prior")
def attach_prior(
    config: Path = DEFAULT_CONFIG,
    app_dir: Path | None = typer.Option(
        None,
        help="Rigged Royale app directory containing the empirical matchup TypeScript code.",
    ),
    max_matrix_rows: int | None = typer.Option(
        None,
        help="Optional smoke-test cap for prepared train rows used to build the prior.",
    ),
    score_batch_size: int = typer.Option(
        32768,
        help="Prepared rows sent to the TypeScript scorer per batch.",
    ),
) -> None:
    """Attach a leakage-safe train-only empirical matrix prior to prepared Parquet splits."""
    result = attach_empirical_prior(
        load_config(config),
        app_dir=app_dir,
        max_matrix_rows=max_matrix_rows,
        score_batch_size=score_batch_size,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def train(config: Path = DEFAULT_CONFIG) -> None:
    """Train and calibrate the antisymmetric matchup model."""
    checkpoint = train_model(load_config(config))
    typer.echo(str(checkpoint))


@app.command()
def evaluate(
    checkpoint: Path,
    config: Path = DEFAULT_CONFIG,
) -> None:
    """Evaluate individual-game discrimination and probability calibration."""
    result = evaluate_checkpoint(load_config(config), checkpoint)
    typer.echo(json.dumps(result, indent=2))


@app.command("evaluate-meta")
def evaluate_meta_command(
    checkpoint: Path,
    config: Path = DEFAULT_CONFIG,
    min_support: int = 100,
) -> None:
    """Measure correct/incorrect matchup classes weighted by the observed meta."""
    loaded = load_config(config)
    result = evaluate_meta(
        loaded.resolve(loaded.data["prepared_dir"]),
        checkpoint,
        loaded.resolve(loaded.training["artifact_dir"]),
        min_support=min_support,
        low=float(loaded.evaluation["neutral_low"]),
        high=float(loaded.evaluation["neutral_high"]),
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def predict(checkpoint: Path, request: Path) -> None:
    """Predict a matchup described by a JSON request."""
    typer.echo(json.dumps(predict_payload(checkpoint, request), indent=2))


if __name__ == "__main__":
    app()
