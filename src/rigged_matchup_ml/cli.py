from __future__ import annotations

import json
from pathlib import Path

import typer

from .api_collect import collect_from_api, download_from_storage, upload_to_storage
from .benchmark import benchmark_model
from .card2vec import pretrain_card_embeddings
from .config import load_config
from .diagnostics import diagnose_ranked_segments
from .empirical_prior import attach_empirical_prior
from .extraction import (
    backfill_ranked_segments,
    drain_from_supabase,
    extract_from_supabase,
)
from .meta_evaluation import evaluate_meta
from .predictor import predict_payload
from .prepare import prepare_splits
from .trainer import evaluate_checkpoint, train_model
from .unseen_evaluation import evaluate_unseen_matchups


app = typer.Typer(no_args_is_help=True, help="Rigged Royale matchup-quality ML pipeline")
DEFAULT_CONFIG = Path("config/default.yaml")


@app.command()
def extract(
    config: Path = DEFAULT_CONFIG,
    max_rows: int | None = typer.Option(None, help="Useful for a smoke test"),
    batch_size: int | None = typer.Option(
        None,
        help="Override database rows fetched per query. Try 25000 or 50000 for speed.",
    ),
) -> None:
    """Stream and deduplicate 1v1 battles from Supabase into Parquet shards."""
    result = extract_from_supabase(
        load_config(config),
        max_rows=max_rows,
        batch_size=batch_size,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("collect-api")
def collect_api(
    config: Path = DEFAULT_CONFIG,
    tags_file: Path | None = typer.Option(
        None, help="Text file with one player tag per line to seed the crawl."
    ),
    from_supabase: int = typer.Option(
        0, help="Seed N most-recent player tags from public.players (0 = off)."
    ),
    snowball: bool = typer.Option(
        True, help="Queue opponents found in each battlelog to expand coverage."
    ),
    max_battles: int | None = typer.Option(None, help="Stop after accepting N battles."),
    max_players: int | None = typer.Option(None, help="Stop after processing N players."),
    requests_per_second: float = typer.Option(
        30.0, help="Global CR API request rate cap across all workers."
    ),
    workers: int = typer.Option(
        16,
        help="Concurrent CR API fetch threads. ~16 saturates the RoyaleAPI proxy "
        "(~16 req/s); raise much higher only against the direct api.clashroyale.com.",
    ),
    upload: bool = typer.Option(
        False, help="Also upload each Parquet shard to Supabase Storage."
    ),
    bucket: str = typer.Option("training-battles", help="Storage bucket for shards."),
    prefix: str = typer.Option("battles", help="Object key prefix inside the bucket."),
    min_trophies: int | None = typer.Option(
        None,
        help="Drop ladder battles/opponents below this trophy count (default from "
        "config collect_min_trophies, 5000). 'En dessous on s'en fout'.",
    ),
    balance: bool = typer.Option(
        True,
        "--balance/--no-balance",
        help="Steer the crawl toward trophy bands under-represented on disk "
        "(scans existing shards at startup to find the gaps).",
    ),
    api_token_mode: str = typer.Option(
        "1",
        "--api-token-mode",
        help="CR API token selection: 1=CR_API_TOKEN, 2=CR_API_TOKEN2, "
        "both=round-robin across both under the same total request rate.",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Render the live tqdm progress bar. Disable it for background runs "
        "inside the IDE terminal.",
    ),
    stats_interval: float = typer.Option(
        10.0,
        "--stats-interval",
        help="Seconds between lightweight status lines when --no-progress is used "
        "(0 disables).",
    ),
) -> None:
    """Fetch battles from the Clash Royale API into training Parquet (local + Storage)."""
    if tags_file is not None and not tags_file.exists():
        raise typer.BadParameter(f"Tags file not found: {tags_file}")
    if tags_file is None and from_supabase <= 0:
        raise typer.BadParameter(
            "Provide a seed: --tags-file <file> or --from-supabase <N>."
        )
    api_token_mode = api_token_mode.strip().lower()
    if api_token_mode not in {"1", "2", "both"}:
        raise typer.BadParameter("--api-token-mode must be one of: 1, 2, both.")
    result = collect_from_api(
        load_config(config),
        tags_file=tags_file,
        from_supabase=from_supabase,
        snowball=snowball,
        max_battles=max_battles,
        max_players=max_players,
        requests_per_second=requests_per_second,
        workers=workers,
        upload=upload,
        bucket=bucket,
        prefix=prefix,
        min_trophies=min_trophies,
        balance=balance,
        api_token_mode=api_token_mode,
        show_progress=progress,
        stats_interval_seconds=stats_interval,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("drain-db")
def drain_db(
    config: Path = DEFAULT_CONFIG,
    delete: bool = typer.Option(
        False,
        "--delete/--no-delete",
        help="Actually DELETE rows from public.battles after archiving them. "
        "Default off = safe dry run that only writes Parquet. Needs a read-WRITE "
        "SUPABASE_DB_URL user.",
    ),
    upload: bool = typer.Option(
        False, help="Also upload each archived shard to Supabase Storage before deleting."
    ),
    batch_size: int | None = typer.Option(
        None, help="Rows fetched (and deleted) per round. Try 25000-50000."
    ),
    max_rows: int | None = typer.Option(None, help="Stop after scanning N rows (smoke test)."),
    bucket: str = typer.Option("training-battles", help="Storage bucket for shards."),
    prefix: str = typer.Option("battles", help="Object key prefix inside the bucket."),
) -> None:
    """Archive public.battles to Parquet, then optionally delete the drained rows.

    Safe to run at the same time as `collect-api`: shards use a separate
    `drain-part-*` prefix and the shared dedup DB is concurrency-safe. With
    --delete, a row leaves Postgres only after its Parquet shard is durably
    written (and uploaded, if --upload). WARNING: deleted battles are gone for
    good; anything still reading `battles.raw` (e.g. the live app) loses them.
    """
    if delete:
        typer.echo(
            "WARNING: --delete will permanently remove drained rows from "
            "public.battles. Ensure shards are safe (use --upload) and that the "
            "app no longer depends on battles.raw.",
            err=True,
        )
    result = drain_from_supabase(
        load_config(config),
        batch_size=batch_size,
        max_rows=max_rows,
        delete=delete,
        upload=upload,
        bucket=bucket,
        prefix=prefix,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("pull-storage")
def pull_storage(
    config: Path = DEFAULT_CONFIG,
    bucket: str = typer.Option("training-battles", help="Storage bucket to download from."),
    prefix: str = typer.Option("battles", help="Object key prefix inside the bucket."),
    overwrite: bool = typer.Option(
        False, help="Re-download shards that already exist locally."
    ),
    workers: int = typer.Option(
        8, help="Parallel Storage downloads. Try 8-16 on RunPod."
    ),
) -> None:
    """Download training Parquet shards from Supabase Storage into data/raw."""
    result = download_from_storage(
        load_config(config),
        bucket=bucket,
        prefix=prefix,
        overwrite=overwrite,
        workers=workers,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("upload-storage")
def upload_storage(
    config: Path = DEFAULT_CONFIG,
    bucket: str = typer.Option("training-battles", help="Storage bucket to upload to."),
    prefix: str = typer.Option("battles", help="Object key prefix inside the bucket."),
    overwrite: bool = typer.Option(
        False, help="Overwrite objects that already exist in Storage."
    ),
) -> None:
    """Upload local data/raw Parquet shards to Supabase Storage."""
    result = upload_to_storage(
        load_config(config), bucket=bucket, prefix=prefix, overwrite=overwrite
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("backfill-ranked-segments")
def backfill_ranked_segments_command(
    config: Path = DEFAULT_CONFIG,
    batch_size: int = typer.Option(
        10000,
        help="Fingerprints fetched from Supabase per backfill query.",
    ),
) -> None:
    """Rewrite old raw extracts so ranked battles are separated by leagueNumber."""
    result = backfill_ranked_segments(load_config(config), batch_size=batch_size)
    typer.echo(json.dumps(result, indent=2))


@app.command("diagnose-ranked-segments")
def diagnose_ranked_segments_command(
    config: Path = DEFAULT_CONFIG,
    sample_size: int = typer.Option(
        100,
        help="ranked:unknown fingerprints sampled to test against the configured database.",
    ),
) -> None:
    """Report ranked league coverage and whether unknown rows can be backfilled."""
    result = diagnose_ranked_segments(load_config(config), sample_size=sample_size)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def prepare(config: Path = DEFAULT_CONFIG, overwrite: bool = False) -> None:
    """Create leakage-safe chronological train/validation/test splits and vocabularies."""
    result = prepare_splits(load_config(config), overwrite=overwrite)
    typer.echo(json.dumps(result, indent=2))


@app.command("pretrain-cards")
def pretrain_cards(
    config: Path = DEFAULT_CONFIG,
    max_rows: int | None = typer.Option(
        None, help="Cap training decks scanned for co-occurrence (smoke test)."
    ),
) -> None:
    """Self-supervised card embeddings from deck co-occurrence (run after prepare).

    Writes data/prepared/card2vec.npy; training loads it as the card-embedding
    warm-start when `card2vec_init` is on. Pure co-occurrence, no labels/stats.
    """
    result = pretrain_card_embeddings(load_config(config), max_rows=max_rows)
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


@app.command("evaluate-unseen")
def evaluate_unseen(
    checkpoint: Path,
    config: Path = DEFAULT_CONFIG,
    split: str = typer.Option(
        "test",
        help="Prepared split to filter against train. Usually 'test' or 'validation'.",
    ),
) -> None:
    """Evaluate only matchups whose unordered deck pair is absent from train."""
    if split not in {"validation", "test"}:
        raise typer.BadParameter("split must be 'validation' or 'test'")
    result = evaluate_unseen_matchups(load_config(config), checkpoint, split=split)
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
def benchmark(
    checkpoint: Path,
    config: Path = DEFAULT_CONFIG,
    split: str = typer.Option(
        "test",
        help="Prepared split to benchmark. Usually 'test' or 'validation'.",
    ),
    min_support: int = typer.Option(
        100,
        help="Minimum games per matchup used to estimate the irreducible noise floor.",
    ),
) -> None:
    """Compare the model to constant 0.5, the matrix prior and the noise floor."""
    if split not in {"validation", "test"}:
        raise typer.BadParameter("split must be 'validation' or 'test'")
    result = benchmark_model(
        load_config(config), checkpoint, split=split, min_support=min_support
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def predict(checkpoint: Path, request: Path) -> None:
    """Predict a matchup described by a JSON request."""
    typer.echo(json.dumps(predict_payload(checkpoint, request), indent=2))


@app.command()
def serve(
    checkpoint: Path = Path("artifacts/matchup-model.pt"),
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8080, help="Port the site's ML_INFERENCE_URL points at."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)."),
) -> None:
    """Serve the matchup model over HTTP (/predict, /health) for the site."""
    import os

    import uvicorn

    os.environ.setdefault("MODEL_CHECKPOINT", str(checkpoint))
    uvicorn.run("rigged_matchup_ml.serve:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
