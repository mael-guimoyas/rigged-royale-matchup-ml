"""Lightweight command line entry point for a collection-only server."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .api_collect import collect_from_api
from .config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigged-collector",
        description="Continuously fetch Clash Royale battles into Parquet and Storage.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--tags-file", type=Path, required=True)
    parser.add_argument("--max-duration-hours", type=float, default=6.0)
    parser.add_argument("--requests-per-second", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--api-token-mode", choices=("1", "2", "both"), default="1")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--bucket", default="training-battles")
    parser.add_argument("--prefix", default="season-87")
    parser.add_argument("--collector-id", default="netcup")
    parser.add_argument("--stats-interval", type=float, default=60.0)
    parser.add_argument("--max-queue", type=int)
    parser.add_argument("--min-trophies", type=int)
    parser.add_argument("--max-age-days", type=float)
    parser.add_argument("--not-before")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if not args.config.is_file():
        raise SystemExit(f"Config file not found: {args.config}")
    if not args.tags_file.is_file():
        raise SystemExit(f"Tags file not found: {args.tags_file}")
    if args.max_duration_hours <= 0:
        raise SystemExit("--max-duration-hours must be positive")
    if args.requests_per_second <= 0:
        raise SystemExit("--requests-per-second must be positive")

    result = collect_from_api(
        load_config(args.config),
        tags_file=args.tags_file,
        max_duration_hours=args.max_duration_hours,
        requests_per_second=args.requests_per_second,
        workers=args.workers,
        upload=args.upload,
        bucket=args.bucket,
        prefix=args.prefix,
        collector_id=args.collector_id,
        max_queue=args.max_queue,
        min_trophies=args.min_trophies,
        max_age_days=args.max_age_days,
        not_before=args.not_before,
        api_token_mode=args.api_token_mode,
        show_progress=False,
        stats_interval_seconds=args.stats_interval,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
