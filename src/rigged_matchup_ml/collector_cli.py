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
    # Band steering. By default the frontier serves bands in proportion to how
    # often the matchmaker actually produces them, which is representative but
    # tracks the live population: season 87 landed 51% of its corpus in
    # ladder:14000-14999 and under 1.5% in every band below 12000.
    # `--deficit-bias p` is the documented catch-up knob (see BalancedFrontier):
    # with probability p the next tag is drawn from the most starved band on
    # disk instead of proportionally. It buys coverage with representativeness
    # -- the last time steering was on, the fetched side won 63.5% of its games
    # in ladder:5000-6999 and 37.2% in ladder:14000+ -- so raise it for a
    # deliberate top-up run and put it back to 0 for a representative one.
    parser.add_argument("--deficit-bias", type=float, default=0.0)
    # Guard rail for the above. Steering a band is stratified sampling, which is
    # unbiased only while the draw inside the band stays random -- and a thin
    # queue drained faster than the crawl refills it is a census of whoever was
    # discovered, not a sample. A band must hold this many candidates before the
    # deficit blend may pick it; below it, the proportional draw applies.
    parser.add_argument("--min-queue-depth", type=int, default=500)
    # The deficit blend needs a baseline of what is already on disk to know
    # which band is starved. --no-balance drops that scan (faster start on a
    # box with thousands of shards) and forces plain proportional draw.
    parser.add_argument("--no-balance", dest="balance", action="store_false")
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
    if not 0.0 <= args.deficit_bias <= 1.0:
        raise SystemExit("--deficit-bias must be between 0.0 and 1.0")
    if args.min_queue_depth < 0:
        raise SystemExit("--min-queue-depth cannot be negative")
    # collect_from_api zeroes the bias when balance is off, so a run asking for
    # both would silently collect proportionally. Fail instead of lying.
    if args.deficit_bias > 0 and not args.balance:
        raise SystemExit("--deficit-bias needs the baseline scan: drop --no-balance")

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
        balance=args.balance,
        deficit_bias=args.deficit_bias,
        min_queue_depth=args.min_queue_depth,
        api_token_mode=args.api_token_mode,
        show_progress=False,
        stats_interval_seconds=args.stats_interval,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
