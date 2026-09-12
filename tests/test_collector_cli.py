from pathlib import Path

import pytest

from rigged_matchup_ml.collector_cli import _build_parser, main

REPO_ROOT = Path(__file__).resolve().parent.parent
COLLECT_SCRIPT = REPO_ROOT / "scripts" / "collect_season87.sh"


def test_collector_cli_has_safe_netcup_defaults() -> None:
    args = _build_parser().parse_args(["--tags-file", "seeds_balanced.txt"])

    assert args.collector_id == "netcup"
    assert args.max_duration_hours == 6.0
    assert args.prefix == "season-87"
    assert args.upload is False


def test_collector_cli_collects_representatively_by_default() -> None:
    # Steering the crawl toward starved bands is a deliberate top-up, never the
    # default: it aims at the players passing through a band instead of the ones
    # living in it. The baseline scan stays on so the bias has something to read.
    args = _build_parser().parse_args(["--tags-file", "seeds_balanced.txt"])

    assert args.deficit_bias == 0.0
    assert args.balance is True


def test_collector_cli_accepts_deficit_bias() -> None:
    args = _build_parser().parse_args(
        ["--tags-file", "seeds_balanced.txt", "--deficit-bias", "0.8"]
    )

    assert args.deficit_bias == pytest.approx(0.8)


def _run(tmp_path: Path, *extra: str) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("data: {}\n", encoding="utf-8")
    tags = tmp_path / "tags.txt"
    tags.write_text("#AAA\n", encoding="utf-8")
    main(["--config", str(config), "--tags-file", str(tags), *extra])


def test_collector_cli_rejects_out_of_range_deficit_bias(tmp_path: Path) -> None:
    # The frontier silently clamps, so "--deficit-bias 80" meaning 80% would run
    # a fully steered crawl without a word. Refuse it instead.
    with pytest.raises(SystemExit, match="between 0.0 and 1.0"):
        _run(tmp_path, "--deficit-bias", "80")


def test_collector_cli_rejects_deficit_bias_without_baseline(tmp_path: Path) -> None:
    # collect_from_api zeroes the bias when balance is off, so this combination
    # would collect proportionally while claiming to top up the starved bands.
    with pytest.raises(SystemExit, match="drop --no-balance"):
        _run(tmp_path, "--deficit-bias", "0.8", "--no-balance")


def test_collector_cli_guards_steering_with_a_queue_depth() -> None:
    # Steering without this is what turned a thin band into a census of whoever
    # had been discovered, which is the documented source of the bias.
    args = _build_parser().parse_args(["--tags-file", "seeds_balanced.txt"])

    assert args.min_queue_depth == 500


def test_collector_cli_rejects_negative_queue_depth(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="cannot be negative"):
        _run(tmp_path, "--min-queue-depth", "-1")


def test_netcup_script_passes_the_steering_flags() -> None:
    # The deployed unit runs this script, so the flags have to survive there
    # too: the CLI gaining an option is worthless if the shell wrapper never
    # sends it. Guards the defaults against an edit that drops or misspells one.
    script = COLLECT_SCRIPT.read_text(encoding="utf-8")

    assert "--deficit-bias" in script
    assert "--min-queue-depth" in script
    assert 'deficit_bias="${COLLECTOR_DEFICIT_BIAS:-0.5}"' in script
    assert 'min_queue_depth="${COLLECTOR_MIN_QUEUE_DEPTH:-500}"' in script
