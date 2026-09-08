from rigged_matchup_ml.collector_cli import _build_parser


def test_collector_cli_has_safe_netcup_defaults() -> None:
    args = _build_parser().parse_args(["--tags-file", "seeds_balanced.txt"])

    assert args.collector_id == "netcup"
    assert args.max_duration_hours == 6.0
    assert args.prefix == "season-87"
    assert args.upload is False
