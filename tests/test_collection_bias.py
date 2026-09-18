from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rigged_matchup_ml.card_stats import CARD_ELIXIR
from rigged_matchup_ml.collection_bias import _gini, _wilson_interval, collection_bias_report
from rigged_matchup_ml.config import AppConfig
from rigged_matchup_ml.extraction import SCHEMA

BUCKETS = [0, 5000, 7000, 9000, 12000, 14000, 999999]
SEGMENTS = [
    ("ladder", "ladder:5000-6999"),
    ("ladder", "ladder:7000-8999"),
    ("ladder", "ladder:9000-11999"),
    ("ladder", "ladder:12000-13999"),
    ("ladder", "ladder:14000-999998"),
    ("ranked", "ranked:league-5"),
]
NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
CONFIG = AppConfig(
    raw={
        "data": {
            "raw_dir": "data/raw",
            "trophy_buckets": BUCKETS,
            "collect_min_trophies": 5000,
            "collect_max_age_days": 2,
        },
        "training": {"artifact_dir": "artifacts"},
    },
    source_path=Path("config/default.yaml"),
)


def _cards(deck_index: int) -> list[int]:
    return [1000 + deck_index * 8 + offset for offset in range(8)]


def _row(
    index: int,
    segment: tuple[str, str],
    team_deck: int,
    opponent_deck: int,
    win: bool,
    battle_time: datetime,
    game_id: str | None = None,
) -> dict[str, Any]:
    mode_key, segment_key = segment
    return {
        "game_id": game_id or f"g{index}",
        "source_fingerprint": f"f{index}",
        "battle_time": battle_time,
        "inserted_at": battle_time,
        "mode_key": mode_key,
        "segment": segment_key,
        "patch": "2026-09",
        "team_card_ids": _cards(team_deck),
        "opponent_card_ids": _cards(opponent_deck),
        "team_evolution_levels": [0] * 8,
        "opponent_evolution_levels": [0] * 8,
        "team_hero_levels": [0] * 8,
        "opponent_hero_levels": [0] * 8,
        "team_card_roles": [1] * 8,
        "opponent_card_roles": [1] * 8,
        "team_tower_troop_id": 1,
        "opponent_tower_troop_id": 1,
        "team_deck_key": f"deck-{team_deck}",
        "opponent_deck_key": f"deck-{opponent_deck}",
        "matrix_prior": 0.5,
        "win": win,
    }


def _clean_rows(deck_count: int = 120, per_segment: int = 40) -> list[dict[str, Any]]:
    """A corpus with no bias: even sides, every tracked band filled, fresh, spread decks."""
    rows: list[dict[str, Any]] = []
    index = 0
    for segment in SEGMENTS:
        for position in range(per_segment):
            rows.append(
                _row(
                    index,
                    segment,
                    team_deck=index % deck_count,
                    opponent_deck=(index + 1) % deck_count,
                    # Alternate inside each segment so every segment is even too.
                    win=position % 2 == 0,
                    battle_time=NOW - timedelta(minutes=index),
                )
            )
            index += 1
    return rows


def _write(rows: list[dict[str, Any]], directory: Path, name: str = "api-part-000000") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), directory / f"{name}.parquet")
    return directory


def _codes(report: dict[str, Any]) -> set[str]:
    return {warning["code"] for warning in report["warnings"]}


def test_wilson_interval_brackets_the_rate() -> None:
    low, high = _wilson_interval(500, 1000)
    assert low < 0.5 < high
    # More data, tighter interval.
    wide = _wilson_interval(50, 100)
    assert (high - low) < (wide[1] - wide[0])


def test_gini_is_zero_when_every_band_is_equal() -> None:
    assert _gini([100, 100, 100]) == pytest.approx(0.0, abs=1e-9)
    assert _gini([0, 0, 300]) > 0.6
    assert _gini([]) == 0.0


def test_clean_corpus_raises_no_warning(tmp_path: Path) -> None:
    raw_dir = _write(_clean_rows(), tmp_path / "raw")
    report = collection_bias_report(CONFIG, raw_dir=raw_dir, include_cards=False)

    assert report["status"] == "ok"
    assert report["warnings"] == []
    assert report["rows"] == 240
    assert report["side_balance"]["win_rate"] == pytest.approx(0.5)
    assert report["segments"]["missing_bands"] == []
    assert report["segments"]["imbalance_ratio"] == pytest.approx(1.0)
    assert report["freshness"]["rows_outside_window"] == 0


def test_side_balance_skew_is_flagged(tmp_path: Path) -> None:
    rows = _clean_rows()
    # Flip most losses to wins: the crawl now looks like it selects winners.
    for position, row in enumerate(rows):
        row["win"] = position % 10 != 0
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir, include_cards=False)

    assert "side_balance" in _codes(report)
    assert report["side_balance"]["win_rate"] == pytest.approx(0.9)
    assert report["status"] == "warn"


def test_tiny_even_skew_is_not_flagged(tmp_path: Path) -> None:
    """One extra win out of 240 is noise, not bias: neither test is met."""
    rows = _clean_rows()
    assert rows[1]["win"] is False
    rows[1]["win"] = True
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir, include_cards=False)

    assert "side_balance" not in _codes(report)


def test_missing_band_and_imbalance_are_flagged(tmp_path: Path) -> None:
    rows = [row for row in _clean_rows() if row["segment"] != "ladder:14000-999998"]
    rows.extend(
        _row(
            1000 + index,
            ("ladder", "ladder:5000-6999"),
            team_deck=index % 120,
            opponent_deck=(index + 1) % 120,
            win=index % 2 == 0,
            battle_time=NOW - timedelta(minutes=index),
        )
        for index in range(600)
    )
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir, include_cards=False)

    assert "missing_bands" in _codes(report)
    assert report["segments"]["missing_bands"] == ["ladder:14000-999998"]
    assert "segment_imbalance" in _codes(report)
    assert report["segments"]["imbalance_ratio"] > 10.0


def test_legacy_band_is_folded_before_the_gap_check(tmp_path: Path) -> None:
    """Old shards use the pre-split 12000+ band; that must not read as two empty bands."""
    rows = _clean_rows()
    for row in rows:
        if row["segment"] in {"ladder:12000-13999", "ladder:14000-999998"}:
            row["segment"] = "ladder:12000-99998"
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir, include_cards=False)

    assert report["segments"]["missing_bands"] == ["ladder:14000-999998"]
    assert report["segments"]["band_rows"]["ladder:12000-13999"] == 80
    assert "ladder:12000-99998" in report["segments"]["legacy_bands_folded"]


def test_untracked_rows_are_flagged(tmp_path: Path) -> None:
    rows = _clean_rows()
    rows.extend(
        _row(
            2000 + index,
            ("ladder", "ladder:0-4999"),
            team_deck=index % 120,
            opponent_deck=(index + 1) % 120,
            win=index % 2 == 0,
            battle_time=NOW - timedelta(minutes=index),
        )
        for index in range(10)
    )
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir, include_cards=False)

    assert "untracked_rows" in _codes(report)
    assert report["segments"]["untracked_rows"] == 10


def test_orientation_drift_is_flagged(tmp_path: Path) -> None:
    """Crawled side always deck 0, opponents always deck 1: two different populations."""
    rows = [
        _row(
            index,
            SEGMENTS[index % len(SEGMENTS)],
            team_deck=0,
            opponent_deck=1,
            win=index % 2 == 0,
            battle_time=NOW - timedelta(minutes=index),
        )
        for index in range(240)
    ]
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir)

    assert "orientation_drift" in _codes(report)
    assert report["orientation"]["card_distribution_tvd"] == pytest.approx(1.0)


def test_symmetric_corpus_has_no_orientation_drift(tmp_path: Path) -> None:
    raw_dir = _write(_clean_rows(), tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir)

    assert report["orientation"]["card_distribution_tvd"] == pytest.approx(0.0, abs=0.02)
    assert "orientation_drift" not in _codes(report)


def test_stale_battles_are_flagged(tmp_path: Path) -> None:
    rows = _clean_rows()
    rows[0]["battle_time"] = NOW - timedelta(days=30)
    rows[1]["battle_time"] = None
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir, include_cards=False)

    assert "stale_battles" in _codes(report)
    assert report["freshness"]["rows_outside_window"] == 1
    assert "missing_battle_time" in _codes(report)
    assert report["freshness"]["rows_without_battle_time"] == 1


def test_duplicate_game_ids_are_flagged(tmp_path: Path) -> None:
    rows = _clean_rows()
    rows[5]["game_id"] = rows[4]["game_id"]
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir, include_cards=False)

    assert "duplicate_games" in _codes(report)
    assert report["duplicate_game_ids"] == 1


def test_deck_concentration_and_mirrors_are_flagged(tmp_path: Path) -> None:
    rows = [
        _row(
            index,
            SEGMENTS[index % len(SEGMENTS)],
            team_deck=index % 3,
            opponent_deck=index % 3,
            win=index % 2 == 0,
            battle_time=NOW - timedelta(minutes=index),
        )
        for index in range(240)
    ]
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir, include_cards=False)

    assert "deck_concentration" in _codes(report)
    assert "mirror_matches" in _codes(report)
    assert report["concentration"]["mirror_share"] == pytest.approx(1.0)
    assert report["concentration"]["distinct_decks"] == 3


def test_card_support_reports_absent_static_cards(tmp_path: Path) -> None:
    known = sorted(CARD_ELIXIR)[:8]
    rows = [
        {
            **_row(
                index,
                SEGMENTS[index % len(SEGMENTS)],
                team_deck=0,
                opponent_deck=1,
                win=index % 2 == 0,
                battle_time=NOW - timedelta(minutes=index),
            ),
            "team_card_ids": known,
            "opponent_card_ids": known,
        }
        for index in range(240)
    ]
    raw_dir = _write(rows, tmp_path / "raw")

    report = collection_bias_report(CONFIG, raw_dir=raw_dir, min_card_support=10)

    assert "cards_absent" in _codes(report)
    support = report["card_support"]
    assert support["cards_with_data"] == len(known)
    assert set(known).isdisjoint(support["absent_cards"])
    assert support["unknown_card_ids"] == []


def test_empty_directory_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "raw"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No raw Parquet shards"):
        collection_bias_report(CONFIG, raw_dir=empty)
