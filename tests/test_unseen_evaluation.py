from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rigged_matchup_ml.unseen_evaluation import (
    build_unseen_matchup_split,
    build_unseen_matchup_splits,
    matchup_key,
)


def _write_split(prepared_dir: Path, split: str, rows: list[dict]) -> None:
    split_dir = prepared_dir / split
    split_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), split_dir / "data.parquet")


def test_matchup_key_is_order_invariant() -> None:
    assert matchup_key("deck-a", "deck-b") == matchup_key("deck-b", "deck-a")


def test_unseen_split_removes_matchups_present_in_train(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_split(
        prepared_dir,
        "train",
        [
            {
                "game_id": "train-seen",
                "team_deck_key": "deck-a",
                "opponent_deck_key": "deck-b",
                "win": True,
            }
        ],
    )
    _write_split(
        prepared_dir,
        "test",
        [
            {
                "game_id": "test-seen-reversed",
                "team_deck_key": "deck-b",
                "opponent_deck_key": "deck-a",
                "win": False,
            },
            {
                "game_id": "test-unseen",
                "team_deck_key": "deck-c",
                "opponent_deck_key": "deck-d",
                "win": True,
            },
        ],
    )

    manifest = build_unseen_matchup_split(
        prepared_dir,
        tmp_path / "strict",
        split="test",
    )

    assert manifest["original_rows"] == 2
    assert manifest["excluded_seen_rows"] == 1
    assert manifest["strict_rows"] == 1
    rows = pq.read_table(tmp_path / "strict" / "data.parquet").to_pylist()
    assert [row["game_id"] for row in rows] == ["test-unseen"]
    assert rows[0]["matchup_key"] == matchup_key("deck-c", "deck-d")


def test_unseen_splits_are_grouped_by_difficulty(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_split(
        prepared_dir,
        "train",
        [
            {
                "game_id": "train-a-b",
                "team_deck_key": "deck-a",
                "opponent_deck_key": "deck-b",
                "win": True,
            },
            {
                "game_id": "train-c-d",
                "team_deck_key": "deck-c",
                "opponent_deck_key": "deck-d",
                "win": False,
            },
        ],
    )
    _write_split(
        prepared_dir,
        "test",
        [
            {
                "game_id": "seen-reversed",
                "team_deck_key": "deck-b",
                "opponent_deck_key": "deck-a",
                "win": False,
            },
            {
                "game_id": "known-decks-new-matchup",
                "team_deck_key": "deck-a",
                "opponent_deck_key": "deck-c",
                "win": True,
            },
            {
                "game_id": "one-new-deck",
                "team_deck_key": "deck-a",
                "opponent_deck_key": "deck-x",
                "win": True,
            },
            {
                "game_id": "two-new-decks",
                "team_deck_key": "deck-y",
                "opponent_deck_key": "deck-z",
                "win": False,
            },
        ],
    )

    manifest = build_unseen_matchup_splits(
        prepared_dir,
        tmp_path / "strict",
        split="test",
    )

    assert manifest["excluded_seen_rows"] == 1
    assert manifest["levels"]["all_unseen_matchups"]["rows"] == 3
    assert manifest["levels"]["known_decks_new_matchup"]["rows"] == 1
    assert manifest["levels"]["one_new_deck"]["rows"] == 1
    assert manifest["levels"]["two_new_decks"]["rows"] == 1

    for level, expected_game_id in [
        ("known_decks_new_matchup", "known-decks-new-matchup"),
        ("one_new_deck", "one-new-deck"),
        ("two_new_decks", "two-new-decks"),
    ]:
        level_dir = Path(manifest["levels"][level]["split_dir"])
        rows = pq.read_table(level_dir / "data.parquet").to_pylist()
        assert [row["game_id"] for row in rows] == [expected_game_id]
