from rigged_matchup_ml.extraction import Deduplicator


def test_deduplicator_keeps_unique_new_records_in_bulk(tmp_path) -> None:
    deduplicator = Deduplicator(tmp_path / "dedup.sqlite3")
    try:
        first = [
            {"game_id": "a", "value": 1},
            {"game_id": "a", "value": 2},
            {"game_id": "b", "value": 3},
        ]
        second = [
            {"game_id": "b", "value": 4},
            {"game_id": "c", "value": 5},
        ]

        assert deduplicator.keep_new(first) == [
            {"game_id": "a", "value": 1},
            {"game_id": "b", "value": 3},
        ]
        assert deduplicator.keep_new(second) == [{"game_id": "c", "value": 5}]
    finally:
        deduplicator.close()
