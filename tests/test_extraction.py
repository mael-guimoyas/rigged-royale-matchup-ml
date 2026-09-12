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


def test_segment_tally_accumulates_and_survives_a_reopen(tmp_path) -> None:
    # The collector box deletes its shards once Storage has them, so the balance
    # baseline cannot be recounted from disk: it lives here, next to `seen`,
    # which survives the purge for the same reason.
    path = tmp_path / "dedup.sqlite3"
    deduplicator = Deduplicator(path)
    try:
        deduplicator.add_segment_counts({"ladder:5000-5999": 3, "ranked:league-1": 5})
        deduplicator.add_segment_counts({"ladder:5000-5999": 4})
    finally:
        deduplicator.close()

    reopened = Deduplicator(path)
    try:
        counts = reopened.segment_counts()
    finally:
        reopened.close()

    assert counts["ladder:5000-5999"] == 7
    assert counts["ranked:league-1"] == 5


def test_segment_tally_starts_empty_and_ignores_zeroes(tmp_path) -> None:
    # An empty tally is the signal to seed it from whatever shards are still on
    # disk, so it must not be polluted by no-op updates.
    deduplicator = Deduplicator(tmp_path / "dedup.sqlite3")
    try:
        assert deduplicator.segment_counts() == {}
        deduplicator.add_segment_counts({})
        deduplicator.add_segment_counts({"ladder:5000-5999": 0})
        assert deduplicator.segment_counts() == {}
    finally:
        deduplicator.close()
