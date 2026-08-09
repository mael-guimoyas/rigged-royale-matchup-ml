from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

import rigged_matchup_ml.api_collect as api_collect
from rigged_matchup_ml.api_collect import (
    BalancedFrontier,
    ClashRoyaleClient,
    CollectorRunLock,
    RateLimiter,
    _battle_fingerprint,
    _battle_time,
    _effective_worker_count,
    _fetch_player,
    _is_fresh,
    _ladder_bucket_label,
    _normalize_baseline,
    _opponent_candidates,
    _resolve_cr_api_tokens,
    _segment_tracked,
    league_from_profile,
    mode_key_for,
    normalize_tag,
)
from rigged_matchup_ml.domain import parse_battle_row

BUCKETS = [0, 5000, 7000, 9000, 12000, 14000, 999999]
DATA_CONFIG = {
    "require_exactly_eight_cards": True,
    "allowed_modes": ["ladder", "ranked"],
    "max_raw_average_level_difference": None,
    "trophy_buckets": BUCKETS,
    "top_ladder_buckets": [100, 1000, 10000],
}


def _deck(start: int) -> list[dict]:
    return [{"id": start + i, "level": 14} for i in range(8)]


def _api_time(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.000Z")


def _battle(battle_type: str = "pathOfLegend", battle_time: str | None = None) -> dict:
    return {
        "type": battle_type,
        "battleTime": battle_time or "20260601T120000.000Z",
        "team": [{"tag": "#AAA", "crowns": 2, "cards": _deck(1000)}],
        "opponent": [{"tag": "#BBB", "crowns": 1, "cards": _deck(2000)}],
    }


def test_normalize_tag() -> None:
    assert normalize_tag(" #abc ") == "#ABC"
    assert normalize_tag("abc") == "#ABC"


def test_resolve_cr_api_tokens_selects_primary_secondary_or_both(monkeypatch) -> None:
    monkeypatch.setenv("CR_API_TOKEN", " token-one ")
    monkeypatch.setenv("CR_API_TOKEN2", "\"token two\"")

    assert _resolve_cr_api_tokens("1") == [("CR_API_TOKEN", "token-one")]
    assert _resolve_cr_api_tokens("2") == [("CR_API_TOKEN2", "tokentwo")]
    assert _resolve_cr_api_tokens("both") == [
        ("CR_API_TOKEN", "token-one"),
        ("CR_API_TOKEN2", "tokentwo"),
    ]


def test_resolve_cr_api_tokens_requires_requested_key(monkeypatch) -> None:
    monkeypatch.setenv("CR_API_TOKEN", "token-one")
    monkeypatch.delenv("CR_API_TOKEN2", raising=False)

    with pytest.raises(RuntimeError, match="CR_API_TOKEN2"):
        _resolve_cr_api_tokens("2")

    with pytest.raises(ValueError, match="1, 2, both"):
        _resolve_cr_api_tokens("bad")


def test_clash_royale_client_round_robins_both_tokens(monkeypatch) -> None:
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout):
        requests.append(request)
        return Response()

    monkeypatch.setenv("CR_API_TOKEN", "token-one")
    monkeypatch.setenv("CR_API_TOKEN2", "token-two")
    monkeypatch.setenv("CR_API_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ClashRoyaleClient(RateLimiter(1_000_000), token_mode="both")
    client.player("#AAA")
    client.player("#BBB")

    assert [request.get_header("Authorization") for request in requests] == [
        "Bearer token-one",
        "Bearer token-two",
    ]
    assert client.token_stats == {
        "CR_API_TOKEN": {"requests": 1, "rate_limited": 0, "errors": 0},
        "CR_API_TOKEN2": {"requests": 1, "rate_limited": 0, "errors": 0},
    }


def test_clash_royale_client_both_shares_total_request_rate(monkeypatch) -> None:
    limiters = []

    class RecordingLimiter:
        def __init__(self, requests_per_second):
            self.requests_per_second = requests_per_second
            self.acquired = 0
            limiters.append(self)

        def acquire(self):
            self.acquired += 1

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"{}"

    monkeypatch.setenv("CR_API_TOKEN", "token-one")
    monkeypatch.setenv("CR_API_TOKEN2", "token-two")
    monkeypatch.setenv("CR_API_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(api_collect, "RateLimiter", RecordingLimiter)
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())

    client = ClashRoyaleClient(
        token_mode="both",
        requests_per_second=30.0,
    )
    client.player("#AAA")
    client.player("#BBB")

    assert [limiter.requests_per_second for limiter in limiters] == [30.0]
    assert [limiter.acquired for limiter in limiters] == [2]


def test_mode_key_for_ranked_and_ladder() -> None:
    assert mode_key_for({"type": "pathOfLegend"}) == "ranked"
    assert mode_key_for({"type": "PvP"}) == "ladder"
    assert mode_key_for({"type": "PvP", "gameMode": {"name": "Ranked1v1"}}) == "ranked"
    assert mode_key_for({"type": "clanMate"}) == "other"


def test_league_from_profile() -> None:
    assert league_from_profile({"currentPathOfLegendSeasonResult": {"leagueNumber": 7}}) == 7
    assert league_from_profile({}) is None
    assert league_from_profile(None) is None


def test_fingerprint_is_deterministic() -> None:
    battle = _battle()
    assert _battle_fingerprint("#AAA", battle) == _battle_fingerprint("#AAA", battle)
    assert _battle_fingerprint("#AAA", battle) != _battle_fingerprint("#ZZZ", battle)


def test_api_battle_parses_into_training_row() -> None:
    data_config = {
        "require_exactly_eight_cards": True,
        "allowed_modes": ["ladder", "ranked"],
        "max_raw_average_level_difference": None,
        "trophy_buckets": [0, 5000, 7000, 9000, 12000, 14000, 999999],
        "top_ladder_buckets": [100, 1000, 10000],
        "ranked_league_buckets": [1, 3, 5, 8],
    }
    row = {
        "raw": _battle(),
        "fingerprint": "fp1",
        "battle_time": "20260601T120000.000Z",
        "inserted_at": "20260601T120001.000Z",
        "mode_key": "ranked",
        "league_number": 7,
    }

    parsed = parse_battle_row(row, data_config)

    assert parsed is not None
    # Collection deliberately retains the exact league; prepare pools it later.
    assert parsed["segment"] == "ranked:league-7"
    assert parsed["win"] is True
    assert len(parsed["team_card_ids"]) == 8
    assert parsed["mode_key"] == "ranked"


def test_battle_time_parses_api_format_and_rejects_junk() -> None:
    assert _battle_time({"battleTime": "20260601T120000.000Z"}) == datetime(
        2026, 6, 1, 12, 0, tzinfo=timezone.utc
    )
    assert _battle_time({"battleTime": "2026-06-01T12:00:00Z"}) == datetime(
        2026, 6, 1, 12, 0, tzinfo=timezone.utc
    )
    assert _battle_time({}) is None
    assert _battle_time({"battleTime": "not-a-date"}) is None


def test_is_fresh_keeps_recent_drops_old_and_undatable() -> None:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=2)
    assert _is_fresh({"battleTime": _api_time(now - timedelta(hours=6))}, cutoff) is True
    assert _is_fresh({"battleTime": _api_time(now - timedelta(days=5))}, cutoff) is False
    # No timestamp = cannot prove freshness = dropped (would otherwise be dated
    # `now()` downstream and poison the chronological split).
    assert _is_fresh({}, cutoff) is False
    # No window configured: everything passes.
    assert _is_fresh({"battleTime": "20200101T000000.000Z"}, None) is True


class _StubClient:
    """Minimal stand-in for ClashRoyaleClient in _fetch_player tests."""

    def __init__(self, battles: list[dict]) -> None:
        self._battles = battles
        self.player_calls = 0

    def battlelog(self, tag: str) -> list[dict]:
        return self._battles

    def player(self, tag: str) -> dict:
        self.player_calls += 1
        return {"currentPathOfLegendSeasonResult": {"leagueNumber": 7}}


def test_fetch_player_keeps_only_battles_inside_the_age_window() -> None:
    now = datetime.now(timezone.utc)
    fresh = _battle(battle_time=_api_time(now - timedelta(hours=3)))
    old = _battle(battle_time=_api_time(now - timedelta(days=9)))
    old["opponent"][0]["tag"] = "#STALEOPP"

    records, candidates, seen, stale = _fetch_player(
        _StubClient([fresh, old]),
        "#AAA",
        DATA_CONFIG,
        {"ladder", "ranked"},
        snowball=True,
        min_trophies=5000,
        max_age_days=2,
    )

    assert seen == 2
    assert stale == 1
    assert len(records) == 1
    # The stale battle contributes no snowball candidates either.
    assert "#STALEOPP" not in {tag for tag, _mode, _trophies in candidates}


def test_fetch_player_without_age_window_keeps_everything() -> None:
    old = _battle(battle_time="20200101T000000.000Z")

    records, _candidates, seen, stale = _fetch_player(
        _StubClient([old]),
        "#AAA",
        DATA_CONFIG,
        {"ladder", "ranked"},
        snowball=True,
        min_trophies=5000,
        max_age_days=None,
    )

    assert (seen, stale, len(records)) == (1, 0, 1)


def test_fetch_player_skips_profile_when_only_ranked_battle_is_stale() -> None:
    now = datetime.now(timezone.utc)
    stale_ranked = _battle(battle_time=_api_time(now - timedelta(days=9)))
    fresh_ladder = _battle("PvP", battle_time=_api_time(now - timedelta(hours=1)))
    fresh_ladder["team"][0]["startingTrophies"] = 8000
    fresh_ladder["opponent"][0]["startingTrophies"] = 8000
    client = _StubClient([stale_ranked, fresh_ladder])

    records, _candidates, seen, stale = _fetch_player(
        client,
        "#AAA",
        DATA_CONFIG,
        {"ladder", "ranked"},
        snowball=True,
        min_trophies=5000,
        max_age_days=2,
    )

    assert (seen, stale, len(records)) == (2, 1, 1)
    assert client.player_calls == 0


def test_fetch_player_loads_profile_for_a_fresh_ranked_battle() -> None:
    now = datetime.now(timezone.utc)
    client = _StubClient([_battle(battle_time=_api_time(now - timedelta(hours=1)))])

    records, _candidates, _seen, _stale = _fetch_player(
        client,
        "#AAA",
        DATA_CONFIG,
        {"ladder", "ranked"},
        snowball=True,
        min_trophies=5000,
        max_age_days=2,
    )

    assert records[0]["segment"] == "ranked:league-7"
    assert client.player_calls == 1


def test_worker_count_auto_sizes_for_target_rate_and_honors_override() -> None:
    assert _effective_worker_count(22, 75) == 22
    assert _effective_worker_count(0, 75) == 188
    assert _effective_worker_count(None, 10) == 25


def test_collector_run_lock_rejects_concurrent_writer_and_releases(tmp_path) -> None:
    lock_path = tmp_path / ".collect-api.lock"
    first = CollectorRunLock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="Another collect-api process"):
            CollectorRunLock(lock_path)
    finally:
        first.close()

    replacement = CollectorRunLock(lock_path)
    replacement.close()


def test_ladder_bucket_label_splits_seasonal_road() -> None:
    assert _ladder_bucket_label(5200, BUCKETS) == "ladder:5000-6999"
    assert _ladder_bucket_label(13000, BUCKETS) == "ladder:12000-13999"
    assert _ladder_bucket_label(16000, BUCKETS) == "ladder:14000-999998"
    assert _ladder_bucket_label(1_000_000, BUCKETS) is None


def test_segment_tracked_keeps_high_drops_low() -> None:
    assert _segment_tracked("ranked:league-3", 5000) is True
    assert _segment_tracked("ranked:unknown", 5000) is True
    assert _segment_tracked("ladder:top-100", 5000) is True
    assert _segment_tracked("ladder:7000-8999", 5000) is True
    assert _segment_tracked("ladder:0-4999", 5000) is False
    assert _segment_tracked("ladder:unknown", 5000) is False
    assert _segment_tracked("ladder:overflow", 5000) is False


def test_opponent_candidates_carry_trophies_and_mode() -> None:
    battle = {
        "type": "PvP",
        "team": [{"tag": "#AAA", "startingTrophies": 8200}],
        "opponent": [{"tag": "#BBB", "startingTrophies": 4000}],
    }
    candidates = _opponent_candidates(battle)
    assert ("#BBB", "ladder", 4000) in candidates
    assert ("#AAA", "ladder", 8200) in candidates


def test_frontier_skips_low_ladder_and_queues_ranked() -> None:
    frontier = BalancedFrontier(
        seeds=[], buckets=BUCKETS, min_trophies=5000, baseline_counts=Counter(), max_queued=None
    )
    skipped = frontier.add(
        [
            ("#LOW", "ladder", 3000),  # below min -> dropped
            ("#MID", "ladder", 6000),  # tracked
            ("#RANK", "ranked", None),  # ranked pooled, always kept
            ("#NOTROPHY", "ladder", None),  # ladder w/o trophy -> dropped
        ]
    )
    assert skipped == 2
    assert frontier.queue_size() == 2


def test_normalize_baseline_folds_legacy_blob_onto_current_buckets() -> None:
    # Legacy "12000-99998" blob must land in the new 12000-14999 band so the
    # split 15000+ band reads empty (not the whole high-trophy mass).
    legacy = Counter(
        {
            "ladder:12000-99998": 6_000_000,
            "ladder:7000-8999": 100,
            "ranked:league-3": 50,
        }
    )
    aligned = _normalize_baseline(legacy, BUCKETS)
    assert aligned["ladder:12000-13999"] == 6_000_000
    assert aligned["ladder:14000-999998"] == 0
    assert aligned["ladder:7000-8999"] == 100
    assert aligned["ranked:league-3"] == 50


def test_frontier_serves_neediest_band_first() -> None:
    # 5000-6999 is starved on disk; the frontier should pop it before the
    # saturated 12000-13999 band even though both are queued.
    baseline = Counter({"ladder:5000-6999": 10, "ladder:12000-13999": 10_000})
    frontier = BalancedFrontier(
        seeds=[], buckets=BUCKETS, min_trophies=5000, baseline_counts=baseline, max_queued=None
    )
    frontier.add([("#HIGH", "ladder", 13000), ("#NEEDY", "ladder", 6000)])
    assert frontier.next_tag() == "#NEEDY"
    assert frontier.next_tag() == "#HIGH"


def test_frontier_bootstraps_from_seeds_when_bands_empty() -> None:
    frontier = BalancedFrontier(
        seeds=["#S1", "#S2"],
        buckets=BUCKETS,
        min_trophies=5000,
        baseline_counts=Counter(),
        max_queued=None,
    )
    # No discoveries yet -> neediest band empty -> seeds drain to bootstrap.
    assert frontier.next_tag() in {"#S1", "#S2"}
    assert frontier.next_tag() in {"#S1", "#S2"}
    assert frontier.next_tag() is None
