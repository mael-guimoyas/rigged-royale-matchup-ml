from collections import Counter

from rigged_matchup_ml.api_collect import (
    BalancedFrontier,
    _battle_fingerprint,
    _ladder_bucket_label,
    _normalize_baseline,
    _opponent_candidates,
    _segment_tracked,
    league_from_profile,
    mode_key_for,
    normalize_tag,
)
from rigged_matchup_ml.domain import parse_battle_row

BUCKETS = [0, 5000, 7000, 9000, 12000, 14000, 999999]


def _deck(start: int) -> list[dict]:
    return [{"id": start + i, "level": 14} for i in range(8)]


def _battle(battle_type: str = "pathOfLegend") -> dict:
    return {
        "type": battle_type,
        "battleTime": "20260601T120000.000Z",
        "team": [{"tag": "#AAA", "crowns": 2, "cards": _deck(1000)}],
        "opponent": [{"tag": "#BBB", "crowns": 1, "cards": _deck(2000)}],
    }


def test_normalize_tag() -> None:
    assert normalize_tag(" #abc ") == "#ABC"
    assert normalize_tag("abc") == "#ABC"


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
    assert parsed["segment"] == "ranked:league-7"
    assert parsed["win"] is True
    assert len(parsed["team_card_ids"]) == 8
    assert parsed["mode_key"] == "ranked"


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
