from rigged_matchup_ml.api_collect import (
    _battle_fingerprint,
    league_from_profile,
    mode_key_for,
    normalize_tag,
)
from rigged_matchup_ml.domain import parse_battle_row


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
        "trophy_buckets": [0, 5000, 7000, 9000, 12000, 99999],
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
