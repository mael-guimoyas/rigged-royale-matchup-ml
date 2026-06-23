from datetime import datetime, timezone

from rigged_matchup_ml.domain import (
    ROLE_CHAMPION,
    canonical_game_id,
    parse_battle_row,
    parse_deck,
    ranked_league_number,
)


def player(tag: str, crowns: int, offset: int = 0) -> dict:
    return {
        "tag": tag,
        "crowns": crowns,
        "cards": [
            {
                "id": 26000000 + offset + index,
                "level": 11,
                "rarity": "champion" if index == 0 else "common",
                "evolutionLevel": 1 if index == 1 else 0,
                "heroLevel": 1 if index == 0 else 0,
            }
            for index in range(8)
        ],
        "supportCards": [{"id": 159000000}],
    }


DATA_CONFIG = {
    "allowed_modes": ["ladder", "ranked"],
    "max_raw_average_level_difference": None,
    "require_exactly_eight_cards": True,
    "top_ladder_buckets": [100, 1000, 10000],
    "trophy_buckets": [0, 5000, 7000, 9000, 12000, 14000, 999999],
}


def test_champion_and_evolution_are_parsed() -> None:
    deck = parse_deck(player("#A", 1))
    assert deck is not None
    assert deck.cards[0].role == ROLE_CHAMPION
    assert deck.cards[1].evolution_level == 1
    assert deck.cards[0].hero_level == 1
    assert deck.tower_troop_id == 159000000


def test_game_id_is_invariant_to_side_order() -> None:
    first = parse_deck(player("#A", 1))
    second = parse_deck(player("#B", 0, 100))
    assert first is not None and second is not None
    time = datetime(2026, 6, 18, tzinfo=timezone.utc)
    assert canonical_game_id(time, first, second) == canonical_game_id(time, second, first)


def test_ranked_segment_uses_league_number() -> None:
    record = parse_battle_row(
        {
            "fingerprint": "ranked-league",
            "battle_time": datetime(2026, 6, 18, tzinfo=timezone.utc),
            "inserted_at": datetime(2026, 6, 18, tzinfo=timezone.utc),
            "mode_key": "ranked",
            "raw": {
                "battleTime": "20260618T000000.000Z",
                "leagueNumber": 7,
                "team": [player("#A", 1)],
                "opponent": [player("#B", 0, 100)],
            },
        },
        DATA_CONFIG,
    )
    assert record is not None
    assert record["segment"] == "ranked:league-7"


def test_ranked_segment_uses_sql_league_number_before_json() -> None:
    record = parse_battle_row(
        {
            "fingerprint": "ranked-sql-league",
            "battle_time": datetime(2026, 6, 18, tzinfo=timezone.utc),
            "inserted_at": datetime(2026, 6, 18, tzinfo=timezone.utc),
            "mode_key": "ranked",
            "league_number": 6,
            "raw": {
                "battleTime": "20260618T000000.000Z",
                "team": [player("#A", 1)],
                "opponent": [player("#B", 0, 100)],
            },
        },
        DATA_CONFIG,
    )
    assert record is not None
    assert record["segment"] == "ranked:league-6"


def test_ranked_segment_does_not_infer_league_from_trophies() -> None:
    team = {**player("#A", 1), "startingTrophies": 2400}
    opponent = {**player("#B", 0, 100), "startingTrophies": 2380}
    record = parse_battle_row(
        {
            "fingerprint": "ranked-trophies-no-league",
            "battle_time": datetime(2026, 6, 18, tzinfo=timezone.utc),
            "inserted_at": datetime(2026, 6, 18, tzinfo=timezone.utc),
            "mode_key": "ranked",
            "raw": {
                "battleTime": "20260618T000000.000Z",
                "team": [team],
                "opponent": [opponent],
            },
        },
        DATA_CONFIG,
    )
    assert record is not None
    assert record["segment"] == "ranked:unknown"


def test_ranked_league_number_falls_back_to_nested_search() -> None:
    assert (
        ranked_league_number(
            {
                "team": [
                    {
                        "profile": {
                            "currentPathOfLegendSeasonResult": {
                                "leagueNumber": 8,
                            }
                        }
                    }
                ]
            }
        )
        == 8
    )
