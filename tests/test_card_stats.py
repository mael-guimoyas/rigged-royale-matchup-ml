from rigged_matchup_ml.card_stats import (
    CHAMPION_CARD_IDS,
    CARD_METADATA_VECTOR_SIZE,
    elixir_for,
    metadata_for,
    metadata_vector_for,
)


def test_card_metadata_known_troop_tags() -> None:
    knight = metadata_for(26000000)
    assert knight["name"] == "Knight"
    assert knight["type"] == "troop"
    assert "mini_tank" in knight["tags"]
    assert knight["numeric"]["elixir"] > 0


def test_card_metadata_win_condition_from_snapshot() -> None:
    hog_rider = metadata_for(26000021)
    assert hog_rider["name"] == "Hog Rider"
    assert "win_condition" in hog_rider["tags"]
    assert "building_chaser" in hog_rider["tags"]


def test_card_metadata_spell_and_building_examples() -> None:
    fireball = metadata_for(28000000)
    assert fireball["type"] == "spell"
    assert {"spell", "splash", "air_target", "ground_target"} <= fireball["tags"]
    assert fireball["numeric"]["damage"] > 0

    cannon = metadata_for(27000000)
    assert cannon["type"] == "building"
    assert "building" in cannon["tags"]
    assert cannon["numeric"]["hitpoints"] > 0


def test_supplemental_metadata_covers_recent_vocab_cards() -> None:
    expected_names = {
        26000093: "Little Prince",
        26000095: "Goblin Demolisher",
        26000096: "Goblin Machine",
        26000097: "Suspicious Bush",
        26000099: "Goblinstein",
        26000101: "Rune Giant",
        26000102: "Berserker",
        26000103: "Boss Bandit",
        28000023: "Void",
        28000024: "Goblin Curse",
        28000025: "Spirit Empress",
        28000026: "Vines",
    }
    for card_id, name in expected_names.items():
        metadata = metadata_for(card_id)
        assert metadata["name"] == name
        assert metadata["type"] != "unknown"
        assert metadata["numeric"]["elixir"] > 0
        assert elixir_for(card_id) > 0


def test_supplemental_metadata_strategy_tags() -> None:
    assert {"building_chaser", "win_condition"} <= metadata_for(26000101)["tags"]
    assert {"spell", "splash", "air_target", "ground_target"} <= metadata_for(28000023)["tags"]
    assert "support" in metadata_for(26000093)["tags"]
    assert {26000093, 26000099, 26000103} <= CHAMPION_CARD_IDS


def test_unknown_and_padding_vectors_are_distinct() -> None:
    unknown = metadata_for(99999999)
    assert unknown["type"] == "unknown"

    unknown_vector = metadata_vector_for(99999999)
    padding_vector = metadata_vector_for(0)
    assert len(unknown_vector) == CARD_METADATA_VECTOR_SIZE
    assert len(padding_vector) == CARD_METADATA_VECTOR_SIZE
    assert sum(unknown_vector) > 0
    assert sum(padding_vector) == 0
