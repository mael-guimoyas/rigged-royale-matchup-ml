from rigged_matchup_ml.card_stats import (
    CARD_METADATA,
    CARD_METADATA_FLAGS,
    CARD_METADATA_ROLES,
    CARD_METADATA_VECTOR_SIZE,
    CHAMPION_CARD_IDS,
    elixir_for,
    metadata_for,
    metadata_vector_for,
)


def test_every_card_has_exactly_one_role() -> None:
    roles = set(CARD_METADATA_ROLES)
    for card_id, meta in CARD_METADATA.items():
        card_roles = meta["tags"] & roles
        assert len(card_roles) == 1, f"{card_id} has roles {card_roles}"
        assert meta["role"] in roles


def test_knight_is_mini_tank_not_win_condition() -> None:
    # The whole point of the taxonomy: support/tank pieces are not win conditions.
    knight = metadata_for(26000000)
    assert knight["name"] == "Knight"
    assert knight["role"] == "mini_tank"
    assert "win_condition" not in knight["tags"]
    assert knight["numeric"]["elixir"] > 0


def test_win_condition_examples() -> None:
    hog_rider = metadata_for(26000021)
    assert hog_rider["name"] == "Hog Rider"
    assert hog_rider["role"] == "win_condition"
    assert "building_target" in hog_rider["tags"]


def test_spell_and_building_roles() -> None:
    fireball = metadata_for(28000000)
    assert fireball["type"] == "spell"
    assert fireball["role"] == "damage_spell"
    assert {"splash", "air_target"} <= fireball["tags"]
    assert fireball["numeric"]["damage"] > 0

    cannon = metadata_for(27000000)
    assert cannon["type"] == "building"
    assert cannon["role"] == "defensive_building"
    assert cannon["numeric"]["dps"] > 0


def test_flag_combos_compose() -> None:
    # tank + high_dps (tank+dps); high_dps + splash (dps+splash).
    pekka = metadata_for(26000004)
    assert pekka["role"] == "tank"
    assert "high_dps" in pekka["tags"]

    mega_knight = metadata_for(26000055)
    assert mega_knight["role"] == "tank"
    assert {"high_dps", "splash"} <= mega_knight["tags"]


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
        assert metadata["role"] in CARD_METADATA_ROLES
        assert metadata["numeric"]["elixir"] > 0
        assert elixir_for(card_id) > 0


def test_champion_flag_matches_champion_ids() -> None:
    for card_id, meta in CARD_METADATA.items():
        if card_id in CHAMPION_CARD_IDS:
            assert "champion" in meta["tags"], card_id
        else:
            assert "champion" not in meta["tags"], card_id


def test_only_known_tags_present() -> None:
    allowed = set(CARD_METADATA_ROLES) | set(CARD_METADATA_FLAGS)
    for card_id, meta in CARD_METADATA.items():
        assert meta["tags"] <= allowed, f"{card_id}: {meta['tags'] - allowed}"


def test_unknown_and_padding_vectors_are_distinct() -> None:
    unknown = metadata_for(99999999)
    assert unknown["type"] == "unknown"
    assert unknown["role"] == ""

    unknown_vector = metadata_vector_for(99999999)
    padding_vector = metadata_vector_for(0)
    assert len(unknown_vector) == CARD_METADATA_VECTOR_SIZE
    assert len(padding_vector) == CARD_METADATA_VECTOR_SIZE
    assert sum(unknown_vector) > 0  # type:unknown one-hot is set
    assert sum(padding_vector) == 0
