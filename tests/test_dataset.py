import torch

from rigged_matchup_ml.dataset import encode_row, encode_rows


VOCABULARY = {
    "cards": {str(26000000 + index): index + 1 for index in range(16)},
    "towers": {"159000000": 1, "159000001": 2},
    "segments": {"ladder:7000-8999": 1},
    "patches": {"2026-06": 1},
}


def row() -> dict:
    return {
        "team_card_ids": [26000000 + index for index in range(8)],
        "opponent_card_ids": [26000008 + index for index in range(8)],
        "team_evolution_levels": [1, 0, 0, 0, 0, 0, 0, 0],
        "opponent_evolution_levels": [0, 1, 0, 0, 0, 0, 0, 0],
        "team_hero_levels": [0, 0, 0, 0, 0, 0, 0, 0],
        "opponent_hero_levels": [1, 0, 0, 0, 0, 0, 0, 0],
        "team_card_roles": [1, 1, 1, 1, 1, 1, 1, 2],
        "opponent_card_roles": [1, 1, 1, 1, 1, 1, 1, 3],
        "team_tower_troop_id": 159000000,
        "opponent_tower_troop_id": 159000001,
        "segment": "ladder:7000-8999",
        "patch": "2026-06",
        "matrix_prior": 0.62,
        "win": True,
    }


def _assert_batch_matches_row(batch: dict, encoded: dict) -> None:
    for key, value in encoded.items():
        assert torch.equal(batch[key][0], value)


def test_encode_rows_matches_encode_row() -> None:
    sample = row()
    _assert_batch_matches_row(
        encode_rows([sample], VOCABULARY),
        encode_row(sample, VOCABULARY),
    )


def test_encode_rows_matches_swapped_encode_row() -> None:
    sample = row()
    _assert_batch_matches_row(
        encode_rows([sample], VOCABULARY, swapped=[True]),
        encode_row(sample, VOCABULARY, swapped=True),
    )
