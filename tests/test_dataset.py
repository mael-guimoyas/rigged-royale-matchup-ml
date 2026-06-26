import numpy as np
import pyarrow as pa
import torch

from rigged_matchup_ml.card_stats import (
    CARD_METADATA_VECTOR_SIZE,
    UNKNOWN_CARD_METADATA_VECTOR,
)
from rigged_matchup_ml.dataset import (
    _assemble_batch,
    _decode_batch,
    _EncodeContext,
    encode_row,
    encode_rows,
)


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
    encoded = encode_row(sample, VOCABULARY)
    batch = encode_rows([sample], VOCABULARY)
    _assert_batch_matches_row(batch, encoded)
    assert encoded["team_card_metadata"].shape == (8, CARD_METADATA_VECTOR_SIZE)
    assert encoded["team_card_present"].tolist() == [True] * 8


def test_encode_rows_matches_swapped_encode_row() -> None:
    sample = row()
    _assert_batch_matches_row(
        encode_rows([sample], VOCABULARY, swapped=[True]),
        encode_row(sample, VOCABULARY, swapped=True),
    )


def _record_batch(rows: list[dict]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {name: [r[name] for r in rows] for name in rows[0]}
    )


def _second_row() -> dict:
    sample = row()
    # A short deck (padding path), an unmapped card (vocab miss -> 0), a different
    # tower / loss, to exercise the branches encode_rows handles per row.
    sample["team_card_ids"] = [26000003, 26000004, 99999999]
    sample["opponent_card_ids"] = [26000010 + index for index in range(8)]
    sample["team_tower_troop_id"] = 159000001
    sample["matrix_prior"] = 0.18
    sample["win"] = False
    return sample


def test_vectorised_batch_matches_encode_rows() -> None:
    rows = [row(), _second_row()]
    context = _EncodeContext(VOCABULARY)
    decoded = _decode_batch(_record_batch(rows), context)
    batch = _assemble_batch(decoded, np.zeros(len(rows), dtype=bool), 0, len(rows))
    expected = encode_rows(rows, VOCABULARY)
    for key, value in expected.items():
        assert torch.equal(batch[key], value), key
    assert batch["team_card_metadata"].shape == (2, 8, CARD_METADATA_VECTOR_SIZE)
    assert batch["team_card_present"].dtype == torch.bool
    assert torch.equal(
        batch["team_card_metadata"][1, 2],
        torch.tensor(UNKNOWN_CARD_METADATA_VECTOR, dtype=torch.float32),
    )
    assert batch["team_card_present"][1, 2].item() is True


def test_vectorised_batch_matches_encode_rows_swapped() -> None:
    rows = [row(), _second_row()]
    swap = np.array([True, False])
    context = _EncodeContext(VOCABULARY)
    decoded = _decode_batch(_record_batch(rows), context)
    batch = _assemble_batch(decoded, swap, 0, len(rows))
    expected = encode_rows(rows, VOCABULARY, swapped=list(swap))
    for key, value in expected.items():
        assert torch.equal(batch[key], value), key
