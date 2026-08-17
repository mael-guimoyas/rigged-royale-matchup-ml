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
    build_encode_context,
    encode_row,
    encode_rows,
    encode_rows_both_directions,
    encode_rows_vectorised,
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


def test_legacy_form_flags_are_decoded_without_recollection() -> None:
    sample = row()
    # Existing Parquet: Hero lived in evolutionLevel bit 2 while the separate
    # hero/role columns remained zero/normal.
    sample["team_evolution_levels"][0] = 2
    sample["team_hero_levels"][0] = 0
    sample["team_card_roles"][0] = 1
    sample["team_evolution_levels"][1] = 3
    sample["team_hero_levels"][1] = 0
    sample["team_card_roles"][1] = 1

    encoded = encode_row(sample, VOCABULARY)
    assert encoded["team_evos"][:2].tolist() == [0, 1]
    assert encoded["team_heroes"][:2].tolist() == [1, 1]
    assert encoded["team_roles"][:2].tolist() == [3, 3]

    _assert_batch_matches_row(encode_rows([sample], VOCABULARY), encoded)

    context = _EncodeContext(VOCABULARY)
    decoded = _decode_batch(_record_batch([sample]), context)
    vectorised = _assemble_batch(decoded, np.zeros(1, dtype=bool), 0, 1)
    _assert_batch_matches_row(vectorised, encoded)


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


def test_vectorised_serving_encoder_matches_encode_rows() -> None:
    """The serving encoder takes dict rows, so it needs its own equivalence guard.

    ``_second_row`` covers the branches the fast path could get wrong: a short
    deck (padding), a card absent from the vocabulary, and a different tower.
    """
    rows = [row(), _second_row()]
    context = build_encode_context(VOCABULARY)
    for swap in (None, [False, True], [True, True]):
        produced = encode_rows_vectorised(rows, VOCABULARY, context, swapped=swap)
        expected = encode_rows(rows, VOCABULARY, swapped=swap)
        assert set(produced) == set(expected)
        for key, value in expected.items():
            assert torch.equal(produced[key], value), (key, swap)


def test_both_directions_matches_two_separate_encodes() -> None:
    """Sharing one decode between the two directions must change nothing."""
    rows = [row(), _second_row()]
    forward, reverse = encode_rows_both_directions(
        rows, VOCABULARY, build_encode_context(VOCABULARY)
    )
    expected_forward = encode_rows(rows, VOCABULARY)
    expected_reverse = encode_rows(rows, VOCABULARY, swapped=[True, True])
    for key, value in expected_forward.items():
        assert torch.equal(forward[key], value), key
    for key, value in expected_reverse.items():
        assert torch.equal(reverse[key], value), key
    # The mirror really is the mirror: the team side of one is the opponent side
    # of the other, which is what the antisymmetry guarantee rests on.
    assert torch.equal(forward["team_cards"], reverse["opponent_cards"])
    assert torch.equal(forward["opponent_cards"], reverse["team_cards"])


def test_vectorised_serving_encoder_tolerates_missing_optional_fields() -> None:
    """Serving rows carry no ``win``; ``matrix_prior`` defaults the same way."""
    sample = row()
    sample.pop("win", None)
    sample.pop("matrix_prior", None)
    produced = encode_rows_vectorised([sample], VOCABULARY)
    assert produced["target"].tolist() == [0.0]
    assert produced["matrix_prior"].tolist() == [0.5]


def test_vectorised_batch_matches_encode_rows_swapped() -> None:
    rows = [row(), _second_row()]
    swap = np.array([True, False])
    context = _EncodeContext(VOCABULARY)
    decoded = _decode_batch(_record_batch(rows), context)
    batch = _assemble_batch(decoded, swap, 0, len(rows))
    expected = encode_rows(rows, VOCABULARY, swapped=list(swap))
    for key, value in expected.items():
        assert torch.equal(batch[key], value), key
