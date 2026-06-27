import pytest
import torch

from rigged_matchup_ml.card_stats import CARD_METADATA_VECTOR_SIZE
from rigged_matchup_ml.model import SymmetricMatchupModel
from rigged_matchup_ml.predictor import _ablation_contributions


def batch() -> dict[str, torch.Tensor]:
    return {
        "team_cards": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "opponent_cards": torch.tensor([[9, 10, 11, 12, 13, 14, 15, 16]]),
        "team_elixir": torch.tensor([[3, 4, 5, 2, 6, 3, 4, 1]]),
        "opponent_elixir": torch.tensor([[2, 5, 3, 4, 7, 1, 3, 5]]),
        "team_card_metadata": torch.zeros((1, 8, CARD_METADATA_VECTOR_SIZE)),
        "opponent_card_metadata": torch.zeros((1, 8, CARD_METADATA_VECTOR_SIZE)),
        "team_card_present": torch.ones((1, 8), dtype=torch.bool),
        "opponent_card_present": torch.ones((1, 8), dtype=torch.bool),
        "team_evos": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]),
        "opponent_evos": torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]]),
        "team_heroes": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]),
        "opponent_heroes": torch.zeros((1, 8), dtype=torch.long),
        "team_roles": torch.ones((1, 8), dtype=torch.long),
        "opponent_roles": torch.ones((1, 8), dtype=torch.long),
        "team_tower": torch.tensor([1]),
        "opponent_tower": torch.tensor([2]),
        "segment": torch.tensor([1]),
        "patch": torch.tensor([1]),
        "matrix_prior": torch.tensor([0.5]),
    }


def test_probability_is_antisymmetric() -> None:
    model = SymmetricMatchupModel(
        32,
        4,
        3,
        3,
        dropout=0.0,
        use_cross_card_interactions=True,
        use_intra_deck_synergies=True,
        card_dropout=0.0,
        use_matchup_transformer=True,
        transformer_layers=1,
        transformer_heads=4,
        use_segment_adapters=True,
    )
    model.eval()
    original = batch()
    reverse = {
        **original,
        "team_cards": original["opponent_cards"],
        "opponent_cards": original["team_cards"],
        "team_elixir": original["opponent_elixir"],
        "opponent_elixir": original["team_elixir"],
        "team_card_metadata": original["opponent_card_metadata"],
        "opponent_card_metadata": original["team_card_metadata"],
        "team_card_present": original["opponent_card_present"],
        "opponent_card_present": original["team_card_present"],
        "team_evos": original["opponent_evos"],
        "opponent_evos": original["team_evos"],
        "team_heroes": original["opponent_heroes"],
        "opponent_heroes": original["team_heroes"],
        "team_roles": original["opponent_roles"],
        "opponent_roles": original["team_roles"],
        "team_tower": original["opponent_tower"],
        "opponent_tower": original["team_tower"],
    }
    with torch.no_grad():
        probability = model.probability(original).item()
        reverse_probability = model.probability(reverse).item()
    assert abs(probability + reverse_probability - 1.0) < 1e-6


def _reverse(original: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        **original,
        "team_cards": original["opponent_cards"],
        "opponent_cards": original["team_cards"],
        "team_elixir": original["opponent_elixir"],
        "opponent_elixir": original["team_elixir"],
        "team_card_metadata": original["opponent_card_metadata"],
        "opponent_card_metadata": original["team_card_metadata"],
        "team_card_present": original["opponent_card_present"],
        "opponent_card_present": original["team_card_present"],
        "team_evos": original["opponent_evos"],
        "opponent_evos": original["team_evos"],
        "team_heroes": original["opponent_heroes"],
        "opponent_heroes": original["team_heroes"],
        "team_roles": original["opponent_roles"],
        "opponent_roles": original["team_roles"],
        "team_tower": original["opponent_tower"],
        "opponent_tower": original["team_tower"],
    }


def test_card_importance_builds_only_with_metadata() -> None:
    kw = dict(use_cross_card_interactions=True, use_intra_deck_synergies=True, dropout=0.0)
    on = SymmetricMatchupModel(32, 4, 3, 3, use_card_importance=True, **kw)
    off = SymmetricMatchupModel(32, 4, 3, 3, use_card_importance=False, **kw)
    dim0 = SymmetricMatchupModel(
        32, 4, 3, 3, use_card_importance=True, card_metadata_dim=0, **kw
    )
    assert on.deck_encoder.card_importance_head is not None
    assert off.deck_encoder.card_importance_head is None
    # No metadata vector -> nothing to read a role from -> importance disabled.
    assert dim0.deck_encoder.card_importance_head is None


def test_card_importance_is_neutral_at_init_and_antisymmetric() -> None:
    model = SymmetricMatchupModel(
        32, 4, 3, 3,
        dropout=0.0,
        use_cross_card_interactions=True,
        use_intra_deck_synergies=True,
        use_card_importance=True,
    )
    model.eval()
    b = batch()
    weights = model.deck_encoder.card_importance(
        b["team_card_metadata"], b["team_card_present"]
    )
    # Neutral init: every present card weighs ~1.0 (uniform-mean baseline).
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-3)
    with torch.no_grad():
        p = model.probability(b).item()
        pr = model.probability(_reverse(b)).item()
    assert abs(p + pr - 1.0) < 1e-6


def test_multihead_cross_and_deck_transformer_stay_antisymmetric() -> None:
    model = SymmetricMatchupModel(
        32,
        4,
        3,
        3,
        dropout=0.0,
        use_cross_card_interactions=True,
        use_intra_deck_synergies=True,
        use_matchup_transformer=True,
        use_segment_adapters=True,
        cross_heads=4,
        use_deck_transformer=True,
        deck_transformer_heads=4,
        deck_transformer_layers=1,
    )
    model.eval()
    original = batch()
    with torch.no_grad():
        probability = model.probability(original).item()
        reverse_probability = model.probability(_reverse(original)).item()
    assert abs(probability + reverse_probability - 1.0) < 1e-6


def test_deck_transformer_archetype_changes_logits() -> None:
    # The archetype path must actually feed the score (not be a dead branch).
    plain = SymmetricMatchupModel(32, 4, 3, 3, dropout=0.0, use_deck_transformer=False)
    with_arch = SymmetricMatchupModel(32, 4, 3, 3, dropout=0.0, use_deck_transformer=True)
    assert with_arch.use_deck_transformer
    # extra archetype parts (4 * embedding_dim, default 64) widen the orientation input
    plain_in = plain.orientation_network[0].in_features
    arch_in = with_arch.orientation_network[0].in_features
    assert arch_in == plain_in + 64 * 4


def test_explain_pairs_survive_multihead_cross() -> None:
    model = SymmetricMatchupModel(
        32, 4, 3, 3, dropout=0.0, use_cross_card_interactions=True, cross_heads=4
    )
    model.eval()
    maps = model.explain(batch())
    assert maps["cross_team_to_opponent"].shape == (1, 8, 8)


def test_explain_uses_same_card_importance_as_forward() -> None:
    model = SymmetricMatchupModel(
        32,
        4,
        3,
        3,
        dropout=0.0,
        use_cross_card_interactions=True,
        use_card_importance=True,
    )
    model.eval()
    b = batch()
    b["team_card_metadata"][0, 0, 0] = 1.0
    assert model.deck_encoder.card_importance_head is not None
    with torch.no_grad():
        model.deck_encoder.card_importance_head.weight.zero_()
        model.deck_encoder.card_importance_head.weight[0, 0] = 3.0

    maps = model.explain(b)
    _, team_cards, team_mask, _ = model.deck_encoder.encode(
        b["team_cards"],
        b["team_evos"],
        b["team_heroes"],
        b["team_roles"],
        b["team_tower"],
        b["team_elixir"],
        b["team_card_metadata"],
        b["team_card_present"],
    )
    _, opponent_cards, opponent_mask, _ = model.deck_encoder.encode(
        b["opponent_cards"],
        b["opponent_evos"],
        b["opponent_heroes"],
        b["opponent_roles"],
        b["opponent_tower"],
        b["opponent_elixir"],
        b["opponent_card_metadata"],
        b["opponent_card_present"],
    )
    importance = model.deck_encoder.card_importance(
        b["team_card_metadata"],
        b["team_card_present"],
        b["team_evos"],
        b["team_heroes"],
    )
    assert model.card_interactions is not None
    _, expected = model.card_interactions.cross(
        team_cards,
        team_mask,
        opponent_cards,
        opponent_mask,
        return_weights=True,
        first_weights=importance,
    )
    _, unweighted = model.card_interactions.cross(
        team_cards,
        team_mask,
        opponent_cards,
        opponent_mask,
        return_weights=True,
    )

    assert torch.allclose(maps["cross_team_to_opponent"].flatten(1), expected)
    assert not torch.allclose(expected, unweighted)


def test_pair_keep_masks_remove_only_selected_attention_slots() -> None:
    model = SymmetricMatchupModel(
        32,
        4,
        3,
        3,
        dropout=0.0,
        use_cross_card_interactions=True,
        use_intra_deck_synergies=True,
    )
    model.eval()
    b = batch()
    _, team_cards, team_mask, _ = model.deck_encoder.encode(
        b["team_cards"],
        b["team_evos"],
        b["team_heroes"],
        b["team_roles"],
        b["team_tower"],
        b["team_elixir"],
        b["team_card_metadata"],
        b["team_card_present"],
    )
    _, opponent_cards, opponent_mask, _ = model.deck_encoder.encode(
        b["opponent_cards"],
        b["opponent_evos"],
        b["opponent_heroes"],
        b["opponent_roles"],
        b["opponent_tower"],
        b["opponent_elixir"],
        b["opponent_card_metadata"],
        b["opponent_card_present"],
    )
    assert model.card_interactions is not None
    keep = torch.ones((1, 64), dtype=torch.bool)
    keep[0, 7] = False
    _, weights = model.card_interactions.cross(
        team_cards,
        team_mask,
        opponent_cards,
        opponent_mask,
        return_weights=True,
        pair_keep_mask=keep,
    )

    assert weights[0, 7] == 0
    assert weights.sum().item() == pytest.approx(1.0)


def test_signed_ablation_matches_masked_forward_delta() -> None:
    model = SymmetricMatchupModel(
        32,
        4,
        3,
        3,
        dropout=0.0,
        use_cross_card_interactions=True,
    )
    model.eval()
    b = batch()
    with torch.no_grad():
        baseline = model(b).item()
    contributions = _ablation_contributions(
        model,
        b,
        baseline,
        "team_cross_pair_keep",
        64,
        [7],
    )
    keep = torch.ones((1, 64), dtype=torch.bool)
    keep[0, 7] = False
    with torch.no_grad():
        masked = model({**b, "team_cross_pair_keep": keep}).item()

    assert contributions[7] == pytest.approx(baseline - masked)


def test_probability_is_antisymmetric_with_learnable_prior() -> None:
    model = SymmetricMatchupModel(
        32,
        4,
        3,
        3,
        dropout=0.0,
        matrix_prior_strength=0.8,
        matrix_prior_learnable=True,
        use_cross_card_interactions=True,
        use_intra_deck_synergies=True,
        use_matchup_transformer=True,
        use_segment_adapters=True,
    )
    model.eval()
    original = {**batch(), "matrix_prior": torch.tensor([0.7])}
    reverse = {
        **original,
        "team_cards": original["opponent_cards"],
        "opponent_cards": original["team_cards"],
        "team_elixir": original["opponent_elixir"],
        "opponent_elixir": original["team_elixir"],
        "team_card_metadata": original["opponent_card_metadata"],
        "opponent_card_metadata": original["team_card_metadata"],
        "team_card_present": original["opponent_card_present"],
        "opponent_card_present": original["team_card_present"],
        "team_evos": original["opponent_evos"],
        "opponent_evos": original["team_evos"],
        "team_heroes": original["opponent_heroes"],
        "opponent_heroes": original["team_heroes"],
        "team_roles": original["opponent_roles"],
        "opponent_roles": original["team_roles"],
        "team_tower": original["opponent_tower"],
        "opponent_tower": original["team_tower"],
        "matrix_prior": torch.tensor([0.3]),
    }
    with torch.no_grad():
        probability = model.probability(original).item()
        reverse_probability = model.probability(reverse).item()
    assert abs(probability + reverse_probability - 1.0) < 1e-6
    assert isinstance(model.matrix_prior_strength, torch.nn.Parameter)
