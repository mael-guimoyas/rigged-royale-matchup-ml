import torch

from rigged_matchup_ml.model import SymmetricMatchupModel


def batch() -> dict[str, torch.Tensor]:
    return {
        "team_cards": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "opponent_cards": torch.tensor([[9, 10, 11, 12, 13, 14, 15, 16]]),
        "team_elixir": torch.tensor([[3, 4, 5, 2, 6, 3, 4, 1]]),
        "opponent_elixir": torch.tensor([[2, 5, 3, 4, 7, 1, 3, 5]]),
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
        "team_evos": original["opponent_evos"],
        "opponent_evos": original["team_evos"],
        "team_heroes": original["opponent_heroes"],
        "opponent_heroes": original["team_heroes"],
        "team_roles": original["opponent_roles"],
        "opponent_roles": original["team_roles"],
        "team_tower": original["opponent_tower"],
        "opponent_tower": original["team_tower"],
    }


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
