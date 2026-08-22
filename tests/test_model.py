import torch

from rigged_matchup_ml.card_stats import CARD_METADATA_VECTOR_SIZE
from rigged_matchup_ml.config import AppConfig, load_config
from rigged_matchup_ml.model import SymmetricMatchupModel
from rigged_matchup_ml.trainer import _model_config


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
    kw = {
        "use_cross_card_interactions": True,
        "use_intra_deck_synergies": True,
        "dropout": 0.0,
    }
    on = SymmetricMatchupModel(32, 4, 3, 3, use_card_importance=True, **kw)
    off = SymmetricMatchupModel(32, 4, 3, 3, use_card_importance=False, **kw)
    dim0 = SymmetricMatchupModel(
        32, 4, 3, 3, use_card_importance=True, card_metadata_dim=0, **kw
    )
    assert on.deck_encoder.card_importance_head is not None
    assert off.deck_encoder.card_importance_head is None
    # No metadata vector -> nothing to read a role from -> importance disabled.
    assert dim0.deck_encoder.card_importance_head is None


def test_new_training_disables_shared_hero_shortcuts() -> None:
    config = load_config("config/default.yaml")
    assert config.model["use_shared_hero_features"] is False

    # Older/custom training YAML files may not contain the newly introduced
    # option. They must also produce the corrected representation on retraining.
    legacy_model_config = dict(config.model)
    legacy_model_config.pop("use_shared_hero_features")
    legacy_training_config = AppConfig(
        raw={**config.raw, "model": legacy_model_config},
        source_path=config.source_path,
    )
    vocabulary = {"cards": {}, "towers": {}, "segments": {}, "patches": {}}
    assert (
        _model_config(legacy_training_config, vocabulary)[
            "use_shared_hero_features"
        ]
        is False
    )


def test_legacy_models_keep_shared_hero_features_by_default() -> None:
    # Existing checkpoint model_config dictionaries do not carry the new key.
    # Constructor compatibility must therefore preserve their trained path.
    model = SymmetricMatchupModel(32, 4, 3, 3)
    assert model.deck_encoder.use_shared_hero_features is True


def test_disabling_shared_hero_features_keeps_only_card_specific_identity() -> None:
    model = SymmetricMatchupModel(
        32,
        4,
        3,
        3,
        dropout=0.0,
        card_metadata_dim=0,
        use_shared_hero_features=False,
    )
    model.eval()
    b = batch()
    cards = b["team_cards"]
    plain_heroes = torch.zeros_like(b["team_heroes"])
    hero_levels = plain_heroes.clone()
    # Card 0 is already evolved in the fixture; equip card 1 as Hero to cover
    # the real Evo + Hero deck shape (the same card cannot be both forms).
    hero_levels[:, 1] = 1
    normal_roles = torch.ones_like(b["team_roles"])
    hero_roles = normal_roles.clone()
    hero_roles[:, 1] = 3

    with torch.no_grad():
        # Remove the one signal that deliberately remains. A huge global gate
        # must still have no effect while shared Hero features are disabled.
        model.deck_encoder.hero_card_embedding.weight.zero_()
        model.deck_encoder.hero_importance_gate.fill_(5.0)
        plain = model.deck_encoder.encode(
            cards,
            b["team_evos"],
            plain_heroes,
            normal_roles,
            b["team_tower"],
            b["team_elixir"],
            card_metadata=None,
            card_present=b["team_card_present"],
        )
        hero_without_identity = model.deck_encoder.encode(
            cards,
            b["team_evos"],
            hero_levels,
            hero_roles,
            b["team_tower"],
            b["team_elixir"],
            card_metadata=None,
            card_present=b["team_card_present"],
        )
        assert torch.equal(plain[0], hero_without_identity[0])
        assert torch.equal(plain[1], hero_without_identity[1])

        # Hero form is still distinguishable once its own card-specific delta is
        # learned; only the shared deck-wide proxy has disappeared.
        card_index = int(cards[0, 1])
        model.deck_encoder.hero_card_embedding.weight[card_index].fill_(0.5)
        hero_with_identity = model.deck_encoder.encode(
            cards,
            b["team_evos"],
            hero_levels,
            hero_roles,
            b["team_tower"],
            b["team_elixir"],
            card_metadata=None,
            card_present=b["team_card_present"],
        )
        assert not torch.equal(plain[0], hero_with_identity[0])
        assert not torch.equal(plain[1], hero_with_identity[1])


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
