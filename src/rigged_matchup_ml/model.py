from __future__ import annotations

import torch
from torch import nn


DECK_PAIR_INDICES = torch.triu_indices(8, 8, offset=1)


class DeckEncoder(nn.Module):
    def __init__(
        self,
        card_count: int,
        tower_count: int,
        embedding_dim: int,
        hidden_dim: int,
        max_evolution_level: int,
        max_hero_level: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_evolution_level = max_evolution_level
        self.max_hero_level = max_hero_level
        self.card_embedding = nn.Embedding(card_count, embedding_dim, padding_idx=0)
        self.evolution_embedding = nn.Embedding(max_evolution_level + 1, embedding_dim // 4)
        self.hero_embedding = nn.Embedding(max_hero_level + 1, embedding_dim // 4)
        self.role_embedding = nn.Embedding(4, embedding_dim // 4, padding_idx=0)
        self.tower_embedding = nn.Embedding(tower_count, embedding_dim // 2, padding_idx=0)
        card_input = embedding_dim + 3 * (embedding_dim // 4)
        self.card_projection = nn.Sequential(
            nn.Linear(card_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.deck_projection = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode(
        self,
        cards: torch.Tensor,
        evolutions: torch.Tensor,
        heroes: torch.Tensor,
        roles: torch.Tensor,
        tower: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = cards.ne(0).unsqueeze(-1)
        evolutions = evolutions.clamp(0, self.max_evolution_level)
        heroes = heroes.clamp(0, self.max_hero_level)
        roles = roles.clamp(0, 3)
        card_features = torch.cat(
            [
                self.card_embedding(cards),
                self.evolution_embedding(evolutions),
                self.hero_embedding(heroes),
                self.role_embedding(roles),
            ],
            dim=-1,
        )
        card_features = self.card_projection(card_features) * mask
        pooled = card_features.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        deck_features = self.deck_projection(
            torch.cat([pooled, self.tower_embedding(tower)], dim=-1)
        )
        return deck_features, card_features, mask.squeeze(-1)

    def forward(
        self,
        cards: torch.Tensor,
        evolutions: torch.Tensor,
        heroes: torch.Tensor,
        roles: torch.Tensor,
        tower: torch.Tensor,
    ) -> torch.Tensor:
        deck_features, _, _ = self.encode(cards, evolutions, heroes, roles, tower)
        return deck_features


class PairSummaryEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Linear(embedding_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, pair_features: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        pair_mask = pair_mask.bool()
        pair_values = pair_features * pair_mask.unsqueeze(-1)
        pair_count = pair_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pair_mean = pair_values.sum(dim=1) / pair_count
        pair_max = pair_features.masked_fill(~pair_mask.unsqueeze(-1), -1e4).max(dim=1).values
        pair_max = torch.where(pair_mask.any(dim=1, keepdim=True), pair_max, torch.zeros_like(pair_max))
        pair_scores = self.score(pair_features).squeeze(-1).masked_fill(~pair_mask, -1e4)
        pair_weights = torch.softmax(pair_scores, dim=1) * pair_mask
        pair_weights = pair_weights / pair_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pair_attention = (pair_values * pair_weights.unsqueeze(-1)).sum(dim=1)
        return self.projection(torch.cat([pair_mean, pair_max, pair_attention], dim=-1))


class CardInteractionEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        use_bilinear_cross: bool = True,
    ) -> None:
        super().__init__()
        self.cross_pairs = PairSummaryEncoder(embedding_dim, hidden_dim, dropout)
        self.deck_pairs = PairSummaryEncoder(embedding_dim, hidden_dim, dropout)
        # A pure Hadamard product only captures "same dimension aligns" and is
        # symmetric in the two cards. A learned bilinear map lets the model encode
        # an oriented "card A counters card B" relation: (A W) ⊙ B. Initialised to
        # identity so it starts as the original elementwise product and learns the
        # counter structure from win/loss only.
        self.use_bilinear_cross = use_bilinear_cross
        self.cross_bilinear = (
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            if use_bilinear_cross
            else None
        )
        if self.cross_bilinear is not None:
            nn.init.eye_(self.cross_bilinear.weight)
        self.register_buffer("pair_first_indices", DECK_PAIR_INDICES[0], persistent=False)
        self.register_buffer("pair_second_indices", DECK_PAIR_INDICES[1], persistent=False)

    def cross(
        self,
        first_cards: torch.Tensor,
        first_mask: torch.Tensor,
        second_cards: torch.Tensor,
        second_mask: torch.Tensor,
    ) -> torch.Tensor:
        first_projected = (
            self.cross_bilinear(first_cards)
            if self.cross_bilinear is not None
            else first_cards
        )
        pair_features = (
            first_projected[:, :, None, :] * second_cards[:, None, :, :]
        ).flatten(1, 2)
        pair_mask = (first_mask[:, :, None] & second_mask[:, None, :]).flatten(1, 2)
        return self.cross_pairs(pair_features, pair_mask)

    def within(self, cards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        first = cards[:, self.pair_first_indices, :]
        second = cards[:, self.pair_second_indices, :]
        pair_features = first * second
        pair_mask = mask[:, self.pair_first_indices] & mask[:, self.pair_second_indices]
        return self.deck_pairs(pair_features, pair_mask)


class SymmetricMatchupModel(nn.Module):
    """Antisymmetric logits guarantee P(A beats B) = 1 - P(B beats A)."""

    def __init__(
        self,
        card_count: int,
        tower_count: int,
        segment_count: int,
        patch_count: int,
        embedding_dim: int = 64,
        hidden_dim: int = 192,
        dropout: float = 0.15,
        max_evolution_level: int = 5,
        max_hero_level: int = 5,
        matrix_prior_strength: float = 1.0,
        use_cross_card_interactions: bool = False,
        use_intra_deck_synergies: bool = False,
        card_dropout: float = 0.0,
        use_matchup_transformer: bool = False,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        use_segment_adapters: bool = False,
        use_bilinear_cross: bool = True,
        matrix_prior_learnable: bool = False,
    ) -> None:
        super().__init__()
        # logit(prior) is antisymmetric (prior swaps to 1-prior), so a scalar
        # weight on it keeps the model antisymmetric. When learnable, the model
        # decides how much to trust the empirical matrix instead of a fixed value;
        # if matrix_prior is a constant 0.5 (no attached prior) its logit is 0 and
        # the weight simply gets no gradient.
        if matrix_prior_learnable:
            self.matrix_prior_strength = nn.Parameter(
                torch.tensor(float(matrix_prior_strength))
            )
        else:
            self.matrix_prior_strength = float(matrix_prior_strength)
        self.use_cross_card_interactions = use_cross_card_interactions
        self.use_intra_deck_synergies = use_intra_deck_synergies
        self.card_dropout = card_dropout
        self.use_matchup_transformer = use_matchup_transformer
        self.use_segment_adapters = use_segment_adapters
        self.deck_encoder = DeckEncoder(
            card_count,
            tower_count,
            embedding_dim,
            hidden_dim,
            max_evolution_level,
            max_hero_level,
            dropout,
        )
        self.card_interactions = (
            CardInteractionEncoder(
                embedding_dim, hidden_dim, dropout, use_bilinear_cross=use_bilinear_cross
            )
            if use_cross_card_interactions or use_intra_deck_synergies
            else None
        )
        context_dim = embedding_dim // 2
        self.segment_embedding = nn.Embedding(segment_count, context_dim, padding_idx=0)
        self.patch_embedding = nn.Embedding(patch_count, context_dim, padding_idx=0)
        self.matchup_side_embedding: nn.Embedding | None = None
        self.matchup_transformer: nn.TransformerEncoder | None = None
        if use_matchup_transformer:
            self.matchup_side_embedding = nn.Embedding(2, embedding_dim)
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.matchup_transformer = nn.TransformerEncoder(
                transformer_layer,
                num_layers=transformer_layers,
                enable_nested_tensor=False,
            )
        interaction_input = embedding_dim * 4 + context_dim * 2
        if use_intra_deck_synergies:
            interaction_input += embedding_dim * 4
        if use_cross_card_interactions:
            interaction_input += embedding_dim
        if use_matchup_transformer:
            interaction_input += embedding_dim
        self.segment_adapter_scale: nn.Embedding | None = None
        self.segment_adapter_bias: nn.Embedding | None = None
        if use_segment_adapters:
            self.segment_adapter_scale = nn.Embedding(
                segment_count, interaction_input, padding_idx=0
            )
            self.segment_adapter_bias = nn.Embedding(
                segment_count, interaction_input, padding_idx=0
            )
            nn.init.zeros_(self.segment_adapter_scale.weight)
            nn.init.zeros_(self.segment_adapter_bias.weight)
        self.orientation_network = nn.Sequential(
            nn.Linear(interaction_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _apply_card_dropout(
        self,
        cards: torch.Tensor,
        evolutions: torch.Tensor,
        heroes: torch.Tensor,
        roles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.card_dropout <= 0:
            return cards, evolutions, heroes, roles
        active = cards.ne(0)
        dropped = torch.rand(cards.shape, device=cards.device) < self.card_dropout
        keep = active & ~dropped
        empty_rows = active.any(dim=1) & ~keep.any(dim=1)
        if empty_rows.any():
            keep[empty_rows] = active[empty_rows]
        keep_or_padding = keep | ~active
        return (
            cards.masked_fill(~keep_or_padding, 0),
            evolutions.masked_fill(~keep_or_padding, 0),
            heroes.masked_fill(~keep_or_padding, 0),
            roles.masked_fill(~keep_or_padding, 0),
        )

    def _matchup_summary(
        self,
        first_cards: torch.Tensor,
        first_mask: torch.Tensor,
        second_cards: torch.Tensor,
        second_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.matchup_transformer is None or self.matchup_side_embedding is None:
            raise ValueError("Matchup transformer is enabled but not initialized")
        first_side = self.matchup_side_embedding(
            torch.zeros(first_cards.shape[:2], dtype=torch.long, device=first_cards.device)
        )
        second_side = self.matchup_side_embedding(
            torch.ones(second_cards.shape[:2], dtype=torch.long, device=second_cards.device)
        )
        tokens = torch.cat([first_cards + first_side, second_cards + second_side], dim=1)
        mask = torch.cat([first_mask, second_mask], dim=1)
        encoded = self.matchup_transformer(tokens, src_key_padding_mask=~mask)
        encoded = encoded * mask.unsqueeze(-1)
        return encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def _orientation_score(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        context: torch.Tensor,
        first_synergy: torch.Tensor | None = None,
        second_synergy: torch.Tensor | None = None,
        cross_interactions: torch.Tensor | None = None,
        matchup_summary: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [first, second, first - second, first * second]
        if self.use_intra_deck_synergies:
            if first_synergy is None or second_synergy is None:
                raise ValueError("Intra-deck synergies are enabled but missing")
            parts.extend(
                [
                    first_synergy,
                    second_synergy,
                    first_synergy - second_synergy,
                    first_synergy * second_synergy,
                ]
            )
        if self.use_cross_card_interactions:
            if cross_interactions is None:
                raise ValueError("Cross-card interactions are enabled but missing")
            parts.append(cross_interactions)
        if self.use_matchup_transformer:
            if matchup_summary is None:
                raise ValueError("Matchup transformer is enabled but missing")
            parts.append(matchup_summary)
        parts.append(context)
        features = torch.cat(parts, dim=-1)
        if self.use_segment_adapters:
            if (
                segment_ids is None
                or self.segment_adapter_scale is None
                or self.segment_adapter_bias is None
            ):
                raise ValueError("Segment adapters are enabled but missing")
            scale = torch.tanh(self.segment_adapter_scale(segment_ids))
            bias = self.segment_adapter_bias(segment_ids)
            features = features * (1.0 + 0.1 * scale) + 0.1 * bias
        return self.orientation_network(features).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        team_cards_input, team_evos, team_heroes, team_roles = self._apply_card_dropout(
            batch["team_cards"],
            batch["team_evos"],
            batch["team_heroes"],
            batch["team_roles"],
        )
        opponent_cards_input, opponent_evos, opponent_heroes, opponent_roles = (
            self._apply_card_dropout(
                batch["opponent_cards"],
                batch["opponent_evos"],
                batch["opponent_heroes"],
                batch["opponent_roles"],
            )
        )
        team, team_cards, team_mask = self.deck_encoder.encode(
            team_cards_input,
            team_evos,
            team_heroes,
            team_roles,
            batch["team_tower"],
        )
        opponent, opponent_cards, opponent_mask = self.deck_encoder.encode(
            opponent_cards_input,
            opponent_evos,
            opponent_heroes,
            opponent_roles,
            batch["opponent_tower"],
        )
        context = torch.cat(
            [self.segment_embedding(batch["segment"]), self.patch_embedding(batch["patch"])],
            dim=-1,
        )
        team_synergy = opponent_synergy = None
        if self.use_intra_deck_synergies:
            if self.card_interactions is None:
                raise ValueError("Intra-deck synergies are enabled but not initialized")
            team_synergy = self.card_interactions.within(team_cards, team_mask)
            opponent_synergy = self.card_interactions.within(opponent_cards, opponent_mask)
        team_to_opponent = opponent_to_team = None
        if self.use_cross_card_interactions:
            if self.card_interactions is None:
                raise ValueError("Cross-card interactions are enabled but not initialized")
            team_to_opponent = self.card_interactions.cross(
                team_cards, team_mask, opponent_cards, opponent_mask
            )
            opponent_to_team = self.card_interactions.cross(
                opponent_cards, opponent_mask, team_cards, team_mask
            )
        team_matchup_summary = opponent_matchup_summary = None
        if self.use_matchup_transformer:
            team_matchup_summary = self._matchup_summary(
                team_cards, team_mask, opponent_cards, opponent_mask
            )
            opponent_matchup_summary = self._matchup_summary(
                opponent_cards, opponent_mask, team_cards, team_mask
            )
        learned_logit = 0.5 * (
            self._orientation_score(
                team,
                opponent,
                context,
                team_synergy,
                opponent_synergy,
                team_to_opponent,
                team_matchup_summary,
                batch["segment"],
            )
            - self._orientation_score(
                opponent,
                team,
                context,
                opponent_synergy,
                team_synergy,
                opponent_to_team,
                opponent_matchup_summary,
                batch["segment"],
            )
        )
        prior = batch["matrix_prior"].clamp(1e-4, 1 - 1e-4)
        prior_logit = torch.logit(prior) * self.matrix_prior_strength
        return learned_logit + prior_logit

    @torch.no_grad()
    def probability(self, batch: dict[str, torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(self.forward(batch) / max(temperature, 1e-4))
