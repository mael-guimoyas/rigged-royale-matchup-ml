from __future__ import annotations

import torch
from torch import nn


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

    def forward(
        self,
        cards: torch.Tensor,
        evolutions: torch.Tensor,
        heroes: torch.Tensor,
        roles: torch.Tensor,
        tower: torch.Tensor,
    ) -> torch.Tensor:
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
        return self.deck_projection(torch.cat([pooled, self.tower_embedding(tower)], dim=-1))


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
    ) -> None:
        super().__init__()
        self.matrix_prior_strength = matrix_prior_strength
        self.deck_encoder = DeckEncoder(
            card_count,
            tower_count,
            embedding_dim,
            hidden_dim,
            max_evolution_level,
            max_hero_level,
            dropout,
        )
        context_dim = embedding_dim // 2
        self.segment_embedding = nn.Embedding(segment_count, context_dim, padding_idx=0)
        self.patch_embedding = nn.Embedding(patch_count, context_dim, padding_idx=0)
        interaction_input = embedding_dim * 4 + context_dim * 2
        self.orientation_network = nn.Sequential(
            nn.Linear(interaction_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _orientation_score(
        self, first: torch.Tensor, second: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            [first, second, first - second, first * second, context], dim=-1
        )
        return self.orientation_network(features).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        team = self.deck_encoder(
            batch["team_cards"],
            batch["team_evos"],
            batch["team_heroes"],
            batch["team_roles"],
            batch["team_tower"],
        )
        opponent = self.deck_encoder(
            batch["opponent_cards"],
            batch["opponent_evos"],
            batch["opponent_heroes"],
            batch["opponent_roles"],
            batch["opponent_tower"],
        )
        context = torch.cat(
            [self.segment_embedding(batch["segment"]), self.patch_embedding(batch["patch"])],
            dim=-1,
        )
        learned_logit = 0.5 * (
            self._orientation_score(team, opponent, context)
            - self._orientation_score(opponent, team, context)
        )
        prior = batch["matrix_prior"].clamp(1e-4, 1 - 1e-4)
        prior_logit = torch.logit(prior) * self.matrix_prior_strength
        return learned_logit + prior_logit

    @torch.no_grad()
    def probability(self, batch: dict[str, torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(self.forward(batch) / max(temperature, 1e-4))
