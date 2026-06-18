from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import pyarrow.dataset as pads
import torch
from torch.utils.data import IterableDataset, get_worker_info


FEATURE_COLUMNS = [
    "team_card_ids",
    "opponent_card_ids",
    "team_evolution_levels",
    "opponent_evolution_levels",
    "team_hero_levels",
    "opponent_hero_levels",
    "team_card_roles",
    "opponent_card_roles",
    "team_tower_troop_id",
    "opponent_tower_troop_id",
    "segment",
    "patch",
    "matrix_prior",
    "win",
]


def load_vocabulary(prepared_dir: Path) -> dict[str, dict[str, int]]:
    return json.loads((prepared_dir / "vocabulary.json").read_text(encoding="utf-8"))


def encode_row(
    row: dict,
    vocabulary: dict[str, dict[str, int]],
    swapped: bool = False,
) -> dict[str, torch.Tensor]:
    def encode_cards(values: list[int]) -> list[int]:
        vocab = vocabulary["cards"]
        encoded = [vocab.get(str(value), 0) for value in values[:8]]
        return encoded + [0] * (8 - len(encoded))

    if swapped:
        team_prefix, opponent_prefix = "opponent", "team"
        win = not bool(row["win"])
        prior = 1.0 - float(row["matrix_prior"])
    else:
        team_prefix, opponent_prefix = "team", "opponent"
        win = bool(row["win"])
        prior = float(row["matrix_prior"])
    return {
        "team_cards": torch.tensor(
            encode_cards(row[f"{team_prefix}_card_ids"]), dtype=torch.long
        ),
        "opponent_cards": torch.tensor(
            encode_cards(row[f"{opponent_prefix}_card_ids"]), dtype=torch.long
        ),
        "team_evos": torch.tensor(
            list(row[f"{team_prefix}_evolution_levels"][:8]), dtype=torch.long
        ),
        "opponent_evos": torch.tensor(
            list(row[f"{opponent_prefix}_evolution_levels"][:8]), dtype=torch.long
        ),
        "team_heroes": torch.tensor(
            list(row[f"{team_prefix}_hero_levels"][:8]), dtype=torch.long
        ),
        "opponent_heroes": torch.tensor(
            list(row[f"{opponent_prefix}_hero_levels"][:8]), dtype=torch.long
        ),
        "team_roles": torch.tensor(
            list(row[f"{team_prefix}_card_roles"][:8]), dtype=torch.long
        ),
        "opponent_roles": torch.tensor(
            list(row[f"{opponent_prefix}_card_roles"][:8]), dtype=torch.long
        ),
        "team_tower": torch.tensor(
            vocabulary["towers"].get(str(row[f"{team_prefix}_tower_troop_id"]), 0),
            dtype=torch.long,
        ),
        "opponent_tower": torch.tensor(
            vocabulary["towers"].get(str(row[f"{opponent_prefix}_tower_troop_id"]), 0),
            dtype=torch.long,
        ),
        "segment": torch.tensor(
            vocabulary["segments"].get(str(row["segment"]), 0), dtype=torch.long
        ),
        "patch": torch.tensor(
            vocabulary["patches"].get(str(row["patch"]), 0), dtype=torch.long
        ),
        "matrix_prior": torch.tensor(prior, dtype=torch.float32),
        "target": torch.tensor(float(win), dtype=torch.float32),
    }


class MatchupIterableDataset(IterableDataset):
    def __init__(
        self,
        split_dir: Path,
        vocabulary: dict[str, dict[str, int]],
        shuffle: bool,
        augment_swap: bool,
        seed: int,
        scan_batch_size: int = 65_536,
    ) -> None:
        super().__init__()
        self.split_dir = split_dir
        self.vocabulary = vocabulary
        self.shuffle = shuffle
        self.augment_swap = augment_swap
        self.seed = seed
        self.scan_batch_size = scan_batch_size

    def _fragments(self) -> list:
        dataset = pads.dataset(self.split_dir, format="parquet")
        fragments = list(dataset.get_fragments())
        worker = get_worker_info()
        if worker is not None:
            fragments = fragments[worker.id :: worker.num_workers]
        return fragments

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_seed = self.seed + (worker.id if worker else 0)
        rng = random.Random(worker_seed)
        fragments = self._fragments()
        if self.shuffle:
            rng.shuffle(fragments)
        for fragment in fragments:
            scanner = fragment.scanner(columns=FEATURE_COLUMNS, batch_size=self.scan_batch_size)
            for record_batch in scanner.to_batches():
                rows = record_batch.to_pylist()
                if self.shuffle:
                    rng.shuffle(rows)
                for row in rows:
                    swapped = self.augment_swap and rng.random() < 0.5
                    yield encode_row(row, self.vocabulary, swapped=swapped)
