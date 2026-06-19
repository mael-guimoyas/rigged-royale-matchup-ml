from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import pyarrow.dataset as pads
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


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


def _encode_card_values(values: list[int], vocabulary: dict[str, int]) -> list[int]:
    encoded = [vocabulary.get(str(value), 0) for value in values[:8]]
    return encoded + [0] * (8 - len(encoded))


def _fixed_length(values: list[int], length: int = 8) -> list[int]:
    clipped = list(values[:length])
    return clipped + [0] * (length - len(clipped))


def encode_rows(
    rows: list[dict],
    vocabulary: dict[str, dict[str, int]],
    swapped: list[bool] | None = None,
) -> dict[str, torch.Tensor]:
    if swapped is None:
        swapped = [False] * len(rows)

    team_cards: list[list[int]] = []
    opponent_cards: list[list[int]] = []
    team_evos: list[list[int]] = []
    opponent_evos: list[list[int]] = []
    team_heroes: list[list[int]] = []
    opponent_heroes: list[list[int]] = []
    team_roles: list[list[int]] = []
    opponent_roles: list[list[int]] = []
    team_towers: list[int] = []
    opponent_towers: list[int] = []
    segments: list[int] = []
    patches: list[int] = []
    priors: list[float] = []
    targets: list[float] = []

    card_vocabulary = vocabulary["cards"]
    tower_vocabulary = vocabulary["towers"]
    segment_vocabulary = vocabulary["segments"]
    patch_vocabulary = vocabulary["patches"]

    for row, should_swap in zip(rows, swapped, strict=True):
        if should_swap:
            team_prefix, opponent_prefix = "opponent", "team"
            target = float(not bool(row["win"]))
            prior = 1.0 - float(row["matrix_prior"])
        else:
            team_prefix, opponent_prefix = "team", "opponent"
            target = float(bool(row["win"]))
            prior = float(row["matrix_prior"])

        team_cards.append(
            _encode_card_values(row[f"{team_prefix}_card_ids"], card_vocabulary)
        )
        opponent_cards.append(
            _encode_card_values(row[f"{opponent_prefix}_card_ids"], card_vocabulary)
        )
        team_evos.append(_fixed_length(row[f"{team_prefix}_evolution_levels"]))
        opponent_evos.append(_fixed_length(row[f"{opponent_prefix}_evolution_levels"]))
        team_heroes.append(_fixed_length(row[f"{team_prefix}_hero_levels"]))
        opponent_heroes.append(_fixed_length(row[f"{opponent_prefix}_hero_levels"]))
        team_roles.append(_fixed_length(row[f"{team_prefix}_card_roles"]))
        opponent_roles.append(_fixed_length(row[f"{opponent_prefix}_card_roles"]))
        team_towers.append(
            tower_vocabulary.get(str(row[f"{team_prefix}_tower_troop_id"]), 0)
        )
        opponent_towers.append(
            tower_vocabulary.get(str(row[f"{opponent_prefix}_tower_troop_id"]), 0)
        )
        segments.append(segment_vocabulary.get(str(row["segment"]), 0))
        patches.append(patch_vocabulary.get(str(row["patch"]), 0))
        priors.append(prior)
        targets.append(target)

    return {
        "team_cards": torch.tensor(team_cards, dtype=torch.long),
        "opponent_cards": torch.tensor(opponent_cards, dtype=torch.long),
        "team_evos": torch.tensor(team_evos, dtype=torch.long),
        "opponent_evos": torch.tensor(opponent_evos, dtype=torch.long),
        "team_heroes": torch.tensor(team_heroes, dtype=torch.long),
        "opponent_heroes": torch.tensor(opponent_heroes, dtype=torch.long),
        "team_roles": torch.tensor(team_roles, dtype=torch.long),
        "opponent_roles": torch.tensor(opponent_roles, dtype=torch.long),
        "team_tower": torch.tensor(team_towers, dtype=torch.long),
        "opponent_tower": torch.tensor(opponent_towers, dtype=torch.long),
        "segment": torch.tensor(segments, dtype=torch.long),
        "patch": torch.tensor(patches, dtype=torch.long),
        "matrix_prior": torch.tensor(priors, dtype=torch.float32),
        "target": torch.tensor(targets, dtype=torch.float32),
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


class BatchedMatchupIterableDataset(IterableDataset):
    def __init__(
        self,
        split_dir: Path,
        vocabulary: dict[str, dict[str, int]],
        shuffle: bool,
        augment_swap: bool,
        seed: int,
        batch_size: int,
        scan_batch_size: int = 65_536,
    ) -> None:
        super().__init__()
        self.split_dir = split_dir
        self.vocabulary = vocabulary
        self.shuffle = shuffle
        self.augment_swap = augment_swap
        self.seed = seed
        self.batch_size = batch_size
        self.scan_batch_size = max(scan_batch_size, batch_size)
        self._epoch = 0

    def _fragments(self) -> list:
        dataset = pads.dataset(self.split_dir, format="parquet")
        fragments = list(dataset.get_fragments())
        worker = get_worker_info()
        if worker is not None:
            fragments = fragments[worker.id :: worker.num_workers]
        return fragments

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        # Vary the seed per epoch so shuffling and swap augmentation differ across
        # epochs instead of repeating the exact same order every pass.
        epoch = self._epoch
        self._epoch += 1
        worker_seed = self.seed + (worker.id if worker else 0) + epoch * 100_003
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
                for offset in range(0, len(rows), self.batch_size):
                    batch_rows = rows[offset : offset + self.batch_size]
                    swapped = (
                        [rng.random() < 0.5 for _ in batch_rows]
                        if self.augment_swap
                        else None
                    )
                    yield encode_rows(batch_rows, self.vocabulary, swapped=swapped)


def matchup_dataloader(
    split_dir: Path,
    vocabulary: dict[str, dict[str, int]],
    shuffle: bool,
    augment_swap: bool,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = BatchedMatchupIterableDataset(
        split_dir,
        vocabulary,
        shuffle=shuffle,
        augment_swap=augment_swap,
        seed=seed,
        batch_size=batch_size,
    )
    loader_options = {
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_options)
