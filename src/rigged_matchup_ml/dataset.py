from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .card_stats import (
    CARD_ELIXIR,
    CARD_METADATA,
    CARD_METADATA_VECTOR_SIZE,
    PADDING_CARD_METADATA_VECTOR,
    UNKNOWN_CARD_METADATA_VECTOR,
    elixir_for,
    metadata_vector_for,
)


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


def _fragments_by_row_group(split_dir: Path) -> list:
    fragments = []
    for fragment in pads.dataset(split_dir, format="parquet").get_fragments():
        split_by_row_group = getattr(fragment, "split_by_row_group", None)
        if split_by_row_group is None:
            fragments.append(fragment)
            continue
        row_group_fragments = list(split_by_row_group())
        fragments.extend(row_group_fragments or [fragment])
    return fragments


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

    def encode_elixir(values: list[int]) -> list[int]:
        # Derived from the raw card ids (not the vocab index) so it needs no
        # column in the Parquet shards; aligned position-wise with encode_cards.
        costs = [elixir_for(value) for value in values[:8]]
        return costs + [0] * (8 - len(costs))

    def encode_metadata(
        values: list[int], evos: list[int], heroes: list[int]
    ) -> list[tuple[float, ...]]:
        cards = values[:8]
        evo = list(evos[:8]) + [0] * (8 - min(len(evos), 8))
        hero = list(heroes[:8]) + [0] * (8 - min(len(heroes), 8))
        vectors = [
            metadata_vector_for(card, evolved=int(evo[i]) > 0, hero=int(hero[i]) > 0)
            for i, card in enumerate(cards)
        ]
        return vectors + [PADDING_CARD_METADATA_VECTOR] * (8 - len(vectors))

    def encode_present(values: list[int]) -> list[bool]:
        present = [int(value) > 0 for value in values[:8]]
        return present + [False] * (8 - len(present))

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
        "team_elixir": torch.tensor(
            encode_elixir(row[f"{team_prefix}_card_ids"]), dtype=torch.long
        ),
        "opponent_elixir": torch.tensor(
            encode_elixir(row[f"{opponent_prefix}_card_ids"]), dtype=torch.long
        ),
        "team_card_metadata": torch.tensor(
            encode_metadata(
                row[f"{team_prefix}_card_ids"],
                row[f"{team_prefix}_evolution_levels"],
                row[f"{team_prefix}_hero_levels"],
            ),
            dtype=torch.float32,
        ),
        "opponent_card_metadata": torch.tensor(
            encode_metadata(
                row[f"{opponent_prefix}_card_ids"],
                row[f"{opponent_prefix}_evolution_levels"],
                row[f"{opponent_prefix}_hero_levels"],
            ),
            dtype=torch.float32,
        ),
        "team_card_present": torch.tensor(
            encode_present(row[f"{team_prefix}_card_ids"]), dtype=torch.bool
        ),
        "opponent_card_present": torch.tensor(
            encode_present(row[f"{opponent_prefix}_card_ids"]), dtype=torch.bool
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
    team_elixir: list[list[int]] = []
    opponent_elixir: list[list[int]] = []
    team_card_metadata: list[list[tuple[float, ...]]] = []
    opponent_card_metadata: list[list[tuple[float, ...]]] = []
    team_card_present: list[list[bool]] = []
    opponent_card_present: list[list[bool]] = []
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
        team_elixir.append(
            _fixed_length([elixir_for(c) for c in row[f"{team_prefix}_card_ids"][:8]])
        )
        opponent_elixir.append(
            _fixed_length([elixir_for(c) for c in row[f"{opponent_prefix}_card_ids"][:8]])
        )
        team_raw_cards = list(row[f"{team_prefix}_card_ids"][:8])
        opponent_raw_cards = list(row[f"{opponent_prefix}_card_ids"][:8])
        team_evo_levels = _fixed_length(row[f"{team_prefix}_evolution_levels"])
        opponent_evo_levels = _fixed_length(row[f"{opponent_prefix}_evolution_levels"])
        team_hero_lv = _fixed_length(row[f"{team_prefix}_hero_levels"])
        opponent_hero_lv = _fixed_length(row[f"{opponent_prefix}_hero_levels"])
        team_card_metadata.append(
            [
                metadata_vector_for(
                    c, evolved=int(team_evo_levels[i]) > 0, hero=int(team_hero_lv[i]) > 0
                )
                for i, c in enumerate(team_raw_cards)
            ]
            + [PADDING_CARD_METADATA_VECTOR] * (8 - len(team_raw_cards))
        )
        opponent_card_metadata.append(
            [
                metadata_vector_for(
                    c,
                    evolved=int(opponent_evo_levels[i]) > 0,
                    hero=int(opponent_hero_lv[i]) > 0,
                )
                for i, c in enumerate(opponent_raw_cards)
            ]
            + [PADDING_CARD_METADATA_VECTOR] * (8 - len(opponent_raw_cards))
        )
        team_card_present.append(_fixed_length([int(c) > 0 for c in team_raw_cards]))
        opponent_card_present.append(
            _fixed_length([int(c) > 0 for c in opponent_raw_cards])
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
        "team_elixir": torch.tensor(team_elixir, dtype=torch.long),
        "opponent_elixir": torch.tensor(opponent_elixir, dtype=torch.long),
        "team_card_metadata": torch.tensor(team_card_metadata, dtype=torch.float32),
        "opponent_card_metadata": torch.tensor(
            opponent_card_metadata, dtype=torch.float32
        ),
        "team_card_present": torch.tensor(team_card_present, dtype=torch.bool),
        "opponent_card_present": torch.tensor(opponent_card_present, dtype=torch.bool),
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


# ---------------------------------------------------------------------------
# Vectorised batch encoding (training hot path).
#
# encode_row / encode_rows above build tensors with a Python loop per row: a
# str()+dict lookup per card, elixir_for per card, padding, a tensor per field.
# On a fast GPU that per-row Python work starves the device -- DataLoader workers
# can't refill batches quickly enough, so the GPU sits near 0%. The helpers below
# do the identical encoding with whole-column numpy ops over a pyarrow
# RecordBatch (no Python per row), so a couple of workers keep the GPU fed.
# Card ids are large and sparse, so a dense lookup table is impossible; we map
# them with a sorted-key np.searchsorted instead. Output is byte-for-byte
# identical to encode_rows (guarded by test_vectorised_batch_matches_encode_rows).
# ---------------------------------------------------------------------------

# Team/opponent fields that exchange places when a row is swap-augmented.
_PAIRED_FIELDS = (
    ("team_cards", "opponent_cards"),
    ("team_elixir", "opponent_elixir"),
    ("team_card_metadata", "opponent_card_metadata"),
    ("team_card_present", "opponent_card_present"),
    ("team_evos", "opponent_evos"),
    ("team_heroes", "opponent_heroes"),
    ("team_roles", "opponent_roles"),
    ("team_tower", "opponent_tower"),
)
_FLOAT_FIELDS = (
    "matrix_prior",
    "target",
    "team_card_metadata",
    "opponent_card_metadata",
)
_BOOL_FIELDS = ("team_card_present", "opponent_card_present")


def _int_lut(mapping: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sorted (keys, values) int arrays for an id->index map; keys may be int or str."""
    keys = np.fromiter((int(key) for key in mapping), dtype=np.int64, count=len(mapping))
    values = np.fromiter((mapping[key] for key in mapping), dtype=np.int64, count=len(mapping))
    order = np.argsort(keys, kind="stable")
    return keys[order], values[order]


def _str_lut(mapping: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted (keys, values) arrays for a str-keyed vocabulary (segment/patch/tower)."""
    items = sorted(mapping.items())
    keys = np.array([key for key, _ in items], dtype=str)
    values = np.array([value for _, value in items], dtype=np.int64)
    return keys, values


class _EncodeContext:
    """Sorted lookup tables for vectorised encoding; built once, reused per batch."""

    __slots__ = (
        "card_keys", "card_values", "elixir_keys", "elixir_values",
        "metadata_keys", "metadata_values", "unknown_metadata",
        "tower_keys", "tower_values", "segment_keys", "segment_values",
        "patch_keys", "patch_values",
    )

    def __init__(self, vocabulary: dict[str, dict[str, int]]) -> None:
        self.card_keys, self.card_values = _int_lut(vocabulary["cards"])
        self.elixir_keys, self.elixir_values = _int_lut(CARD_ELIXIR)
        self.metadata_keys, self.metadata_values = _metadata_lut(CARD_METADATA)
        self.unknown_metadata = np.asarray(
            UNKNOWN_CARD_METADATA_VECTOR, dtype=np.float32
        )
        self.tower_keys, self.tower_values = _str_lut(vocabulary["towers"])
        self.segment_keys, self.segment_values = _str_lut(vocabulary["segments"])
        self.patch_keys, self.patch_values = _str_lut(vocabulary["patches"])


def _lookup(values: np.ndarray, keys: np.ndarray, mapped: np.ndarray) -> np.ndarray:
    """Map ``values`` through a sorted (keys -> mapped) table; misses become 0."""
    if keys.size == 0:
        return np.zeros(values.shape, dtype=np.int64)
    flat = values.reshape(-1)
    position = np.searchsorted(keys, flat)
    np.clip(position, 0, keys.size - 1, out=position)
    hit = keys[position] == flat
    result = np.where(hit, mapped[position], 0)
    return result.reshape(values.shape).astype(np.int64, copy=False)


# Card metadata is form-dependent: the same card id has different ability / evo
# blocks as base / evolved / hero. We key the lookup table by a composite
# ``card_id * 4 + variant`` where variant = 2*(hero) + evolved, and expand every
# card into its four forms.
_METADATA_VARIANTS = ((False, False), (True, False), (False, True), (True, True))


def _metadata_lut(mapping: dict[int, dict]) -> tuple[np.ndarray, np.ndarray]:
    composite_keys: list[int] = []
    vectors: list[tuple[float, ...]] = []
    for key in mapping:
        cid = int(key)
        for variant, (evolved, hero) in enumerate(_METADATA_VARIANTS):
            composite_keys.append(cid * 4 + variant)
            vectors.append(metadata_vector_for(cid, evolved=evolved, hero=hero))
    keys = np.asarray(composite_keys, dtype=np.int64)
    mat = np.asarray(vectors, dtype=np.float32)
    if keys.size == 0:
        return keys, mat.reshape(0, CARD_METADATA_VECTOR_SIZE)
    order = np.argsort(keys, kind="stable")
    return keys[order], mat[order]


def _lookup_metadata(
    values: np.ndarray,
    evos: np.ndarray,
    heroes: np.ndarray,
    keys: np.ndarray,
    mapped: np.ndarray,
    unknown: np.ndarray,
) -> np.ndarray:
    out = np.zeros(values.shape + (CARD_METADATA_VECTOR_SIZE,), dtype=np.float32)
    flat = values.reshape(-1)
    out_flat = out.reshape(flat.shape[0], CARD_METADATA_VECTOR_SIZE)
    real_card = flat > 0
    # Unknown real cards get the form-independent unknown base (no ability/evo).
    out_flat[real_card] = unknown
    if keys.size == 0:
        return out
    variant = 2 * (heroes.reshape(-1) > 0).astype(np.int64) + (
        evos.reshape(-1) > 0
    ).astype(np.int64)
    composite = flat * 4 + variant
    position = np.searchsorted(keys, composite)
    np.clip(position, 0, keys.size - 1, out=position)
    hit = (keys[position] == composite) & real_card
    out_flat[hit] = mapped[position[hit]]
    return out


def _list_matrix(column, width: int = 8) -> np.ndarray:
    """A list<int> arrow column -> dense (n, width) int64, truncated/zero-padded."""
    count = len(column)
    out = np.zeros((count, width), dtype=np.int64)
    if count == 0:
        return out
    flat = column.flatten().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    lengths = pc.list_value_length(column).to_numpy(zero_copy_only=False).astype(np.int64)
    if bool((lengths == width).all()):
        return flat.reshape(count, width)
    starts = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(lengths, out=starts[1:])
    rows = np.repeat(np.arange(count), lengths)
    within = np.arange(flat.shape[0], dtype=np.int64) - np.repeat(starts[:-1], lengths)
    keep = within < width
    out[rows[keep], within[keep]] = flat[keep]
    return out


def _decode_batch(batch, context: _EncodeContext) -> dict[str, np.ndarray]:
    """Decode a whole pyarrow RecordBatch into named numpy arrays (no swap applied)."""
    column = batch.column
    team_raw = _list_matrix(column("team_card_ids"))
    opponent_raw = _list_matrix(column("opponent_card_ids"))
    team_evos = _list_matrix(column("team_evolution_levels"))
    opponent_evos = _list_matrix(column("opponent_evolution_levels"))
    team_heroes = _list_matrix(column("team_hero_levels"))
    opponent_heroes = _list_matrix(column("opponent_hero_levels"))
    tower_team = column("team_tower_troop_id").to_numpy(zero_copy_only=False).astype(str)
    tower_opponent = column("opponent_tower_troop_id").to_numpy(zero_copy_only=False).astype(str)
    return {
        "team_cards": _lookup(team_raw, context.card_keys, context.card_values),
        "opponent_cards": _lookup(opponent_raw, context.card_keys, context.card_values),
        "team_elixir": _lookup(team_raw, context.elixir_keys, context.elixir_values),
        "opponent_elixir": _lookup(opponent_raw, context.elixir_keys, context.elixir_values),
        "team_card_metadata": _lookup_metadata(
            team_raw,
            team_evos,
            team_heroes,
            context.metadata_keys,
            context.metadata_values,
            context.unknown_metadata,
        ),
        "opponent_card_metadata": _lookup_metadata(
            opponent_raw,
            opponent_evos,
            opponent_heroes,
            context.metadata_keys,
            context.metadata_values,
            context.unknown_metadata,
        ),
        "team_card_present": team_raw > 0,
        "opponent_card_present": opponent_raw > 0,
        "team_evos": team_evos,
        "opponent_evos": opponent_evos,
        "team_heroes": team_heroes,
        "opponent_heroes": opponent_heroes,
        "team_roles": _list_matrix(column("team_card_roles")),
        "opponent_roles": _list_matrix(column("opponent_card_roles")),
        "team_tower": _lookup(tower_team, context.tower_keys, context.tower_values),
        "opponent_tower": _lookup(tower_opponent, context.tower_keys, context.tower_values),
        "segment": _lookup(
            column("segment").to_numpy(zero_copy_only=False).astype(str),
            context.segment_keys,
            context.segment_values,
        ),
        "patch": _lookup(
            column("patch").to_numpy(zero_copy_only=False).astype(str),
            context.patch_keys,
            context.patch_values,
        ),
        "win": column("win").to_numpy(zero_copy_only=False).astype(np.float32),
        "matrix_prior": column("matrix_prior").to_numpy(zero_copy_only=False).astype(np.float32),
    }


def _assemble_batch(
    decoded: dict[str, np.ndarray], swap: np.ndarray, start: int, stop: int
) -> dict[str, torch.Tensor]:
    """Slice [start:stop], apply per-row swap, and convert to the model's tensor dict."""
    section = slice(start, stop)
    swapped = swap[section]
    out: dict[str, np.ndarray] = {}
    for team_key, opponent_key in _PAIRED_FIELDS:
        team = decoded[team_key][section]
        opponent = decoded[opponent_key][section]
        mask = swapped.reshape((-1,) + (1,) * (team.ndim - 1))
        out[team_key] = np.where(mask, opponent, team)
        out[opponent_key] = np.where(mask, team, opponent)
    out["segment"] = decoded["segment"][section]
    out["patch"] = decoded["patch"][section]
    win = decoded["win"][section]
    prior = decoded["matrix_prior"][section]
    out["target"] = np.where(swapped, 1.0 - win, win)
    out["matrix_prior"] = np.where(swapped, 1.0 - prior, prior)
    tensors: dict[str, torch.Tensor] = {}
    for key, value in out.items():
        if key in _FLOAT_FIELDS:
            dtype = np.float32
        elif key in _BOOL_FIELDS:
            dtype = np.bool_
        else:
            dtype = np.int64
        tensors[key] = torch.from_numpy(np.ascontiguousarray(value, dtype=dtype))
    return tensors


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
        fragments = _fragments_by_row_group(self.split_dir)
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
        self._context: _EncodeContext | None = None

    def _fragments(self) -> list:
        fragments = _fragments_by_row_group(self.split_dir)
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
        rng = np.random.default_rng(worker_seed)
        if self._context is None:
            self._context = _EncodeContext(self.vocabulary)
        fragments = self._fragments()
        if self.shuffle:
            fragments = [fragments[i] for i in rng.permutation(len(fragments))]
        for fragment in fragments:
            scanner = fragment.scanner(columns=FEATURE_COLUMNS, batch_size=self.scan_batch_size)
            for record_batch in scanner.to_batches():
                decoded = _decode_batch(record_batch, self._context)
                count = decoded["win"].shape[0]
                if count == 0:
                    continue
                if self.shuffle:
                    order = rng.permutation(count)
                    decoded = {key: value[order] for key, value in decoded.items()}
                swap = (
                    rng.random(count) < 0.5
                    if self.augment_swap
                    else np.zeros(count, dtype=bool)
                )
                for offset in range(0, count, self.batch_size):
                    yield _assemble_batch(decoded, swap, offset, offset + self.batch_size)


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
