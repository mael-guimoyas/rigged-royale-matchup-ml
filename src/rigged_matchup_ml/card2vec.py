"""Self-supervised card embeddings from deck co-occurrence (no labels, no stats).

Cards that appear in the same deck are functionally related (archetype / synergy
structure), exactly like words sharing a context window. We count card-card
co-occurrence over every training deck, build a PPMI matrix, and take a truncated
SVD -- the closed-form equivalent of word2vec (Levy & Goldberg 2014). The result
is a ``card_id -> vector`` table used to *initialise* the model's card embedding,
so rare cards start near functionally-similar common cards instead of at random.

Pure co-occurrence: no hardcoded card stats (hp/damage/type), no win/loss labels.
The only inputs are which cards share decks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
from tqdm import tqdm

from .config import AppConfig
from .dataset import load_vocabulary

_DECK_PAIRS = [(a, b) for a in range(8) for b in range(a + 1, 8)]


def _build_vocab_lookup(card_vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted (raw_card_id, vocab_index) arrays for vectorised id mapping.

    Raw Clash Royale card ids are sparse 8-digit numbers, so a dense lookup table
    would waste hundreds of MB. ``searchsorted`` over the sorted key array maps a
    whole batch of ids at once without any Python-level dict lookups.
    """
    keys = np.array(sorted(int(k) for k in card_vocab), dtype=np.int64)
    vals = np.array([card_vocab[str(int(k))] for k in keys], dtype=np.int64)
    return keys, vals


def _map_ids(flat_ids: np.ndarray, keys: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Map raw card ids -> vocab indices (0 for unknown), fully vectorised."""
    pos = np.searchsorted(keys, flat_ids)
    pos = np.clip(pos, 0, keys.size - 1)
    matched = keys[pos] == flat_ids
    return np.where(matched, vals[pos], 0)


def _batch_card_indices(
    record_batch, column: str, card_vocab: dict[str, int], keys: np.ndarray, vals: np.ndarray
) -> np.ndarray:
    """Return (B, 8) vocab indices for a column, vectorised when decks are size 8.

    Decks are guaranteed 8 cards when ``require_exactly_eight_cards`` is set, so the
    Arrow list column flattens to a clean ``B*8`` child array we reshape in one shot.
    Falls back to the per-row path for ragged batches.
    """
    col = record_batch.column(column)
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    if flat.size == len(col) * 8:
        return _map_ids(flat, keys, vals).reshape(len(col), 8)
    return _decks_to_indices(col.to_pylist(), card_vocab)


def _accumulate(cooccurrence: np.ndarray, deck_indices: np.ndarray) -> None:
    """Add a batch of decks (B, 8 vocab indices) to the co-occurrence counts.

    Builds a binary multi-hot matrix ``M`` (B, n_cards) and adds ``M.T @ M``: entry
    (i, j) becomes the number of decks containing both card i and card j. The matmul
    runs in BLAS (multi-threaded, all cores) instead of millions of ``np.add.at``
    scatter ops. The diagonal (self pairs) is dropped to match pairwise semantics.
    """
    size = cooccurrence.shape[0]
    multi_hot = np.zeros((deck_indices.shape[0], size), dtype=np.float32)
    rows = np.arange(deck_indices.shape[0])[:, None]
    multi_hot[rows, deck_indices] = 1.0
    batch_cooc = multi_hot.T @ multi_hot
    np.fill_diagonal(batch_cooc, 0.0)
    cooccurrence += batch_cooc


def _decks_to_indices(decks: list[list[int]], card_vocab: dict[str, int]) -> np.ndarray:
    rows = [
        [card_vocab.get(str(card), 0) for card in deck[:8]] + [0] * (8 - len(deck[:8]))
        for deck in decks
    ]
    return np.asarray(rows, dtype=np.int64)


def _ppmi_svd(cooccurrence: np.ndarray, dim: int) -> np.ndarray:
    """PPMI factorisation -> (n, dim) dense card vectors. Row 0 (padding) stays 0."""
    cooccurrence[0, :] = 0.0
    cooccurrence[:, 0] = 0.0
    total = cooccurrence.sum()
    if total <= 0:
        return np.zeros((cooccurrence.shape[0], dim), dtype=np.float32)
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    col_sums = cooccurrence.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((cooccurrence * total) / (row_sums * col_sums))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)
    # n_cards is small (hundreds), so a full dense SVD is cheap and exact.
    u_matrix, singular_values, _ = np.linalg.svd(ppmi)
    keep = min(dim, singular_values.shape[0])
    vectors = u_matrix[:, :keep] * np.sqrt(singular_values[:keep])
    out = np.zeros((cooccurrence.shape[0], dim), dtype=np.float32)
    out[:, :keep] = vectors.astype(np.float32)
    # Standardise to a small init scale so it slots in as an embedding init without
    # dwarfing the other learned signals, then re-zero the padding row.
    spread = out[1:].std()
    if spread > 0:
        out = out / spread * 0.1
    out[0] = 0.0
    return out


def pretrain_card_embeddings(
    config: AppConfig,
    dim: int | None = None,
    max_rows: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build and save self-supervised card vectors aligned to the card vocabulary.

    Reads the prepared train decks + vocabulary, writes ``card2vec.npy`` (shape
    ``(card_count, embedding_dim)``, row index = card vocab index) and
    ``card2vec.json`` (metadata). Defaults to the prepared dir; pass ``output_dir``
    to write elsewhere (e.g. a writable scratch dir when the prepared dataset is
    read-only, as on Kaggle).
    """
    prepared_dir = config.resolve(config.data["prepared_dir"])
    destination = output_dir or prepared_dir
    destination.mkdir(parents=True, exist_ok=True)
    card_vocab = load_vocabulary(prepared_dir)["cards"]
    dim = int(dim or config.model["embedding_dim"])
    size = len(card_vocab) + 1
    cooccurrence = np.zeros((size, size), dtype=np.float64)

    keys, vals = _build_vocab_lookup(card_vocab)
    dataset = pads.dataset(prepared_dir / "train", format="parquet")
    scanner = dataset.scanner(
        columns=["team_card_ids", "opponent_card_ids"], batch_size=65_536
    )
    scanned = 0
    for record_batch in tqdm(
        scanner.to_batches(), desc="card2vec train scan", unit="batch"
    ):
        for column in ("team_card_ids", "opponent_card_ids"):
            indices = _batch_card_indices(record_batch, column, card_vocab, keys, vals)
            _accumulate(cooccurrence, indices)
        scanned += record_batch.num_rows
        if max_rows is not None and scanned >= max_rows:
            break

    vectors = _ppmi_svd(cooccurrence, dim)
    np.save(destination / "card2vec.npy", vectors)
    meta = {
        "dim": dim,
        "card_count": size,
        "rows_scanned": scanned,
        "method": "ppmi_svd_deck_cooccurrence",
    }
    (destination / "card2vec.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def write_card_frequencies(config: AppConfig, output_dir: Path | None = None) -> dict[str, Any]:
    """Count per-card train frequency and write ``card_frequencies.json``.

    ``prepare`` already emits this; this standalone path lets a read-only prepared
    dataset (Kaggle) regenerate it into a writable dir for the loss weighting.
    """
    prepared_dir = config.resolve(config.data["prepared_dir"])
    destination = output_dir or prepared_dir
    destination.mkdir(parents=True, exist_ok=True)
    card_vocab = load_vocabulary(prepared_dir)["cards"]
    keys, vals = _build_vocab_lookup(card_vocab)
    totals = np.zeros(len(card_vocab) + 1, dtype=np.int64)
    dataset = pads.dataset(prepared_dir / "train", format="parquet")
    scanner = dataset.scanner(
        columns=["team_card_ids", "opponent_card_ids"], batch_size=65_536
    )
    for record_batch in tqdm(
        scanner.to_batches(), desc="card frequencies train scan", unit="batch"
    ):
        for column in ("team_card_ids", "opponent_card_ids"):
            col = record_batch.column(column)
            flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            idx = _map_ids(flat, keys, vals)
            totals += np.bincount(idx, minlength=totals.size)
    # Re-key by raw card id (index 0 is padding/unknown and is dropped).
    counts = {str(int(raw)): int(totals[idx]) for raw, idx in zip(keys, vals) if totals[idx] > 0}
    (destination / "card_frequencies.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"cards": len(counts), "output_dir": str(destination)}


def load_card2vec(prepared_dir: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
    """Return saved card vectors if present and shape-compatible, else None."""
    path = prepared_dir / "card2vec.npy"
    if not path.exists():
        return None
    vectors = np.load(path)
    if tuple(vectors.shape) != tuple(expected_shape):
        return None
    return vectors
