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

from .config import AppConfig
from .dataset import load_vocabulary

_DECK_PAIRS = [(a, b) for a in range(8) for b in range(a + 1, 8)]


def _accumulate(cooccurrence: np.ndarray, deck_indices: np.ndarray) -> None:
    """Add this batch of decks (B, 8 vocab indices) to the co-occurrence counts."""
    for first, second in _DECK_PAIRS:
        left = deck_indices[:, first]
        right = deck_indices[:, second]
        np.add.at(cooccurrence, (left, right), 1.0)
        np.add.at(cooccurrence, (right, left), 1.0)


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

    dataset = pads.dataset(prepared_dir / "train", format="parquet")
    scanner = dataset.scanner(
        columns=["team_card_ids", "opponent_card_ids"], batch_size=65_536
    )
    scanned = 0
    for record_batch in scanner.to_batches():
        teams = record_batch.column("team_card_ids").to_pylist()
        opponents = record_batch.column("opponent_card_ids").to_pylist()
        for decks in (teams, opponents):
            _accumulate(cooccurrence, _decks_to_indices(decks, card_vocab))
        scanned += len(teams)
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
    counts: dict[str, int] = {}
    dataset = pads.dataset(prepared_dir / "train", format="parquet")
    scanner = dataset.scanner(
        columns=["team_card_ids", "opponent_card_ids"], batch_size=65_536
    )
    for record_batch in scanner.to_batches():
        for column in ("team_card_ids", "opponent_card_ids"):
            for deck in record_batch.column(column).to_pylist():
                for card in deck:
                    key = str(card)
                    counts[key] = counts.get(key, 0) + 1
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
