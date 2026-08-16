"""Bagging has to resample the corpus, and resample it the same way every epoch.

Seed-only ensemble members disagreed by 0.0132 on emitted decks against a model
error of 0.127, so the training process explains almost none of that error. The
next question is whether the *data* does, and that needs members trained on
different draws of the corpus rather than different initialisations.

Two properties make or break the experiment, and neither is visible from a
training log, so they are pinned here:

* a bagged member must see a genuinely different corpus from an unbagged one,
  and from a member drawn under another seed;
* it must see the *same* corpus on every epoch. Redrawing per epoch would show
  the member all the data eventually, turning bagging into noise and making the
  comparison against a seed-only member measure nothing.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rigged_matchup_ml.dataset import BatchedMatchupIterableDataset

VOCABULARY = {
    "cards": {str(26000000 + index): index + 1 for index in range(16)},
    "towers": {"159000000": 1},
    "segments": {"ladder:7000-8999": 1},
    "patches": {"2026-06": 1},
}


def write_split(directory: Path, row_groups: int) -> None:
    """A split whose row groups are individually identifiable."""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(row_groups):
        table = pa.table({"marker": pa.array([index], type=pa.int64())})
        pq.write_table(table, directory / f"part-{index:03d}.parquet")


def markers(dataset: BatchedMatchupIterableDataset) -> list[int]:
    """Which row groups a pass would read, in order."""
    return [
        fragment.to_table(columns=["marker"]).column("marker")[0].as_py()
        for fragment in dataset._fragments()
    ]


def dataset_for(directory: Path, bootstrap_seed: int | None) -> BatchedMatchupIterableDataset:
    return BatchedMatchupIterableDataset(
        directory,
        VOCABULARY,
        shuffle=True,
        augment_swap=True,
        seed=42,
        batch_size=8,
        bootstrap_seed=bootstrap_seed,
    )


def test_without_a_bootstrap_seed_every_row_group_is_read_once(tmp_path: Path) -> None:
    write_split(tmp_path / "train", 24)
    assert sorted(markers(dataset_for(tmp_path / "train", None))) == list(range(24))


def test_a_bootstrap_seed_draws_with_replacement(tmp_path: Path) -> None:
    write_split(tmp_path / "train", 24)
    drawn = markers(dataset_for(tmp_path / "train", 101))
    assert len(drawn) == 24
    # With replacement, so some row groups repeat and others are left out. A
    # draw that happened to be a permutation would silently be no bagging at all.
    assert len(set(drawn)) < 24
    assert set(drawn) != set(range(24))


def test_two_bootstrap_seeds_train_on_different_corpora(tmp_path: Path) -> None:
    write_split(tmp_path / "train", 24)
    assert markers(dataset_for(tmp_path / "train", 101)) != markers(
        dataset_for(tmp_path / "train", 202)
    )


def test_the_draw_is_stable_across_epochs(tmp_path: Path) -> None:
    write_split(tmp_path / "train", 24)
    dataset = dataset_for(tmp_path / "train", 101)
    first = markers(dataset)
    dataset._epoch += 1
    assert markers(dataset) == first


def test_the_draw_is_reproducible_between_runs(tmp_path: Path) -> None:
    write_split(tmp_path / "train", 24)
    assert markers(dataset_for(tmp_path / "train", 101)) == markers(
        dataset_for(tmp_path / "train", 101)
    )
