"""Is this new deck an interpolation, or an extrapolation?

A deck nobody has played is not automatically a bad recommendation — inventing
one is the whole point of the optimizer. But there are two kinds of new. A deck
whose every three-card and four-card group already appears in decks people play
is new only in its full combination: the model has seen every local interaction
it contains, and its prediction rests on something. A deck containing groups
nobody has ever assembled is asking the model to extrapolate, and off there its
error is 2.7x more dispersed with no bias to warn anyone.

Pairs cannot tell the two apart: the decks the search emits have zero unseen
pairs, so a pair-level test passes them all. This measures the higher orders,
where a combination has room to be genuinely unprecedented.

The counts are battle-weighted, so a group attested only by decks somebody
played twice does not count as established. The threshold is a parameter,
because what "attested" should mean is a product decision, not a fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decks",
        type=Path,
        default=root / "artifacts" / "deck-plays-all.parquet",
        help="Every distinct deck with its battle count, from deck_plausibility.py.",
    )
    parser.add_argument(
        "--envelopes",
        type=Path,
        nargs="*",
        default=None,
        help="plausibility-envelope JSON files whose trajectories are placed.",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "deck-manifold.json"
    )
    parser.add_argument(
        "--attested-battles",
        type=int,
        nargs="+",
        default=(1, 30, 300),
        help="Battle floors a card group needs to count as attested.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=(3, 4),
        help="Card-group sizes measured.",
    )
    parser.add_argument(
        "--max-dense-cells",
        type=int,
        default=50_000_000,
        help="Above this table size the sparse attested-set path is used instead.",
    )
    parser.add_argument("--reference-min-plays", type=int, default=30)
    parser.add_argument(
        "--reference-max-plays",
        type=int,
        default=None,
        help=(
            "Upper bound on the reference population's battles. Set it below "
            "the attestation floor so a reference deck cannot attest its own "
            "groups, which would make its unattested count trivially zero."
        ),
    )
    parser.add_argument("--reference-sample", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def load_decks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    connection = duckdb.connect()
    cached = path.resolve().as_posix().replace("'", "''")
    columns = connection.execute(f"select * from read_parquet('{cached}')").fetchnumpy()
    connection.close()
    decks = np.column_stack(
        [np.asarray(columns[f"card{slot}"], dtype=np.int64) for slot in range(1, 9)]
    )
    plays = np.asarray(columns["plays"], dtype=np.float64)
    return decks, plays


class GroupCounts:
    """Battle-weighted counts of every card group of a given size.

    Each group is addressed by mixed-radix packing of its sorted card indexes,
    so the whole table is one dense array and lookup is arithmetic. With 122
    cards a three-card table holds 1.8 million cells and a four-card table 221
    million, both of which fit far more comfortably than a Python set holding
    hundreds of millions of tuples.
    """

    def __init__(self, indexed: np.ndarray, weights: np.ndarray, cards: int, size: int) -> None:
        self.cards = cards
        self.size = size
        self.slots = list(combinations(range(8), size))
        self.counts = np.zeros(cards**size, dtype=np.float64)
        for slots in self.slots:
            self.counts += np.bincount(
                self.keys(indexed, slots), weights=weights, minlength=cards**size
            )

    def keys(self, indexed: np.ndarray, slots: tuple[int, ...]) -> np.ndarray:
        picked = np.sort(indexed[:, list(slots)], axis=1)
        key = np.zeros(indexed.shape[0], dtype=np.int64)
        for position in range(self.size):
            key = key * self.cards + picked[:, position]
        return key

    def deck_counts(self, indexed: np.ndarray) -> np.ndarray:
        """One row per deck, one column per group, holding that group's weight."""
        output = np.empty((indexed.shape[0], len(self.slots)), dtype=np.float64)
        for position, slots in enumerate(self.slots):
            output[:, position] = self.counts[self.keys(indexed, slots)]
        return output


class AttestedGroups:
    """Sparse alternative to `GroupCounts` when the dense table cannot fit.

    Four-card groups over 122 cards address 221 million cells, and `np.bincount`
    would allocate that table once per group position. Only the groups that
    actually occur matter, so this stores the sorted distinct keys of decks that
    clear a battle floor and answers membership by binary search. The trade is
    that it knows whether a group is attested, not how heavily.
    """

    def __init__(
        self, indexed: np.ndarray, weights: np.ndarray, cards: int, size: int, floor: int
    ) -> None:
        self.cards = cards
        self.size = size
        self.floor = floor
        self.slots = list(combinations(range(8), size))
        rows = indexed[weights >= floor]
        chunks = [self.keys(rows, slots) for slots in self.slots]
        self.attested = np.unique(np.concatenate(chunks)) if chunks else np.empty(0, np.int64)

    def keys(self, indexed: np.ndarray, slots: tuple[int, ...]) -> np.ndarray:
        picked = np.sort(indexed[:, list(slots)], axis=1)
        key = np.zeros(indexed.shape[0], dtype=np.int64)
        for position in range(self.size):
            key = key * self.cards + picked[:, position]
        return key

    def unattested_counts(self, indexed: np.ndarray) -> np.ndarray:
        """Groups per deck with no attested occurrence at this floor."""
        missing = np.zeros(indexed.shape[0], dtype=np.int64)
        for slots in self.slots:
            key = self.keys(indexed, slots)
            position = np.searchsorted(self.attested, key)
            position = np.clip(position, 0, max(self.attested.size - 1, 0))
            found = self.attested.size > 0
            missing += (~(found & (self.attested[position] == key))).astype(np.int64)
        return missing


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability a random `positive` scores above a random `negative`.

    Ties split evenly, which matters here: unattested-group counts are small
    integers and a detector that assigns both populations the same value should
    read as 0.5, not as a win.
    """
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positive, negative]), kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    values = np.concatenate([positive, negative])
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return float(
        (ranks[: positive.size].sum() - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def summarise(values: np.ndarray, floors: tuple[int, ...]) -> dict[str, Any]:
    return {
        "groups": int(values.shape[1]),
        "unattested": {
            str(floor): {
                "mean": float((values < floor).sum(axis=1).mean()),
                "median": float(np.median((values < floor).sum(axis=1))),
                "share_with_any": float(((values < floor).sum(axis=1) > 0).mean()),
            }
            for floor in floors
        },
        "weakest_group_battles": {
            "p05": float(np.percentile(values.min(axis=1), 5)),
            "p50": float(np.percentile(values.min(axis=1), 50)),
            "p95": float(np.percentile(values.min(axis=1), 95)),
        },
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    decks, plays = load_decks(args.decks)
    cards = np.unique(decks)
    lookup = np.full(int(cards.max()) + 1, -1, dtype=np.int64)
    lookup[cards] = np.arange(len(cards))
    indexed = lookup[decks]
    log(f"{len(decks):,} decks, {len(cards)} cards")

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decks": int(len(decks)),
        "cards": int(len(cards)),
        "attested_battles": list(args.attested_battles),
        "sizes": {},
    }

    reference_mask = plays >= args.reference_min_plays
    if args.reference_max_plays is not None:
        reference_mask &= plays <= args.reference_max_plays
    reference_rows = np.flatnonzero(reference_mask)
    if reference_rows.size > args.reference_sample:
        reference_rows = rng.choice(reference_rows, args.reference_sample, replace=False)

    trajectories: list[dict[str, Any]] = []
    for path in args.envelopes or []:
        if not Path(path).exists():
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for segment, entry in payload.get("segments", {}).items():
            for search in entry.get("searches") or []:
                for step in search["trajectory"]:
                    trajectories.append(
                        {
                            "source": Path(path).name,
                            "segment": segment,
                            "round": step["round"],
                            "model_score": step["profile"]["score"],
                            "cards": step["cards"],
                        }
                    )
    log(f"{len(trajectories)} search steps to place")

    for size in args.sizes:
        if len(cards) ** size > args.max_dense_cells:
            log(f"{size}-card groups: dense table too large, using attested sets")
            entry = {"mode": "attested-sets", "reference_decks": int(reference_rows.size)}
            walk = (
                lookup[np.asarray([step["cards"] for step in trajectories], dtype=np.int64)]
                if trajectories
                else None
            )
            for floor in args.attested_battles:
                if floor < args.reference_min_plays:
                    continue
                groups = AttestedGroups(indexed, plays, len(cards), size, floor)
                reference_missing = groups.unattested_counts(indexed[reference_rows])
                record: dict[str, Any] = {
                    "attested_groups": int(groups.attested.size),
                    "reference": {
                        "mean": float(reference_missing.mean()),
                        "share_with_any": float((reference_missing > 0).mean()),
                    },
                }
                if walk is not None:
                    missing = groups.unattested_counts(walk)
                    by_round: dict[int, list[int]] = {}
                    for position, step in enumerate(trajectories):
                        by_round.setdefault(step["round"], []).append(position)
                    record["by_round"] = [
                        {
                            "round": key,
                            "decks": len(rows),
                            "mean": float(missing[rows].mean()),
                            "share_with_any": float((missing[rows] > 0).mean()),
                            # How separable this round is from the reference
                            # population: the chance a random emitted deck has
                            # more unattested groups than a random real one.
                            "auc_vs_reference": float(
                                auc(missing[rows], reference_missing)
                            ),
                        }
                        for key, rows in sorted(by_round.items())
                    ]
                entry[str(floor)] = record
                del groups
            report["sizes"][str(size)] = entry
            continue

        log(f"counting {size}-card groups")
        table = GroupCounts(indexed, plays, len(cards), size)
        entry: dict[str, Any] = {}

        entry["reference"] = summarise(
            table.deck_counts(indexed[reference_rows]), tuple(args.attested_battles)
        )
        entry["reference_decks"] = int(reference_rows.size)

        if trajectories:
            walk = lookup[np.asarray([step["cards"] for step in trajectories], dtype=np.int64)]
            values = table.deck_counts(walk)
            by_round: dict[int, list[int]] = {}
            for position, step in enumerate(trajectories):
                by_round.setdefault(step["round"], []).append(position)
            entry["by_round"] = [
                {
                    "round": key,
                    "decks": len(rows),
                    **summarise(values[rows], tuple(args.attested_battles)),
                }
                for key, rows in sorted(by_round.items())
            ]
        report["sizes"][str(size)] = entry
        del table

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done — {args.output}")


if __name__ == "__main__":
    main()
