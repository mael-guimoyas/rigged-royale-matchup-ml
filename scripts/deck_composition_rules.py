"""What a real Clash Royale deck is made of, counted over the whole corpus.

The score envelope says how *well* a deck may plausibly do. This says what a deck
plausibly *is*. The two catch different failures: a search can stay under a score
ceiling and still emit eight cards no human would ever sleeve together.

Every rule here is measured, never assumed. For each candidate rule the output
reports the share of genuinely played decks that satisfy it, weighted by play
count as well as by distinct deck, so a rule can be adopted with its real false-
positive rate in hand rather than a guess.

Roles come from the site catalog (`data/cards.json`), so a rule expressed here
can be re-expressed verbatim in TypeScript against the same role strings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

POOLED_SEGMENT_SQL = """case
  when mode_key = 'ranked'
    and try_cast(regexp_extract(segment, '^ranked:league-([0-9]+)$', 1) as integer)
      between 1 and 2 then 'ranked:league-1-2'
  when mode_key = 'ranked'
    and try_cast(regexp_extract(segment, '^ranked:league-([0-9]+)$', 1) as integer)
      between 3 and 4 then 'ranked:league-3-4'
  when mode_key = 'ranked'
    and try_cast(regexp_extract(segment, '^ranked:league-([0-9]+)$', 1) as integer)
      between 5 and 7 then 'ranked:league-5-7'
  else segment
end"""


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / "data" / "raw")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root.parent / "riggedroyale" / "data" / "cards.json",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "deck-composition-rules.json"
    )
    parser.add_argument("--min-plays", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


class Catalog:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.by_id: dict[int, dict[str, Any]] = {}
        for card in data["cards"]:
            if card.get("id") is not None:
                self.by_id[int(card["id"])] = card

    def known(self, cards: tuple[int, ...]) -> bool:
        return all(card in self.by_id for card in cards)

    def roles(self, cards: tuple[int, ...]) -> list[list[str]]:
        return [list(self.by_id[card].get("roles") or []) for card in cards]

    def elixir(self, cards: tuple[int, ...]) -> list[int]:
        return [int(self.by_id[card].get("elixir") or 0) for card in cards]

    def kinds(self, cards: tuple[int, ...]) -> list[str]:
        return [str(self.by_id[card].get("kind") or "") for card in cards]

    def targets(self, cards: tuple[int, ...]) -> list[str]:
        return [str(self.by_id[card].get("targets") or "") for card in cards]


def features(catalog: Catalog, cards: tuple[int, ...]) -> dict[str, float]:
    """The handful of composition numbers a hardcoded rule could ever read."""
    roles = catalog.roles(cards)
    elixir = catalog.elixir(cards)
    kinds = catalog.kinds(cards)
    targets = catalog.targets(cards)
    flat = [role for card_roles in roles for role in card_roles]

    def count(role: str) -> int:
        return sum(1 for card_roles in roles if role in card_roles)

    known_elixir = [value for value in elixir if value > 0]
    hits_air = sum(
        1
        for index in range(len(cards))
        if targets[index] == "air-and-ground"
        or "anti-air" in roles[index]
        or (kinds[index] == "spell" and "spell-utility" not in roles[index])
    )
    return {
        "win_conditions": count("win-condition"),
        "small_spells": count("spell-small"),
        "any_spells": sum(1 for kind in kinds if kind == "spell"),
        "medium_spells": count("spell-medium"),
        "heavy_spells": count("spell-heavy"),
        "anti_air": hits_air,
        "buildings": sum(1 for kind in kinds if kind == "building"),
        "splash": count("splash-damage"),
        "tank_killers": count("tank-killer"),
        "cheap_cards": sum(1 for value in elixir if 0 < value <= 3),
        "cheapest": min(known_elixir) if known_elixir else 0,
        "average_elixir": sum(known_elixir) / len(known_elixir) if known_elixir else 0.0,
        "distinct_roles": len(set(flat)),
    }


ENVELOPES: dict[str, Any] = {
    "wide": lambda f: (
        f["anti_air"] >= 2
        and f["average_elixir"] <= 5.0
        and f["buildings"] <= 2
        and f["distinct_roles"] >= 8
    ),
    "tight": lambda f: (
        f["anti_air"] >= 2
        and f["average_elixir"] <= 5.0
        and f["buildings"] <= 2
        and f["distinct_roles"] >= 8
        and f["cheap_cards"] >= 2
        and f["any_spells"] >= 1
    ),
}
"""The two composition gates under test, defined once so the benchmark that
prices them and the search that runs under them cannot drift apart."""


def admits_deck(catalog: Catalog, envelope: str, cards: tuple[int, ...]) -> bool:
    """Admit a deck under one envelope; unknown cards are admitted, not judged."""
    if not catalog.known(cards):
        return True
    return bool(ENVELOPES[envelope](features(catalog, cards)))


def rule_table(rows: list[tuple[dict[str, float], int]]) -> list[dict[str, Any]]:
    """Pass rate of each candidate rule, by distinct deck and by battle played."""
    rules: dict[str, Any] = {
        "win_conditions >= 1": lambda f: f["win_conditions"] >= 1,
        "win_conditions >= 1 and <= 3": lambda f: 1 <= f["win_conditions"] <= 3,
        "any_spells >= 1": lambda f: f["any_spells"] >= 1,
        "any_spells >= 2": lambda f: f["any_spells"] >= 2,
        "small_spells >= 1": lambda f: f["small_spells"] >= 1,
        "anti_air >= 2": lambda f: f["anti_air"] >= 2,
        "anti_air >= 3": lambda f: f["anti_air"] >= 3,
        "cheap_cards >= 2": lambda f: f["cheap_cards"] >= 2,
        "cheap_cards >= 3": lambda f: f["cheap_cards"] >= 3,
        "cheapest <= 2": lambda f: f["cheapest"] <= 2,
        "average_elixir <= 4.5": lambda f: f["average_elixir"] <= 4.5,
        "average_elixir <= 5.0": lambda f: f["average_elixir"] <= 5.0,
        "average_elixir in [2.4, 4.6]": lambda f: 2.4 <= f["average_elixir"] <= 4.6,
        "splash >= 1": lambda f: f["splash"] >= 1,
        "tank_killers >= 1": lambda f: f["tank_killers"] >= 1,
    }
    total_decks = len(rows)
    total_plays = sum(plays for _, plays in rows)
    output = []
    for label, test in rules.items():
        decks = sum(1 for feature, _ in rows if test(feature))
        plays = sum(plays for feature, plays in rows if test(feature))
        output.append(
            {
                "rule": label,
                "deck_pass_rate": decks / total_decks if total_decks else 0.0,
                "play_pass_rate": plays / total_plays if total_plays else 0.0,
            }
        )
    return output


def distribution(values: list[float], weights: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    order = np.argsort(array)
    array, weight = array[order], weight[order]
    cumulative = np.cumsum(weight) / weight.sum()

    def at(quantile: float) -> float:
        return float(array[int(np.searchsorted(cumulative, quantile, side="left"))])

    return {
        "min": float(array.min()),
        "p01": at(0.01),
        "p05": at(0.05),
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    catalog = Catalog(args.catalog)
    connection = duckdb.connect()
    connection.execute(f"pragma threads={args.threads}")
    parquet = (args.raw.resolve() / "*.parquet").as_posix().replace("'", "''")

    log("aggregating played decks")
    rows = connection.execute(
        f"""
        with base as (
          select {POOLED_SEGMENT_SQL} as seg, team_deck_key, opponent_deck_key
          from read_parquet('{parquet}')
          where segment not in ('ladder:top-100', 'ladder:top-1000')
        ), sides as (
          select seg, team_deck_key as deck_key from base
          union all
          select seg, opponent_deck_key as deck_key from base
        )
        select seg, deck_key, count(*) as plays
        from sides group by seg, deck_key
        having count(*) >= {args.min_plays}
        """
    ).fetchall()
    log(f"played decks: {len(rows):,}")

    by_segment: dict[str, list[tuple[dict[str, float], int]]] = defaultdict(list)
    skipped = 0
    for segment, deck_key, plays in rows:
        cards = tuple(int(value) for value in str(deck_key).split(","))
        if len(cards) != 8 or not catalog.known(cards):
            skipped += 1
            continue
        by_segment[str(segment)].append((features(catalog, cards), int(plays)))
    log(f"usable decks: {sum(len(v) for v in by_segment.values()):,} (skipped {skipped:,})")

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "min_plays": args.min_plays,
        "segments": {},
        "pooled": {},
    }
    pooled: list[tuple[dict[str, float], int]] = []
    for segment, entries in sorted(by_segment.items()):
        pooled.extend(entries)
        report["segments"][segment] = {
            "decks": len(entries),
            "rules": rule_table(entries),
            "average_elixir": distribution(
                [feature["average_elixir"] for feature, _ in entries],
                [plays for _, plays in entries],
            ),
        }
    report["pooled"] = {
        "decks": len(pooled),
        "rules": rule_table(pooled),
        "features": {
            name: distribution(
                [feature[name] for feature, _ in pooled], [plays for _, plays in pooled]
            )
            for name in (
                "win_conditions",
                "any_spells",
                "small_spells",
                "anti_air",
                "buildings",
                "splash",
                "tank_killers",
                "cheap_cards",
                "cheapest",
                "average_elixir",
                "distinct_roles",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"written {args.output}")

    print(f"\n{'rule':34} {'decks':>8} {'plays':>8}")
    for rule in report["pooled"]["rules"]:
        print(f"{rule['rule']:34} {rule['deck_pass_rate']:8.4f} {rule['play_pass_rate']:8.4f}")
    print(f"\n{'feature':18} {'min':>6} {'p01':>6} {'p05':>6} {'p50':>6} {'p95':>6} {'p99':>6} {'max':>6}")
    for name, values in report["pooled"]["features"].items():
        print(
            f"{name:18} {values['min']:6.2f} {values['p01']:6.2f} {values['p05']:6.2f} "
            f"{values['p50']:6.2f} {values['p95']:6.2f} {values['p99']:6.2f} {values['max']:6.2f}"
        )


if __name__ == "__main__":
    main()
