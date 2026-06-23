"""Regenerate the CARD_ELIXIR / CHAMPION_CARD_IDS tables in card_stats.py.

Pulls RoyaleAPI's cr-api-data card list (ids match the battlelog ``id``) and
prints a ready-to-paste Python block. Run when a balance patch adds cards or
champions::

    python scripts/refresh_card_stats.py

Then paste the output over the literals in ``src/rigged_matchup_ml/card_stats.py``.
Kept as a print-only helper so the bundled table stays an auditable literal
(works offline, e.g. in the Kaggle trainer) rather than a runtime fetch.
"""

from __future__ import annotations

import json
import urllib.request

CARDS_URL = "https://royaleapi.github.io/cr-api-data/json/cards.json"


def main() -> int:
    with urllib.request.urlopen(CARDS_URL, timeout=30) as response:
        cards = json.loads(response.read().decode("utf-8"))

    elixir = {int(c["id"]): int(c["elixir"]) for c in cards if c.get("elixir") is not None}
    champions = sorted(int(c["id"]) for c in cards if str(c.get("rarity")) == "Champion")
    max_elixir = max(elixir.values())

    print(f"# {len(elixir)} cards, max elixir {max_elixir}")
    print("CARD_ELIXIR: dict[int, int] = {")
    ids = sorted(elixir)
    for offset in range(0, len(ids), 5):
        chunk = ids[offset : offset + 5]
        print("    " + " ".join(f"{cid}: {elixir[cid]}," for cid in chunk))
    print("}")
    print(f"CHAMPION_CARD_IDS = frozenset({set(champions)})")
    print(f"MAX_CARD_ELIXIR = {max_elixir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
