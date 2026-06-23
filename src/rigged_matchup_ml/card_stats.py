"""Static per-card metadata keyed by the Supercell card id used in battlelogs.

The Clash Royale API battlelog only carries ``id`` / ``level`` / ``evolutionLevel``
/ ``rarity`` per card -- never the elixir cost. Elixir is a dominant matchup
signal (cycle vs beatdown), so we bundle a static ``card_id -> elixir`` table
here instead of fetching it at runtime (keeps training reproducible and works in
offline environments like the Kaggle trainer).

Source: RoyaleAPI cr-api-data (``json/cards.json``); ids match the battlelog
``id``. ``CHAMPION_CARD_IDS`` is every card with ``rarity == "Champion"`` and
lets the HTTP server reconstruct the champion role at inference, since the site
payload does not send card rarities. Regenerate with::

    python scripts/refresh_card_stats.py
"""

from __future__ import annotations

# card_id -> elixir cost. Missing ids resolve to 0 ("unknown") via elixir_for.
CARD_ELIXIR: dict[int, int] = {
    26000000: 3, 26000001: 3, 26000002: 2, 26000003: 5, 26000004: 7,
    26000005: 3, 26000006: 5, 26000007: 5, 26000008: 5, 26000009: 8,
    26000010: 1, 26000011: 4, 26000012: 3, 26000013: 2, 26000014: 4,
    26000015: 4, 26000016: 5, 26000017: 5, 26000018: 4, 26000019: 2,
    26000020: 6, 26000021: 4, 26000022: 5, 26000023: 3, 26000024: 6,
    26000025: 3, 26000026: 3, 26000027: 4, 26000028: 9, 26000029: 7,
    26000030: 1, 26000031: 1, 26000032: 3, 26000033: 6, 26000034: 5,
    26000035: 4, 26000036: 4, 26000037: 4, 26000038: 2, 26000039: 3,
    26000040: 3, 26000041: 3, 26000042: 4, 26000043: 6, 26000044: 4,
    26000045: 5, 26000046: 3, 26000047: 7, 26000048: 4, 26000049: 2,
    26000050: 3, 26000051: 5, 26000052: 4, 26000053: 5, 26000054: 5,
    26000055: 7, 26000056: 3, 26000057: 4, 26000058: 2, 26000059: 5,
    26000060: 6, 26000061: 3, 26000062: 4, 26000063: 5, 26000064: 3,
    26000065: 4, 26000066: 6, 26000067: 3, 26000068: 4, 26000069: 4,
    26000070: 8, 26000071: 5, 26000072: 5, 26000073: 5, 26000074: 4,
    26000075: 4, 26000077: 5, 26000078: 3, 26000080: 4, 26000081: 4,
    26000082: 5, 26000083: 4, 26000084: 1, 26000085: 7, 26000086: 5,
    26000087: 4,
    27000000: 3, 27000001: 5, 27000002: 4, 27000003: 5, 27000004: 4,
    27000005: 6, 27000006: 4, 27000007: 6, 27000008: 6, 27000009: 3,
    27000010: 4, 27000012: 4, 27000013: 4, 27000014: 5,
    28000000: 4, 28000001: 3, 28000002: 2, 28000003: 6, 28000004: 3,
    28000005: 4, 28000006: 1, 28000007: 6, 28000008: 2, 28000009: 4,
    28000010: 5, 28000011: 2, 28000012: 3, 28000013: 3, 28000014: 3,
    28000015: 2, 28000016: 1, 28000017: 2, 28000018: 3, 28000020: 5,
}

# Cards whose rarity is "Champion" (role 2). Used server-side to set the
# champion role from card ids alone -- the site payload carries no rarity.
CHAMPION_CARD_IDS: frozenset[int] = frozenset(
    {26000065, 26000069, 26000072, 26000074, 26000077, 26000081}
)

# Highest real card elixir is 9; the embedding table sizes to this.
MAX_CARD_ELIXIR = 9


def elixir_for(card_id: int) -> int:
    """Elixir cost for a card id, or 0 when unknown (new/unmapped card)."""
    return CARD_ELIXIR.get(int(card_id), 0)
