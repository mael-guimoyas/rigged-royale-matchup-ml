"""Static per-card metadata keyed by the Supercell card id used in battlelogs.

The Clash Royale API battlelog only carries ``id`` / ``level`` / ``evolutionLevel``
/ ``rarity`` per card -- never the elixir cost. Elixir is a dominant matchup
signal (cycle vs beatdown), so we bundle a static ``card_id -> elixir`` table
here instead of fetching it at runtime (keeps training reproducible and works in
offline environments like the Kaggle trainer).

Source: RoyaleAPI cr-api-data (``json/cards.json``); ids match the battlelog
``id``. ``CHAMPION_CARD_IDS`` is every card with ``rarity == "Champion"`` and
lets the HTTP server reconstruct the champion role at inference, since the site
payload does not send card rarities.

Richer per-card gameplay metadata (role / flags / bucketed numerics) lives in the
packaged ``card_metadata_snapshot.json`` and is exposed via ``metadata_for`` /
``metadata_vector_for``. Regenerate the snapshot with::

    python scripts/refresh_card_metadata.py
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

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
    26000087: 4, 26000093: 3, 26000095: 4, 26000096: 5, 26000097: 2,
    26000099: 5, 26000101: 4, 26000102: 2, 26000103: 6,
    27000000: 3, 27000001: 5, 27000002: 4, 27000003: 5, 27000004: 4,
    27000005: 6, 27000006: 4, 27000007: 6, 27000008: 6, 27000009: 3,
    27000010: 4, 27000012: 4, 27000013: 4, 27000014: 5,
    28000000: 4, 28000001: 3, 28000002: 2, 28000003: 6, 28000004: 3,
    28000005: 4, 28000006: 1, 28000007: 6, 28000008: 2, 28000009: 4,
    28000010: 5, 28000011: 2, 28000012: 3, 28000013: 3, 28000014: 3,
    28000015: 2, 28000016: 1, 28000017: 2, 28000018: 3, 28000020: 5,
    28000023: 3, 28000024: 2, 28000025: 6, 28000026: 3,
}

# Cards whose rarity is "Champion" (role 2). Used server-side to set the
# champion role from card ids alone -- the site payload carries no rarity.
CHAMPION_CARD_IDS: frozenset[int] = frozenset(
    {
        26000065,
        26000069,
        26000072,
        26000074,
        26000077,
        26000081,
        26000093,
        26000099,
        26000103,
    }
)

# Highest real card elixir is 9; the embedding table sizes to this.
MAX_CARD_ELIXIR = 9

CARD_METADATA_TYPE_NAMES: tuple[str, ...] = ("unknown", "troop", "building", "spell")

# Exactly one role tag per card (mutually exclusive primary function). Curated by
# hand so the model can separate offensive win conditions from support / tank /
# defensive pieces -- e.g. Knight is ``mini_tank``, never ``win_condition``.
CARD_METADATA_ROLES: tuple[str, ...] = (
    "win_condition",
    "tank",
    "mini_tank",
    "support",
    "dps",
    "swarm",
    "spawner_building",
    "defensive_building",
    "damage_spell",
    "utility_spell",
)
# Boolean modifier flags; compose to express combos the user asked for, e.g.
# ``tank`` + ``high_dps`` (tank+dps) or ``high_dps`` + ``splash`` (dps+splash).
CARD_METADATA_FLAGS: tuple[str, ...] = (
    "splash",
    "air_target",
    "flying",
    "high_dps",
    "reset_control",
    "spawner",
    "building_target",
    "champion",
)
CARD_METADATA_TAGS: tuple[str, ...] = (*CARD_METADATA_ROLES, *CARD_METADATA_FLAGS)
# Bucketed (ordinal) numerics in [0, 1]; coarse on purpose so balance changes do
# not invalidate the snapshot the way exact RoyaleAPI stats did.
CARD_METADATA_NUMERIC_FEATURES: tuple[str, ...] = (
    "elixir",
    "hitpoints",
    "damage",
    "dps",
    "speed",
    "range",
)
CARD_METADATA_VECTOR_FIELDS: tuple[str, ...] = (
    *(f"type:{name}" for name in CARD_METADATA_TYPE_NAMES),
    *(f"tag:{name}" for name in CARD_METADATA_TAGS),
    *(f"num:{name}" for name in CARD_METADATA_NUMERIC_FEATURES),
)
CARD_METADATA_VECTOR_SIZE = len(CARD_METADATA_VECTOR_FIELDS)

UNKNOWN_CARD_METADATA: dict[str, Any] = {
    "name": "<unknown>",
    "key": "<unknown>",
    "type": "unknown",
    "role": "",
    "tags": frozenset(),
    "numeric": {feature: 0.0 for feature in CARD_METADATA_NUMERIC_FEATURES},
}


def elixir_for(card_id: int) -> int:
    """Elixir cost for a card id, or 0 when unknown (new/unmapped card)."""
    return CARD_ELIXIR.get(int(card_id), 0)


def _load_metadata_snapshot() -> dict[str, Any]:
    try:
        resource = files(__package__).joinpath("card_metadata_snapshot.json")
        return json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError):
        return {"cards": {}, "schema_version": 1}


def _normalise_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    card_type = str(raw.get("type") or "unknown").lower()
    if card_type not in CARD_METADATA_TYPE_NAMES:
        card_type = "unknown"
    raw_tags = raw.get("tags") or []
    tags = frozenset(str(tag) for tag in raw_tags if str(tag) in CARD_METADATA_TAGS)
    role = next((tag for tag in tags if tag in CARD_METADATA_ROLES), "")
    raw_numeric = raw.get("numeric") or {}
    numeric = {
        feature: max(0.0, min(1.0, float(raw_numeric.get(feature, 0.0) or 0.0)))
        for feature in CARD_METADATA_NUMERIC_FEATURES
    }
    return {
        "name": str(raw.get("name") or "<unknown>"),
        "key": str(raw.get("key") or "<unknown>"),
        "type": card_type,
        "role": role,
        "tags": tags,
        "numeric": numeric,
    }


CARD_METADATA_SNAPSHOT = _load_metadata_snapshot()
CARD_METADATA_SOURCE_VERSION = str(
    CARD_METADATA_SNAPSHOT.get("generated_at") or CARD_METADATA_SNAPSHOT.get("schema_version") or ""
)
CARD_METADATA: dict[int, dict[str, Any]] = {
    int(card_id): _normalise_metadata(raw)
    for card_id, raw in (CARD_METADATA_SNAPSHOT.get("cards") or {}).items()
}
UNKNOWN_CARD_METADATA_VECTOR: tuple[float, ...] = (
    tuple(1.0 if name == "unknown" else 0.0 for name in CARD_METADATA_TYPE_NAMES)
    + tuple(0.0 for _ in CARD_METADATA_TAGS)
    + tuple(0.0 for _ in CARD_METADATA_NUMERIC_FEATURES)
)
PADDING_CARD_METADATA_VECTOR: tuple[float, ...] = tuple(
    0.0 for _ in range(CARD_METADATA_VECTOR_SIZE)
)


def metadata_for(card_id: int) -> dict[str, Any]:
    """Static gameplay metadata for a raw card id.

    Unknown real card ids return an explicit ``unknown`` metadata record; padding
    id 0 should be handled by callers with ``metadata_vector_for``.
    """
    return CARD_METADATA.get(int(card_id), UNKNOWN_CARD_METADATA)


def metadata_vector_for(card_id: int) -> tuple[float, ...]:
    """Stable float vector for a raw card id; id 0 is all-zero padding."""
    if int(card_id) == 0:
        return PADDING_CARD_METADATA_VECTOR
    metadata = metadata_for(card_id)
    card_type = str(metadata["type"])
    tags = metadata["tags"]
    numeric = metadata["numeric"]
    return (
        tuple(1.0 if name == card_type else 0.0 for name in CARD_METADATA_TYPE_NAMES)
        + tuple(1.0 if tag in tags else 0.0 for tag in CARD_METADATA_TAGS)
        + tuple(float(numeric[feature]) for feature in CARD_METADATA_NUMERIC_FEATURES)
    )
