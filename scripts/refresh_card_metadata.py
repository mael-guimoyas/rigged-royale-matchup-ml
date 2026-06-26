"""Regenerate the packaged static card metadata snapshot.

The model must not fetch external card data at training/inference time. This
script snapshots the current external sources into
``src/rigged_matchup_ml/card_metadata_snapshot.json`` for review and versioning.

Sources:
- RoyaleAPI cr-api-data for card ids and mechanical stats.
- Deck Shop card detail pages for strategy flags/properties.

Run manually when adding cards or refreshing the taxonomy:

    python scripts/refresh_card_metadata.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rigged_matchup_ml.card_stats import (  # noqa: E402
    CARD_METADATA_NUMERIC_FEATURES,
    CARD_METADATA_TAGS,
    CARD_METADATA_TYPE_NAMES,
    MAX_CARD_ELIXIR,
)


ROYALE_API_BASE = "https://royaleapi.github.io/cr-api-data/json"
DECKSHOP_BASE = "https://www.deckshop.pro/card/detail"
USER_AGENT = "Mozilla/5.0 (compatible; rigged-royale-matchup-ml/metadata-refresh)"
OUT_PATH = ROOT / "src" / "rigged_matchup_ml" / "card_metadata_snapshot.json"

SUPPLEMENTAL_CARDS: tuple[dict[str, Any], ...] = (
    {
        "id": 26000093,
        "key": "little-prince",
        "name": "Little Prince",
        "type": "troop",
        "elixir": 3,
        "raw": {
            "hitpoints": 698,
            "damage": 104,
            "dps": 104 / 1.2,
            "hit_speed": 1200,
            "range": 5500,
            "speed": 60,
            "radius": 0,
        },
        "extra_tags": ("air_target", "bait", "ground_target", "support"),
    },
    {
        "id": 26000095,
        "key": "goblin-demolisher",
        "name": "Goblin Demolisher",
        "type": "troop",
        "elixir": 4,
        "raw": {
            "hitpoints": 614,
            "damage": 88,
            "dps": 88 / 1.1,
            "hit_speed": 1100,
            "range": 5000,
            "speed": 60,
            "radius": 1500,
        },
        "extra_tags": ("ground_target", "mini_tank", "splash"),
    },
    {
        "id": 26000096,
        "key": "goblin-machine",
        "name": "Goblin Machine",
        "type": "troop",
        "elixir": 5,
        "raw": {
            "hitpoints": 1780,
            "damage": 175,
            "dps": 175 / 1.2,
            "hit_speed": 1200,
            "range": 1200,
            "speed": 60,
            "radius": 0,
        },
        "extra_tags": ("air_target", "ground_target", "mini_tank", "splash"),
    },
    {
        "id": 26000097,
        "key": "suspicious-bush",
        "name": "Suspicious Bush",
        "type": "troop",
        "elixir": 2,
        "raw": {
            "hitpoints": 324,
            "damage": 242,
            "dps": 242 / 1.1,
            "hit_speed": 1100,
            "range": 500,
            "speed": 60,
            "radius": 0,
        },
        "extra_tags": ("building_chaser", "cycle", "ground_target", "win_condition"),
    },
    {
        "id": 26000099,
        "key": "goblinstein",
        "name": "Goblinstein",
        "type": "troop",
        "elixir": 5,
        "raw": {
            "hitpoints": 3114,
            "damage": 220,
            "dps": 220 / 1.8,
            "hit_speed": 1800,
            "range": 5500,
            "speed": 60,
            "radius": 0,
        },
        "extra_tags": ("air_target", "building_chaser", "ground_target", "tank"),
    },
    {
        "id": 26000101,
        "key": "rune-giant",
        "name": "Rune Giant",
        "type": "troop",
        "elixir": 4,
        "raw": {
            "hitpoints": 1664,
            "damage": 95,
            "dps": 95 / 1.5,
            "hit_speed": 1500,
            "range": 1200,
            "speed": 60,
            "radius": 0,
        },
        "extra_tags": ("building_chaser", "ground_target", "tank", "win_condition"),
    },
    {
        "id": 26000102,
        "key": "berserker",
        "name": "Berserker",
        "type": "troop",
        "elixir": 2,
        "raw": {
            "hitpoints": 350,
            "damage": 40,
            "dps": 40 / 0.6,
            "hit_speed": 600,
            "range": 800,
            "speed": 90,
            "radius": 0,
        },
        "extra_tags": ("cycle", "ground_target"),
    },
    {
        "id": 26000103,
        "key": "boss-bandit",
        "name": "Boss Bandit",
        "type": "troop",
        "elixir": 6,
        "raw": {
            "hitpoints": 2624,
            "damage": 245,
            "dps": 245 / 1.1,
            "hit_speed": 1100,
            "range": 800,
            "speed": 90,
            "radius": 0,
        },
        "extra_tags": ("ground_target", "mini_tank"),
    },
    {
        "id": 28000023,
        "key": "void",
        "name": "Void",
        "type": "spell",
        "elixir": 3,
        "raw": {
            "hitpoints": 0,
            "damage": 633,
            "dps": 633 / 4,
            "hit_speed": 4000,
            "range": 0,
            "speed": 0,
            "radius": 2500,
        },
        "extra_tags": ("air_target", "ground_target", "spell", "splash"),
    },
    {
        "id": 28000024,
        "key": "goblin-curse",
        "name": "Goblin Curse",
        "type": "spell",
        "elixir": 2,
        "raw": {
            "hitpoints": 0,
            "damage": 130,
            "dps": 130 / 6,
            "hit_speed": 6000,
            "range": 0,
            "speed": 0,
            "radius": 3000,
        },
        "extra_tags": ("air_target", "cycle", "ground_target", "spell", "splash"),
    },
    {
        "id": 28000025,
        "key": "spirit-empress",
        "name": "Spirit Empress",
        "type": "troop",
        "elixir": 6,
        "raw": {
            "hitpoints": 927,
            "damage": 254,
            "dps": 254 / 1.4,
            "hit_speed": 1400,
            "range": 5000,
            "speed": 60,
            "radius": 0,
        },
        "extra_tags": ("air_target", "bait", "ground_target", "mini_tank", "support"),
    },
    {
        "id": 28000026,
        "key": "vines",
        "name": "Vines",
        "type": "spell",
        "elixir": 3,
        "raw": {
            "hitpoints": 0,
            "damage": 190,
            "dps": 190 / 2,
            "hit_speed": 2000,
            "range": 0,
            "speed": 0,
            "radius": 2500,
        },
        "extra_tags": ("air_target", "ground_target", "spell", "splash"),
    },
)


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value[4:] if value.startswith("the-") else value


def _slug_variants(value: str) -> list[str]:
    slug = _slug(value)
    variants = [slug]
    parts = slug.split("-")
    if parts:
        last = parts[-1]
        if last.endswith("ies"):
            variants.append("-".join([*parts[:-1], last[:-3] + "y"]))
        if last.endswith("s"):
            variants.append("-".join([*parts[:-1], last[:-1]]))
    return list(dict.fromkeys(variants))


def _lookup_slug(index: dict[str, dict[str, Any]], *names: str | None) -> dict[str, Any]:
    for name in names:
        if not name:
            continue
        for slug in _slug_variants(str(name)):
            if slug in index:
                return index[slug]
    return {}


def _index_by_slug(rows: list[dict[str, Any]], *fields: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value:
                out.setdefault(_slug(str(value)), row)
    return out


def _deckshop_taxonomy(key: str) -> tuple[list[str], list[str]]:
    try:
        html = _fetch_text(f"{DECKSHOP_BASE}/{key}")
    except Exception as exc:  # noqa: BLE001 - snapshot refresh should continue.
        print(f"warning: Deck Shop fetch failed for {key}: {exc}", file=sys.stderr)
        return [], []
    flags = sorted(set(re.findall(r'href="/card/flag/([^"]+)"', html)))
    properties = sorted(set(re.findall(r'href="/card/property/([^"]+)"', html)))
    return flags, properties


def _dps(damage: float, hit_speed_ms: float) -> float:
    return damage / (hit_speed_ms / 1000.0) if damage > 0 and hit_speed_ms > 0 else 0.0


def _projectile_for(
    projectiles: dict[str, dict[str, Any]],
    *names: str | None,
) -> dict[str, Any]:
    candidates: list[str] = []
    for name in names:
        if not name:
            continue
        slug = _slug(str(name))
        candidates.extend([slug, f"{slug}spell", f"{slug}projectile"])
    for candidate in candidates:
        if candidate in projectiles:
            return projectiles[candidate]
    for candidate in candidates:
        for slug, row in projectiles.items():
            if slug.startswith(candidate) or candidate.startswith(slug):
                return row
    return {}


def _merge_projectile_stats(
    source: dict[str, Any], projectile: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(source)
    for key, value in projectile.items():
        if key in {"name", "rarity"} or value in (None, False, ""):
            continue
        if merged.get(key) in (None, 0, False, ""):
            merged[key] = value
    return merged


def _mechanical_stats(
    card: dict[str, Any],
    troop_cards: dict[str, dict[str, Any]],
    characters: dict[str, dict[str, Any]],
    buildings: dict[str, dict[str, Any]],
    spells: dict[str, dict[str, Any]],
    projectiles: dict[str, dict[str, Any]],
) -> dict[str, float]:
    key = str(card.get("key") or "")
    name = str(card.get("name") or key)
    card_type = str(card.get("type") or "").lower()
    troop_card = _lookup_slug(troop_cards, key, name)
    summon_name = troop_card.get("summon_character") or troop_card.get("name") or name
    source = _lookup_slug(characters, str(summon_name), name)
    if card_type == "building":
        source = _lookup_slug(buildings, str(summon_name), name) or source
    if card_type == "spell":
        source = _lookup_slug(spells, name, key)
    projectile = _projectile_for(projectiles, source.get("projectile"), name, key)
    if projectile:
        source = _merge_projectile_stats(source, projectile)

    units = max(1, int(troop_card.get("summon_number") or 1))
    damage = float(source.get("damage") or source.get("damage_special") or 0)
    hit_speed = float(source.get("hit_speed") or 0)
    hitpoints = float(source.get("hitpoints") or 0)
    return {
        "elixir": float(card.get("elixir") or troop_card.get("mana_cost") or 0),
        "hitpoints": hitpoints * units,
        "damage": damage * units,
        "dps": float(source.get("dps") or _dps(damage, hit_speed)) * units,
        "hit_speed": hit_speed,
        "range": float(source.get("range") or 0),
        "speed": float(source.get("speed") or 0),
        "radius": float(source.get("radius") or source.get("death_damage_radius") or 0),
    }


def _mechanical_tags(
    card: dict[str, Any],
    raw: dict[str, float],
    flags: list[str],
    properties: list[str],
    troop_cards: dict[str, dict[str, Any]],
    characters: dict[str, dict[str, Any]],
    buildings: dict[str, dict[str, Any]],
    spells: dict[str, dict[str, Any]],
    projectiles: dict[str, dict[str, Any]],
) -> list[str]:
    tags: set[str] = set()
    card_type = str(card.get("type") or "unknown").lower()
    if card_type == "spell":
        tags.add("spell")
    if card_type == "building":
        tags.add("building")
    if raw["elixir"] > 0 and raw["elixir"] <= 2:
        tags.add("cycle")

    taxonomy = set(flags) | set(properties)
    if "win-condition" in taxonomy:
        tags.add("win_condition")
    if any("support" in item for item in taxonomy):
        tags.add("support")
    if "tank" in taxonomy or "big-tank" in taxonomy or "heavy-tank" in taxonomy:
        tags.add("tank")
    if "mini-tank" in taxonomy:
        tags.add("mini_tank")
    if any("bait" in item for item in taxonomy):
        tags.add("bait")
    if "building-chaser" in taxonomy:
        tags.add("building_chaser")
    if any("splash" in item or "area" in item for item in taxonomy):
        tags.add("splash")

    key = str(card.get("key") or "")
    name = str(card.get("name") or key)
    troop_card = _lookup_slug(troop_cards, key, name)
    summon_name = troop_card.get("summon_character") or troop_card.get("name") or name
    source = _lookup_slug(characters, str(summon_name), name)
    if card_type == "building":
        source = _lookup_slug(buildings, str(summon_name), name) or source
    if card_type == "spell":
        source = _lookup_slug(spells, name, key)
    projectile = _projectile_for(projectiles, source.get("projectile"), name, key)
    if projectile:
        source = _merge_projectile_stats(source, projectile)

    if bool(source.get("attacks_air") or source.get("aoe_to_air")):
        tags.add("air_target")
    if bool(source.get("attacks_ground") or source.get("aoe_to_ground")):
        tags.add("ground_target")
    if (
        raw["radius"] > 0
        or int(source.get("multiple_targets") or 0) > 0
        or bool(source.get("all_targets_hit"))
        or float(source.get("death_damage_radius") or 0) > 0
    ):
        tags.add("splash")
    if int(troop_card.get("summon_number") or 1) > 1:
        tags.add("swarm")
    if source.get("spawn_character") or float(source.get("spawn_interval") or 0) > 0:
        tags.add("spawner")

    return sorted(tag for tag in tags if tag in CARD_METADATA_TAGS)


def _normalise_cards(cards: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    maxima = {
        feature: max((card["raw"].get(feature, 0.0) for card in cards.values()), default=0.0)
        for feature in CARD_METADATA_NUMERIC_FEATURES
    }
    maxima["elixir"] = float(MAX_CARD_ELIXIR)
    for card in cards.values():
        numeric: dict[str, float] = {}
        for feature in CARD_METADATA_NUMERIC_FEATURES:
            value = float(card["raw"].get(feature, 0.0) or 0.0)
            denom = maxima.get(feature, 0.0) or 0.0
            numeric[feature] = round(value / denom, 6) if denom > 0 else 0.0
        card["numeric"] = numeric
    return cards


def main() -> None:
    card_rows = _fetch_json(f"{ROYALE_API_BASE}/cards.json")
    stats = _fetch_json(f"{ROYALE_API_BASE}/cards_stats.json")
    troop_cards = _index_by_slug(stats.get("troop", []), "name", "icon_file")
    characters = _index_by_slug(
        _fetch_json(f"{ROYALE_API_BASE}/cards_stats_characters.json"), "name"
    )
    buildings = _index_by_slug(
        _fetch_json(f"{ROYALE_API_BASE}/cards_stats_building.json"), "name"
    )
    spells = _index_by_slug(_fetch_json(f"{ROYALE_API_BASE}/cards_stats_spell.json"), "name")
    projectiles = _index_by_slug(
        _fetch_json(f"{ROYALE_API_BASE}/cards_stats_projectile.json"), "name"
    )

    cards: dict[str, dict[str, Any]] = {}
    for index, card in enumerate(card_rows, start=1):
        if card.get("elixir") is None:
            continue
        key = str(card["key"])
        flags, properties = _deckshop_taxonomy(key)
        raw = _mechanical_stats(
            card, troop_cards, characters, buildings, spells, projectiles
        )
        card_type = str(card.get("type") or "unknown").lower()
        if card_type not in CARD_METADATA_TYPE_NAMES:
            card_type = "unknown"
        tags = _mechanical_tags(
            card,
            raw,
            flags,
            properties,
            troop_cards,
            characters,
            buildings,
            spells,
            projectiles,
        )
        cards[str(int(card["id"]))] = {
            "name": card.get("name"),
            "key": key,
            "type": card_type,
            "tags": tags,
            "raw": {feature: round(float(raw.get(feature, 0.0) or 0.0), 6) for feature in raw},
            "deckshop_flags": flags,
            "deckshop_properties": properties,
        }
        if index % 20 == 0:
            time.sleep(0.5)

    for card in SUPPLEMENTAL_CARDS:
        key = str(card["key"])
        flags, properties = _deckshop_taxonomy(key)
        raw = dict(card["raw"])
        raw["elixir"] = float(card["elixir"])
        card_type = str(card.get("type") or "unknown").lower()
        if card_type not in CARD_METADATA_TYPE_NAMES:
            card_type = "unknown"
        tags = set(
            _mechanical_tags(
                card,
                raw,
                flags,
                properties,
                troop_cards,
                characters,
                buildings,
                spells,
                projectiles,
            )
        )
        tags.update(str(tag) for tag in card.get("extra_tags", ()) if tag in CARD_METADATA_TAGS)
        cards[str(int(card["id"]))] = {
            "name": card["name"],
            "key": key,
            "type": card_type,
            "tags": sorted(tags),
            "raw": {feature: round(float(raw.get(feature, 0.0) or 0.0), 6) for feature in raw},
            "deckshop_flags": flags,
            "deckshop_properties": properties,
        }

    snapshot = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "royaleapi_cards": f"{ROYALE_API_BASE}/cards.json",
            "royaleapi_stats": f"{ROYALE_API_BASE}/cards_stats.json",
            "deckshop_detail": f"{DECKSHOP_BASE}/<card-key>",
            "supplemental_cards": "scripts/refresh_card_metadata.py::SUPPLEMENTAL_CARDS",
        },
        "vector": {
            "types": list(CARD_METADATA_TYPE_NAMES),
            "tags": list(CARD_METADATA_TAGS),
            "numeric_features": list(CARD_METADATA_NUMERIC_FEATURES),
        },
        "cards": _normalise_cards(cards),
    }
    OUT_PATH.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cards": len(cards), "output": str(OUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
