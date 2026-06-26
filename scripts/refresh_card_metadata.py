"""Regenerate the packaged static card metadata snapshot.

The model must not fetch external card data at training/inference time, and the
old RoyaleAPI / Deck Shop scrape produced stale numbers (~42% of cards had no
stats) plus noisy tags (``bait`` landed on 73% of cards). This generator is
fully hand-curated and offline: it emits a clean, low-noise taxonomy into
``src/rigged_matchup_ml/card_metadata_snapshot.json``.

Design:
- ``role``  -- exactly one primary function per card (offense vs support/defense
  is the signal the model was missing: Knight is ``mini_tank``, not a win
  condition).
- ``flags`` -- sparse boolean modifiers that compose (``tank`` + ``high_dps`` =
  tank+dps; ``high_dps`` + ``splash`` = dps+splash).
- numeric  -- coarse ordinal *buckets* in [0, 1], so a balance tweak does not
  invalidate the snapshot.

Run when adding cards or revising the taxonomy::

    python scripts/refresh_card_metadata.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rigged_matchup_ml.card_stats import (  # noqa: E402
    CARD_ABILITY_TAGS,
    CARD_ELIXIR,
    CARD_EVO_TAGS,
    CARD_METADATA_FLAGS,
    CARD_METADATA_NUMERIC_FEATURES,
    CARD_METADATA_ROLES,
    CARD_METADATA_TYPE_NAMES,
    CHAMPION_CARD_IDS,
)

OUT_PATH = ROOT / "src" / "rigged_matchup_ml" / "card_metadata_snapshot.json"

# Numeric resolution. The CARD_TABLE below stores coarse 0-5 tiers that are easy
# to review and stay valid across balance patches. We expand those onto a finer
# 0-20 grid (tier * GRID_STEP) and then apply small per-card NUDGES so cards that
# share a coarse tier (e.g. every tier-5 tank) still get separated -- 20 levels
# is the honest sweet spot: 4x finer than the tiers, without the false precision
# of pretending to know exact hp/dps to 1/100th.
GRID = 20         # final ordinal scale for hp / dmg / dps / range
GRID_STEP = 4     # 0-5 tier -> 0-20 grid (tier * 4)
SPEED_GRID = 8    # speed has fewer meaningful levels; tier * 2 -> 0-8
SPEED_STEP = 2
MAX_ELIXIR = 9

# Per-card fine adjustments on the 0-20 grid, applied on top of (tier * GRID_STEP)
# and clamped to [0, GRID]. Only cards that share a coarse tier with a clearly
# stronger/weaker sibling are listed; everything else inherits the tier centre.
NUDGES: dict[int, dict[str, int]] = {
    # tier-5 hitpoints, separated by real body size (Golem > Giant > P.E.K.K.A).
    26000029: {"hitpoints": -1},               # Lava Hound
    26000003: {"hitpoints": -2},               # Giant
    26000060: {"hitpoints": -2},               # Goblin Giant
    26000067: {"hitpoints": -3},               # Elixir Golem
    26000099: {"hitpoints": -3},               # Goblinstein
    26000055: {"hitpoints": -3},               # Mega Knight
    26000020: {"hitpoints": -4},               # Giant Skeleton
    26000004: {"hitpoints": -4, "damage": -1},  # P.E.K.K.A (mid tier-5 body, top punch)
    # tier-5 dps, separated by how fast they shred (Inferno ramps to the top).
    26000012: {"dps": -1},                     # Skeleton Army
    26000035: {"dps": -2},                     # Lumberjack
    26000103: {"dps": -2},                     # Boss Bandit
    # tier-5 damage, separated by per-hit punch (Sparky / Rocket on top).
    26000018: {"damage": -2},                  # Mini P.E.K.K.A
    26000006: {"damage": -2},                  # Balloon
    # tier-5 range, separated by reach (X-Bow / Mortar / Princess longest).
    26000040: {"range": -1},                   # Dart Goblin
    26000064: {"range": -1},                   # Firecracker
    26000062: {"range": -2},                   # Magic Archer
}

# id -> (type, role, (flags...), hp, dmg, dps, speed, range)
# Tiers are game-knowledge buckets, not exact stats. type is stored explicitly
# because a few "28*" ids are troops (Heal Spirit, Spirit Empress), not spells.
CARD_TABLE: dict[int, tuple[str, str, tuple[str, ...], int, int, int, int, int]] = {
    # --- Troops -----------------------------------------------------------
    26000000: ("troop", "mini_tank", (), 3, 3, 3, 2, 1),                                  # Knight
    26000001: ("troop", "support", ("air_target",), 2, 2, 2, 2, 4),                       # Archers
    26000002: ("troop", "swarm", (), 1, 3, 4, 4, 1),                                      # Goblins
    26000003: ("troop", "win_condition", ("building_target",), 5, 3, 2, 1, 1),            # Giant
    26000004: ("troop", "tank", ("high_dps",), 5, 5, 4, 2, 1),                            # P.E.K.K.A
    26000005: ("troop", "support", ("air_target", "flying"), 2, 2, 3, 3, 2),              # Minions
    26000006: ("troop", "win_condition", ("flying", "building_target"), 4, 5, 4, 2, 1),   # Balloon
    26000007: ("troop", "support", ("splash", "air_target", "spawner"), 3, 2, 3, 2, 3),   # Witch
    26000008: ("troop", "swarm", (), 3, 3, 4, 2, 1),                                      # Barbarians
    26000009: ("troop", "win_condition", ("building_target",), 5, 3, 2, 1, 1),            # Golem
    26000010: ("troop", "swarm", (), 1, 2, 2, 3, 1),                                      # Skeletons
    26000011: ("troop", "mini_tank", ("splash",), 3, 3, 3, 2, 1),                         # Valkyrie
    26000012: ("troop", "swarm", (), 1, 3, 5, 3, 1),                                      # Skeleton Army
    26000013: ("troop", "support", ("splash",), 2, 3, 3, 2, 3),                           # Bomber
    26000014: ("troop", "support", ("air_target",), 2, 3, 3, 2, 4),                       # Musketeer
    26000015: ("troop", "support", ("splash", "air_target", "flying"), 3, 3, 3, 3, 2),    # Baby Dragon
    26000016: ("troop", "dps", ("high_dps",), 3, 4, 4, 2, 1),                             # Prince
    26000017: ("troop", "support", ("splash", "air_target"), 2, 3, 3, 2, 3),              # Wizard
    26000018: ("troop", "dps", ("high_dps",), 3, 5, 4, 2, 1),                             # Mini P.E.K.K.A
    26000019: ("troop", "swarm", ("air_target",), 1, 2, 2, 4, 3),                         # Spear Goblins
    26000020: ("troop", "tank", ("splash",), 5, 3, 2, 2, 1),                              # Giant Skeleton
    26000021: ("troop", "win_condition", ("building_target",), 3, 3, 4, 4, 1),            # Hog Rider
    26000022: ("troop", "swarm", ("air_target", "flying"), 2, 2, 3, 3, 2),                # Minion Horde
    26000023: ("troop", "support", ("splash", "air_target", "reset_control"), 2, 1, 1, 2, 3),  # Ice Wizard
    26000024: ("troop", "win_condition", ("building_target",), 4, 3, 3, 1, 3),            # Royal Giant
    26000025: ("troop", "swarm", (), 2, 2, 3, 3, 1),                                      # Guards
    26000026: ("troop", "support", ("splash", "air_target"), 1, 2, 1, 2, 5),              # Princess
    26000027: ("troop", "dps", ("splash", "high_dps"), 3, 3, 3, 2, 1),                    # Dark Prince
    26000028: ("troop", "support", ("air_target",), 2, 3, 4, 2, 4),                       # Three Musketeers
    26000029: ("troop", "win_condition", ("flying", "building_target"), 5, 1, 1, 1, 2),   # Lava Hound
    26000030: ("troop", "swarm", ("reset_control",), 1, 1, 1, 4, 1),                      # Ice Spirit
    26000031: ("troop", "swarm", ("splash",), 1, 3, 2, 4, 2),                             # Fire Spirit
    26000032: ("troop", "win_condition", (), 3, 2, 3, 3, 1),                              # Miner
    26000033: ("troop", "support", ("splash", "high_dps"), 3, 5, 2, 1, 4),                # Sparky
    26000034: ("troop", "support", ("splash", "reset_control"), 3, 3, 3, 2, 3),           # Bowler
    26000035: ("troop", "dps", ("high_dps",), 3, 3, 5, 4, 1),                             # Lumberjack
    26000036: ("troop", "win_condition", ("building_target",), 3, 3, 3, 4, 1),            # Battle Ram
    26000037: ("troop", "dps", ("air_target", "flying", "high_dps"), 2, 1, 5, 3, 3),      # Inferno Dragon
    26000038: ("troop", "mini_tank", ("reset_control", "building_target"), 3, 1, 1, 2, 1),  # Ice Golem
    26000039: ("troop", "support", ("air_target", "flying"), 2, 3, 3, 2, 2),              # Mega Minion
    26000040: ("troop", "support", ("air_target",), 1, 2, 3, 4, 5),                       # Dart Goblin
    26000041: ("troop", "swarm", ("air_target",), 1, 3, 4, 4, 1),                         # Goblin Gang
    26000042: ("troop", "support", ("air_target", "reset_control"), 2, 2, 2, 2, 3),       # Electro Wizard
    26000043: ("troop", "dps", ("high_dps",), 3, 4, 4, 3, 1),                             # Elite Barbarians
    26000044: ("troop", "dps", ("splash", "air_target"), 3, 5, 3, 2, 2),                  # Hunter
    26000045: ("troop", "support", ("splash", "air_target"), 3, 3, 3, 2, 3),              # Executioner
    26000046: ("troop", "dps", ("high_dps",), 3, 3, 4, 4, 1),                             # Bandit
    26000047: ("troop", "swarm", (), 3, 2, 3, 2, 1),                                      # Royal Recruits
    26000048: ("troop", "support", ("spawner",), 3, 3, 4, 2, 1),                          # Night Witch
    26000049: ("troop", "swarm", ("air_target", "flying"), 1, 2, 3, 4, 1),                # Bats
    26000050: ("troop", "dps", ("splash",), 3, 3, 3, 3, 1),                               # Royal Ghost
    26000051: ("troop", "win_condition", ("building_target", "reset_control"), 3, 3, 4, 4, 1),  # Ram Rider
    26000052: ("troop", "support", ("air_target", "reset_control"), 2, 2, 2, 2, 3),       # Zappies
    26000053: ("troop", "mini_tank", ("air_target",), 3, 2, 3, 2, 2),                     # Rascals
    26000054: ("troop", "mini_tank", (), 3, 3, 3, 2, 3),                                  # Cannon Cart
    26000055: ("troop", "tank", ("splash", "high_dps"), 5, 4, 3, 2, 1),                   # Mega Knight
    26000056: ("troop", "win_condition", ("flying", "building_target"), 2, 2, 2, 3, 1),   # Skeleton Barrel
    26000057: ("troop", "support", ("air_target", "flying"), 2, 3, 3, 3, 4),              # Flying Machine
    26000058: ("troop", "win_condition", ("building_target",), 1, 4, 4, 4, 1),            # Wall Breakers
    26000059: ("troop", "win_condition", ("building_target",), 3, 2, 3, 4, 1),            # Royal Hogs
    26000060: ("troop", "win_condition", ("building_target", "spawner", "air_target"), 5, 3, 3, 2, 1),  # Goblin Giant
    26000061: ("troop", "support", ("reset_control",), 2, 3, 3, 2, 2),                    # Fisherman
    26000062: ("troop", "support", ("splash", "air_target"), 2, 2, 3, 2, 5),              # Magic Archer
    26000063: ("troop", "support", ("splash", "air_target", "flying", "reset_control"), 3, 3, 3, 2, 3),  # Electro Dragon
    26000064: ("troop", "support", ("splash", "air_target"), 1, 3, 3, 3, 5),              # Firecracker
    26000065: ("troop", "dps", ("high_dps", "champion"), 4, 3, 4, 3, 1),                  # Mighty Miner
    26000066: ("troop", "support", ("splash", "air_target", "spawner"), 3, 3, 3, 2, 3),   # Super Witch
    26000067: ("troop", "win_condition", ("building_target",), 5, 1, 1, 2, 1),            # Elixir Golem
    26000068: ("troop", "mini_tank", (), 3, 2, 2, 2, 1),                                  # Battle Healer
    26000069: ("troop", "tank", ("splash", "spawner", "champion"), 4, 3, 3, 2, 1),        # Skeleton King
    26000070: ("troop", "win_condition", ("flying", "building_target"), 5, 1, 1, 1, 2),   # Super Lava Hound
    26000071: ("troop", "support", ("splash", "air_target"), 2, 3, 3, 2, 5),              # Super Magic Archer
    26000072: ("troop", "support", ("air_target", "champion"), 3, 4, 4, 2, 4),            # Archer Queen
    26000073: ("troop", "win_condition", ("building_target",), 3, 3, 4, 4, 1),            # Santa Hog Rider
    26000074: ("troop", "dps", ("high_dps", "champion"), 4, 3, 4, 3, 1),                  # Golden Knight
    26000075: ("troop", "mini_tank", ("splash", "reset_control", "building_target"), 4, 1, 1, 2, 1),  # Super Ice Golem
    26000077: ("troop", "tank", ("reset_control", "champion"), 4, 3, 3, 2, 1),            # Monk
    26000078: ("troop", "support", ("air_target",), 2, 4, 3, 2, 4),                       # Super Archers
    26000080: ("troop", "support", ("splash", "air_target", "flying"), 2, 2, 3, 3, 2),    # Skeleton Dragons
    26000081: ("troop", "dps", ("high_dps", "champion"), 4, 3, 4, 3, 1),                  # Terry
    26000082: ("troop", "dps", ("high_dps",), 3, 5, 5, 3, 1),                             # Super Mini P.E.K.K.A
    26000083: ("troop", "support", ("splash", "air_target"), 2, 2, 2, 2, 4),              # Mother Witch
    26000084: ("troop", "swarm", ("air_target", "reset_control"), 1, 1, 1, 4, 2),         # Electro Spirit
    26000085: ("troop", "win_condition", ("building_target", "reset_control", "splash"), 5, 3, 2, 1, 3),  # Electro Giant
    26000086: ("troop", "dps", ("high_dps",), 3, 4, 4, 3, 1),                             # Raging Prince
    26000087: ("troop", "support", ("air_target", "flying"), 3, 3, 3, 3, 2),              # Phoenix
    26000093: ("troop", "support", ("air_target", "champion"), 2, 3, 3, 2, 4),            # Little Prince
    26000095: ("troop", "mini_tank", ("splash", "building_target"), 3, 3, 3, 2, 2),       # Goblin Demolisher
    26000096: ("troop", "mini_tank", ("splash", "building_target"), 4, 4, 3, 2, 2),       # Goblin Machine
    26000097: ("troop", "win_condition", ("building_target", "spawner"), 1, 3, 3, 3, 1),  # Suspicious Bush
    26000099: ("troop", "tank", ("high_dps", "champion"), 5, 4, 3, 2, 1),                 # Goblinstein
    26000101: ("troop", "win_condition", ("building_target",), 4, 2, 2, 2, 3),            # Rune Giant
    26000102: ("troop", "dps", ("high_dps",), 2, 2, 4, 3, 1),                             # Berserker
    26000103: ("troop", "dps", ("high_dps", "champion"), 4, 4, 5, 4, 1),                  # Boss Bandit
    28000016: ("troop", "swarm", (), 1, 1, 1, 4, 2),                                      # Heal Spirit
    28000025: ("troop", "support", ("splash", "air_target"), 2, 3, 3, 2, 4),              # Spirit Empress
    # --- Buildings --------------------------------------------------------
    27000000: ("building", "defensive_building", (), 2, 3, 3, 0, 3),                      # Cannon
    27000001: ("building", "spawner_building", ("spawner",), 3, 0, 0, 0, 0),              # Goblin Hut
    27000002: ("building", "win_condition", ("splash", "building_target"), 2, 3, 2, 0, 5),  # Mortar
    27000003: ("building", "defensive_building", ("high_dps",), 3, 1, 5, 0, 3),           # Inferno Tower
    27000004: ("building", "defensive_building", ("splash",), 3, 3, 3, 0, 3),             # Bomb Tower
    27000005: ("building", "spawner_building", ("spawner",), 4, 0, 0, 0, 0),              # Barbarian Hut
    27000006: ("building", "defensive_building", ("air_target",), 2, 3, 3, 0, 3),         # Tesla
    27000007: ("building", "spawner_building", (), 2, 0, 0, 0, 0),                        # Elixir Collector
    27000008: ("building", "win_condition", ("building_target",), 2, 3, 4, 0, 5),         # X-Bow
    27000009: ("building", "spawner_building", ("spawner",), 2, 0, 0, 0, 0),              # Tombstone
    27000010: ("building", "spawner_building", ("spawner", "splash"), 3, 0, 0, 0, 0),     # Furnace
    27000012: ("building", "defensive_building", ("spawner",), 3, 0, 0, 0, 0),            # Goblin Cage
    27000013: ("building", "win_condition", ("building_target", "spawner"), 2, 0, 0, 0, 0),  # Goblin Drill
    27000014: ("building", "spawner_building", ("spawner",), 3, 0, 0, 0, 0),              # Party Hut
    # --- Spells -----------------------------------------------------------
    28000000: ("spell", "damage_spell", ("splash", "air_target"), 0, 4, 0, 0, 3),         # Fireball
    28000001: ("spell", "damage_spell", ("splash", "air_target"), 0, 2, 0, 0, 4),         # Arrows
    28000002: ("spell", "utility_spell", (), 0, 0, 0, 0, 3),                              # Rage
    28000003: ("spell", "damage_spell", ("splash", "air_target"), 0, 5, 0, 0, 2),         # Rocket
    28000004: ("spell", "win_condition", ("building_target", "spawner"), 0, 3, 0, 0, 1),  # Goblin Barrel
    28000005: ("spell", "utility_spell", ("reset_control", "air_target", "splash"), 0, 0, 0, 0, 3),  # Freeze
    28000006: ("spell", "utility_spell", (), 0, 0, 0, 0, 0),                              # Mirror
    28000007: ("spell", "damage_spell", ("air_target", "reset_control"), 0, 5, 0, 0, 2),  # Lightning
    28000008: ("spell", "damage_spell", ("splash", "air_target", "reset_control"), 0, 1, 0, 0, 3),  # Zap
    28000009: ("spell", "damage_spell", ("splash", "air_target"), 0, 2, 0, 0, 4),         # Poison
    28000010: ("spell", "win_condition", ("building_target", "spawner"), 0, 2, 0, 0, 4),  # Graveyard
    28000011: ("spell", "damage_spell", ("splash", "reset_control"), 0, 2, 0, 0, 4),      # The Log
    28000012: ("spell", "utility_spell", ("reset_control", "air_target", "splash"), 0, 1, 0, 0, 4),  # Tornado
    28000013: ("spell", "utility_spell", (), 0, 0, 0, 0, 3),                              # Clone
    28000014: ("spell", "damage_spell", ("splash", "building_target"), 0, 2, 0, 0, 4),    # Earthquake
    28000015: ("spell", "damage_spell", ("splash", "spawner"), 0, 2, 0, 0, 3),            # Barbarian Barrel
    28000017: ("spell", "damage_spell", ("splash", "air_target", "reset_control"), 0, 1, 0, 0, 3),  # Giant Snowball
    28000018: ("spell", "damage_spell", ("splash", "spawner"), 0, 2, 0, 0, 3),            # Royal Delivery
    28000020: ("spell", "damage_spell", ("splash", "air_target"), 0, 4, 0, 0, 2),         # Party Rocket
    28000023: ("spell", "damage_spell", ("splash", "air_target"), 0, 3, 0, 0, 3),         # Void
    28000024: ("spell", "utility_spell", ("splash", "air_target", "spawner"), 0, 2, 0, 0, 4),  # Goblin Curse
    28000026: ("spell", "utility_spell", ("splash", "air_target", "reset_control"), 0, 2, 0, 0, 3),  # Vines
}

CARD_NAMES: dict[int, str] = {
    26000000: "Knight", 26000001: "Archers", 26000002: "Goblins", 26000003: "Giant",
    26000004: "P.E.K.K.A", 26000005: "Minions", 26000006: "Balloon", 26000007: "Witch",
    26000008: "Barbarians", 26000009: "Golem", 26000010: "Skeletons", 26000011: "Valkyrie",
    26000012: "Skeleton Army", 26000013: "Bomber", 26000014: "Musketeer", 26000015: "Baby Dragon",
    26000016: "Prince", 26000017: "Wizard", 26000018: "Mini P.E.K.K.A", 26000019: "Spear Goblins",
    26000020: "Giant Skeleton", 26000021: "Hog Rider", 26000022: "Minion Horde", 26000023: "Ice Wizard",
    26000024: "Royal Giant", 26000025: "Guards", 26000026: "Princess", 26000027: "Dark Prince",
    26000028: "Three Musketeers", 26000029: "Lava Hound", 26000030: "Ice Spirit", 26000031: "Fire Spirit",
    26000032: "Miner", 26000033: "Sparky", 26000034: "Bowler", 26000035: "Lumberjack",
    26000036: "Battle Ram", 26000037: "Inferno Dragon", 26000038: "Ice Golem", 26000039: "Mega Minion",
    26000040: "Dart Goblin", 26000041: "Goblin Gang", 26000042: "Electro Wizard", 26000043: "Elite Barbarians",
    26000044: "Hunter", 26000045: "Executioner", 26000046: "Bandit", 26000047: "Royal Recruits",
    26000048: "Night Witch", 26000049: "Bats", 26000050: "Royal Ghost", 26000051: "Ram Rider",
    26000052: "Zappies", 26000053: "Rascals", 26000054: "Cannon Cart", 26000055: "Mega Knight",
    26000056: "Skeleton Barrel", 26000057: "Flying Machine", 26000058: "Wall Breakers", 26000059: "Royal Hogs",
    26000060: "Goblin Giant", 26000061: "Fisherman", 26000062: "Magic Archer", 26000063: "Electro Dragon",
    26000064: "Firecracker", 26000065: "Mighty Miner", 26000066: "Super Witch", 26000067: "Elixir Golem",
    26000068: "Battle Healer", 26000069: "Skeleton King", 26000070: "Super Lava Hound",
    26000071: "Super Magic Archer", 26000072: "Archer Queen", 26000073: "Santa Hog Rider",
    26000074: "Golden Knight", 26000075: "Super Ice Golem", 26000077: "Monk", 26000078: "Super Archers",
    26000080: "Skeleton Dragons", 26000081: "Terry", 26000082: "Super Mini P.E.K.K.A", 26000083: "Mother Witch",
    26000084: "Electro Spirit", 26000085: "Electro Giant", 26000086: "Raging Prince", 26000087: "Phoenix",
    26000093: "Little Prince", 26000095: "Goblin Demolisher", 26000096: "Goblin Machine",
    26000097: "Suspicious Bush", 26000099: "Goblinstein", 26000101: "Rune Giant", 26000102: "Berserker",
    26000103: "Boss Bandit",
    27000000: "Cannon", 27000001: "Goblin Hut", 27000002: "Mortar", 27000003: "Inferno Tower",
    27000004: "Bomb Tower", 27000005: "Barbarian Hut", 27000006: "Tesla", 27000007: "Elixir Collector",
    27000008: "X-Bow", 27000009: "Tombstone", 27000010: "Furnace", 27000012: "Goblin Cage",
    27000013: "Goblin Drill", 27000014: "Party Hut",
    28000000: "Fireball", 28000001: "Arrows", 28000002: "Rage", 28000003: "Rocket",
    28000004: "Goblin Barrel", 28000005: "Freeze", 28000006: "Mirror", 28000007: "Lightning",
    28000008: "Zap", 28000009: "Poison", 28000010: "Graveyard", 28000011: "The Log",
    28000012: "Tornado", 28000013: "Clone", 28000014: "Earthquake", 28000015: "Barbarian Barrel",
    28000016: "Heal Spirit", 28000017: "Giant Snowball", 28000018: "Royal Delivery", 28000020: "Party Rocket",
    28000023: "Void", 28000024: "Goblin Curse", 28000025: "Spirit Empress", 28000026: "Vines",
}


# --- State-dependent effects ------------------------------------------------
# Champion always-on button ability, keyed by the champion's own card id.
# (cost in elixir to activate, effect tags). Sources: Clash Royale Wiki / RoyaleAPI.
CHAMPION_ABILITY: dict[int, tuple[int, tuple[str, ...]]] = {
    26000065: (1, ("ability_damage", "ability_dash")),     # Mighty Miner - Explosive Escape
    26000069: (2, ("ability_spawn",)),                     # Skeleton King - Soul Summoning
    26000072: (1, ("ability_shield", "ability_buff")),     # Archer Queen - Cloaking Cape
    26000074: (1, ("ability_dash",)),                      # Golden Knight - Dashing Dash
    26000077: (1, ("ability_shield", "ability_control")),  # Monk - Brushing Strike (deflect)
    # Terry: defunct temporary event champion, kept for historical battle data.
    # "Hoggy Height" -- leaps and slams for 360 area damage + knockback (1 elixir).
    26000081: (1, ("ability_damage", "ability_control")),  # Terry - Hoggy Height
    26000093: (3, ("ability_spawn",)),                     # Little Prince - Summon Guardian
    26000099: (2, ("ability_damage",)),                    # Goblinstein - electric surge
    26000103: (1, ("ability_dash",)),                      # Boss Bandit - reset + dash
}

# Hero button ability, keyed by the BASE card id; applies only when the card is
# fielded in hero form (heroLevel > 0). cost = elixir to activate the ability.
# Several ability costs are best-effort estimates -- verify against the wiki.
HERO_ABILITY: dict[int, tuple[int, tuple[str, ...]]] = {
    26000000: (2, ("ability_control",)),                   # Hero Knight - Royal Taunt
    26000003: (3, ("ability_control",)),                   # Hero Giant - Heroic Hurl
    26000018: (2, ("ability_buff",)),                      # Hero Mini P.E.K.K.A - Breakfast Boost
    26000014: (3, ("ability_spawn",)),                     # Hero Musketeer - Trusty Turret
    26000038: (2, ("ability_control",)),                   # Hero Ice Golem - Blizzard
    26000017: (1, ("ability_damage",)),                    # Hero Wizard - fire blast
    26000002: (2, ("ability_spawn",)),                     # Hero Goblins - Banner Brigade (respawn)
    26000039: (2, ("ability_dash", "ability_damage")),     # Hero Mega Minion - Wounding Warp
    28000015: (2, ("ability_damage",)),                    # Hero Barbarian Barrel - second roll
    26000062: (1, ("ability_damage",)),                    # Hero Magic Archer - Triple Threat
    26000006: (2, ("ability_spawn",)),                     # Hero Balloon - Coffin Cadets
    26000027: (3, ("ability_damage", "ability_dash")),     # Hero Dark Prince - Destructive Dismount
    26000034: (2, ("ability_damage",)),                    # Hero Bowler - Stone Swish
    27000009: (6, ("ability_spawn",)),                     # Hero Tombstone - Regal Revival
}

# Evolution effects, keyed by base card id; apply only when fielded evolved
# (evolutionLevel > 0). (cycle = cycles to charge the evo, effect tags). Covers
# the evos curated with confidence; any other evolved card still gets the generic
# ``evolved`` flag at runtime. cycles/tags are best-effort -- verify.
EVO_EFFECT: dict[int, tuple[int, tuple[str, ...]]] = {
    26000000: (2, ("evo_shield", "evo_control")),    # Evo Knight - shield + taunt/pull troops
    26000001: (2, ("evo_damage",)),                  # Evo Archers - Power Shot + range
    26000004: (2, ("evo_buff",)),                    # Evo P.E.K.K.A - heals on kill
    26000008: (2, ("evo_buff",)),                    # Evo Barbarians - +HP, speed, attack speed
    26000010: (1, ("evo_spawn", "evo_buff")),        # Evo Skeletons - extra skeleton + atk speed
    26000011: (2, ("evo_splash", "evo_control")),    # Evo Valkyrie - tornado lures troops in
    26000012: (2, ("evo_spawn",)),                   # Evo Skeleton Army - ghosts survive
    26000013: (2, ("evo_splash", "evo_damage")),     # Evo Bomber - bomb bounces twice
    26000014: (2, ("evo_damage",)),                  # Evo Musketeer - long-range triple shot
    26000015: (2, ("evo_control", "evo_buff")),      # Evo Baby Dragon - gusts slow foes/speed allies
    26000017: (2, ("evo_shield", "evo_splash")),     # Evo Wizard - shield + bigger splash
    26000024: (2, ("evo_control", "evo_damage")),    # Evo Royal Giant - knockback on every shot
    26000030: (1, ("evo_control",)),                 # Evo Ice Spirit - freezes twice
    26000035: (2, ("evo_spawn", "evo_buff")),        # Evo Lumberjack - rage ghost on death
    26000044: (2, ("evo_control",)),                 # Evo Hunter - net slows/stuns tanks
    26000045: (2, ("evo_control",)),                 # Evo Executioner - axe pushes + pulls
    26000047: (3, ("evo_shield", "evo_charge")),     # Evo Royal Recruits - shield + dash
    26000049: (1, ("evo_buff",)),                    # Evo Bats - lifesteal / heal
    26000050: (2, ("evo_spawn",)),                   # Evo Royal Ghost - spawns 2 Souldiers
    26000055: (2, ("evo_splash", "evo_control")),    # Evo Mega Knight - jump + knockback
    26000056: (2, ("evo_damage",)),                  # Evo Skeleton Barrel - destroys tower intact
    26000059: (2, ("evo_buff",)),                    # Evo Royal Hogs - gain flight
    26000064: (2, ("evo_splash", "evo_damage")),     # Evo Firecracker - lingering AoE field
    27000002: (3, ("evo_spawn", "evo_damage")),      # Evo Mortar - spawns Goblins on hit
    27000010: (2, ("evo_spawn",)),                   # Evo Furnace - spawns Fire Spirit each attack
    28000004: (2, ("evo_spawn",)),                   # Evo Goblin Barrel - spawns two barrels
    28000008: (2, ("evo_control", "evo_damage")),    # Evo Zap - double zap, second stuns wider
}


def _ability_payload(table: dict[int, tuple[int, tuple[str, ...]]], card_id: int):
    entry = table.get(card_id)
    if entry is None:
        return None
    cost, tags = entry
    bad = [t for t in tags if t not in CARD_ABILITY_TAGS]
    if bad:
        raise ValueError(f"{card_id}: bad ability tags {bad}")
    return {"cost": int(cost), "tags": list(tags)}


def _evo_payload(card_id: int):
    entry = EVO_EFFECT.get(card_id)
    if entry is None:
        return None
    cycle, tags = entry
    bad = [t for t in tags if t not in CARD_EVO_TAGS]
    if bad:
        raise ValueError(f"{card_id}: bad evo tags {bad}")
    return {"cycle": int(cycle), "tags": list(tags)}


def _slug(name: str) -> str:
    value = name.lower().replace(".", "").replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value


def _build_card(card_id: int) -> dict[str, Any]:
    card_type, role, flags, hp, dmg, dps, speed, rng = CARD_TABLE[card_id]
    if card_type not in CARD_METADATA_TYPE_NAMES:
        raise ValueError(f"{card_id}: bad type {card_type!r}")
    if role not in CARD_METADATA_ROLES:
        raise ValueError(f"{card_id}: bad role {role!r}")
    bad_flags = [flag for flag in flags if flag not in CARD_METADATA_FLAGS]
    if bad_flags:
        raise ValueError(f"{card_id}: bad flags {bad_flags}")
    # Champion flag must agree with the authoritative rarity-derived set.
    champion = card_id in CHAMPION_CARD_IDS
    if champion and "champion" not in flags:
        raise ValueError(f"{card_id}: champion id missing 'champion' flag")
    if "champion" in flags and not champion:
        raise ValueError(f"{card_id}: 'champion' flag but not in CHAMPION_CARD_IDS")

    nudge = NUDGES.get(card_id, {})
    bad_nudge = set(nudge) - {"hitpoints", "damage", "dps", "range"}
    if bad_nudge:
        raise ValueError(f"{card_id}: nudge on unsupported field(s) {bad_nudge}")

    def grid(tier: int, field: str) -> int:
        """Expand a 0-5 tier onto the 0-GRID grid, then apply a per-card nudge."""
        value = tier * GRID_STEP + nudge.get(field, 0)
        return max(0, min(GRID, value))

    elixir = CARD_ELIXIR.get(card_id, 0)
    grid_values = {
        "hitpoints": grid(hp, "hitpoints"),
        "damage": grid(dmg, "damage"),
        "dps": grid(dps, "dps"),
        "range": grid(rng, "range"),
        "speed": speed * SPEED_STEP,  # 0-8, no nudges
    }
    numeric = {
        "elixir": round(elixir / MAX_ELIXIR, 4),
        "hitpoints": round(grid_values["hitpoints"] / GRID, 4),
        "damage": round(grid_values["damage"] / GRID, 4),
        "dps": round(grid_values["dps"] / GRID, 4),
        "speed": round(grid_values["speed"] / SPEED_GRID, 4),
        "range": round(grid_values["range"] / GRID, 4),
    }
    missing = set(CARD_METADATA_NUMERIC_FEATURES) - set(numeric)
    if missing:
        raise ValueError(f"{card_id}: numeric missing {missing}")

    tags = (role, *flags)
    champion_ability = _ability_payload(CHAMPION_ABILITY, card_id)
    if champion and champion_ability is None:
        raise ValueError(f"{card_id}: champion missing CHAMPION_ABILITY entry")
    if champion_ability is not None and not champion:
        raise ValueError(f"{card_id}: CHAMPION_ABILITY entry but not a champion")
    card: dict[str, Any] = {
        "name": CARD_NAMES[card_id],
        "key": _slug(CARD_NAMES[card_id]),
        "type": card_type,
        "role": role,
        "tags": list(tags),
        "tiers": {  # coarse 0-5 anchors (reviewable, patch-stable)
            "hitpoints": hp, "damage": dmg, "dps": dps, "speed": speed, "range": rng,
            "elixir": elixir,
        },
        "grid": grid_values,  # fine 0-20 (speed 0-8) after nudges
        "numeric": numeric,
    }
    if champion_ability is not None:
        card["champion_ability"] = champion_ability
    hero_ability = _ability_payload(HERO_ABILITY, card_id)
    if hero_ability is not None:
        card["hero_ability"] = hero_ability
    evo = _evo_payload(card_id)
    if evo is not None:
        card["evo"] = evo
    return card


def main() -> None:
    missing_in_table = sorted(set(CARD_ELIXIR) - set(CARD_TABLE))
    extra_in_table = sorted(set(CARD_TABLE) - set(CARD_ELIXIR))
    if missing_in_table:
        names = ", ".join(str(cid) for cid in missing_in_table)
        raise SystemExit(f"CARD_TABLE missing ids present in CARD_ELIXIR: {names}")
    if extra_in_table:
        raise SystemExit(f"CARD_TABLE has ids absent from CARD_ELIXIR: {extra_in_table}")
    if set(CARD_NAMES) != set(CARD_TABLE):
        raise SystemExit("CARD_NAMES and CARD_TABLE id sets differ")
    for label, table in (
        ("HERO_ABILITY", HERO_ABILITY),
        ("EVO_EFFECT", EVO_EFFECT),
        ("CHAMPION_ABILITY", CHAMPION_ABILITY),
    ):
        unknown = sorted(set(table) - set(CARD_TABLE))
        if unknown:
            raise SystemExit(f"{label} has ids absent from CARD_TABLE: {unknown}")

    cards = {str(card_id): _build_card(card_id) for card_id in sorted(CARD_TABLE)}
    snapshot = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "method": "hand-curated taxonomy (offline, no scraping)",
            "generator": "scripts/refresh_card_metadata.py::CARD_TABLE",
        },
        "schema": {
            "types": list(CARD_METADATA_TYPE_NAMES),
            "roles": list(CARD_METADATA_ROLES),
            "flags": list(CARD_METADATA_FLAGS),
            "numeric_features": list(CARD_METADATA_NUMERIC_FEATURES),
            "ability_tags": list(CARD_ABILITY_TAGS),
            "evo_tags": list(CARD_EVO_TAGS),
            "grid_scales": {
                "hitpoints": GRID, "damage": GRID, "dps": GRID, "range": GRID,
                "speed": SPEED_GRID, "elixir": MAX_ELIXIR,
            },
        },
        "cards": cards,
    }
    OUT_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(cards)} cards -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
