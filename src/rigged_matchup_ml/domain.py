from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


ROLE_NORMAL = 1
ROLE_CHAMPION = 2
ROLE_HERO = 3


@dataclass(frozen=True)
class ParsedCard:
    card_id: int
    evolution_level: int
    hero_level: int
    role: int
    raw_level: int | None


@dataclass(frozen=True)
class Deck:
    cards: tuple[ParsedCard, ...]
    tower_troop_id: int
    tag: str
    crowns: int
    starting_trophies: int | None
    global_rank: int | None

    @property
    def key(self) -> str:
        return ",".join(str(card.card_id) for card in sorted(self.cards, key=lambda c: c.card_id))


def _as_datetime(value: datetime | str | None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value:
        text = str(value).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return datetime.strptime(str(value), "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _card_role(card: dict[str, Any]) -> int:
    rarity = str(card.get("rarity", "")).lower()
    if rarity == "champion" or card.get("isChampion") is True:
        return ROLE_CHAMPION
    if (
        rarity == "hero"
        or card.get("isHero") is True
        or int(card.get("heroLevel") or card.get("hero_level") or 0) > 0
    ):
        return ROLE_HERO
    return ROLE_NORMAL


def parse_deck(player: dict[str, Any]) -> Deck | None:
    raw_cards = player.get("cards") or []
    cards: list[ParsedCard] = []
    for raw_card in raw_cards:
        try:
            card_id = int(raw_card["id"])
        except (KeyError, TypeError, ValueError):
            return None
        cards.append(
            ParsedCard(
                card_id=card_id,
                evolution_level=max(0, int(raw_card.get("evolutionLevel") or 0)),
                hero_level=max(
                    0, int(raw_card.get("heroLevel") or raw_card.get("hero_level") or 0)
                ),
                role=_card_role(raw_card),
                raw_level=int(raw_card["level"]) if raw_card.get("level") is not None else None,
            )
        )
    support_cards = player.get("supportCards") or []
    tower_id = int(support_cards[0].get("id") or 0) if support_cards else 0
    return Deck(
        cards=tuple(cards),
        tower_troop_id=tower_id,
        tag=str(player.get("tag") or ""),
        crowns=int(player.get("crowns") or 0),
        starting_trophies=_optional_int(player.get("startingTrophies")),
        global_rank=_optional_int(player.get("globalRank")),
    )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _find_first_numeric_key(value: Any, names: set[str]) -> int | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in names:
                parsed = _optional_int(child)
                if parsed is not None and parsed > 0:
                    return parsed
        for child in value.values():
            parsed = _find_first_numeric_key(child, names)
            if parsed is not None:
                return parsed
    elif isinstance(value, list):
        for child in value:
            parsed = _find_first_numeric_key(child, names)
            if parsed is not None:
                return parsed
    return None


def canonical_game_id(battle_time: datetime, team: Deck, opponent: Deck) -> str:
    sides = sorted(
        [
            (team.tag, team.key, team.crowns),
            (opponent.tag, opponent.key, opponent.crowns),
        ]
    )
    payload = json.dumps(
        [battle_time.astimezone(timezone.utc).isoformat(), sides],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ranked_league_number(raw: dict[str, Any]) -> int | None:
    candidates = [
        raw.get("leagueNumber"),
        raw.get("league_number"),
        (raw.get("currentPathOfLegendSeasonResult") or {}).get("leagueNumber")
        if isinstance(raw.get("currentPathOfLegendSeasonResult"), dict)
        else None,
    ]
    for side in ("team", "opponent"):
        participants = raw.get(side) or []
        if participants and isinstance(participants[0], dict):
            candidates.extend(
                [
                    participants[0].get("leagueNumber"),
                    participants[0].get("league_number"),
                ]
            )
    for value in candidates:
        league = _optional_int(value)
        if league is not None and league > 0:
            return league
    return _find_first_numeric_key(raw, {"leagueNumber", "league_number"})


def ranked_segment(raw: dict[str, Any]) -> str:
    league = ranked_league_number(raw)
    return f"ranked:league-{league}" if league is not None else "ranked:unknown"


def segment_for(
    deck: Deck,
    mode_key: str,
    data_config: dict[str, Any],
    raw: dict[str, Any] | None = None,
) -> str:
    mode = (mode_key or "other").lower()
    if mode == "ranked":
        return ranked_segment(raw or {})
    if mode != "ladder":
        return mode
    rank = deck.global_rank
    if rank is not None and rank > 0:
        for upper in data_config["top_ladder_buckets"]:
            if rank <= upper:
                return f"ladder:top-{upper}"
        return "ladder:ranked-other"
    trophies = deck.starting_trophies
    if trophies is None:
        return "ladder:unknown"
    buckets = data_config["trophy_buckets"]
    for lower, upper in zip(buckets, buckets[1:], strict=True):
        if lower <= trophies < upper:
            return f"ladder:{lower}-{upper - 1}"
    return "ladder:overflow"


def raw_average_level(deck: Deck) -> float | None:
    levels = [card.raw_level for card in deck.cards if card.raw_level is not None]
    return sum(levels) / len(levels) if levels else None


def parse_battle_row(row: dict[str, Any], data_config: dict[str, Any]) -> dict[str, Any] | None:
    raw = row["raw"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    teams = raw.get("team") or []
    opponents = raw.get("opponent") or []
    if len(teams) != 1 or len(opponents) != 1:
        return None
    team = parse_deck(teams[0])
    opponent = parse_deck(opponents[0])
    if team is None or opponent is None or team.crowns == opponent.crowns:
        return None
    if data_config["require_exactly_eight_cards"] and (
        len(team.cards) != 8 or len(opponent.cards) != 8
    ):
        return None
    mode_key = str(row.get("mode_key") or "other")
    allowed = set(data_config.get("allowed_modes") or [])
    if allowed and mode_key not in allowed:
        return None
    max_level_diff = data_config.get("max_raw_average_level_difference")
    team_level = raw_average_level(team)
    opponent_level = raw_average_level(opponent)
    if (
        max_level_diff is not None
        and team_level is not None
        and opponent_level is not None
        and abs(team_level - opponent_level) > float(max_level_diff)
    ):
        return None
    battle_time = _as_datetime(row.get("battle_time") or raw.get("battleTime"))
    inserted_at = _as_datetime(row.get("inserted_at"))
    sql_league = _optional_int(row.get("league_number"))
    if mode_key == "ranked" and sql_league is not None and sql_league > 0:
        segment = f"ranked:league-{sql_league}"
    else:
        segment = segment_for(team, mode_key, data_config, raw)
    return {
        "game_id": canonical_game_id(battle_time, team, opponent),
        "source_fingerprint": str(row["fingerprint"]),
        "battle_time": battle_time,
        "inserted_at": inserted_at,
        "mode_key": mode_key,
        "segment": segment,
        "patch": battle_time.strftime("%Y-%m"),
        "team_card_ids": [card.card_id for card in team.cards],
        "opponent_card_ids": [card.card_id for card in opponent.cards],
        "team_evolution_levels": [card.evolution_level for card in team.cards],
        "opponent_evolution_levels": [card.evolution_level for card in opponent.cards],
        "team_hero_levels": [card.hero_level for card in team.cards],
        "opponent_hero_levels": [card.hero_level for card in opponent.cards],
        "team_card_roles": [card.role for card in team.cards],
        "opponent_card_roles": [card.role for card in opponent.cards],
        "team_tower_troop_id": team.tower_troop_id,
        "opponent_tower_troop_id": opponent.tower_troop_id,
        "team_deck_key": team.key,
        "opponent_deck_key": opponent.key,
        "matrix_prior": 0.5,
        "win": team.crowns > opponent.crowns,
    }
