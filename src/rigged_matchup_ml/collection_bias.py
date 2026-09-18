"""Sampling-bias audit of the collected raw corpus.

`collect-api` decides *which* players it crawls (balanced frontier, `min_trophies`
cut, freshness window, snowball from opponents). Every one of those decisions can
skew the corpus in a way the training metrics never surface: a model trained on a
biased corpus still reports a fine AUC, it is simply confident about the wrong
population.

This module reads the raw Parquet shards -- never the collector's internals -- so
it keeps working when the collector is rewritten, and it can audit a corpus pulled
from Storage on a machine that never ran `collect-api`.

The checks, and what a failure actually means:

- side balance: raw records are oriented "crawled player = team". Battlelogs carry
  the player's wins *and* losses, so the team-side win rate must sit at 0.5. Any
  real deviation means the orientation correlates with selection, and since the
  model is antisymmetric it cannot absorb that -- it becomes irreducible noise.
- orientation drift: the card distribution of the team side versus the opponent
  side. Crawled players are steered by the balancer, their opponents are not, so a
  large total-variation distance means the two sides are not samples of the same
  population.
- segment coverage: which tracked trophy bands are starved or missing entirely --
  the same question `--balance` asks at startup, asked after the fact.
- freshness: battles older than the collection window, which silently blend two
  metas into one corpus.
- concentration: deck/matchup mass piled on a few entries, the signature of a
  snowball crawl that walked a dense cluster (a clan, a region) instead of the meta.
- card support: cards of the static table with little or no data, which end up
  predicted from metadata alone.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from .api_collect import _ladder_bucket_label, _normalize_baseline, _segment_tracked
from .card_stats import CARD_ELIXIR
from .config import AppConfig

# Practical-significance thresholds. On millions of rows every deviation is
# statistically significant, so a bare p-value would flag noise: each check needs
# both a 95% interval excluding the neutral value *and* an effect size past these.
WIN_RATE_TOLERANCE = 0.01
ORIENTATION_TVD_TOLERANCE = 0.02
SEGMENT_IMBALANCE_RATIO = 10.0
TOP_DECK_SHARE_TOLERANCE = 0.25
MIRROR_SHARE_TOLERANCE = 0.02
SEGMENT_MIN_ROWS = 5000
DEFAULT_MIN_CARD_SUPPORT = 1000
DEFAULT_TOP_N = 20

# Mirrors `unseen_evaluation.matchup_key`: A vs B and B vs A collapse together.
_MATCHUP_KEY_SQL = (
    "case when team_deck_key <= opponent_deck_key "
    "then team_deck_key || '::' || opponent_deck_key "
    "else opponent_deck_key || '::' || team_deck_key end"
)


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion (stable at the tiny rates we check)."""
    if total <= 0:
        return (0.0, 0.0)
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (centre - margin, centre + margin)


def _gini(values: list[int]) -> float:
    """0 = every band holds the same number of battles, 1 = one band holds them all."""
    data = sorted(int(value) for value in values)
    count = len(data)
    total = sum(data)
    if count == 0 or total == 0:
        return 0.0
    cumulative = sum((index + 1) * value for index, value in enumerate(data))
    return (2 * cumulative) / (count * total) - (count + 1) / count


def _skewed(successes: int, total: int, neutral: float, tolerance: float) -> bool:
    """True when the rate is both statistically and practically away from `neutral`."""
    if total <= 0:
        return False
    rate = successes / total
    low, high = _wilson_interval(successes, total)
    return abs(rate - neutral) >= tolerance and not (low <= neutral <= high)


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def collection_bias_report(
    config: AppConfig,
    raw_dir: Path | None = None,
    min_card_support: int = DEFAULT_MIN_CARD_SUPPORT,
    top_n: int = DEFAULT_TOP_N,
    include_cards: bool = True,
) -> dict[str, Any]:
    """Audit the raw shards for sampling bias and return a JSON-ready report."""
    directory = raw_dir if raw_dir is not None else config.resolve(config.data["raw_dir"])
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"No raw Parquet shards found in {directory}")

    buckets = [int(edge) for edge in config.data["trophy_buckets"]]
    min_trophies = int(config.data.get("collect_min_trophies", 5000))
    window_days = float(config.data.get("collect_max_age_days", 2) or 0)
    raw_glob = _quoted(directory / "*.parquet")

    connection = duckdb.connect()
    try:
        connection.execute("set preserve_insertion_order=false")
        connection.execute(
            f"create or replace temp view battles as select * from read_parquet('{raw_glob}')"
        )
        row_count, distinct_games, wins, missing_time, first_battle, last_battle = (
            connection.execute(
                """
                select
                    count(*) as row_count,
                    count(distinct game_id) as distinct_games,
                    sum(case when win then 1 else 0 end) as wins,
                    count(*) filter (where battle_time is null) as missing_time,
                    min(battle_time) as first_battle,
                    max(battle_time) as last_battle
                from battles
                """
            ).fetchone()
        )
        row_count = int(row_count)
        if row_count == 0:
            raise RuntimeError(f"Raw shards in {directory} hold no rows")
        wins = int(wins or 0)

        segment_rows = connection.execute(
            """
            select segment, count(*) as row_count, sum(case when win then 1 else 0 end) as wins
            from battles
            group by segment
            order by row_count desc
            """
        ).fetchall()
        mode_rows = connection.execute(
            """
            select mode_key, count(*) as row_count
            from battles
            group by mode_key
            order by row_count desc
            """
        ).fetchall()
        day_rows = connection.execute(
            """
            select cast(battle_time as date) as day, count(*) as row_count
            from battles
            where battle_time is not null
            group by day
            order by day
            """
        ).fetchall()

        outside_window = 0
        if window_days > 0 and isinstance(last_battle, datetime):
            cutoff = last_battle - timedelta(days=window_days)
            outside_window = int(
                connection.execute(
                    "select count(*) from battles where battle_time < ?", [cutoff]
                ).fetchone()[0]
            )

        connection.execute(
            """
            create or replace temp view deck_slots as
            select team_deck_key as deck_key from battles
            union all
            select opponent_deck_key as deck_key from battles
            """
        )
        deck_slots, distinct_decks = connection.execute(
            "select count(*), count(distinct deck_key) from deck_slots"
        ).fetchone()
        top_decks = connection.execute(
            f"""
            select deck_key, count(*) as row_count
            from deck_slots
            group by deck_key
            order by row_count desc
            limit {int(top_n)}
            """
        ).fetchall()
        distinct_matchups, mirror_share = connection.execute(
            f"""
            select
                count(distinct {_MATCHUP_KEY_SQL}),
                avg(case when team_deck_key = opponent_deck_key then 1.0 else 0.0 end)
            from battles
            """
        ).fetchone()
        top_matchups = connection.execute(
            f"""
            select {_MATCHUP_KEY_SQL} as matchup_key, count(*) as row_count
            from battles
            group by matchup_key
            order by row_count desc
            limit {int(top_n)}
            """
        ).fetchall()

        card_rows: list[tuple[Any, Any, Any]] = []
        if include_cards:
            card_rows = connection.execute(
                """
                select
                    card_id,
                    sum(team_count) as team_count,
                    sum(opponent_count) as opponent_count
                from (
                    select unnest(team_card_ids) as card_id,
                           1 as team_count,
                           0 as opponent_count
                    from battles
                    union all
                    select unnest(opponent_card_ids) as card_id,
                           0 as team_count,
                           1 as opponent_count
                    from battles
                )
                group by card_id
                order by card_id
                """
            ).fetchall()
    finally:
        connection.close()

    warnings: list[dict[str, Any]] = []

    def flag(code: str, message: str, **details: Any) -> None:
        warnings.append({"code": code, "message": message, **details})

    # --- side balance -------------------------------------------------------
    win_rate = wins / row_count
    win_low, win_high = _wilson_interval(wins, row_count)
    if _skewed(wins, row_count, 0.5, WIN_RATE_TOLERANCE):
        flag(
            "side_balance",
            f"Team-side win rate is {win_rate:.4f}, not 0.5. Battlelogs contain a player's "
            "wins and losses alike, so a real gap means the crawl selects players by "
            "outcome. The antisymmetric model cannot fit that skew.",
            win_rate=win_rate,
        )

    segments: list[dict[str, Any]] = []
    skewed_segments: list[str] = []
    for segment, segment_count, segment_wins in segment_rows:
        segment = str(segment)
        segment_count = int(segment_count)
        segment_wins = int(segment_wins or 0)
        low, high = _wilson_interval(segment_wins, segment_count)
        entry = {
            "segment": segment,
            "rows": segment_count,
            "share": segment_count / row_count,
            "win_rate": segment_wins / segment_count if segment_count else None,
            "win_rate_ci_95": [low, high],
            "tracked": _segment_tracked(segment, min_trophies),
        }
        if segment_count >= SEGMENT_MIN_ROWS and _skewed(
            segment_wins, segment_count, 0.5, WIN_RATE_TOLERANCE
        ):
            skewed_segments.append(segment)
            entry["win_rate_skewed"] = True
        segments.append(entry)
    if skewed_segments:
        flag(
            "segment_side_balance",
            "Team-side win rate is off 0.5 in: " + ", ".join(skewed_segments[:10]),
            segments=skewed_segments,
        )

    # --- segment coverage ---------------------------------------------------
    raw_counts = Counter({str(row[0]): int(row[1]) for row in segment_rows})
    aligned = _normalize_baseline(raw_counts, buckets)
    tracked_bands = [
        label
        for lower in buckets[:-1]
        if lower >= min_trophies and (label := _ladder_bucket_label(lower, buckets)) is not None
    ]
    # `ranked` is pooled here exactly like BalancedFrontier pools it: Path of
    # Legends hides trophies, so the crawl cannot steer inside it.
    band_counts = {band: int(aligned.get(band, 0)) for band in tracked_bands}
    band_counts["ranked"] = sum(
        count for segment, count in aligned.items() if segment.startswith("ranked:")
    )
    missing_bands = [band for band, count in band_counts.items() if count == 0]
    populated = [count for count in band_counts.values() if count > 0]
    imbalance_ratio = max(populated) / min(populated) if populated else None
    legacy_folded = sorted(set(raw_counts) - set(aligned))
    untracked_rows = sum(
        count
        for segment, count in raw_counts.items()
        if not _segment_tracked(segment, min_trophies)
    )

    if missing_bands:
        flag(
            "missing_bands",
            "Tracked bands with zero battles: " + ", ".join(missing_bands),
            bands=missing_bands,
        )
    if imbalance_ratio is not None and imbalance_ratio > SEGMENT_IMBALANCE_RATIO:
        flag(
            "segment_imbalance",
            f"The fullest tracked band holds {imbalance_ratio:.1f}x the emptiest one. The "
            "model will be confident on the saturated band and guess on the starved ones.",
            ratio=imbalance_ratio,
        )
    if untracked_rows:
        flag(
            "untracked_rows",
            f"{untracked_rows:,} battles sit in segments the collector no longer keeps "
            f"(sub-{min_trophies} ladder or unplaceable). They still reach training.",
            rows=untracked_rows,
        )

    # --- orientation drift --------------------------------------------------
    support: dict[int, int] = {}
    team_total = sum(int(row[1]) for row in card_rows)
    opponent_total = sum(int(row[2]) for row in card_rows)
    divergences: list[dict[str, Any]] = []
    orientation_tvd: float | None = None
    if card_rows and team_total and opponent_total:
        total_variation = 0.0
        for card_id, team_count, opponent_count in card_rows:
            team_count = int(team_count)
            opponent_count = int(opponent_count)
            support[int(card_id)] = team_count + opponent_count
            delta = team_count / team_total - opponent_count / opponent_total
            total_variation += abs(delta)
            divergences.append(
                {
                    "card_id": int(card_id),
                    "team_share_minus_opponent_share": delta,
                    "appearances": team_count + opponent_count,
                }
            )
        orientation_tvd = total_variation / 2
        divergences.sort(key=lambda item: -abs(item["team_share_minus_opponent_share"]))
        if orientation_tvd > ORIENTATION_TVD_TOLERANCE:
            flag(
                "orientation_drift",
                f"Crawled-side and opponent-side card distributions differ by "
                f"{orientation_tvd:.4f} (total variation). The two sides are not drawn from "
                "the same population, so side-inversion augmentation is hiding a real skew.",
                total_variation_distance=orientation_tvd,
            )

    # --- concentration ------------------------------------------------------
    deck_slots = int(deck_slots)
    top_deck_rows = sum(int(row[1]) for row in top_decks)
    top_deck_share = top_deck_rows / deck_slots if deck_slots else 0.0
    mirror_share = float(mirror_share or 0.0)
    top_matchup_share = sum(int(row[1]) for row in top_matchups) / row_count
    if top_deck_share > TOP_DECK_SHARE_TOLERANCE:
        flag(
            "deck_concentration",
            f"The {len(top_decks)} most common decks cover {top_deck_share:.1%} of deck slots. "
            "A snowball crawl that walks one dense cluster looks exactly like this.",
            share=top_deck_share,
        )
    if mirror_share > MIRROR_SHARE_TOLERANCE:
        flag(
            "mirror_matches",
            f"{mirror_share:.2%} of battles are deck mirrors. They carry no matchup signal "
            "and inflate the apparent support of the decks involved.",
            share=mirror_share,
        )

    # --- freshness ----------------------------------------------------------
    if outside_window:
        flag(
            "stale_battles",
            f"{outside_window:,} battles are older than the {window_days:g}-day collection "
            "window measured from the newest battle. Two metas are mixed in this corpus.",
            rows=outside_window,
        )
    if missing_time:
        flag(
            "missing_battle_time",
            f"{int(missing_time):,} battles have no battle_time. The chronological split "
            "cannot place them, and they would be dated to now().",
            rows=int(missing_time),
        )

    duplicate_games = row_count - int(distinct_games)
    if duplicate_games > 0:
        flag(
            "duplicate_games",
            f"{duplicate_games:,} rows repeat a game_id already present. Deduplication "
            "leaked, and those matchups now weigh more than they should.",
            rows=duplicate_games,
        )

    # --- card support -------------------------------------------------------
    card_report: dict[str, Any] = {"checked": include_cards}
    if include_cards:
        known_cards = sorted(CARD_ELIXIR)
        absent = [card for card in known_cards if support.get(card, 0) == 0]
        thin = [
            {"card_id": card, "appearances": support[card]}
            for card in known_cards
            if 0 < support.get(card, 0) < min_card_support
        ]
        card_report.update(
            {
                "cards_in_static_table": len(known_cards),
                "cards_with_data": sum(1 for card in known_cards if support.get(card, 0) > 0),
                "min_support": min_card_support,
                "absent_cards": absent,
                "thin_cards": thin,
                "unknown_card_ids": sorted(set(support) - set(known_cards)),
            }
        )
        if absent:
            flag(
                "cards_absent",
                f"{len(absent)} cards of the static table never appear in the corpus: "
                + ", ".join(str(card) for card in absent[:10]),
                cards=absent,
            )
        if thin:
            flag(
                "cards_thin",
                f"{len(thin)} cards appear fewer than {min_card_support:,} times; their "
                "predictions rest on metadata, not on observed results.",
                cards=[entry["card_id"] for entry in thin],
            )

    return {
        "raw_dir": str(directory),
        "shards": len(shards),
        "rows": row_count,
        "distinct_games": int(distinct_games),
        "duplicate_game_ids": duplicate_games,
        "thresholds": {
            "win_rate_tolerance": WIN_RATE_TOLERANCE,
            "orientation_tvd_tolerance": ORIENTATION_TVD_TOLERANCE,
            "segment_imbalance_ratio": SEGMENT_IMBALANCE_RATIO,
            "top_deck_share_tolerance": TOP_DECK_SHARE_TOLERANCE,
            "mirror_share_tolerance": MIRROR_SHARE_TOLERANCE,
            "segment_min_rows": SEGMENT_MIN_ROWS,
            "min_card_support": min_card_support,
        },
        "side_balance": {
            "wins": wins,
            "win_rate": win_rate,
            "win_rate_ci_95": [win_low, win_high],
            "deviation_from_even": win_rate - 0.5,
        },
        "orientation": {
            "card_distribution_tvd": orientation_tvd,
            "most_divergent_cards": divergences[:top_n],
        },
        "segments": {
            "min_trophies": min_trophies,
            "tracked_bands": tracked_bands,
            "band_rows": band_counts,
            "missing_bands": missing_bands,
            "imbalance_ratio": imbalance_ratio,
            "gini": _gini(list(band_counts.values())),
            "untracked_rows": untracked_rows,
            "legacy_bands_folded": legacy_folded,
            "rows_by_segment": segments,
        },
        "modes": [
            {"mode_key": str(mode), "rows": int(count), "share": int(count) / row_count}
            for mode, count in mode_rows
        ],
        "freshness": {
            "first_battle": _iso(first_battle),
            "last_battle": _iso(last_battle),
            "window_days": window_days,
            "rows_outside_window": outside_window,
            "rows_without_battle_time": int(missing_time),
            "rows_by_day": [
                {"day": str(day), "rows": int(count)} for day, count in day_rows
            ],
        },
        "concentration": {
            "deck_slots": deck_slots,
            "distinct_decks": int(distinct_decks),
            "top_deck_share": top_deck_share,
            "top_decks": [
                {"deck_key": str(deck), "slots": int(count)} for deck, count in top_decks
            ],
            "distinct_matchups": int(distinct_matchups),
            "top_matchup_share": top_matchup_share,
            "top_matchups": [
                {"matchup_key": str(key), "rows": int(count)} for key, count in top_matchups
            ],
            "mirror_share": mirror_share,
        },
        "card_support": card_report,
        "warnings": warnings,
        "status": "warn" if warnings else "ok",
    }
