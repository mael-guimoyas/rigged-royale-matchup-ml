from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

from rigged_matchup_ml.card_stats import CHAMPION_CARD_IDS
from rigged_matchup_ml.domain import ROLE_CHAMPION, ROLE_NORMAL
from rigged_matchup_ml.predictor import load_bundle, predict_from_rows

SIGMA_THIN = 0.12
SIGMA_SUPPORTED = 0.05
SUPPORT_CUTOFF = 300
DEFAULT_K_VALUES = (0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 1.96, 2.25, 2.5, 3.0)
TOWER_PRINCESS_ID = 159000000
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
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


@dataclass
class DeckStat:
    segment: str
    key: str
    cards: tuple[int, ...]
    train_n: int
    validation_n: int
    validation_wins: int
    validation_team_n: int
    validation_team_wins: int
    validation_opponent_n: int
    validation_opponent_wins: int
    test_n: int
    test_wins: int
    test_team_n: int
    test_team_wins: int
    test_opponent_n: int
    test_opponent_wins: int
    train_rank: int
    model_score: float | None = None
    model_matchup_sigma: float | None = None
    model_unfavorable_share: float | None = None
    model_hard_counter_share: float | None = None
    model_worst_matchup: float | None = None
    validation_team_residual_sum: float = 0.0
    validation_team_residual_n: int = 0
    validation_opponent_residual_sum: float = 0.0
    validation_opponent_residual_n: int = 0
    test_team_residual_sum: float = 0.0
    test_team_residual_n: int = 0
    test_opponent_residual_sum: float = 0.0
    test_opponent_residual_n: int = 0

    @property
    def sigma(self) -> float:
        return SIGMA_SUPPORTED if self.train_n >= SUPPORT_CUTOFF else SIGMA_THIN

    def observed(self, split: str, prior_strength: float) -> float:
        if split == "validation":
            values = (
                self.validation_team_wins,
                self.validation_team_n,
                self.validation_opponent_wins,
                self.validation_opponent_n,
            )
        else:
            values = (
                self.test_team_wins,
                self.test_team_n,
                self.test_opponent_wins,
                self.test_opponent_n,
            )
        # A deck does not occur equally often on the player-log side and the
        # opponent side. Pooling both orientations therefore folds the crawler's
        # player-side win-rate prior into the deck target. Estimate each side
        # separately, then give them equal weight. Splitting the beta prior keeps
        # the same total prior strength as the former pooled estimate.
        return orientation_balanced_shrunk_mean(*values, 0.5, prior_strength)

    def corrected_panel_score(self, split: str, prior_strength: float) -> float:
        """Panel score plus a shrunk held-out deck-level model residual."""
        if self.model_score is None:
            raise RuntimeError("Candidate has not been scored against the panel")
        if split == "validation":
            values = (
                self.validation_team_residual_sum,
                self.validation_team_residual_n,
                self.validation_opponent_residual_sum,
                self.validation_opponent_residual_n,
            )
        else:
            values = (
                self.test_team_residual_sum,
                self.test_team_residual_n,
                self.test_opponent_residual_sum,
                self.test_opponent_residual_n,
            )
        residual = orientation_balanced_shrunk_mean(*values, 0.0, prior_strength)
        return max(0.0, min(1.0, self.model_score + residual))


def orientation_balanced_shrunk_mean(
    team_sum: float,
    team_count: int,
    opponent_sum: float,
    opponent_count: int,
    prior_mean: float,
    prior_strength: float,
) -> float:
    """Equal-side mean with one shared prior split across both orientations."""
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")
    side_prior = prior_strength / 2.0

    def side_mean(total: float, count: int) -> float:
        denominator = count + side_prior
        if denominator <= 0:
            return prior_mean
        return (total + prior_mean * side_prior) / denominator

    return 0.5 * (
        side_mean(team_sum, team_count)
        + side_mean(opponent_sum, opponent_count)
    )


@dataclass
class SearchPool:
    segment: str
    objective: str
    label: str
    candidates: tuple[DeckStat, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the optimizer uncertainty multiplier on chronological holdouts."
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--raw", type=Path, default=root / "data" / "raw")
    parser.add_argument("--checkpoint", type=Path, default=root / "artifacts" / "matchup-model.pt")
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "optimizer-variance-benchmark.json"
    )
    parser.add_argument("--panel-size", type=int, default=100)
    parser.add_argument("--min-eval-support", type=int, default=25)
    parser.add_argument("--prior-strength", type=float, default=25.0)
    parser.add_argument(
        "--residual-samples",
        type=int,
        default=50,
        help="Held-out battles sampled per deck and split to estimate model residuals.",
    )
    parser.add_argument("--starts-per-segment", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Inference device. 'auto' selects CUDA and refuses to run without "
            "it; a silent CPU fallback turns this sweep into an hours-long "
            "RAM-bound run. Pass 'cpu' explicitly to accept that."
        ),
    )
    parser.add_argument(
        "--k-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
    )
    return parser.parse_args()


CPU_VENV_HINT = (
    "CUDA is not available to this interpreter. The repository's own .venv holds "
    "a CPU-only torch build; the CUDA one lives in "
    r"%LOCALAPPDATA%\rigged-matchup-cuda-venv. Run this script with that "
    "interpreter, or pass --device cpu to accept a CPU sweep on purpose."
)


def resolve_device(choice: str) -> str:
    """'auto' has to become a real device name; torch.device('auto') raises.

    'auto' never resolves to CPU. Launching this from the CPU-only interpreter
    used to look identical to a GPU run in the log for the first minute, then
    take hours and evict the machine, so the absence of CUDA is an error here
    and CPU has to be asked for by name.
    """
    if choice != "auto":
        return choice
    if not torch.cuda.is_available():
        raise RuntimeError(CPU_VENV_HINT)
    return "cuda"


def raw_glob(raw_dir: Path) -> str:
    return (raw_dir.resolve() / "*.parquet").as_posix().replace("'", "''")


def chronological_cutoffs(
    connection: duckdb.DuckDBPyConnection, parquet: str
) -> tuple[float, float]:
    row_count, game_count, cutoffs = connection.execute(
        f"""
        select count(*), count(distinct game_id),
               quantile_cont(epoch(battle_time), [0.70, 0.85])
        from read_parquet('{parquet}')
        """
    ).fetchone()
    if row_count != game_count:
        raise RuntimeError(
            f"Corpus contains {row_count - game_count:,} duplicate game ids; "
            "deduplicate before benchmarking."
        )
    log(f"corpus rows={row_count:,}, unique games={game_count:,}")
    return float(cutoffs[0]), float(cutoffs[1])


def load_deck_stats(
    connection: duckdb.DuckDBPyConnection,
    parquet: str,
    train_cutoff: float,
    validation_cutoff: float,
    panel_size: int,
    min_eval_support: int,
) -> dict[str, list[DeckStat]]:
    log("aggregating exact-deck train support and chronological holdouts")
    rows = connection.execute(
        f"""
        with base as (
          select {POOLED_SEGMENT_SQL} as pooled_segment,
                 epoch(battle_time) as battle_epoch,
                 team_deck_key,
                 opponent_deck_key,
                 win
          from read_parquet('{parquet}')
          where segment not in ('ladder:top-100', 'ladder:top-1000')
        ), sides as (
          select pooled_segment, battle_epoch, team_deck_key as deck_key,
                 'team' as source_orientation, win as won
          from base
          union all
          select pooled_segment, battle_epoch, opponent_deck_key as deck_key,
                 'opponent' as source_orientation, not win as won
          from base
        ), stats as (
          select pooled_segment, deck_key,
                 count(*) filter (where battle_epoch <= {train_cutoff}) as train_n,
                 count(*) filter (
                   where battle_epoch > {train_cutoff}
                     and battle_epoch <= {validation_cutoff}
                 ) as validation_n,
                 count(*) filter (
                   where battle_epoch > {train_cutoff}
                     and battle_epoch <= {validation_cutoff}
                     and won
                 ) as validation_wins,
                 count(*) filter (
                   where battle_epoch > {train_cutoff}
                     and battle_epoch <= {validation_cutoff}
                     and source_orientation = 'team'
                 ) as validation_team_n,
                 count(*) filter (
                   where battle_epoch > {train_cutoff}
                     and battle_epoch <= {validation_cutoff}
                     and source_orientation = 'team' and won
                 ) as validation_team_wins,
                 count(*) filter (
                   where battle_epoch > {train_cutoff}
                     and battle_epoch <= {validation_cutoff}
                     and source_orientation = 'opponent'
                 ) as validation_opponent_n,
                 count(*) filter (
                   where battle_epoch > {train_cutoff}
                     and battle_epoch <= {validation_cutoff}
                     and source_orientation = 'opponent' and won
                 ) as validation_opponent_wins,
                 count(*) filter (where battle_epoch > {validation_cutoff}) as test_n,
                 count(*) filter (
                   where battle_epoch > {validation_cutoff} and won
                 ) as test_wins,
                 count(*) filter (
                   where battle_epoch > {validation_cutoff}
                     and source_orientation = 'team'
                 ) as test_team_n,
                 count(*) filter (
                   where battle_epoch > {validation_cutoff}
                     and source_orientation = 'team' and won
                 ) as test_team_wins,
                 count(*) filter (
                   where battle_epoch > {validation_cutoff}
                     and source_orientation = 'opponent'
                 ) as test_opponent_n,
                 count(*) filter (
                   where battle_epoch > {validation_cutoff}
                     and source_orientation = 'opponent' and won
                 ) as test_opponent_wins
          from sides
          group by pooled_segment, deck_key
        ), ranked as (
          select *, row_number() over (
            partition by pooled_segment order by train_n desc, deck_key
          ) as train_rank
          from stats
        )
        select pooled_segment, deck_key, train_n,
               validation_n, validation_wins,
               validation_team_n, validation_team_wins,
               validation_opponent_n, validation_opponent_wins,
               test_n, test_wins,
               test_team_n, test_team_wins,
               test_opponent_n, test_opponent_wins,
               train_rank
        from ranked
        where train_rank <= {panel_size}
           or (validation_n >= {min_eval_support} and test_n >= {min_eval_support})
        order by pooled_segment, train_rank
        """
    ).fetchall()
    by_segment: dict[str, list[DeckStat]] = defaultdict(list)
    for row in rows:
        cards = tuple(int(value) for value in str(row[1]).split(","))
        if len(cards) != 8:
            continue
        by_segment[str(row[0])].append(
            DeckStat(
                segment=str(row[0]),
                key=str(row[1]),
                cards=cards,
                train_n=int(row[2]),
                validation_n=int(row[3]),
                validation_wins=int(row[4]),
                validation_team_n=int(row[5]),
                validation_team_wins=int(row[6]),
                validation_opponent_n=int(row[7]),
                validation_opponent_wins=int(row[8]),
                test_n=int(row[9]),
                test_wins=int(row[10]),
                test_team_n=int(row[11]),
                test_team_wins=int(row[12]),
                test_opponent_n=int(row[13]),
                test_opponent_wins=int(row[14]),
                train_rank=int(row[15]),
            )
        )
    for segment, stats in sorted(by_segment.items()):
        eligible = sum(
            stat.validation_n >= min_eval_support and stat.test_n >= min_eval_support
            for stat in stats
        )
        log(f"{segment}: eligible decks={eligible:,}, panel records={min(panel_size, len(stats))}")
    return dict(by_segment)


def inference_row(
    team: tuple[int, ...], opponent: tuple[int, ...], segment: str, patch: str
) -> dict[str, Any]:
    zeros = [0] * 8
    roles_for = lambda cards: [
        ROLE_CHAMPION if card in CHAMPION_CARD_IDS else ROLE_NORMAL for card in cards
    ]
    return {
        "team_card_ids": list(team),
        "opponent_card_ids": list(opponent),
        "team_evolution_levels": zeros,
        "opponent_evolution_levels": zeros,
        "team_hero_levels": zeros,
        "opponent_hero_levels": zeros,
        "team_card_roles": roles_for(team),
        "opponent_card_roles": roles_for(opponent),
        "team_tower_troop_id": TOWER_PRINCESS_ID,
        "opponent_tower_troop_id": TOWER_PRINCESS_ID,
        "segment": segment,
        "patch": patch,
        "matrix_prior": 0.5,
        "win": False,
    }


def chunks(
    values: list[tuple[DeckStat, DeckStat, float]], size: int
) -> Iterable[list[tuple[DeckStat, DeckStat, float]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def matchup_sigma(mean: float, weighted_second_moment: float) -> float:
    """Weighted matchup standard deviation with float-noise protection."""
    return math.sqrt(max(0.0, weighted_second_moment - mean * mean))


def score_candidates(
    bundle: dict[str, Any],
    by_segment: dict[str, list[DeckStat]],
    panel_size: int,
    min_eval_support: int,
    batch_size: int,
) -> None:
    patch = max(str(value) for value in bundle["vocabulary"]["patches"])
    for segment, stats in sorted(by_segment.items()):
        candidates = [
            stat
            for stat in stats
            if stat.validation_n >= min_eval_support and stat.test_n >= min_eval_support
        ]
        panel = sorted(stats, key=lambda stat: (stat.train_rank, stat.key))[:panel_size]
        total_weight = sum(max(0, stat.train_n) for stat in panel)
        if not candidates or total_weight <= 0:
            continue
        jobs = [
            (candidate, opponent, opponent.train_n / total_weight)
            for candidate in candidates
            for opponent in panel
        ]
        weighted_scores: dict[str, float] = defaultdict(float)
        weighted_squares: dict[str, float] = defaultdict(float)
        unfavorable_shares: dict[str, float] = defaultdict(float)
        hard_counter_shares: dict[str, float] = defaultdict(float)
        worst_matchups: dict[str, float] = defaultdict(lambda: 1.0)
        log(f"{segment}: scoring {len(jobs):,} candidate-panel matchups")
        completed = 0
        for batch in chunks(jobs, batch_size):
            rows = [
                inference_row(candidate.cards, opponent.cards, segment, patch)
                for candidate, opponent, _ in batch
            ]
            predictions = predict_from_rows(bundle, rows)
            for (candidate, _, weight), prediction in zip(batch, predictions, strict=True):
                # Project the calibrated forward/reverse pair onto exact
                # antisymmetry, matching the correction consumed by the site.
                probability = 0.5 * (
                    float(prediction["team_win_probability"])
                    + 1.0
                    - float(prediction["opponent_win_probability"])
                )
                weighted_scores[candidate.key] += weight * probability
                weighted_squares[candidate.key] += weight * probability * probability
                if probability <= 0.45:
                    unfavorable_shares[candidate.key] += weight
                if probability <= 0.35:
                    hard_counter_shares[candidate.key] += weight
                worst_matchups[candidate.key] = min(worst_matchups[candidate.key], probability)
            completed += len(batch)
            if completed == len(batch) or completed % (batch_size * 20) == 0:
                log(f"{segment}: predictions {completed:,}/{len(jobs):,}")
        for candidate in candidates:
            candidate.model_score = weighted_scores[candidate.key]
            candidate.model_matchup_sigma = matchup_sigma(
                candidate.model_score,
                weighted_squares[candidate.key],
            )
            candidate.model_unfavorable_share = unfavorable_shares[candidate.key]
            candidate.model_hard_counter_share = hard_counter_shares[candidate.key]
            candidate.model_worst_matchup = worst_matchups[candidate.key]


def score_heldout_residuals(
    connection: duckdb.DuckDBPyConnection,
    parquet: str,
    bundle: dict[str, Any],
    by_segment: dict[str, list[DeckStat]],
    train_cutoff: float,
    validation_cutoff: float,
    min_eval_support: int,
    samples_per_split: int,
    batch_size: int,
) -> None:
    """Measure each deck's systematic held-out error in its real matchups.

    The optimizer score uses one common meta panel. Raw observed win rate does
    not: it also reflects whichever opponents, levels, and players happened to
    carry a deck. The mean residual (outcome - model probability) controls for
    that context. Adding its shrunk value to the common-panel score gives a
    held-out quality proxy on the same yardstick the optimizer maximizes.
    """
    connection.execute(
        "create or replace temporary table optimizer_benchmark_eligible "
        "(segment varchar, deck_key varchar)"
    )
    eligible_rows = [
        (segment, stat.key)
        for segment, stats in by_segment.items()
        for stat in stats
        if stat.model_score is not None
        and stat.validation_n >= min_eval_support
        and stat.test_n >= min_eval_support
    ]
    connection.executemany("insert into optimizer_benchmark_eligible values (?, ?)", eligible_rows)
    stat_lookup = {
        (segment, stat.key): stat
        for segment, stats in by_segment.items()
        for stat in stats
        if stat.model_score is not None
    }
    log(f"sampling up to {samples_per_split} real held-out matchups per deck and split")
    query = f"""
        with base as (
          select game_id,
                 {POOLED_SEGMENT_SQL} as pooled_segment,
                 epoch(battle_time) as battle_epoch,
                 patch,
                 team_deck_key,
                 opponent_deck_key,
                 team_card_ids,
                 opponent_card_ids,
                 team_evolution_levels,
                 opponent_evolution_levels,
                 team_hero_levels,
                 opponent_hero_levels,
                 team_card_roles,
                 opponent_card_roles,
                 team_tower_troop_id,
                 opponent_tower_troop_id,
                 matrix_prior,
                 win
          from read_parquet('{parquet}')
          where segment not in ('ladder:top-100', 'ladder:top-1000')
            and epoch(battle_time) > {train_cutoff}
        ), oriented as (
          select game_id, pooled_segment, battle_epoch, patch,
                 team_deck_key as deck_key,
                 'team' as source_orientation,
                 team_card_ids, opponent_card_ids,
                 team_evolution_levels, opponent_evolution_levels,
                 team_hero_levels, opponent_hero_levels,
                 team_card_roles, opponent_card_roles,
                 team_tower_troop_id, opponent_tower_troop_id,
                 matrix_prior, win as won
          from base
          union all
          select game_id, pooled_segment, battle_epoch, patch,
                 opponent_deck_key as deck_key,
                 'opponent' as source_orientation,
                 opponent_card_ids as team_card_ids,
                 team_card_ids as opponent_card_ids,
                 opponent_evolution_levels as team_evolution_levels,
                 team_evolution_levels as opponent_evolution_levels,
                 opponent_hero_levels as team_hero_levels,
                 team_hero_levels as opponent_hero_levels,
                 opponent_card_roles as team_card_roles,
                 team_card_roles as opponent_card_roles,
                 opponent_tower_troop_id as team_tower_troop_id,
                 team_tower_troop_id as opponent_tower_troop_id,
                 1.0 - matrix_prior as matrix_prior, not win as won
          from base
        ), sampled as (
          select o.*,
                 case when battle_epoch <= {validation_cutoff}
                      then 'validation' else 'test' end as split
          from oriented o
          inner join optimizer_benchmark_eligible e
            on e.segment = o.pooled_segment and e.deck_key = o.deck_key
          qualify row_number() over (
            partition by o.pooled_segment, o.deck_key,
                         case when battle_epoch <= {validation_cutoff}
                              then 'validation' else 'test' end
            order by hash(game_id, o.deck_key)
          ) <= {samples_per_split}
        )
        select split, pooled_segment, deck_key, source_orientation, won,
               team_card_ids, opponent_card_ids,
               team_evolution_levels, opponent_evolution_levels,
               team_hero_levels, opponent_hero_levels,
               team_card_roles, opponent_card_roles,
               team_tower_troop_id, opponent_tower_troop_id,
               patch, matrix_prior
        from sampled
        order by pooled_segment, deck_key, split
    """
    cursor = connection.execute(query)
    columns = [description[0] for description in cursor.description]
    completed = 0
    while batch := cursor.fetchmany(batch_size):
        rows: list[dict[str, Any]] = []
        metadata: list[tuple[str, str, str, str, bool]] = []
        for values in batch:
            record = dict(zip(columns, values, strict=True))
            split = str(record.pop("split"))
            segment = str(record.pop("pooled_segment"))
            deck_key = str(record.pop("deck_key"))
            source_orientation = str(record.pop("source_orientation"))
            won = bool(record.pop("won"))
            record["segment"] = segment
            record["win"] = won
            rows.append(record)
            metadata.append((split, segment, deck_key, source_orientation, won))
        predictions = predict_from_rows(bundle, rows)
        for (split, segment, deck_key, source_orientation, won), prediction in zip(
            metadata, predictions, strict=True
        ):
            probability = 0.5 * (
                float(prediction["team_win_probability"])
                + 1.0
                - float(prediction["opponent_win_probability"])
            )
            stat = stat_lookup[(segment, deck_key)]
            residual = float(won) - probability
            if split == "validation":
                if source_orientation == "team":
                    stat.validation_team_residual_sum += residual
                    stat.validation_team_residual_n += 1
                else:
                    stat.validation_opponent_residual_sum += residual
                    stat.validation_opponent_residual_n += 1
            else:
                if source_orientation == "team":
                    stat.test_team_residual_sum += residual
                    stat.test_team_residual_n += 1
                else:
                    stat.test_opponent_residual_sum += residual
                    stat.test_opponent_residual_n += 1
        completed += len(batch)
        if completed == len(batch) or completed % (batch_size * 20) == 0:
            log(f"held-out residual predictions={completed:,}")


def deterministic_subsets(
    cards: tuple[int, ...], count: int, limit: int = 3
) -> list[frozenset[int]]:
    combinations = list(itertools.combinations(cards, count))
    if len(combinations) <= limit:
        return [frozenset(values) for values in combinations]
    indices = np.linspace(0, len(combinations) - 1, num=limit, dtype=int)
    return [frozenset(combinations[int(index)]) for index in indices]


def build_search_pools(
    by_segment: dict[str, list[DeckStat]],
    min_eval_support: int,
    starts_per_segment: int,
) -> list[SearchPool]:
    pools: list[SearchPool] = []
    for segment, stats in sorted(by_segment.items()):
        candidates = [
            stat
            for stat in stats
            if stat.model_score is not None
            and stat.validation_n >= min_eval_support
            and stat.test_n >= min_eval_support
        ]
        starts = sorted(candidates, key=lambda stat: (-stat.train_n, stat.key))[:starts_per_segment]
        sets = {stat.key: frozenset(stat.cards) for stat in candidates}
        for start in starts:
            start_set = sets[start.key]
            single = tuple(
                stat
                for stat in candidates
                if stat.key != start.key and len(start_set & sets[stat.key]) == 7
            )
            if len(single) >= 2:
                pools.append(SearchPool(segment, "single", start.key, single))

            multi = tuple(
                stat
                for stat in candidates
                if stat.key != start.key and len(start_set & sets[stat.key]) >= 4
            )
            if len(multi) >= 2:
                pools.append(SearchPool(segment, "multi", start.key, multi))

            for mutable_count in (2, 3, 4):
                for mutable in deterministic_subsets(start.cards, mutable_count):
                    locked = start_set - mutable
                    group = tuple(
                        stat
                        for stat in candidates
                        if locked <= sets[stat.key] and not (mutable & sets[stat.key])
                    )
                    if len(group) >= 2:
                        pools.append(
                            SearchPool(
                                segment,
                                "group",
                                f"{start.key}:replace:{','.join(map(str, sorted(mutable)))}",
                                group,
                            )
                        )

            for fixed_count in (2, 4, 6):
                for fixed in deterministic_subsets(start.cards, fixed_count):
                    completions = tuple(stat for stat in candidates if fixed <= sets[stat.key])
                    if len(completions) >= 2:
                        pools.append(
                            SearchPool(
                                segment,
                                "complete",
                                f"{start.key}:keep:{','.join(map(str, sorted(fixed)))}",
                                completions,
                            )
                        )
    return pools


def selection_key(stat: DeckStat, k: float) -> tuple[float, float, int, str]:
    assert stat.model_score is not None
    conservative = stat.model_score - k * stat.sigma
    return conservative, stat.model_score, stat.train_n, stat.key


def evaluate_k_values(
    pools: list[SearchPool],
    k_values: list[float],
    prior_strength: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for k in k_values:
        for pool in pools:
            selected = max(pool.candidates, key=lambda stat: selection_key(stat, k))
            row = {
                "k": k,
                "segment": pool.segment,
                "objective": pool.objective,
                "pool_size": len(pool.candidates),
                "selected_key": selected.key,
                "selected_support": selected.train_n,
                "selected_sigma": selected.sigma,
                "selected_model_score": selected.model_score,
                "selected_matchup_sigma": selected.model_matchup_sigma,
                "selected_unfavorable_share": selected.model_unfavorable_share,
                "selected_hard_counter_share": selected.model_hard_counter_share,
                "selected_worst_matchup": selected.model_worst_matchup,
                "selected_validation_rate": selected.observed("validation", prior_strength),
                "selected_test_rate": selected.observed("test", prior_strength),
                "selected_validation_quality": selected.corrected_panel_score(
                    "validation", prior_strength
                ),
                "selected_test_quality": selected.corrected_panel_score("test", prior_strength),
            }
            detail.append(row)
            grouped[(k, pool.segment, pool.objective)].append(row)

    summary: list[dict[str, Any]] = []
    for (k, segment, objective), rows in sorted(grouped.items()):
        summary.append(
            {
                "k": k,
                "segment": segment,
                "objective": objective,
                "pools": len(rows),
                "median_pool_size": float(np.median([row["pool_size"] for row in rows])),
                "thin_selection_share": float(
                    np.mean([row["selected_support"] < SUPPORT_CUTOFF for row in rows])
                ),
                "mean_model_score": float(np.mean([row["selected_model_score"] for row in rows])),
                "mean_validation_rate": float(
                    np.mean([row["selected_validation_rate"] for row in rows])
                ),
                "mean_test_rate": float(np.mean([row["selected_test_rate"] for row in rows])),
                "mean_validation_quality": float(
                    np.mean([row["selected_validation_quality"] for row in rows])
                ),
                "mean_test_quality": float(np.mean([row["selected_test_quality"] for row in rows])),
            }
        )
    return summary, detail


def macro_summary(summary: list[dict[str, Any]], k_values: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in k_values:
        selected = [row for row in summary if math.isclose(row["k"], k)]
        if not selected:
            continue
        rows.append(
            {
                "k": k,
                "segment_objective_cells": len(selected),
                "mean_validation_rate": float(
                    np.mean([row["mean_validation_quality"] for row in selected])
                ),
                "mean_test_rate": float(np.mean([row["mean_test_quality"] for row in selected])),
                "thin_selection_share": float(
                    np.mean([row["thin_selection_share"] for row in selected])
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    device = torch.device(resolve_device(args.device))
    parquet = raw_glob(args.raw)
    connection = duckdb.connect()
    connection.execute(f"set threads={max(1, args.threads)}")
    train_cutoff, validation_cutoff = chronological_cutoffs(connection, parquet)
    by_segment = load_deck_stats(
        connection,
        parquet,
        train_cutoff,
        validation_cutoff,
        args.panel_size,
        args.min_eval_support,
    )
    bundle = load_bundle(args.checkpoint.resolve())
    bundle["model"].to(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    log(f"loaded model {bundle['resolved_model_version']} on {device_name}")
    score_candidates(
        bundle,
        by_segment,
        args.panel_size,
        args.min_eval_support,
        args.batch_size,
    )
    score_heldout_residuals(
        connection,
        parquet,
        bundle,
        by_segment,
        train_cutoff,
        validation_cutoff,
        args.min_eval_support,
        args.residual_samples,
        args.batch_size,
    )
    pools = build_search_pools(by_segment, args.min_eval_support, args.starts_per_segment)
    log(
        "search pools "
        + ", ".join(
            f"{objective}={sum(pool.objective == objective for pool in pools):,}"
            for objective in ("single", "multi", "group", "complete")
        )
    )
    k_values = sorted({float(value) for value in args.k_values})
    summary, detail = evaluate_k_values(pools, k_values, args.prior_strength)
    macro = macro_summary(summary, k_values)
    best_validation = max(macro, key=lambda row: (row["mean_validation_rate"], -row["k"]))
    output = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint.resolve()),
        "model_version": bundle["resolved_model_version"],
        "corpus_glob": parquet,
        "train_cutoff_epoch": train_cutoff,
        "validation_cutoff_epoch": validation_cutoff,
        "settings": {
            "panel_size": args.panel_size,
            "min_eval_support": args.min_eval_support,
            "prior_strength": args.prior_strength,
            "residual_samples_per_deck_split": args.residual_samples,
            "starts_per_segment": args.starts_per_segment,
            "support_cutoff": SUPPORT_CUTOFF,
            "sigma_thin": SIGMA_THIN,
            "sigma_supported": SIGMA_SUPPORTED,
            "k_values": k_values,
            "forms": "base forms on both candidate and panel decks",
            "device": str(device),
            "device_name": device_name,
        },
        "segments": {
            segment: {
                "records_loaded": len(stats),
                "eligible_decks": sum(
                    stat.validation_n >= args.min_eval_support
                    and stat.test_n >= args.min_eval_support
                    for stat in stats
                ),
            }
            for segment, stats in sorted(by_segment.items())
        },
        "pool_count": len(pools),
        "best_k_on_validation": best_validation["k"],
        "macro": macro,
        "by_segment_objective": summary,
        "selections": detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("k\tvalidation\ttest\tthin-selected")
    for row in macro:
        print(
            f"{row['k']:.2f}\t{row['mean_validation_rate']:.4f}\t"
            f"{row['mean_test_rate']:.4f}\t{row['thin_selection_share']:.3f}"
        )
    print(f"best_k_on_validation={best_validation['k']:.2f}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
