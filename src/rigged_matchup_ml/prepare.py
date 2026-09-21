from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import duckdb

from .api_collect import _segment_tracked
from .card_stats import CARD_ELIXIR
from .config import AppConfig
from .domain import ranked_league_buckets
from .training_weights import segment_weights, weighting_report


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _ranked_segment_sql(data_config: dict[str, Any]) -> str:
    """DuckDB expression pooling exact raw leagues for model preparation."""
    buckets = ranked_league_buckets(data_config)
    if not buckets:
        return "segment"
    league = (
        "try_cast(regexp_extract(segment, "
        "'^ranked:league-([0-9]+)$', 1) as integer)"
    )
    clauses = [
        (
            f"when lower(mode_key) = 'ranked' and {league} >= {lower} "
            f"and {league} < {upper} then 'ranked:league-{lower}-{upper - 1}'"
        )
        for lower, upper in pairwise(buckets)
    ]
    return "case " + " ".join(clauses) + " else segment end"


def prepare_splits(config: AppConfig, overwrite: bool = False) -> dict[str, Any]:
    raw_dir = config.resolve(config.data["raw_dir"])
    prepared_dir = config.resolve(config.data["prepared_dir"])
    if not list(raw_dir.glob("*.parquet")):
        raise RuntimeError(f"No extracted Parquet files found in {raw_dir}")
    if prepared_dir.exists() and overwrite:
        shutil.rmtree(prepared_dir)
    if prepared_dir.exists() and any(prepared_dir.iterdir()):
        raise RuntimeError(f"{prepared_dir} is not empty. Pass --overwrite to rebuild it.")
    prepared_dir.mkdir(parents=True, exist_ok=True)

    raw_glob = _quoted(raw_dir / "*.parquet")
    ranked_buckets = ranked_league_buckets(config.data)
    ranked_segment_sql = _ranked_segment_sql(config.data)
    train_fraction = float(config.data["train_fraction"])
    validation_fraction = float(config.data["validation_fraction"])
    validation_boundary = train_fraction + validation_fraction
    connection = duckdb.connect()
    connection.execute("set preserve_insertion_order=false")
    # Let the dedup temp table spill to disk instead of OOM-ing on a RAM-limited
    # Pod; the in-memory connection has no spill target otherwise.
    connection.execute(f"set temp_directory='{_quoted(prepared_dir)}'")
    # Collect's SQLite dedup only spans a single collect run; if it was reset
    # between runs, the same game_id was written into multiple Storage shards.
    # pull-storage downloads them all, so dedup globally here (one row per
    # game_id) before splitting -- otherwise ~1M duplicate battles leak into
    # train/val/test. A duplicate game_id is the same battle (same battle_time),
    # so it always lands in the same split; deduping at read is safe.
    # Materialize the deduped set ONCE into a temp table. A view would re-run
    # the partition-by-game_id window over all rows on every reference (count,
    # quantile, each split copy = ~5 full dedup passes over tens of millions of
    # rows), which crawls and spills RAM. The table pays the window cost once.
    _log("prepare: deduplicating raw shards by game_id")
    if ranked_buckets:
        labels = [
            f"{lower}-{upper - 1}"
            for lower, upper in pairwise(ranked_buckets)
        ]
        _log(f"prepare: pooling ranked leagues into {labels}")
    raw_total = connection.execute(
        f"select count(*) from read_parquet('{raw_glob}')"
    ).fetchone()[0]
    # The deduplicated set is written ONCE to a compressed Parquet file rather than a
    # temp table. Uncompressed, ~70M rows spill well past what a workstation's disk
    # can hold (DuckDB caps the spill at 90% of free space and aborts), whereas the
    # zstd file is about the size of the raw shards. Only the rare repeated game_ids
    # go through the window: sorting every row just to drop ~0.1% of them is what
    # made the spill so large. NULL game_ids stay in the windowed branch, so they
    # still collapse to one row exactly as the plain window did.
    dedup_file = prepared_dir / "_raw_dedup.parquet"
    dedup_path = _quoted(dedup_file)
    connection.execute(
        f"""
        create temporary table repeated_games as
        select game_id from read_parquet('{raw_glob}')
        where game_id is not null
        group by game_id having count(*) > 1
        """
    )
    connection.execute(
        f"""
        copy (
          select * replace ({ranked_segment_sql} as segment)
          from read_parquet('{raw_glob}')
          where game_id is not null
            and game_id not in (select game_id from repeated_games)
          union all
          select * replace ({ranked_segment_sql} as segment)
          from read_parquet('{raw_glob}')
          where game_id is null or game_id in (select game_id from repeated_games)
          qualify row_number() over (partition by game_id order by inserted_at) = 1
        ) to '{dedup_path}' (format parquet, compression zstd, row_group_size 100000)
        """
    )
    # Later filters (history window, out-of-scope segments) narrow this view instead
    # of deleting rows from a table.
    kept_filters: list[str] = []

    def _refresh_dedup_view() -> None:
        where = " and ".join(kept_filters) if kept_filters else "true"
        connection.execute(
            f"create or replace temp view raw_dedup as "
            f"select * from read_parquet('{dedup_path}') where {where}"
        )

    _refresh_dedup_view()
    dedup_total = connection.execute("select count(*) from raw_dedup").fetchone()[0]
    _log(
        f"prepare: rows raw={raw_total:,} unique={dedup_total:,} "
        f"duplicates_removed={raw_total - dedup_total:,}"
    )

    # Rolling history window. The split is chronological, so a card released
    # yesterday only ever lands in validation/test if the corpus still carries
    # years of pre-release battles -- the model then never trains on it. Trimming
    # the oldest battles moves the train cutoff forward so the new card falls
    # inside the train period too. Null = keep everything (previous behaviour).
    max_history_days = config.data.get("max_history_days")
    history_cutoff_epoch: float | None = None
    dropped_old = 0
    if max_history_days is not None:
        latest = connection.execute(
            "select max(epoch(battle_time)) from raw_dedup"
        ).fetchone()[0]
        history_cutoff_epoch = float(latest) - float(max_history_days) * 86400.0
        # Same rows as `delete ... where epoch(battle_time) < cutoff`: a NULL
        # battle_time was never deleted, so it is kept here too.
        kept_filters.append(
            f"not coalesce(epoch(battle_time) < {history_cutoff_epoch}, false)"
        )
        _refresh_dedup_view()
        kept = connection.execute("select count(*) from raw_dedup").fetchone()[0]
        dropped_old = dedup_total - kept
        dedup_total = kept
        _log(
            f"prepare: history window {max_history_days}d "
            f"dropped_old={dropped_old:,} kept={dedup_total:,}"
        )
        if dedup_total == 0:
            raise RuntimeError(
                f"max_history_days={max_history_days} dropped every battle; widen the window."
            )

    # Out-of-scope segments. The collector already refuses sub-`collect_min_trophies`
    # ladder, but legacy shards still carry it, and `ladder:unknown` cannot be mapped
    # to any band at all. Neither is a population this model is ever asked about, so
    # they skew the training mix for nothing. Dropped BEFORE the quantiles: cutoffs
    # computed on rows that are deleted afterwards no longer split 70/15/15.
    min_trophies = int(config.data.get("collect_min_trophies", 5000))
    drop_out_of_scope = bool(config.data.get("drop_out_of_scope_segments", True))
    dropped_out_of_scope: dict[str, int] = {}
    if drop_out_of_scope:
        out_of_scope = {
            str(segment): int(row_count)
            for segment, row_count in connection.execute(
                "select segment, count(*) from raw_dedup group by segment"
            ).fetchall()
            if not _segment_tracked(str(segment), min_trophies)
        }
        if out_of_scope:
            values = ", ".join(
                "'" + segment.replace("'", "''") + "'" for segment in out_of_scope
            )
            kept_filters.append(f"not coalesce(segment in ({values}), false)")
            _refresh_dedup_view()
            dropped_out_of_scope = out_of_scope
            dedup_total -= sum(out_of_scope.values())
            _log(
                f"prepare: dropped {sum(out_of_scope.values()):,} out-of-scope rows "
                f"from {sorted(out_of_scope)}"
            )
            if dedup_total == 0:
                raise RuntimeError(
                    "Dropping out-of-scope segments removed every battle; check "
                    "collect_min_trophies or set data.drop_out_of_scope_segments to false."
                )

    _log("prepare: computing chronological train/validation cutoffs")
    quantiles = connection.execute(
        """
        select quantile_cont(epoch(battle_time), [?, ?])
        from raw_dedup
        """,
        [train_fraction, validation_boundary],
    ).fetchone()[0]
    train_cutoff, validation_cutoff = quantiles

    split_conditions = {
        "train": f"epoch(battle_time) <= {train_cutoff}",
        "validation": (
            f"epoch(battle_time) > {train_cutoff} and epoch(battle_time) <= {validation_cutoff}"
        ),
        "test": f"epoch(battle_time) > {validation_cutoff}",
    }
    counts: dict[str, int] = {}
    for split, condition in split_conditions.items():
        destination = prepared_dir / split
        destination.mkdir(parents=True, exist_ok=True)
        output = _quoted(destination / "data.parquet")
        _log(f"prepare: writing {split} split")
        connection.execute(
            f"""
            copy (
              select * from raw_dedup where {condition}
            ) to '{output}' (format parquet, compression zstd, row_group_size 100000)
            """
        )
        counts[split] = connection.execute(
            f"select count(*) from read_parquet('{output}')"
        ).fetchone()[0]
        _log(f"prepare: {split} rows={counts[split]:,}")

    train_file = _quoted(prepared_dir / "train" / "*.parquet")
    all_file = _quoted(prepared_dir / "*" / "*.parquet")
    _log("prepare: building vocabularies from train split")
    train_card_ids = [
        row[0]
        for row in connection.execute(
            f"""
            select distinct card_id from (
              select unnest(team_card_ids) card_id from read_parquet('{train_file}')
              union all
              select unnest(opponent_card_ids) card_id from read_parquet('{train_file}')
            ) order by card_id
            """
        ).fetchall()
    ]
    # A card absent from train (a card released after the train cutoff, e.g.
    # Ronin) would otherwise encode to index 0 -- the padding slot -- so it would
    # be indistinguishable from an empty deck position at validation/test time.
    # Seeding the vocabulary from the static card table gives every known card its
    # own embedding row; an unseen row simply stays at its init and the card is
    # then carried by its metadata vector instead of being silently erased.
    seed_vocabulary = bool(config.data.get("seed_card_vocabulary", True))
    seeded_ids = sorted(set(CARD_ELIXIR) - set(train_card_ids)) if seed_vocabulary else []
    card_ids = sorted(set(train_card_ids) | set(seeded_ids))
    if seeded_ids:
        _log(f"prepare: seeded {len(seeded_ids)} card ids absent from train: {seeded_ids}")
    tower_ids = [
        row[0]
        for row in connection.execute(
            f"""
            select distinct tower_id from (
              select team_tower_troop_id tower_id from read_parquet('{train_file}')
              union all
              select opponent_tower_troop_id tower_id from read_parquet('{train_file}')
            ) order by tower_id
            """
        ).fetchall()
    ]
    segments = [
        row[0]
        for row in connection.execute(
            f"select distinct segment from read_parquet('{train_file}') order by segment"
        ).fetchall()
    ]
    patches = [
        row[0]
        for row in connection.execute(
            f"select distinct patch from read_parquet('{train_file}') order by patch"
        ).fetchall()
    ]
    vocabulary = {
        "cards": {str(value): index + 1 for index, value in enumerate(card_ids)},
        "towers": {str(value): index + 1 for index, value in enumerate(tower_ids)},
        "segments": {str(value): index + 1 for index, value in enumerate(segments)},
        "patches": {str(value): index + 1 for index, value in enumerate(patches)},
    }
    (prepared_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Per-card train frequency, for inverse-frequency loss weighting: rare cards
    # are under-sampled, so without this the model just learns the popular meta.
    _log("prepare: counting per-card train frequencies")
    card_counts = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            f"""
            select card_id, count(*) from (
              select unnest(team_card_ids) card_id from read_parquet('{train_file}')
              union all
              select unnest(opponent_card_ids) card_id from read_parquet('{train_file}')
            ) group by card_id
            """
        ).fetchall()
    }
    (prepared_dir / "card_frequencies.json").write_text(
        json.dumps(card_counts, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Per-segment train counts, for the label-blind segment weighting in the loss.
    # The crawl does not sample segments evenly, so without this the loss belongs to
    # the saturated band. Counts and wins are read together, but only the counts feed
    # the weight -- the win column exists solely to report the population shift the
    # weighting causes, never to compute it.
    _log("prepare: counting per-segment train frequencies")
    segment_frequency_rows = connection.execute(
        f"""
        select segment, count(*) as row_count, sum(case when win then 1 else 0 end) as wins
        from read_parquet('{train_file}')
        group by segment
        """
    ).fetchall()
    segment_counts = {str(row[0]): int(row[1]) for row in segment_frequency_rows}
    segment_wins = {str(row[0]): int(row[2] or 0) for row in segment_frequency_rows}
    (prepared_dir / "segment_frequencies.json").write_text(
        json.dumps(segment_counts, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Read defensively: `prepare` has never needed the training section, and a
    # prepare-only config must not start failing because of a reporting block.
    training_config = config.raw.get("training") or {}
    segment_power = float(training_config.get("loss_segment_power", 0.3))
    segment_cap = float(training_config.get("loss_segment_cap", 3.0))
    weights = segment_weights(segment_counts, power=segment_power, cap=segment_cap)
    weighting = weighting_report(segment_counts, segment_wins, weights)
    for warning in weighting["warnings"]:
        _log(f"prepare: WARNING {warning['message']}")
    _log(
        "prepare: segment weighting "
        f"effective_sample={weighting['effective_sample_fraction']:.1%} "
        f"weight_ratio={weighting['weight_ratio']:.2f} "
        f"win_rate_shift={weighting['win_rate_shift']:+.4f}"
    )
    # Cards the corpus contains but the train period does not: a card released
    # after the train cutoff cannot be learned, only evaluated. Surfacing it here
    # is the signal to collect more post-release battles or to shorten
    # ``max_history_days`` before retraining.
    _log("prepare: checking card coverage of the train split")
    corpus_card_ids = {
        int(row[0])
        for row in connection.execute(
            f"""
            select distinct card_id from (
              select unnest(team_card_ids) card_id from read_parquet('{all_file}')
              union all
              select unnest(opponent_card_ids) card_id from read_parquet('{all_file}')
            )
            """
        ).fetchall()
    }
    untrained_cards = sorted(corpus_card_ids - set(train_card_ids))
    if untrained_cards:
        _log(
            "prepare: WARNING cards present in the corpus but never in train "
            f"(no learned embedding): {untrained_cards}"
        )
    manifest = {
        "counts": counts,
        "raw_rows": raw_total,
        "unique_rows": dedup_total,
        "duplicates_removed": raw_total - dedup_total,
        "dropped_before_history_window": dropped_old,
        "dropped_out_of_scope": dropped_out_of_scope,
        "drop_out_of_scope_segments": drop_out_of_scope,
        "sample_weighting": {
            **weighting,
            "power": segment_power,
            "cap": segment_cap,
            "weights": weights,
            "applies_to": "train loss only; validation/test stay unweighted",
        },
        "max_history_days": max_history_days,
        "history_cutoff_epoch": history_cutoff_epoch,
        "train_cutoff_epoch": train_cutoff,
        "validation_cutoff_epoch": validation_cutoff,
        "vocabulary_sizes": {key: len(value) + 1 for key, value in vocabulary.items()},
        "seeded_card_ids": seeded_ids,
        "cards_absent_from_train": untrained_cards,
        "ranked_league_buckets": list(ranked_buckets),
        "split_policy": "chronological 70/15/15 by battle_time",
    }
    (prepared_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    connection.close()
    # The splits hold every kept row; the intermediate file only costs disk now.
    dedup_file.unlink(missing_ok=True)
    _log("prepare: done")
    return manifest
