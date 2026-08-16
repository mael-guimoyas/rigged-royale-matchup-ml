"""Measure what a *real* deck can actually look like against the meta panel.

Every attempt so far tried to make the model's number more accurate. This script
asks a different question, the one a hardcoded guard rail needs answered: over
decks that human beings genuinely play, what range does the panel score occupy,
and what does the matchup profile behind a high score look like?

The answer cannot come from the optimizer benchmark. That benchmark only ever
evaluates decks with enough held-out battles to measure (>= 25 per split), so
the decks the optimizer actually emits — played by nobody, zero battles — are
structurally absent from it. Their score is unfalsifiable by construction. The
only handle left is the *shape* of the real population, which is measurable.

Three stages:

1. Body — a uniform random sample of decks with real play, scored against the
   segment's weighted top-100 panel. Gives the distribution: median, spread, and
   the joint (score, unfavorable share).
2. Ceiling — every deck with real play screened against a small weighted panel,
   then the screen leaders rescored against the full panel. Gives the empirical
   maximum a real deck reaches, per segment.
3. Off-manifold — the site's own greedy beam re-implemented here, run from real
   seeds, so the emitted decks can be placed against the two clouds above.

Output is a JSON envelope: per-segment percentiles, the score→unfavorable-share
floor observed among real decks, and where the searched decks land.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

from rigged_matchup_ml.card_stats import CARD_ELIXIR, CHAMPION_CARD_IDS
from rigged_matchup_ml.dataset import encode_rows
from rigged_matchup_ml.domain import ROLE_CHAMPION, ROLE_NORMAL
from rigged_matchup_ml.predictor import (
    _antisymmetric_projection,
    _calibration_for_segment,
    load_bundle,
    predict_from_rows,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deck_composition_rules import ENVELOPES, Catalog, admits_deck

TOWER_PRINCESS_ID = 159000000
UNFAVORABLE_THRESHOLD = 0.45
HARD_COUNTER_THRESHOLD = 0.35
FAVORABLE_THRESHOLD = 0.55

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


@dataclass
class Profile:
    """A deck's whole answer against one weighted panel, not just its mean."""

    score: float
    unfavorable_share: float
    hard_counter_share: float
    favorable_share: float
    worst_matchup: float
    sigma: float
    elixir: float


@dataclass
class Deck:
    cards: tuple[int, ...]
    plays: int
    wins: int
    profile: Profile | None = None
    screen_score: float | None = None

    @property
    def key(self) -> str:
        return ",".join(str(card) for card in self.cards)

    @property
    def win_rate(self) -> float:
        return self.wins / self.plays if self.plays else float("nan")


@dataclass
class SegmentPanel:
    segment: str
    decks: list[tuple[int, ...]]
    weights: list[float]
    screen_decks: list[tuple[int, ...]] = field(default_factory=list)
    screen_weights: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / "data" / "raw")
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "artifacts" / "matchup-model.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "plausibility-envelope.json"
    )
    parser.add_argument("--panel-size", type=int, default=100)
    parser.add_argument(
        "--screen-panel-size",
        type=int,
        default=16,
        help="Panel used for the exhaustive ceiling screen, before rescoring.",
    )
    parser.add_argument(
        "--min-plays",
        type=int,
        default=30,
        help="Battles a deck needs before it counts as genuinely played.",
    )
    parser.add_argument("--body-sample", type=int, default=1500)
    parser.add_argument(
        "--ceiling-rescore",
        type=int,
        default=400,
        help="Screen leaders rescored per segment against the full panel.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root.parent / "riggedroyale" / "data" / "cards.json",
        help="Card roles and elixir, read for the composition admission rule.",
    )
    parser.add_argument(
        "--search-admission",
        choices=("none", "wide", "tight"),
        default="tight",
        help=(
            "Admission rule the constrained search runs under, alongside the "
            "unconstrained one. 'none' skips the constrained run."
        ),
    )
    parser.add_argument("--search-starts", type=int, default=2)
    parser.add_argument(
        "--search-seed-source",
        choices=("top", "sample"),
        default="top",
        help=(
            "Which real decks the search starts from. 'top' takes the most "
            "played decks of the segment; 'sample' draws from a play band, "
            "which is what a visitor's own deck actually looks like."
        ),
    )
    parser.add_argument(
        "--search-seed-plays",
        type=int,
        nargs=2,
        default=(30, 300),
        metavar=("MIN", "MAX"),
        help="Play band the sampled seeds are drawn from.",
    )
    parser.add_argument(
        "--search-segments",
        nargs="*",
        default=None,
        help="Segments to run the beam search on (default: every measured segment).",
    )
    parser.add_argument(
        "--search-pool",
        type=int,
        default=40,
        help="Distinct swap-in cards the beam may reach for.",
    )
    parser.add_argument("--search-rounds", type=int, default=4)
    parser.add_argument("--search-beam", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--encode-chunk",
        type=int,
        default=4096,
        help="Decks encoded per call when filling the scorer's cache.",
    )
    parser.add_argument(
        "--verify-scorer",
        type=int,
        default=512,
        help=(
            "Matchups per segment re-scored through predict_from_rows to check "
            "the encoded-deck cache. 0 skips the check."
        ),
    )
    parser.add_argument(
        "--memory-limit",
        default="4GB",
        help=(
            "DuckDB memory ceiling for the aggregation. The default keeps the "
            "whole-corpus group-by from evicting the rest of the machine."
        ),
    )
    parser.add_argument(
        "--population-cache",
        type=Path,
        default=root / "artifacts" / "played-deck-population.parquet",
        help=(
            "Where the per-segment play counts are cached. Rescanning the raw "
            "corpus costs minutes and gigabytes, and the answer never changes "
            "between runs, so it is written once and reused."
        ),
    )
    parser.add_argument(
        "--rebuild-population",
        action="store_true",
        help="Ignore the cache and re-aggregate from the raw parquet.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep the segments already present in the output file and only "
            "measure the missing ones."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Inference device. 'auto' selects CUDA and refuses to run without "
            "it; falling back silently to CPU turns a one-hour sweep into a "
            "machine-eating one. Pass 'cpu' explicitly to accept that."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--segments",
        nargs="*",
        default=None,
        help="Restrict to these pooled segments (default: all eight).",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip stage 3 (the beam search) and only measure the real population.",
    )
    return parser.parse_args()


def raw_glob(raw_dir: Path) -> str:
    return (raw_dir.resolve() / "*.parquet").as_posix().replace("'", "''")


CPU_VENV_HINT = (
    "CUDA is not available to this interpreter. The repository's own .venv holds "
    "a CPU-only torch build; the CUDA one lives in "
    r"%LOCALAPPDATA%\rigged-matchup-cuda-venv. Run this script with that "
    "interpreter, or pass --device cpu to accept a CPU sweep on purpose."
)


def resolve_device(choice: str) -> str:
    """'auto' has to become a real device name; torch.device('auto') raises.

    'auto' never resolves to CPU. A silent fallback is what turned a previous
    sweep into an hours-long RAM-bound run on the wrong interpreter, so the
    absence of CUDA is an error here and CPU has to be asked for by name.
    """
    if choice != "auto":
        return choice
    if not torch.cuda.is_available():
        raise RuntimeError(CPU_VENV_HINT)
    return "cuda"


POPULATION_SQL = """
        with base as (
          select {pooled} as seg, team_deck_key, opponent_deck_key, win
          from read_parquet('{parquet}')
          where segment not in ('ladder:top-100', 'ladder:top-1000')
        ), sides as (
          select seg, team_deck_key as deck_key, win as won from base
          union all
          select seg, opponent_deck_key as deck_key, not win as won from base
        )
        select seg, deck_key, count(*) as plays,
               count(*) filter (where won) as wins
        from sides
        group by seg, deck_key
        having count(*) >= {min_plays}
        order by seg, plays desc
"""


def cache_signature(raw: Path, min_plays: int) -> dict[str, Any]:
    return {"raw": str(raw.resolve()), "min_plays": min_plays}


def population_rows(
    connection: duckdb.DuckDBPyConnection,
    raw: Path,
    parquet: str,
    min_plays: int,
    cache: Path | None,
    rebuild: bool,
) -> list[tuple[Any, ...]]:
    """The aggregation itself, cached to parquet because it dominates startup.

    Scanning the whole corpus twice per run costs minutes and several gigabytes
    of working set, and the result depends on nothing but the raw directory and
    ``min_plays``. A sidecar records both so a stale cache is never reused.
    """
    query = POPULATION_SQL.format(
        pooled=POOLED_SEGMENT_SQL, parquet=parquet, min_plays=min_plays
    )
    if cache is None:
        log("aggregating deck play counts per pooled segment")
        return connection.execute(query).fetchall()

    sidecar = cache.with_suffix(".meta.json")
    signature = cache_signature(raw, min_plays)
    fresh = (
        not rebuild
        and cache.exists()
        and sidecar.exists()
        and json.loads(sidecar.read_text(encoding="utf-8")) == signature
    )
    if fresh:
        log(f"reading cached deck play counts from {cache}")
    else:
        log("aggregating deck play counts per pooled segment")
        cache.parent.mkdir(parents=True, exist_ok=True)
        target = cache.resolve().as_posix().replace("'", "''")
        connection.execute(f"copy ({query}) to '{target}' (format parquet)")
        sidecar.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        log(f"cached deck play counts to {cache}")
    cached = cache.resolve().as_posix().replace("'", "''")
    return connection.execute(
        f"select seg, deck_key, plays, wins from read_parquet('{cached}') order by seg, plays desc"
    ).fetchall()


def load_population(
    rows: list[tuple[Any, ...]],
    panel_size: int,
) -> tuple[dict[str, list[Deck]], dict[str, SegmentPanel]]:
    """Every genuinely played deck per segment, plus that segment's meta panel.

    Both orientations are folded in: a deck appears once per battle it took part
    in, whichever side of the log it sat on. Win counts stay orientation-split
    downstream; here only the play count matters, and that one is symmetric.
    """
    population: dict[str, list[Deck]] = defaultdict(list)
    for segment, deck_key, plays, wins in rows:
        cards = tuple(int(value) for value in str(deck_key).split(","))
        if len(cards) != 8:
            continue
        population[str(segment)].append(Deck(cards=cards, plays=int(plays), wins=int(wins)))

    panels: dict[str, SegmentPanel] = {}
    for segment, decks in population.items():
        top = decks[:panel_size]
        total = float(sum(deck.plays for deck in top))
        panels[segment] = SegmentPanel(
            segment=segment,
            decks=[deck.cards for deck in top],
            weights=[deck.plays / total for deck in top],
        )
        log(f"{segment}: played decks={len(decks):,}, panel={len(top)}")
    return dict(population), panels


def build_screen_panel(panel: SegmentPanel, size: int) -> None:
    """A stratified slice of the panel, renormalised, for the cheap first pass.

    Taking the top `size` decks would screen against the head of the meta only,
    which flatters mirror-adjacent decks. Sampling evenly across the ranked panel
    keeps the archetype spread that decides whether a deck has a bad matchup.
    """
    count = min(size, len(panel.decks))
    indexes = np.linspace(0, len(panel.decks) - 1, count).round().astype(int).tolist()
    indexes = sorted(set(indexes))
    weights = [panel.weights[index] for index in indexes]
    total = sum(weights)
    panel.screen_decks = [panel.decks[index] for index in indexes]
    panel.screen_weights = [weight / total for weight in weights]


def inference_row(
    team: tuple[int, ...], opponent: tuple[int, ...], segment: str, patch: str
) -> dict[str, Any]:
    zeros = [0] * 8

    def roles(cards: tuple[int, ...]) -> list[int]:
        return [ROLE_CHAMPION if card in CHAMPION_CARD_IDS else ROLE_NORMAL for card in cards]

    return {
        "team_card_ids": list(team),
        "opponent_card_ids": list(opponent),
        "team_evolution_levels": zeros,
        "opponent_evolution_levels": zeros,
        "team_hero_levels": zeros,
        "opponent_hero_levels": zeros,
        "team_card_roles": roles(team),
        "opponent_card_roles": roles(opponent),
        "team_tower_troop_id": TOWER_PRINCESS_ID,
        "opponent_tower_troop_id": TOWER_PRINCESS_ID,
        "segment": segment,
        "patch": patch,
        "matrix_prior": 0.5,
        "win": False,
    }


SIDE_FIELDS = (
    "cards",
    "elixir",
    "card_metadata",
    "card_present",
    "evos",
    "heroes",
    "roles",
    "tower",
)


class DeckScorer:
    """Score matchups from an encoded-deck cache instead of re-encoding rows.

    `predict_from_rows` rebuilds every tensor field in Python for each matchup:
    a dict lookup and a 45-wide metadata vector per card, twice per direction.
    Measured on this checkpoint that is 3.15 s per 4,096 matchups against 0.20 s
    of actual GPU work, so the card sits idle while the sweep eats CPU and RAM —
    the sweep is bound by encoding, not by the model.

    Nothing in the encoding of a side depends on its opponent, and this script
    holds evolutions, heroes, towers, segment and patch fixed. So each distinct
    deck is encoded once, kept on the device, and every matchup is assembled by
    indexing those rows. Encoding then costs O(decks) rather than O(matchups):
    a segment's 44,000 played decks are encoded once instead of appearing in
    700,000 screen rows.

    The reverse direction reuses the same cache with the two sides exchanged,
    which is what `encode_rows(..., swapped=True)` produces here — `target` is a
    label the model never reads, and `matrix_prior` is 0.5, its own mirror.
    Calibration and the antisymmetric projection are applied exactly as in
    `predict_from_rows`; `--verify-scorer` checks the two agree.
    """

    def __init__(
        self,
        bundle: dict[str, Any],
        segment: str,
        patch: str,
        device: torch.device,
        encode_chunk: int = 4096,
    ) -> None:
        self.bundle = bundle
        self.model = bundle["model"]
        self.vocabulary = bundle["vocabulary"]
        self.segment = segment
        self.patch = patch
        self.device = device
        self.encode_chunk = encode_chunk
        temperature, bias, _ = _calibration_for_segment(bundle, segment)
        self.temperature = max(float(temperature), 1e-4)
        self.bias = float(bias)
        self.index: dict[tuple[int, ...], int] = {}
        self.sides: dict[str, torch.Tensor] = {}
        vocabulary = bundle["vocabulary"]
        self.segment_id = int(vocabulary["segments"].get(str(segment), 0))
        self.patch_id = int(vocabulary["patches"].get(str(patch), 0))

    def register(self, decks: list[tuple[int, ...]]) -> None:
        """Encode any deck not seen yet, in chunks, and keep it on the device."""
        missing: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for deck in decks:
            if deck in self.index or deck in seen:
                continue
            seen.add(deck)
            missing.append(deck)
        if not missing:
            return
        for start in range(0, len(missing), self.encode_chunk):
            chunk = missing[start : start + self.encode_chunk]
            encoded = encode_rows(
                [inference_row(deck, deck, self.segment, self.patch) for deck in chunk],
                self.vocabulary,
            )
            for field in SIDE_FIELDS:
                values = encoded[f"team_{field}"].to(self.device)
                existing = self.sides.get(field)
                self.sides[field] = (
                    values if existing is None else torch.cat((existing, values), dim=0)
                )
            for deck in chunk:
                self.index[deck] = len(self.index)

    def _batch(self, team: torch.Tensor, opponent: torch.Tensor) -> dict[str, torch.Tensor]:
        count = team.shape[0]
        batch: dict[str, torch.Tensor] = {}
        for field in SIDE_FIELDS:
            values = self.sides[field]
            batch[f"team_{field}"] = values.index_select(0, team)
            batch[f"opponent_{field}"] = values.index_select(0, opponent)
        batch["segment"] = torch.full((count,), self.segment_id, dtype=torch.long, device=self.device)
        batch["patch"] = torch.full((count,), self.patch_id, dtype=torch.long, device=self.device)
        batch["matrix_prior"] = torch.full(
            (count,), 0.5, dtype=torch.float32, device=self.device
        )
        batch["target"] = torch.zeros(count, dtype=torch.float32, device=self.device)
        return batch

    def probabilities(self, team: torch.Tensor, opponent: torch.Tensor) -> torch.Tensor:
        """Projected win probability for the team side of each index pair."""
        with torch.no_grad():
            logits = self.model(self._batch(team, opponent))
            reverse_logits = self.model(self._batch(opponent, team))
            forward = torch.sigmoid(logits / self.temperature + self.bias)
            reverse = torch.sigmoid(reverse_logits / self.temperature + self.bias)
            projected, _, _ = _antisymmetric_projection(forward, reverse)
        return projected

    def score_pairs(
        self,
        decks: list[tuple[int, ...]],
        panel: list[tuple[int, ...]],
        batch_size: int,
        label: str,
        report_every: int,
    ) -> torch.Tensor:
        """Every deck against every panel opponent, as a (decks, panel) matrix."""
        self.register(decks)
        self.register(panel)
        deck_indexes = torch.tensor(
            [self.index[deck] for deck in decks], dtype=torch.long, device=self.device
        )
        panel_indexes = torch.tensor(
            [self.index[deck] for deck in panel], dtype=torch.long, device=self.device
        )
        total = len(decks) * len(panel)
        log(f"{self.segment}: {label} - {total:,} matchups over {len(decks):,} decks")
        rows_per_batch = max(1, batch_size // max(1, len(panel)))
        chunks: list[torch.Tensor] = []
        for start in range(0, len(decks), rows_per_batch):
            block = deck_indexes[start : start + rows_per_batch]
            team = block.repeat_interleave(len(panel))
            opponent = panel_indexes.repeat(block.shape[0])
            chunks.append(self.probabilities(team, opponent).view(block.shape[0], len(panel)))
            done = min(start + rows_per_batch, len(decks)) * len(panel)
            if done == total or (start // rows_per_batch) % report_every == 0:
                log(f"{self.segment}: {label} {done:,}/{total:,}")
        return torch.cat(chunks, dim=0)


def deck_elixir(cards: tuple[int, ...]) -> float:
    values = [CARD_ELIXIR.get(int(card), 0) for card in cards]
    known = [value for value in values if value > 0]
    return sum(known) / len(known) if known else 0.0


def score_decks(
    scorer: DeckScorer,
    decks: list[tuple[int, ...]],
    panel_decks: list[tuple[int, ...]],
    panel_weights: list[float],
    batch_size: int,
    label: str,
) -> list[Profile]:
    """Full weighted sweep of every deck against every panel opponent.

    Mirrors `finalizeDeckSweep` in the site's `ml-inference.ts`: the antisymmetric
    projection, the same three share thresholds, and the same weighted variance.
    """
    if not decks:
        return []
    matrix = scorer.score_pairs(decks, panel_decks, batch_size, label, report_every=40)
    weights = torch.tensor(panel_weights, dtype=matrix.dtype, device=matrix.device)
    scores = (matrix * weights).sum(dim=1)
    squares = (matrix * matrix * weights).sum(dim=1)
    unfavorable = ((matrix <= UNFAVORABLE_THRESHOLD) * weights).sum(dim=1)
    hard = ((matrix <= HARD_COUNTER_THRESHOLD) * weights).sum(dim=1)
    favorable = ((matrix >= FAVORABLE_THRESHOLD) * weights).sum(dim=1)
    worst = matrix.min(dim=1).values

    scores_list = scores.tolist()
    squares_list = squares.tolist()
    unfavorable_list = unfavorable.tolist()
    hard_list = hard.tolist()
    favorable_list = favorable.tolist()
    worst_list = worst.tolist()

    return [
        Profile(
            score=float(scores_list[index]),
            unfavorable_share=float(unfavorable_list[index]),
            hard_counter_share=float(hard_list[index]),
            favorable_share=float(favorable_list[index]),
            worst_matchup=float(worst_list[index]),
            sigma=float(math.sqrt(max(0.0, squares_list[index] - scores_list[index] ** 2))),
            elixir=deck_elixir(decks[index]),
        )
        for index in range(len(decks))
    ]


def screen_scores(
    scorer: DeckScorer,
    decks: list[tuple[int, ...]],
    panel: SegmentPanel,
    batch_size: int,
) -> np.ndarray:
    """Mean-only pass against the reduced panel, to find the tail cheaply."""
    matrix = scorer.score_pairs(decks, panel.screen_decks, batch_size, "screen", report_every=60)
    weights = torch.tensor(panel.screen_weights, dtype=matrix.dtype, device=matrix.device)
    return (matrix * weights).sum(dim=1).cpu().numpy()


def verify_scorer(
    scorer: DeckScorer,
    bundle: dict[str, Any],
    patch: str,
    decks: list[tuple[int, ...]],
    panel_decks: list[tuple[int, ...]],
    sample: int,
) -> float:
    """Largest gap between the cached scorer and `predict_from_rows`.

    The cache is only ever worth using if it reproduces the reference path, so
    the run measures that rather than assuming it.
    """
    rng = random.Random(0)
    pairs = [
        (rng.choice(decks), rng.choice(panel_decks))
        for _ in range(min(sample, len(decks) * len(panel_decks)))
    ]
    rows = [inference_row(team, opponent, scorer.segment, patch) for team, opponent in pairs]
    reference = [
        0.5
        * (
            float(prediction["team_win_probability"])
            + 1.0
            - float(prediction["opponent_win_probability"])
        )
        for prediction in predict_from_rows(bundle, rows)
    ]
    scorer.register([team for team, _ in pairs])
    scorer.register([opponent for _, opponent in pairs])
    team_index = torch.tensor(
        [scorer.index[team] for team, _ in pairs], dtype=torch.long, device=scorer.device
    )
    opponent_index = torch.tensor(
        [scorer.index[opponent] for _, opponent in pairs], dtype=torch.long, device=scorer.device
    )
    cached = scorer.probabilities(team_index, opponent_index).tolist()
    return max(abs(left - right) for left, right in zip(cached, reference, strict=True))


def legal(cards: tuple[int, ...]) -> bool:
    """One champion at most, no duplicates — the game's own deck rules."""
    if len(set(cards)) != 8:
        return False
    return sum(1 for card in cards if card in CHAMPION_CARD_IDS) <= 1


def beam_search(
    scorer: DeckScorer,
    seed: tuple[int, ...],
    pool: list[int],
    panel: SegmentPanel,
    rounds: int,
    beam_width: int,
    batch_size: int,
    admits: Callable[[tuple[int, ...]], bool] | None = None,
) -> list[dict[str, Any]]:
    """The site's greedy beam, reproduced: maximise panel score, nothing else.

    Each round swaps one further slot, keeps the `beam_width` best decks, and
    records the leader's full profile so the trajectory out of the real
    population is visible round by round.

    `admits` is the admission rule under test. Passing one turns this into the
    constrained search the product would actually ship: candidates the rule
    rejects never enter the beam, so the comparison against the unconstrained
    run measures what the guard rail changes on decks nobody has ever played —
    exactly the population the optimizer benchmark cannot reach, because those
    decks have no battles to hold out.
    """
    seed_profile = score_decks(
        scorer, [seed], panel.decks, panel.weights, batch_size, "search seed"
    )[0]
    trajectory = [{"round": 0, "cards": list(seed), "profile": vars(seed_profile)}]
    beam: list[tuple[tuple[int, ...], frozenset[int]]] = [(seed, frozenset())]

    for round_index in range(1, rounds + 1):
        variants: list[tuple[tuple[int, ...], frozenset[int]]] = []
        seen: set[tuple[int, ...]] = set()
        rejected = 0
        for deck, changed in beam:
            for slot in range(8):
                if slot in changed:
                    continue
                for card in pool:
                    if card in deck:
                        continue
                    candidate = deck[:slot] + (card,) + deck[slot + 1 :]
                    if not legal(candidate):
                        continue
                    fingerprint = tuple(sorted(candidate))
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    if admits is not None and not admits(candidate):
                        rejected += 1
                        continue
                    variants.append((candidate, changed | {slot}))
        if not variants:
            break
        profiles = score_decks(
            scorer,
            [variant for variant, _ in variants],
            panel.decks,
            panel.weights,
            batch_size,
            f"search round {round_index}",
        )
        ranked = sorted(
            zip(variants, profiles, strict=True), key=lambda item: item[1].score, reverse=True
        )
        beam = [variant for variant, _ in ranked[:beam_width]]
        best_variant, best_profile = ranked[0]
        considered = len(variants) + rejected
        trajectory.append(
            {
                "round": round_index,
                "cards": list(best_variant[0]),
                "profile": vars(best_profile),
                "candidates_considered": considered,
                "candidates_rejected": rejected,
                "rejected_share": rejected / considered if considered else 0.0,
            }
        )
        log(
            f"{scorer.segment}: search round {round_index} best={best_profile.score:.4f} "
            f"unfavorable={best_profile.unfavorable_share:.4f} elixir={best_profile.elixir:.2f}"
        )
    return trajectory


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "p999": float(np.percentile(array, 99.9)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def score_band_floors(profiles: list[Profile]) -> list[dict[str, Any]]:
    """Per score band, how little unfavorable exposure a real deck ever shows.

    This is the joint the guard rail needs. A ceiling on the mean alone cannot
    catch a deck that stays under it while claiming an impossible profile; the
    floors say what exposure the real population still carries at that level.
    """
    bands = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]
    output: list[dict[str, Any]] = []
    for low, high in bands:
        inside = [
            profile for profile in profiles if low <= profile.score < high
        ]
        if not inside:
            output.append({"band": [low, high], "decks": 0})
            continue
        unfavorable = np.asarray([profile.unfavorable_share for profile in inside])
        hard = np.asarray([profile.hard_counter_share for profile in inside])
        worst = np.asarray([profile.worst_matchup for profile in inside])
        output.append(
            {
                "band": [low, high],
                "decks": len(inside),
                "unfavorable_min": float(unfavorable.min()),
                "unfavorable_p01": float(np.percentile(unfavorable, 1)),
                "unfavorable_p05": float(np.percentile(unfavorable, 5)),
                "unfavorable_p50": float(np.percentile(unfavorable, 50)),
                "hard_counter_p05": float(np.percentile(hard, 5)),
                "hard_counter_p50": float(np.percentile(hard, 50)),
                "worst_matchup_p95": float(np.percentile(worst, 95)),
                "worst_matchup_max": float(worst.max()),
                "zero_unfavorable_decks": int((unfavorable <= 1e-9).sum()),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    catalog = Catalog(args.catalog)
    admits = (
        None
        if args.search_admission == "none"
        else lambda cards: admits_deck(catalog, args.search_admission, cards)
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    device = torch.device(resolve_device(args.device))

    bundle = load_bundle(args.checkpoint)
    bundle["model"].to(device)
    log(
        f"loaded model {bundle['resolved_model_version']} on "
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}"
    )
    patch = max(str(value) for value in bundle["vocabulary"]["patches"])
    vocabulary_cards = {int(card) for card in bundle["vocabulary"]["cards"]}

    connection = duckdb.connect()
    connection.execute(f"pragma threads={args.threads}")
    connection.execute(f"pragma memory_limit='{args.memory_limit}'")
    parquet = raw_glob(args.raw)
    rows = population_rows(
        connection,
        args.raw,
        parquet,
        args.min_plays,
        args.population_cache,
        args.rebuild_population,
    )
    connection.close()
    population, panels = load_population(rows, args.panel_size)
    del rows

    segments = args.segments or sorted(population)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "min_plays": args.min_plays,
        "panel_size": args.panel_size,
        "screen_panel_size": args.screen_panel_size,
        "body_sample": args.body_sample,
        "search_admission": args.search_admission,
        "segments": {},
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        report["segments"] = previous.get("segments", {})
        log(f"resuming — {len(report['segments'])} segment(s) already measured")

    for segment in segments:
        decks = population.get(segment)
        panel = panels.get(segment)
        if not decks or not panel:
            continue
        wants_search = not args.skip_search and (
            args.search_segments is None or segment in args.search_segments
        )
        done = report["segments"].get(segment)
        # A segment written by an earlier --skip-search run carries no searches,
        # so resuming into a searching run has to redo it rather than inherit it.
        if args.resume and done is not None and (not wants_search or "searches" in done):
            log(f"{segment}: already in {args.output}, skipped")
            continue
        build_screen_panel(panel, args.screen_panel_size)
        scorer = DeckScorer(bundle, segment, patch, device, args.encode_chunk)
        if args.verify_scorer:
            gap = verify_scorer(
                scorer,
                bundle,
                patch,
                [deck.cards for deck in decks[:200]],
                panel.decks,
                args.verify_scorer,
            )
            log(f"{segment}: scorer agrees with predict_from_rows to {gap:.2e}")

        # Stage 1 — the body of the real population.
        sample = decks if len(decks) <= args.body_sample else rng.sample(decks, args.body_sample)
        body_profiles = score_decks(
            scorer,
            [deck.cards for deck in sample],
            panel.decks,
            panel.weights,
            args.batch_size,
            "body",
        )
        for deck, profile in zip(sample, body_profiles, strict=True):
            deck.profile = profile

        # Stage 2 — the ceiling, screened over every played deck then rescored.
        screen = screen_scores(scorer, [deck.cards for deck in decks], panel, args.batch_size)
        for deck, value in zip(decks, screen.tolist(), strict=True):
            deck.screen_score = value
        leaders = sorted(decks, key=lambda deck: deck.screen_score or 0.0, reverse=True)[
            : args.ceiling_rescore
        ]
        leader_profiles = score_decks(
            scorer,
            [deck.cards for deck in leaders],
            panel.decks,
            panel.weights,
            args.batch_size,
            "ceiling rescore",
        )
        for deck, profile in zip(leaders, leader_profiles, strict=True):
            deck.profile = profile

        real_profiles = body_profiles + leader_profiles
        ceiling_index = int(np.argmax([profile.score for profile in leader_profiles]))
        ceiling_deck = leaders[ceiling_index]
        ceiling_profile = leader_profiles[ceiling_index]

        entry: dict[str, Any] = {
            "played_decks": len(decks),
            "body_sample": len(sample),
            "ceiling_rescored": len(leaders),
            "body_score": percentiles([profile.score for profile in body_profiles]),
            "body_unfavorable_share": percentiles(
                [profile.unfavorable_share for profile in body_profiles]
            ),
            "body_elixir": percentiles([profile.elixir for profile in body_profiles]),
            "ceiling": {
                "score": ceiling_profile.score,
                "unfavorable_share": ceiling_profile.unfavorable_share,
                "hard_counter_share": ceiling_profile.hard_counter_share,
                "worst_matchup": ceiling_profile.worst_matchup,
                "sigma": ceiling_profile.sigma,
                "elixir": ceiling_profile.elixir,
                "plays": ceiling_deck.plays,
                "observed_win_rate": ceiling_deck.win_rate,
                "cards": list(ceiling_deck.cards),
            },
            "ceiling_score": percentiles([profile.score for profile in leader_profiles]),
            "score_bands": score_band_floors(real_profiles),
        }

        # Stage 3 — where an unconstrained search actually goes.
        if wants_search:
            # Swap-ins are the cards the segment's own meta actually uses, most
            # played first — the same shortlist shape the site's optimizer builds.
            usage: dict[int, int] = defaultdict(int)
            for deck in decks[:400]:
                for card in deck.cards:
                    if card in vocabulary_cards:
                        usage[card] += deck.plays
            pool = [
                card
                for card, _ in sorted(usage.items(), key=lambda item: (-item[1], item[0]))
            ][: args.search_pool]
            if args.search_seed_source == "sample":
                low, high = args.search_seed_plays
                band = [deck for deck in decks if low <= deck.plays <= high]
                seeds = (
                    rng.sample(band, min(args.search_starts, len(band))) if band else []
                )
            else:
                seeds = decks[: args.search_starts]
            searches = []
            for seed_deck in seeds:
                unconstrained = beam_search(
                    scorer,
                    seed_deck.cards,
                    pool,
                    panel,
                    args.search_rounds,
                    args.search_beam,
                    args.batch_size,
                )
                for step in unconstrained:
                    cards = tuple(step["cards"])
                    step["admitted"] = {
                        name: admits_deck(catalog, name, cards) for name in ENVELOPES
                    }
                search: dict[str, Any] = {
                    "seed_plays": seed_deck.plays,
                    "seed_win_rate": seed_deck.win_rate,
                    "trajectory": unconstrained,
                }
                # The same search under the guard rail. Its emitted deck has no
                # battles either, so the honest comparison is not "which one is
                # better" but how much panel score the rule gives up and whether
                # what it emits stays inside the real population's envelope.
                if admits is not None:
                    search["constrained"] = {
                        "envelope": args.search_admission,
                        "trajectory": beam_search(
                            scorer,
                            seed_deck.cards,
                            pool,
                            panel,
                            args.search_rounds,
                            args.search_beam,
                            args.batch_size,
                            admits,
                        ),
                    }
                searches.append(search)
            entry["searches"] = searches

        report["segments"][segment] = entry
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"{segment}: written to {args.output}")

    log(f"done — {args.output}")


if __name__ == "__main__":
    main()
