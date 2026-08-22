"""FastAPI inference server for the antisymmetric matchup model.

Serves the trained PyTorch ``SymmetricMatchupModel`` over the exact HTTP
contract the Rigged Royale site speaks (``POST /predict`` and vectorised
``POST /predict/batch``, see ``riggedroyale/src/lib/ml-inference.ts``). The
site sends a thin request
(decks, mode, towers, average levels, evolved-card ids); this module adapts it
into the rich row :func:`encode_row` expects, fills the fields the site does not
send with safe defaults, and maps the model output back to the site's
``{bad,neutral,good}`` / ``{low,medium,high}`` response shape.

Run locally:  ``rigged-matchup serve``  (or ``uvicorn rigged_matchup_ml.serve:app``).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from .card_stats import CHAMPION_CARD_IDS
from .domain import ROLE_CHAMPION, ROLE_HERO, ROLE_NORMAL, Deck, segment_for
from .hero_evo_correction import CORRECTION_KEY
from .predictor import (
    load_bundle,
    predict_from_row,
    predict_from_rows,
    resolve_device,
    warm_up_bundle,
)

DEFAULT_CHECKPOINT = "artifacts/matchup-model.pt"
DEFAULT_MODEL_NAME = "symmetric-matchup"

# Mirror config/default.yaml so the current checkpoint (which predates the
# embedded ``data_config``) still resolves real segments. Newer checkpoints carry
# their own ``data_config`` and override this (see ``_data_config``).
DEFAULT_DATA_CONFIG = {
    "trophy_buckets": [0, 5000, 7000, 9000, 12000, 14000, 999999],
    "top_ladder_buckets": [100, 1000, 10000],
    "ranked_league_buckets": [1, 3, 5, 8],
}


# --- request / response contract (mirrors the site's ml-inference.ts) ---------


class MatchupRequest(BaseModel):
    team_card_ids: list[int] = Field(..., min_length=8, max_length=8)
    opponent_card_ids: list[int] = Field(..., min_length=8, max_length=8)
    mode_key: str = "ladder"
    team_tower_troop_id: int | None = None
    opponent_tower_troop_id: int | None = None
    team_avg_card_level: float | None = None
    opponent_avg_card_level: float | None = None
    trophy_diff: int | None = 0
    # Segmentation inputs (the site's player bracket). Without these the request
    # cannot be placed in its trained segment and falls back to the mode default,
    # so per-segment calibration never applies. ``team_trophies`` /
    # ``team_global_rank`` drive the ladder bucket; ``league_number`` the ranked
    # league. All optional for backward compatibility.
    team_trophies: int | None = None
    team_global_rank: int | None = None
    league_number: int | None = None
    team_evolution_card_ids: list[int] = Field(default_factory=list)
    opponent_evolution_card_ids: list[int] = Field(default_factory=list)
    # Cards fielded in their Hero form (March 2026 Hero slot) -- distinct from
    # Champion class and from Evolution. Mirrors the evolution-ids pattern; without
    # it the Hero form is invisible at inference even though training sees it.
    team_hero_card_ids: list[int] = Field(default_factory=list)
    opponent_hero_card_ids: list[int] = Field(default_factory=list)
    # Per-card counter / synergy attributions are only needed for the headline
    # matchup of each battle, not the expected-context or meta-panel requests, so
    # the caller opts in to avoid the extra attribution/ablation passes on the
    # high-fan-out requests.
    include_interactions: bool = False

    @field_validator("team_card_ids", "opponent_card_ids")
    @classmethod
    def _unique_cards(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("deck cards must be unique")
        return value


class CardInteraction(BaseModel):
    source_card_id: int
    target_card_id: int
    weight: float
    # Signed raw-logit effect from leave-one-pair-out ablation. Positive favors
    # the requested team; negative favors the opponent.
    contribution: float | None = None


class CardInteractions(BaseModel):
    answers: list[CardInteraction]
    threats: list[CardInteraction]


class PredictionResponse(BaseModel):
    win_probability: float
    matchup_label: str
    confidence: str
    model_run_id: int | None
    model_name: str | None
    model_version: str | None
    explanation: dict
    # Model-derived attention plus signed ablation; absent when the request did not
    # opt in or the checkpoint has no interaction terms.
    card_interactions: CardInteractions | None = None
    synergies: list[CardInteraction] | None = None


def _max_batch_requests() -> int:
    """Hard cap on one ``/predict/batch`` payload.

    Read once at import, because pydantic bakes the bound into the model class.
    The cap exists to keep a single request from monopolising the service and to
    bound its memory, not to tune throughput — the caller picks the actual batch
    size. Raise it when the site raises ``ML_INFERENCE_BATCH_SIZE``; the two must
    move together or the larger payloads bounce with a 422.
    """
    try:
        return max(1, int(os.getenv("MAX_BATCH_REQUESTS", "2048")))
    except ValueError:
        return 2048


MAX_BATCH_REQUESTS = _max_batch_requests()


class BatchMatchupRequest(BaseModel):
    requests: list[MatchupRequest] = Field(
        ..., min_length=1, max_length=MAX_BATCH_REQUESTS
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


# --- output mapping (ported from the obsolete sklearn container) ---------------


def probability_to_label(probability: float) -> str:
    if probability >= 0.55:
        return "good"
    if probability <= 0.45:
        return "bad"
    return "neutral"


def probability_to_confidence(probability: float) -> str:
    distance = abs(probability - 0.5)
    if distance < 0.04:
        return "low"
    if distance >= 0.12:
        return "high"
    return "medium"


# --- request -> model row adapter ---------------------------------------------


def default_segment(vocabulary: dict[str, Any], mode_key: str) -> str:
    """Pick a segment that exists in the checkpoint vocabulary for this mode.

    The model only learned ladder / ranked segments, so non-ladder/ranked modes
    fall back to the representative ladder segment. Choosing a segment present in
    the vocabulary means the fitted per-segment calibration applies.
    """
    segments = vocabulary.get("segments", {})
    mode = (mode_key or "").strip().lower()
    if mode.startswith("ranked"):
        if "ranked:unknown" in segments:
            return "ranked:unknown"
        ranked = sorted(s for s in segments if s.startswith("ranked"))
        if ranked:
            return ranked[0]
    if "ladder:9000-11999" in segments:
        return "ladder:9000-11999"
    ladder = sorted(s for s in segments if s.startswith("ladder"))
    if ladder:
        return ladder[0]
    return next(iter(segments), "ladder:9000-11999")


def latest_patch(vocabulary: dict[str, Any]) -> str:
    """Latest patch key in the vocabulary (YYYY-MM strings sort lexicographically)."""
    patches = [str(p) for p in vocabulary.get("patches", {}) if p not in (None, "", "0")]
    return max(patches) if patches else ""


def _data_config(bundle: dict[str, Any]) -> dict[str, Any]:
    """Trophy / top-ladder bucket edges for segment resolution.

    Prefers the buckets embedded in the checkpoint (newer trainings) and falls
    back to the config/default.yaml values so checkpoints saved before the
    embedded ``data_config`` still segment correctly.
    """
    return bundle.get("data_config") or DEFAULT_DATA_CONFIG


def resolve_segment(
    request: MatchupRequest, vocabulary: dict[str, Any], data_config: dict[str, Any]
) -> str:
    """Place the request in its trained segment from the site's bracket inputs.

    Reuses the exact training-time rule (:func:`domain.segment_for`): ladder
    splits by global rank then trophy bucket; ranked by configured league group.
    Falls back to :func:`default_segment` when the request carries no bracket
    info OR the resolved segment is not in the checkpoint vocabulary (an unknown
    segment has no fitted calibration, so the representative default is safer).
    """
    mode = (request.mode_key or "").strip().lower()
    has_bracket = (
        request.team_trophies is not None
        or request.team_global_rank is not None
        or request.league_number is not None
    )
    if not has_bracket:
        return default_segment(vocabulary, request.mode_key)

    deck = Deck(
        cards=(),
        tower_troop_id=request.team_tower_troop_id or 0,
        tag="",
        crowns=0,
        starting_trophies=request.team_trophies,
        global_rank=request.team_global_rank,
    )
    raw = {"leagueNumber": request.league_number} if request.league_number else {}
    segment = segment_for(deck, mode, data_config, raw)

    segments = vocabulary.get("segments", {})
    if segment in segments:
        return segment
    return default_segment(vocabulary, request.mode_key)


def request_to_row(request: MatchupRequest, bundle: dict[str, Any]) -> dict[str, Any]:
    """Adapt the site's thin request into a full row for :func:`encode_row`.

    Evolution and Hero forms arrive as lists of card ids and are spread back onto
    the per-card-position arrays. Card roles are reconstructed from the ids: the
    Champion role from the static ``CHAMPION_CARD_IDS`` set (the site sends no card
    rarity) and the Hero role from the hero-form ids. Elixir is derived from the
    card ids inside :func:`encode_row`, so it needs no field here. The segment is
    resolved from the request's bracket inputs (trophies / rank / league) and only
    falls back to the mode default when those are absent.
    """
    vocabulary = bundle["vocabulary"]
    team_cards = list(request.team_card_ids)
    opponent_cards = list(request.opponent_card_ids)
    team_evos = set(request.team_evolution_card_ids)
    opponent_evos = set(request.opponent_evolution_card_ids)
    team_heroes = set(request.team_hero_card_ids)
    opponent_heroes = set(request.opponent_hero_card_ids)

    def roles_for(cards: list[int], hero_ids: set[int]) -> list[int]:
        # Hero form takes precedence over Champion class when a card is both sent
        # as a hero and is a champion id (should not normally overlap).
        return [
            ROLE_HERO
            if card in hero_ids
            else ROLE_CHAMPION
            if card in CHAMPION_CARD_IDS
            else ROLE_NORMAL
            for card in cards
        ]

    return {
        "team_card_ids": team_cards,
        "opponent_card_ids": opponent_cards,
        "team_evolution_levels": [1 if card in team_evos else 0 for card in team_cards],
        "opponent_evolution_levels": [1 if card in opponent_evos else 0 for card in opponent_cards],
        "team_hero_levels": [1 if card in team_heroes else 0 for card in team_cards],
        "opponent_hero_levels": [1 if card in opponent_heroes else 0 for card in opponent_cards],
        "team_card_roles": roles_for(team_cards, team_heroes),
        "opponent_card_roles": roles_for(opponent_cards, opponent_heroes),
        "team_tower_troop_id": request.team_tower_troop_id,
        "opponent_tower_troop_id": request.opponent_tower_troop_id,
        "segment": resolve_segment(request, vocabulary, _data_config(bundle)),
        "patch": latest_patch(vocabulary),
        "matrix_prior": 0.5,
    }


def response_from_result(
    bundle: dict[str, Any],
    request: MatchupRequest,
    result: dict[str, Any],
) -> PredictionResponse:
    probability = max(0.0, min(1.0, float(result["team_win_probability"])))
    model_version = bundle.get("resolved_model_version")

    interactions = result.get("interactions")
    card_interactions: CardInteractions | None = None
    synergies: list[CardInteraction] | None = None
    if interactions is not None:
        card_interactions = CardInteractions(
            answers=[CardInteraction(**hit) for hit in interactions["answers"]],
            threats=[CardInteraction(**hit) for hit in interactions["threats"]],
        )
        synergies = [CardInteraction(**hit) for hit in interactions["synergies"]]

    return PredictionResponse(
        win_probability=round(probability, 6),
        matchup_label=probability_to_label(probability),
        confidence=probability_to_confidence(probability),
        model_run_id=None,
        model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME,
        model_version=model_version,
        explanation={
            "segment": result["segment"],
            "patch": result["patch"],
            "raw_win_probability": round(float(result["raw_team_win_probability"]), 6),
            "pre_correction_win_probability": round(
                float(result["pre_correction_team_win_probability"]), 6
            ),
            "hero_evo_logit_adjustment": round(
                float(result["hero_evo_logit_adjustment"]), 6
            ),
            "symmetry_error": round(float(result["symmetry_error"]), 6),
            "calibration_symmetry_error": round(float(result["calibration_symmetry_error"]), 6),
            "temperature": result["temperature"],
            "bias": result["bias"],
        },
        card_interactions=card_interactions,
        synergies=synergies,
    )


def build_response(bundle: dict[str, Any], request: MatchupRequest) -> PredictionResponse:
    row = request_to_row(request, bundle)
    result = predict_from_row(bundle, row, include_interactions=request.include_interactions)
    return response_from_result(bundle, request, result)


def build_batch_response(
    bundle: dict[str, Any], payload: BatchMatchupRequest
) -> BatchPredictionResponse:
    """Vectorise regular predictions; preserve opt-in interaction semantics."""
    rows = [request_to_row(item, bundle) for item in payload.requests]
    results: list[dict[str, Any] | None] = [None] * len(rows)
    regular_indices = [
        index for index, item in enumerate(payload.requests) if not item.include_interactions
    ]
    regular_results = predict_from_rows(bundle, [rows[index] for index in regular_indices])
    for index, result in zip(regular_indices, regular_results, strict=True):
        results[index] = result

    for index, item in enumerate(payload.requests):
        if item.include_interactions:
            results[index] = predict_from_row(bundle, rows[index], include_interactions=True)

    return BatchPredictionResponse(
        predictions=[
            response_from_result(bundle, request, result)
            for request, result in zip(payload.requests, results, strict=True)
            if result is not None
        ]
    )


# --- app ----------------------------------------------------------------------


def _checkpoint_path() -> Path:
    return Path(os.getenv("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))


def _warm_up_enabled() -> bool:
    """Whether to burn one throwaway batch at startup. On by default.

    Worth turning off (``MODEL_WARMUP=0``) only when boot time matters more than
    first-request latency — which is the opposite of the serverless GPU case.
    """
    return (os.getenv("MODEL_WARMUP", "1") or "1").strip().lower() not in ("0", "false", "no")


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = resolve_device()
    bundle = load_bundle(_checkpoint_path(), device=device)
    print(f"Matchup model loaded on device={bundle['device']}", flush=True)
    if _warm_up_enabled():
        warm_up_bundle(bundle)
    app.state.bundle = bundle
    yield


app = FastAPI(title="Rigged Royale Matchup ML Inference", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def server_timing(request: Request, call_next):
    """Report what the service itself spent, so callers can subtract the network.

    A remote batch caller sees one number: the round trip. That number is
    useless on its own — it cannot tell a slow model from a slow link, and for
    this service the two differ by an order of magnitude. The standard
    ``Server-Timing`` header splits it:

      total  everything inside the process: reading and validating the payload,
             encoding, the forward passes, and serialising the response
      model  just the encode + forward + calibration work

    ``round_trip - total`` is therefore transfer and routing, and
    ``total - model`` is JSON and validation overhead. Both are things you would
    otherwise have to guess at.
    """
    started = time.perf_counter()
    response = await call_next(request)
    parts = [f"total;dur={(time.perf_counter() - started) * 1000:.1f}"]
    model_ms = getattr(request.state, "model_ms", None)
    if model_ms is not None:
        parts.append(f"model;dur={model_ms:.1f}")
    response.headers["Server-Timing"] = ", ".join(parts)
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("PREDICT_API_KEY", "").strip() or None
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _supports_heroes(bundle: dict[str, Any] | None) -> bool:
    """Whether the loaded checkpoint has trained Hero-form parameters.

    The site fails closed on this flag (see ``runtimeFromHealth`` in
    ``ml-inference.ts``): without it, every request is stripped of its Hero card
    ids and a Hero deck predicts exactly like its base deck. Report it from the
    checkpoint itself rather than a constant so an older Hero-less model keeps
    the stripping behaviour it needs.
    """
    if bundle is None:
        return False
    state = bundle.get("model_state") or {}
    return any("hero" in key for key in state)


@app.get("/ping")
def ping(request: Request) -> Response:
    """Readiness probe for a RunPod load-balancing endpoint.

    RunPod's load balancer polls this path (override it with ``HEALTH_CHECK_PATH``)
    and reads only the status code: 200 means the worker may receive traffic, 204
    means it is still initialising, anything else means unhealthy. It must stay
    unauthenticated — the balancer sends no API key — and stay cheap, since it is
    polled continuously for every running worker.

    The model loads during FastAPI's lifespan, so in practice the socket is not
    accepting connections until the bundle is ready; the 204 branch covers the
    case where the app is reachable without one.
    """
    bundle = getattr(request.app.state, "bundle", None)
    return Response(status_code=200 if bundle is not None else 204)


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    bundle = getattr(request.app.state, "bundle", None)
    supports_heroes = _supports_heroes(bundle)
    correction = bundle.get(CORRECTION_KEY) if bundle else None
    return {
        "ok": bundle is not None,
        "model_name": os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
        "model_version": bundle.get("resolved_model_version") if bundle else None,
        "supports_heroes": supports_heroes,
        "capabilities": {
            "heroes": supports_heroes,
            "evolutions": bundle is not None,
            "hero_evo_statistical_correction": correction is not None,
        },
        "hero_evo_correction": {
            "enabled": correction is not None,
            "shrinkage": correction.get("shrinkage") if correction else None,
            "heroes_covered": len(correction.get("coefficients_by_hero_card_id", {}))
            if correction
            else 0,
        },
        # Which device actually served the request, so a GPU worker that silently
        # fell back to CPU is visible from the outside instead of only in logs.
        "device": bundle.get("device") if bundle else None,
        # The largest payload /predict/batch will accept. Published so a caller
        # sending oversized batches can be diagnosed from /health rather than
        # from a wall of 422s.
        "max_batch_requests": MAX_BATCH_REQUESTS,
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(request: Request, payload: MatchupRequest) -> PredictionResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:  # pragma: no cover - lifespan always loads it
        raise HTTPException(status_code=503, detail="Model not loaded")
    return build_response(bundle, payload)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict_batch(request: Request, payload: BatchMatchupRequest) -> BatchPredictionResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:  # pragma: no cover - lifespan always loads it
        raise HTTPException(status_code=503, detail="Model not loaded")
    started = time.perf_counter()
    response = build_batch_response(bundle, payload)
    # Read back by the Server-Timing middleware. Measured here rather than around
    # the whole request so it excludes payload validation and response
    # serialisation, which is exactly the distinction that makes the header
    # worth having.
    request.state.model_ms = (time.perf_counter() - started) * 1000
    return response
