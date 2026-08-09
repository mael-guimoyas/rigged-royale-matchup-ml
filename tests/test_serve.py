from __future__ import annotations

from pathlib import Path

import pytest

from rigged_matchup_ml.serve import (
    MatchupRequest,
    build_response,
    default_segment,
    latest_patch,
    probability_to_confidence,
    probability_to_label,
    request_to_row,
)

CHECKPOINT = Path("artifacts/matchup-model.pt")

FAKE_VOCAB = {
    "segments": {
        "ladder:5000-6999": 1,
        "ladder:9000-11999": 2,
        "ranked:unknown": 3,
    },
    "patches": {"2026-05": 1, "2026-06": 2},
}

# A web-shaped request (mirrors riggedroyale ml-inference.ts PredictRequest).
WEB_PAYLOAD = {
    "team_card_ids": [26000000, 26000030, 26000021, 26000014, 27000006, 28000000, 28000011, 26000064],
    "opponent_card_ids": [26000055, 26000011, 26000007, 26000012, 27000003, 28000001, 28000004, 26000018],
    "mode_key": "ladder",
    "team_tower_troop_id": 159000000,
    "opponent_tower_troop_id": 159000000,
    "team_avg_card_level": 14.0,
    "opponent_avg_card_level": 14.0,
    "trophy_diff": 0,
    "team_evolution_card_ids": [26000000, 27000006, 26000064],
    "opponent_evolution_card_ids": [26000055, 26000011],
}


# --- pure adapter / mapping tests (no model needed) ---------------------------


def test_default_segment_picks_vocab_member_per_mode() -> None:
    assert default_segment(FAKE_VOCAB, "ladder") == "ladder:9000-11999"
    assert default_segment(FAKE_VOCAB, "ranked") == "ranked:unknown"
    # Modes the model never saw fall back to the representative ladder segment.
    assert default_segment(FAKE_VOCAB, "events") == "ladder:9000-11999"


def test_latest_patch_is_lexicographic_max() -> None:
    assert latest_patch(FAKE_VOCAB) == "2026-06"
    assert latest_patch({"patches": {}}) == ""


def test_request_to_row_spreads_evolution_ids_onto_positions() -> None:
    request = MatchupRequest(**WEB_PAYLOAD)
    row = request_to_row(request, {"vocabulary": FAKE_VOCAB})

    assert row["team_card_ids"] == WEB_PAYLOAD["team_card_ids"]
    # Evolved-card ids become 1s at the matching deck positions, 0 elsewhere.
    assert row["team_evolution_levels"] == [1, 0, 0, 0, 1, 0, 0, 1]
    assert row["opponent_evolution_levels"] == [1, 1, 0, 0, 0, 0, 0, 0]
    # Defaults for fields the site does not send.
    assert row["team_hero_levels"] == [0] * 8
    assert row["team_card_roles"] == [1] * 8
    assert row["segment"] == "ladder:9000-11999"
    assert row["patch"] == "2026-06"
    assert row["matrix_prior"] == 0.5


def test_request_to_row_routes_ranked_leagues_to_trained_groups() -> None:
    vocabulary = {
        **FAKE_VOCAB,
        "segments": {
            "ranked:league-1-2": 1,
            "ranked:league-3-4": 2,
            "ranked:league-5-7": 3,
        },
    }
    bundle = {
        "vocabulary": vocabulary,
        "data_config": {
            "trophy_buckets": [0, 5000, 7000, 9000, 12000, 14000, 999999],
            "top_ladder_buckets": [100, 1000, 10000],
            "ranked_league_buckets": [1, 3, 5, 8],
        },
    }
    expected = {
        1: "ranked:league-1-2",
        3: "ranked:league-3-4",
        5: "ranked:league-5-7",
        7: "ranked:league-5-7",
    }
    for league, segment in expected.items():
        request = MatchupRequest(
            **{**WEB_PAYLOAD, "mode_key": "ranked", "league_number": league}
        )
        assert request_to_row(request, bundle)["segment"] == segment


def test_request_to_row_keeps_legacy_exact_ranked_checkpoint_compatible() -> None:
    vocabulary = {
        **FAKE_VOCAB,
        "segments": {"ranked:league-6": 1},
    }
    bundle = {
        "vocabulary": vocabulary,
        "data_config": {
            "trophy_buckets": [0, 5000, 7000, 9000, 12000, 14000, 999999],
            "top_ladder_buckets": [100, 1000, 10000],
        },
    }
    request = MatchupRequest(
        **{**WEB_PAYLOAD, "mode_key": "ranked", "league_number": 6}
    )
    assert request_to_row(request, bundle)["segment"] == "ranked:league-6"


def test_request_to_row_reconstructs_champion_and_hero_roles() -> None:
    # Golden Knight (26000074) is a champion id; 26000064 is sent as a hero form.
    payload = {
        **WEB_PAYLOAD,
        "team_card_ids": [
            26000074, 26000030, 26000021, 26000014, 27000006, 28000000, 28000011, 26000064,
        ],
        "team_evolution_card_ids": [],
        "team_hero_card_ids": [26000064],
    }
    request = MatchupRequest(**payload)
    row = request_to_row(request, {"vocabulary": FAKE_VOCAB})

    # Champion class -> role 2 at the Golden Knight position.
    assert row["team_card_roles"][0] == 2
    # Hero form -> role 3 and hero level 1 at the hero card's position.
    assert row["team_card_roles"][7] == 3
    assert row["team_hero_levels"][7] == 1
    # Everything else stays normal.
    assert row["team_card_roles"][1:7] == [1] * 6


def test_probability_to_label_three_class() -> None:
    assert probability_to_label(0.60) == "good"
    assert probability_to_label(0.40) == "bad"
    assert probability_to_label(0.50) == "neutral"


def test_probability_to_confidence() -> None:
    assert probability_to_confidence(0.50, has_context=True) == "low"
    assert probability_to_confidence(0.70, has_context=True) == "high"
    assert probability_to_confidence(0.57, has_context=True) == "medium"
    # Missing average levels => always low confidence.
    assert probability_to_confidence(0.90, has_context=False) == "low"


# --- integration tests against a synthesized, code-matching checkpoint ---------
#
# We build a tiny checkpoint from the CURRENT model.py instead of depending on
# the trained artifact on disk: that keeps these tests fast, deterministic and
# immune to drift between a stale checkpoint and an evolving model definition.
# Antisymmetry is architectural, so it holds for random weights too.


def _make_checkpoint(path: Path) -> None:
    import torch

    from rigged_matchup_ml.card_stats import CARD_METADATA_VECTOR_SIZE
    from rigged_matchup_ml.model import SymmetricMatchupModel

    cards = {str(cid): i + 1 for i, cid in enumerate(
        WEB_PAYLOAD["team_card_ids"] + WEB_PAYLOAD["opponent_card_ids"]
    )}
    vocabulary = {
        "cards": cards,
        "towers": {"159000000": 1},
        "segments": {"ladder:5000-6999": 1, "ladder:9000-11999": 2, "ranked:unknown": 3},
        "patches": {"2026-05": 1, "2026-06": 2},
    }
    config = {
        "card_count": len(cards) + 1,
        "tower_count": 2,
        "segment_count": 4,
        "patch_count": 3,
        "embedding_dim": 16,
        "hidden_dim": 32,
        "card_metadata_dim": CARD_METADATA_VECTOR_SIZE,
        "dropout": 0.0,
        "use_cross_card_interactions": True,
        "use_intra_deck_synergies": True,
        "use_matchup_transformer": True,
        "use_segment_adapters": True,
        "use_bilinear_cross": True,
        "matrix_prior_strength": 0.0,
    }
    model = SymmetricMatchupModel(**config)
    model.eval()
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": config,
            "vocabulary": vocabulary,
            "temperature": 1.0,
            "segment_temperatures": {},
            "calibration": {},
            "feature_version": 5,
        },
        path,
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("fastapi")
    import os

    from fastapi.testclient import TestClient

    from rigged_matchup_ml import serve

    checkpoint = tmp_path_factory.mktemp("model") / "matchup-model.pt"
    _make_checkpoint(checkpoint)
    os.environ["MODEL_CHECKPOINT"] = str(checkpoint)
    with TestClient(serve.app) as test_client:
        yield test_client


def test_health_reports_loaded(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["model_name"]


def test_predict_returns_valid_contract(client) -> None:
    response = client.post("/predict", json=WEB_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["win_probability"] <= 1.0
    assert body["matchup_label"] in {"bad", "neutral", "good"}
    assert body["confidence"] in {"low", "medium", "high"}
    assert body["explanation"]["segment"] == "ladder:9000-11999"
    # Antisymmetric model => self-symmetry error is essentially zero.
    assert body["explanation"]["symmetry_error"] < 1e-3


def test_predict_is_antisymmetric_across_swap(client) -> None:
    forward = client.post("/predict", json=WEB_PAYLOAD).json()
    swapped_payload = {
        **WEB_PAYLOAD,
        "team_card_ids": WEB_PAYLOAD["opponent_card_ids"],
        "opponent_card_ids": WEB_PAYLOAD["team_card_ids"],
        "team_tower_troop_id": WEB_PAYLOAD["opponent_tower_troop_id"],
        "opponent_tower_troop_id": WEB_PAYLOAD["team_tower_troop_id"],
        "team_evolution_card_ids": WEB_PAYLOAD["opponent_evolution_card_ids"],
        "opponent_evolution_card_ids": WEB_PAYLOAD["team_evolution_card_ids"],
    }
    swapped = client.post("/predict", json=swapped_payload).json()
    assert forward["win_probability"] + swapped["win_probability"] == pytest.approx(
        1.0, abs=1e-3
    )


def test_predict_rejects_duplicate_cards(client) -> None:
    bad = {**WEB_PAYLOAD, "team_card_ids": [26000000] * 8}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_omits_interactions_by_default(client) -> None:
    body = client.post("/predict", json=WEB_PAYLOAD).json()
    assert body["card_interactions"] is None
    assert body["synergies"] is None


def test_predict_includes_model_interactions_when_requested(client) -> None:
    payload = {**WEB_PAYLOAD, "include_interactions": True}
    body = client.post("/predict", json=payload).json()

    interactions = body["card_interactions"]
    assert interactions is not None
    team = set(WEB_PAYLOAD["team_card_ids"])
    opponent = set(WEB_PAYLOAD["opponent_card_ids"])

    # Answers = your card vs their card; threats = their card vs your card.
    assert 1 <= len(interactions["answers"]) <= 3
    for hit in interactions["answers"]:
        assert hit["source_card_id"] in team
        assert hit["target_card_id"] in opponent
        assert 0.0 <= hit["weight"] <= 1.0
    for hit in interactions["threats"]:
        assert hit["source_card_id"] in opponent
        assert hit["target_card_id"] in team
        assert 0.0 <= hit["weight"] <= 1.0

    # Synergies are unordered pairs inside the player's own deck.
    assert len(body["synergies"]) >= 1
    for hit in body["synergies"]:
        assert hit["source_card_id"] in team
        assert hit["target_card_id"] in team
        assert hit["source_card_id"] != hit["target_card_id"]

    # Weights are normalised against the peak attention over ALL pairs, so they
    # land in (0, 1]. They do not have to reach 1.0: selection ranks by signed
    # ablation contribution and reserves a slot per opposing win condition, so
    # the pair holding the attention peak is not necessarily among those returned.
    weights = [hit["weight"] for hit in interactions["answers"]]
    assert all(0.0 < weight <= 1.0 for weight in weights)
    # Ablation gives every returned pair a signed effect on the team logit.
    assert all(hit["contribution"] is not None for hit in interactions["answers"])


# --- batch endpoint -----------------------------------------------------------


def _rotated_payload(shift: int) -> dict:
    """A distinct but valid payload: rotate the opponent deck so rows differ."""
    opponent = WEB_PAYLOAD["opponent_card_ids"]
    return {
        **WEB_PAYLOAD,
        "opponent_card_ids": opponent[shift:] + opponent[:shift],
        "opponent_evolution_card_ids": [],
    }


def test_predict_batch_matches_single_predict(client) -> None:
    payloads = [_rotated_payload(shift) for shift in range(4)]
    batch = client.post("/predict/batch", json={"requests": payloads})
    assert batch.status_code == 200
    predictions = batch.json()["predictions"]
    assert len(predictions) == len(payloads)

    for payload, prediction in zip(payloads, predictions, strict=True):
        single = client.post("/predict", json=payload).json()
        # The batched pass must be numerically identical to the single-row path,
        # not merely close: the site mixes both and caches the results.
        assert prediction["win_probability"] == pytest.approx(
            single["win_probability"], abs=1e-6
        )
        assert prediction["matchup_label"] == single["matchup_label"]
        assert prediction["confidence"] == single["confidence"]
        assert prediction["model_version"] == single["model_version"]
        assert prediction["explanation"]["segment"] == single["explanation"]["segment"]


def test_predict_batch_preserves_request_order(client) -> None:
    # Segments differ per row, so a reordered response would show up here: the
    # site indexes predictions positionally back into its own job list.
    payloads = [
        {**WEB_PAYLOAD, "team_trophies": 5500},
        {**WEB_PAYLOAD, "team_trophies": 10000},
        {**WEB_PAYLOAD, "mode_key": "ranked", "league_number": 3},
    ]
    predictions = client.post("/predict/batch", json={"requests": payloads}).json()[
        "predictions"
    ]
    segments = [prediction["explanation"]["segment"] for prediction in predictions]
    assert segments == ["ladder:5000-6999", "ladder:9000-11999", "ranked:unknown"]


def test_predict_batch_routes_interaction_requests_to_the_single_path(client) -> None:
    payloads = [
        _rotated_payload(1),
        {**WEB_PAYLOAD, "include_interactions": True},
        _rotated_payload(2),
    ]
    predictions = client.post("/predict/batch", json={"requests": payloads}).json()[
        "predictions"
    ]
    assert len(predictions) == 3
    # Only the opted-in row carries attributions, and it stays in position 1.
    assert predictions[0]["card_interactions"] is None
    assert predictions[1]["card_interactions"] is not None
    assert predictions[2]["card_interactions"] is None


def test_predict_batch_stays_antisymmetric(client) -> None:
    swapped_payload = {
        **WEB_PAYLOAD,
        "team_card_ids": WEB_PAYLOAD["opponent_card_ids"],
        "opponent_card_ids": WEB_PAYLOAD["team_card_ids"],
        "team_evolution_card_ids": WEB_PAYLOAD["opponent_evolution_card_ids"],
        "opponent_evolution_card_ids": WEB_PAYLOAD["team_evolution_card_ids"],
    }
    predictions = client.post(
        "/predict/batch", json={"requests": [WEB_PAYLOAD, swapped_payload]}
    ).json()["predictions"]
    assert predictions[0]["win_probability"] + predictions[1][
        "win_probability"
    ] == pytest.approx(1.0, abs=1e-3)


def test_predict_batch_rejects_empty_and_oversized(client) -> None:
    # 512 is the cap the site's META_BATCH_SIZE is tuned against; keep them in step.
    assert client.post("/predict/batch", json={"requests": []}).status_code == 422
    oversized = {"requests": [WEB_PAYLOAD] * 513}
    assert client.post("/predict/batch", json=oversized).status_code == 422


def test_real_checkpoint_loads_if_compatible() -> None:
    """Smoke-load the on-disk trained checkpoint when one exists and matches the
    current model. Skips (does not fail) when absent or stale so an evolving
    model.py never breaks the suite — retrain to make this run."""
    if not CHECKPOINT.exists():
        pytest.skip("no trained checkpoint on disk")
    from rigged_matchup_ml.predictor import load_bundle

    try:
        bundle = load_bundle(CHECKPOINT)
    except RuntimeError as exc:
        pytest.skip(f"trained checkpoint is stale vs current model.py: {exc}")
    assert "vocabulary" in bundle and "model" in bundle
