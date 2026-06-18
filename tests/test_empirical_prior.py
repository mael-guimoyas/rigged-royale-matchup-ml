import json
import os
import subprocess
from pathlib import Path

import pytest

from rigged_matchup_ml.empirical_prior import clamp_prior, fallback_prior, reverse_prior


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT.parent / "riggedroyale"
TSX = APP_DIR / "node_modules" / ".bin" / ("tsx.cmd" if os.name == "nt" else "tsx")
HELPER = ROOT / "tools" / "empirical-prior-helper.ts"


EMPTY_MATRIX = {
    "version": 3,
    "archGlobal": [],
    "archBucket": [],
    "planGlobal": [],
    "planBucket": [],
    "spellAnswerGlobal": [],
    "metaPanel": [],
    "totalGames": 0,
}


HOG_CYCLE = [
    26000012,
    26000000,
    26000010,
    27000000,
    26000014,
    26000031,
    28000000,
    26000084,
]
GOLEM_BEATDOWN = [
    26000061,
    26000063,
    26000019,
    26000043,
    26000029,
    26000077,
    26000031,
    27000004,
]


def test_prior_bounds_reverse_and_fallback() -> None:
    assert clamp_prior(-10.0) == 0.001
    assert clamp_prior(10.0) == 0.999
    assert fallback_prior() == 0.5
    assert reverse_prior(0.8) == pytest.approx(0.2)
    assert reverse_prior(0.999) == pytest.approx(0.001)


def _score_with_helper(tmp_path: Path, matrix: dict, rows: list[dict]) -> list[dict]:
    if not TSX.exists():
        pytest.skip(f"tsx is not installed at {TSX}")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    input_text = "".join(json.dumps(row) + "\n" for row in rows)
    result = subprocess.run(
        [str(TSX), str(HELPER), "score-records", "--matrix", str(matrix_path)],
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()]


def test_prior_helper_falls_back_to_neutral_without_edges(tmp_path: Path) -> None:
    scored = _score_with_helper(
        tmp_path,
        EMPTY_MATRIX,
        [
            {
                "segment": "ranked",
                "team_card_ids": HOG_CYCLE,
                "opponent_card_ids": GOLEM_BEATDOWN,
            }
        ],
    )
    assert scored[0]["prior"] == pytest.approx(0.5)
    assert scored[0]["coverage"] == {
        "archGlobal": False,
        "archBucket": False,
        "planGlobal": False,
        "planBucket": False,
        "spellAnswerGlobal": False,
    }


def test_prior_helper_is_bounded_and_symmetric(tmp_path: Path) -> None:
    matrix = {
        **EMPTY_MATRIX,
        "planGlobal": [
            {"key": "shell:cycle>lava-hound", "delta": 100.0, "n": 1000, "weight": 1.0},
            {"key": "lava-hound>shell:cycle", "delta": -100.0, "n": 1000, "weight": 1.0},
        ],
    }
    scored = _score_with_helper(
        tmp_path,
        matrix,
        [
            {
                "segment": "ranked",
                "team_card_ids": HOG_CYCLE,
                "opponent_card_ids": GOLEM_BEATDOWN,
            },
            {
                "segment": "ranked",
                "team_card_ids": GOLEM_BEATDOWN,
                "opponent_card_ids": HOG_CYCLE,
            },
        ],
    )
    assert scored[0]["prior"] == pytest.approx(0.999)
    assert scored[1]["prior"] == pytest.approx(0.001)
    assert scored[0]["prior"] + scored[1]["prior"] == pytest.approx(1.0)
    assert scored[0]["coverage"]["planGlobal"] is True
    assert scored[1]["coverage"]["planGlobal"] is True


def test_prior_helper_keeps_ranked_league_buckets(tmp_path: Path) -> None:
    matrix = {
        **EMPTY_MATRIX,
        "planBucket": [
            {
                "key": "ranked:league-7#shell:cycle>lava-hound",
                "delta": 0.5,
                "n": 120,
                "weight": 1.0,
            },
            {
                "key": "ranked:league-7#lava-hound>shell:cycle",
                "delta": -0.5,
                "n": 120,
                "weight": 1.0,
            },
        ],
    }
    scored = _score_with_helper(
        tmp_path,
        matrix,
        [
            {
                "segment": "ranked:league-7",
                "team_card_ids": HOG_CYCLE,
                "opponent_card_ids": GOLEM_BEATDOWN,
            }
        ],
    )
    assert scored[0]["context"]["bucket"] == "ranked:league-7"
    assert scored[0]["coverage"]["planBucket"] is True
    assert scored[0]["prior"] > 0.5
