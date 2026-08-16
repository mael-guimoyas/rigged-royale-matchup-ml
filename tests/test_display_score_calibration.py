from scripts.fit_display_score_calibration import calibration_report, distribution_signal_report


def row(objective: str, score: float, validation: float, test: float) -> dict[str, object]:
    return {
        "k": 0.0,
        "objective": objective,
        "selected_model_score": score,
        "selected_validation_quality": validation,
        "selected_test_quality": test,
    }


def test_recommends_identity_when_validation_lines_hurt_every_test_mode() -> None:
    selections = []
    for objective in ("single", "multi", "group", "complete"):
        selections.extend(
            [
                row(objective, 0.4, 0.45, 0.4),
                row(objective, 0.6, 0.55, 0.6),
            ]
        )

    report = calibration_report(
        {
            "generated_at": "2026-08-16T00:00:00+00:00",
            "model_version": "model",
            "selections": selections,
        }
    )

    assert report["recommendedMapping"] == "identity"
    assert report["overall"]["test"]["rawMae"] == 0.0
    assert report["overall"]["test"]["linearMae"] > 0.0


def test_distribution_signal_requires_a_positive_cluster_bootstrap_interval() -> None:
    rows = []
    for index in range(20):
        score = 0.4 + index / 100
        rows.append(
            {
                "segment": "segment",
                "selected_key": f"deck-{index}",
                "selected_model_score": score,
                "selected_validation_quality": score,
                "selected_test_quality": score,
                "selected_matchup_sigma": index / 100,
                "selected_unfavorable_share": 0.0,
                "selected_hard_counter_share": 0.0,
            }
        )

    report = distribution_signal_report(rows, bootstrap_samples=100)

    assert report is not None
    assert report["recommended"] is False
    assert report["zeroUnfavorable"]["n"] == 20
