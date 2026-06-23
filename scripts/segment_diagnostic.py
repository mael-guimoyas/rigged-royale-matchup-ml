"""Sanity-check segmentation + matchup-class balance of a trained checkpoint.

Confirms the inference server places requests in distinct trained segments from
the site's bracket inputs (trophies / league) and that no segment collapses into
a ~90%-bad matchup distribution. Use after training or when investigating the
"every deck is a bad matchup" report.

Run:  python scripts/segment_diagnostic.py [artifacts/matchup-model.pt]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from rigged_matchup_ml import serve
from rigged_matchup_ml.predictor import load_bundle


def main() -> None:
    checkpoint = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/matchup-model.pt")
    bundle = load_bundle(checkpoint)
    vocab = bundle["vocabulary"]
    segments = sorted(vocab.get("segments", {}).keys())
    print(f"checkpoint: {checkpoint}")
    print(f"data_config embedded: {'data_config' in bundle}")
    print(f"segments ({len(segments)}): {segments}")

    cards_vocab = vocab.get("cards", {}) or vocab.get("card_ids", {})
    ids = [
        int(c)
        for c in (cards_vocab.keys() if isinstance(cards_vocab, dict) else cards_vocab)
        if int(c) > 0
    ]
    random.seed(1)

    def deck() -> list[int]:
        return random.sample(ids, 8)

    def predict(mode: str, *, trophies=None, league=None, team=None, opp=None):
        req = serve.MatchupRequest(
            team_card_ids=team or deck(),
            opponent_card_ids=opp or deck(),
            mode_key=mode,
            team_trophies=trophies,
            league_number=league,
        )
        row = serve.request_to_row(req, bundle)
        resp = serve.build_response(bundle, req)
        return row["segment"], resp.win_probability, resp.matchup_label

    team, opp = deck(), deck()
    print("\nsame decks, different bracket (segment must differ):")
    cases = [
        ("ladder@4000", dict(mode="ladder", trophies=4000)),
        ("ladder@8000", dict(mode="ladder", trophies=8000)),
        ("ladder@13000", dict(mode="ladder", trophies=13000)),
        ("ranked L5", dict(mode="ranked", league=5)),
        ("ladder (no bracket)", dict(mode="ladder")),
    ]
    for label, kw in cases:
        seg, p, lab = predict(team=team, opp=opp, **kw)
        print(f"  {label:22s} seg={seg:22s} p={p:.3f} {lab}")

    print("\nmatchup-class share over 400 random matchups per segment:")
    for mode, kw in [
        ("ladder", dict(trophies=8000)),
        ("ladder", dict(trophies=4000)),
        ("ranked", dict(league=4)),
    ]:
        bad = good = neutral = 0
        seg = ""
        for _ in range(400):
            seg, _p, lab = predict(mode=mode, **kw)
            bad += lab == "bad"
            good += lab == "good"
            neutral += lab == "neutral"
        print(
            f"  {mode} {kw!s:18s} seg={seg:20s} "
            f"bad={bad / 4:.1f}% good={good / 4:.1f}% neutral={neutral / 4:.1f}%"
        )


if __name__ == "__main__":
    main()
