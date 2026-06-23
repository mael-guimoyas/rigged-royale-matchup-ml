import numpy as np
import torch

from rigged_matchup_ml.card2vec import _accumulate, _ppmi_svd
from rigged_matchup_ml.trainer import _card_weight_tensor, _weighted_bce


def test_ppmi_svd_groups_co_occurring_cards() -> None:
    # Two disjoint deck families: {1,2,3} always together, {4,5,6} always together.
    size = 7
    cooc = np.zeros((size, size), dtype=np.float64)
    family_a = np.array([[1, 2, 3, 0, 0, 0, 0, 0]] * 50)
    family_b = np.array([[4, 5, 6, 0, 0, 0, 0, 0]] * 50)
    _accumulate(cooc, family_a)
    _accumulate(cooc, family_b)
    vectors = _ppmi_svd(cooc, dim=4)
    assert vectors.shape == (size, 4)
    assert np.allclose(vectors[0], 0.0)  # padding row stays zero

    def cosine(i: int, j: int) -> float:
        a, b = vectors[i], vectors[j]
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

    # within-family more aligned than cross-family
    assert cosine(1, 2) > cosine(1, 4)
    assert cosine(4, 5) > cosine(4, 2)


def test_card_weight_tensor_normalises_and_boosts_rare(tmp_path) -> None:
    (tmp_path / "card_frequencies.json").write_text(
        '{"100": 1000, "200": 10}', encoding="utf-8"
    )
    vocabulary = {"cards": {"100": 1, "200": 2}}
    weights = _card_weight_tensor(tmp_path, vocabulary, power=0.5, cap=50.0, device=torch.device("cpu"))
    assert weights is not None
    assert weights[0].item() == 0.0  # padding
    # rare card (200) weighted higher than common (100)
    assert weights[2].item() > weights[1].item()
    # frequency-weighted mean weight ~= 1 (loss scale preserved)
    counts = torch.tensor([0.0, 1000.0, 10.0])
    mean = float((counts * weights).sum() / counts.sum())
    assert abs(mean - 1.0) < 1e-5


def test_weighted_bce_matches_manual() -> None:
    loss_none = torch.nn.BCEWithLogitsLoss(reduction="none")
    logits = torch.tensor([0.5, -0.5])
    target = torch.tensor([1.0, 0.0])
    card_weight = torch.tensor([0.0, 2.0, 4.0])  # idx 0 padding
    team = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]])
    opp = torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]])
    out = _weighted_bce(loss_none, logits, target, card_weight, team, opp)
    # sample 0: cards {1,2} -> mean weight 3 ; sample 1: cards {2,2} -> 4
    per = loss_none(logits, target)
    expected = (per * torch.tensor([3.0, 4.0])).mean()
    assert torch.allclose(out, expected)
