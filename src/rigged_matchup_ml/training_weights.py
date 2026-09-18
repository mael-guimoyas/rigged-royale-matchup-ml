"""Label-blind segment weighting: correct collection bias without adding a new one.

The crawl does not sample the player population uniformly -- `--balance` steers it,
`min_trophies` cuts it, snowball clusters it -- so some segments carry ten times the
battles of others. Left alone, the loss is dominated by the saturated band and the
model is merely confident about the wrong population.

Every choice here exists to avoid *replacing* that bias with a worse one:

- **Weights, never resampling.** Upsampling a starved band repeats the same
  `win/loss` label: the effective sample size does not grow, the model just gets
  more chances to memorise one noisy battle. Downsampling the saturated band throws
  away real observations. Re-weighting keeps every row and only moves gradient mass.
- **A bounded ratio.** Full inverse frequency hands a band of 2k battles a weight of
  ~500, and the whole model ends up steered by 2k noisy labels -- a coverage bias
  traded for a variance bias. `cap` bounds the largest/smallest weight ratio, and
  `power` < 1 moves only part of the way toward a uniform population.
- **Mean exactly 1.** The loss scale, and with it the effective learning rate, must
  not change when weighting is switched on.
- **A function of `segment` alone.** The weight cannot see `win`, so inside a segment
  it is constant and cannot teach the model a spurious outcome signal. Reweighting
  *does* shift the aggregate win rate when segments differ -- that is the intended
  population change, and `weighting_report` states it instead of hiding it.

Train split only. Validation and test stay unweighted: they exist to describe the
population the model will actually meet, and calibration is fitted on them.
"""

from __future__ import annotations

from typing import Any

# Below this Kish ratio the weighting has thrown away more than half the corpus's
# statistical value -- too much variance bought for the coverage gained.
MIN_EFFECTIVE_FRACTION = 0.5
# A global win-rate move this large means segments disagree on the base rate; the
# antisymmetric model cannot fit a global skew, so it would land as pure noise.
MAX_WIN_RATE_SHIFT = 0.005


def segment_weights(
    counts: dict[str, int],
    power: float = 0.3,
    cap: float = 3.0,
) -> dict[str, float]:
    """Capped inverse-frequency weight per segment, count-weighted mean of exactly 1.

    Two guarantees the tests pin down: `max(w) / min(w) <= cap`, and the mean weight
    over train rows is 1. Ratio first, normalisation second -- capping *after*
    normalising would silently shrink the loss scale.
    """
    positive = {segment: int(rows) for segment, rows in counts.items() if int(rows) > 0}
    if not positive:
        return {}
    total = sum(positive.values())
    raw = {segment: (1.0 / rows) ** power for segment, rows in positive.items()}
    smallest = min(raw.values())
    # Ratios live in [1, cap] by construction, so the cap cannot be re-violated by
    # the normalisation that follows (it scales every weight by the same factor).
    ratios = {segment: min(value / smallest, cap) for segment, value in raw.items()}
    mean = sum(positive[segment] * ratios[segment] for segment in positive) / total
    return {segment: ratio / mean for segment, ratio in ratios.items()}


def effective_sample_size(counts: dict[str, int], weights: dict[str, float]) -> float:
    """Kish effective sample size: how many unweighted battles this corpus is worth.

    Constant-within-segment weights make this exact from the per-segment counts, no
    row scan: sum(n*w)^2 / sum(n*w^2).
    """
    total = 0.0
    total_squared = 0.0
    for segment, rows in counts.items():
        weight = float(weights.get(segment, 0.0))
        total += rows * weight
        total_squared += rows * weight * weight
    return (total * total) / total_squared if total_squared > 0 else 0.0


def weighting_report(
    counts: dict[str, int],
    wins: dict[str, int],
    weights: dict[str, float],
    min_effective_fraction: float = MIN_EFFECTIVE_FRACTION,
    max_win_rate_shift: float = MAX_WIN_RATE_SHIFT,
) -> dict[str, Any]:
    """Describe what the weighting did, and flag the new biases it could have added."""
    rows = sum(int(value) for value in counts.values())
    if rows == 0:
        return {"rows": 0, "segments": [], "warnings": []}

    total_wins = sum(int(value) for value in wins.values())
    weighted_rows = sum(counts[segment] * weights.get(segment, 0.0) for segment in counts)
    weighted_wins = sum(wins.get(segment, 0) * weights.get(segment, 0.0) for segment in counts)
    win_rate = total_wins / rows
    weighted_win_rate = weighted_wins / weighted_rows if weighted_rows > 0 else win_rate
    ess = effective_sample_size(counts, weights)
    effective_fraction = ess / rows

    segments = sorted(
        (
            {
                "segment": segment,
                "rows": int(row_count),
                "share": row_count / rows,
                "weighted_share": (
                    row_count * weights.get(segment, 0.0) / weighted_rows
                    if weighted_rows > 0
                    else 0.0
                ),
                "weight": weights.get(segment, 0.0),
                "win_rate": wins.get(segment, 0) / row_count if row_count else None,
            }
            for segment, row_count in counts.items()
        ),
        key=lambda entry: -entry["rows"],
    )
    applied = [weight for weight in weights.values() if weight > 0]

    warnings: list[dict[str, Any]] = []
    if effective_fraction < min_effective_fraction:
        warnings.append(
            {
                "code": "effective_sample_size",
                "message": (
                    f"Weighting leaves {effective_fraction:.1%} of the corpus's statistical "
                    f"value ({ess:,.0f} of {rows:,} battles). Lower loss_segment_power or "
                    "loss_segment_cap: coverage bought here is paid in variance."
                ),
                "effective_sample_fraction": effective_fraction,
            }
        )
    if abs(weighted_win_rate - win_rate) > max_win_rate_shift:
        warnings.append(
            {
                "code": "win_rate_shift",
                "message": (
                    f"Reweighting moves the global win rate {win_rate:.4f} -> "
                    f"{weighted_win_rate:.4f}. Segments disagree on the base rate, which "
                    "an antisymmetric model cannot fit; check per-segment side balance "
                    "before trusting this corpus."
                ),
                "shift": weighted_win_rate - win_rate,
            }
        )

    return {
        "rows": rows,
        "effective_sample_size": ess,
        "effective_sample_fraction": effective_fraction,
        "win_rate": win_rate,
        "weighted_win_rate": weighted_win_rate,
        "win_rate_shift": weighted_win_rate - win_rate,
        "weight_min": min(applied) if applied else 0.0,
        "weight_max": max(applied) if applied else 0.0,
        "weight_ratio": (max(applied) / min(applied)) if applied else 1.0,
        "segments": segments,
        "warnings": warnings,
    }
