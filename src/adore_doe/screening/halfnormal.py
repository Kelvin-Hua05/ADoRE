"""Half-normal plot coordinates (statistics only; plotting lives in plots.py).

Sort the m absolute effects ascending, a_(1) <= ... <= a_(m).  With plotting positions
p_i = 0.5 + 0.5 * (i - 0.5) / m, the half-normal quantiles are  q_i = Phi^{-1}(p_i)
(the quantiles of |Z|, Z ~ N(0,1)).  If all effects were pure noise ~ N(0, s^2) the points
(q_i, a_(i)) fall on a line through the origin with slope s.  Points far above the line are
candidate real effects.

The reference line is computed independently of any selection rule: least squares through
the origin on the smallest ``fraction`` of the points (default 50 %), which assumes
effect sparsity (the smallest effects are noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def half_normal_quantiles(m: int) -> np.ndarray:
    if m < 1:
        raise ValueError("m must be >= 1")
    i = np.arange(1, m + 1)
    return stats.norm.ppf(0.5 + 0.5 * (i - 0.5) / m)


def half_normal_points(labels, effects, kinds=None) -> pd.DataFrame:
    """DataFrame sorted ascending by |effect| with columns
    Label, Effect, AbsEffect, Sign, Rank, Quantile, Kind ('factor'/'dummy')."""
    eff = np.asarray(effects, dtype=float)
    if len(labels) != eff.size:
        raise ValueError("labels and effects differ in length")
    kinds = ["factor"] * eff.size if kinds is None else list(kinds)
    order = np.argsort(np.abs(eff), kind="stable")
    q = half_normal_quantiles(eff.size)
    return pd.DataFrame({
        "Label": [labels[i] for i in order], "Effect": eff[order], "AbsEffect": np.abs(eff[order]),
        "Sign": np.where(eff[order] >= 0, "+", "-"), "Rank": np.arange(1, eff.size + 1),
        "Quantile": q, "Kind": [kinds[i] for i in order]})


def reference_slope(points: pd.DataFrame, fraction: float = 0.5) -> float:
    """Slope of the through-origin LS line on the smallest `fraction` of points."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n = max(2, int(np.ceil(fraction * len(points))))
    q = points["Quantile"].to_numpy()[:n]
    a = points["AbsEffect"].to_numpy()[:n]
    return float(np.sum(q * a) / np.sum(q * q))
