"""Decision limits for screening. Pure statistics - no plotting, no design objects.

Notation: alpha = significance level, df = error degrees of freedom, m = number of effects
(factors) tested simultaneously, T = |effect| / SE(effect) ~ t_df under H0 (classical OLS).

1. Single-factor test (no multiplicity correction)
       t_limit = t_{1 - alpha/2, df}                       (two-sided)
   Each factor on its own has type-I error alpha.  Across m factors the chance of at least one
   false flag is 1 - (1 - alpha)^m (>> alpha for large m).

2. Bonferroni correction (family-wise error <= alpha, by the union bound)
       bonferroni_limit = t_{1 - alpha/(2m), df}
   This is exact Bonferroni on two-sided t-tests: each of the m tests is run at alpha/m.
   Same definition as the Design-Expert "Bonferroni Limit" (verified against the
   PDF figure: df=10, m=19, alpha=0.05 -> t-limit 2.22814, Bonferroni 3.97276).

3. Lenth's pseudo standard error (for designs WITHOUT error df, e.g. saturated PB)
   Given m contrasts c_j (effect estimates):
       s0  = 1.5 * median |c_j|
       PSE = 1.5 * median{ |c_j| : |c_j| < 2.5 * s0 }
       ME  = t_{1 - alpha/2, d} * PSE,                     d = m / 3
       SME = t_{gamma, d} * PSE,   gamma = (1 + (1 - alpha)^(1/m)) / 2     (Lenth, 1989)
   ME/SME are the analogues of t-limit/Bonferroni limit on the *effect* scale.
   Assumptions: effect sparsity (most true effects ~ 0), orthogonal contrasts with equal
   variance, approximately normal errors.  The t_d reference with d = m/3 is itself an
   approximation (Lenth's own simulation-based critical values differ slightly), so
   Lenth limits are guidance, NOT exact tests; PSE is not an OLS standard error and no
   p-values are produced from it.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..exceptions import InferenceError


def _check(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")


def t_limit(alpha: float, df: float) -> float:
    """Two-sided single-test critical value t_{1-alpha/2, df}."""
    _check(alpha)
    if df is None or df < 1:
        raise InferenceError("t-limit needs at least 1 error degree of freedom.")
    return float(stats.t.ppf(1.0 - alpha / 2.0, df))


def bonferroni_limit(alpha: float, df: float, m: int) -> float:
    """Bonferroni critical value t_{1-alpha/(2m), df} for m simultaneous two-sided tests."""
    _check(alpha)
    if m < 1:
        raise ValueError("m must be >= 1.")
    if df is None or df < 1:
        raise InferenceError("Bonferroni limit needs at least 1 error degree of freedom.")
    return float(stats.t.ppf(1.0 - alpha / (2.0 * m), df))


def exceeds(values, limit: float) -> np.ndarray:
    """Boolean mask: |value| > limit (NaN -> False)."""
    v = np.abs(np.asarray(values, dtype=float))
    with np.errstate(invalid="ignore"):
        return v > limit


def classify(t_values, t_lim: float, bonf_lim: float) -> list[str]:
    """'Bonferroni' / 't-limit' / 'not significant' / 'n/a' per t-value (single-test status)."""
    out = []
    for t in np.asarray(t_values, dtype=float):
        if not np.isfinite(t):
            out.append("n/a")
        elif abs(t) > bonf_lim:
            out.append("Bonferroni")
        elif abs(t) > t_lim:
            out.append("t-limit")
        else:
            out.append("not significant")
    return out


def lenth(contrasts, alpha: float = 0.05) -> dict:
    """Lenth's PSE, margin of error (ME) and simultaneous margin of error (SME)."""
    _check(alpha)
    c = np.abs(np.asarray(contrasts, dtype=float))
    c = c[np.isfinite(c)]
    m = c.size
    if m < 3:
        raise InferenceError("Lenth's method needs at least 3 contrasts.")
    s0 = 1.5 * np.median(c)
    keep = c[c < 2.5 * s0]
    pse = 1.5 * np.median(keep) if keep.size else s0
    d = m / 3.0
    gamma = (1.0 + (1.0 - alpha) ** (1.0 / m)) / 2.0
    return {"pse": float(pse), "s0": float(s0), "df": float(d), "m": int(m), "alpha": alpha,
            "ME": float(stats.t.ppf(1.0 - alpha / 2.0, d) * pse),
            "SME": float(stats.t.ppf(gamma, d) * pse), "gamma": float(gamma)}
