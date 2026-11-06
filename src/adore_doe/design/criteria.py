"""Design criteria with explicit rank / conditioning checks (research_methodology §4.2).

I-criterion (σ² = 1):   I(X) = tr[(X'X)^-1 B] = average prediction variance over the region,
                         B = ∫ f(x) f(x)' w(x) dx / ∫ w(x) dx     (smaller is better)
D-criterion:            log det(X'X), computed from a QR factor (2 Σ log|R_ii|), never from
                         a raw determinant; D_eff = det(X'X)^(1/p) / n    (larger is better)
A singular X'X is REPORTED (RankDeficientError); the pseudo-inverse is never used silently.
"""
from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from ..exceptions import InferenceWarning, RankDeficientError

COND_WARN = 1e8


class DesignCheck(NamedTuple):
    n: int
    p: int
    rank: int
    cond: float          # condition number of X (SVD)
    singular: bool


def check_design(X) -> DesignCheck:
    X = np.atleast_2d(np.asarray(X, float))
    n, p = X.shape
    if n == 0:
        return DesignCheck(0, p, 0, np.inf, True)
    s = np.linalg.svd(X, compute_uv=False)
    tol = s.max() * max(n, p) * np.finfo(float).eps if s.size and s.max() > 0 else 0.0
    rank = int(np.sum(s > tol))
    cond = float(s.max() / s.min()) if s.size and s.min() > 0 else float("inf")
    return DesignCheck(n, p, rank, cond, rank < p)


def require_full_rank(X, names=None, what: str = "design") -> DesignCheck:
    c = check_design(X)
    if c.singular:
        extra = ""
        if c.n < c.p:
            extra = f" ({c.n} runs < {c.p} model terms)"
        raise RankDeficientError(f"{what} is not identifiable: rank {c.rank} < {c.p} model terms"
                                 f"{extra}. Add runs or simplify the model.")
    if c.cond > COND_WARN:
        warnings.warn(f"{what} is ill-conditioned (cond(X) = {c.cond:.2e}); estimates may be "
                      f"numerically unstable.", InferenceWarning, stacklevel=3)
    return c


def i_criterion(X, B) -> float:
    """tr[(X'X)^-1 B]; raises RankDeficientError for a singular design."""
    X = np.asarray(X, float)
    require_full_rank(X)
    M = X.T @ X
    return float(np.trace(np.linalg.solve(M, np.asarray(B, float))))


def log_det(X) -> float:
    """log det(X'X) via QR (stable); -inf is never returned: singular designs raise."""
    X = np.asarray(X, float)
    require_full_rank(X)
    R = np.linalg.qr(X, mode="r")
    return float(2.0 * np.sum(np.log(np.abs(np.diag(R)))))


def d_efficiency(X) -> float:
    """(det X'X)^(1/p) / n  (per-run D-efficiency)."""
    X = np.asarray(X, float)
    n, p = X.shape
    return float(np.exp(log_det(X) / p) / n)


def prediction_variance(X, F_points) -> np.ndarray:
    """σ²=1 prediction variance f(x)'(X'X)^-1 f(x) for model rows F_points (m x p)."""
    X = np.asarray(X, float)
    require_full_rank(X)
    F = np.atleast_2d(np.asarray(F_points, float))
    S = np.linalg.solve(X.T @ X, F.T)
    return np.einsum("ij,ji->i", F, S)
