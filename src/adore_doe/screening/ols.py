"""Minimal, transparent OLS (numpy/scipy). Cross-checked against scipy/statsmodels in tests.

    beta_hat = argmin ||y - Z beta||^2,   sigma^2 = RSS / df,   df = n - rank(Z)
    Cov(beta_hat) = sigma^2 (Z'Z)^{-1},   t_j = beta_j / SE_j,   p_j = 2 P(T_df > |t_j|)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, stats

from ..exceptions import RankDeficientError


@dataclass
class OLSFit:
    beta: np.ndarray
    rank: int
    n: int
    df: int
    rss: float
    sigma2: float          # nan when df < 1
    cov: np.ndarray        # nan matrix when df < 1

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def t(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.beta / self.se

    @property
    def p(self) -> np.ndarray:
        if self.df < 1:
            return np.full(self.beta.shape, np.nan)
        return 2.0 * stats.t.sf(np.abs(self.t), self.df)


def nonestimable_columns(Z: np.ndarray, names: list[str], tol: float = 1e-9) -> list[str]:
    """Names of columns that are linear combinations of the preceding (pivoted) columns."""
    _, R, piv = linalg.qr(Z, pivoting=True, mode="economic")
    d = np.abs(np.diag(R))
    scale = d[0] if d.size and d[0] > 0 else 1.0
    rank = int(np.sum(d > tol * max(Z.shape) * scale))
    return [names[j] for j in sorted(piv[rank:])]


def fit_ols(Z: np.ndarray, y: np.ndarray, names: list[str] | None = None) -> OLSFit:
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = Z.shape
    names = names or [f"x{j}" for j in range(p)]
    rank = int(np.linalg.matrix_rank(Z))
    if rank < p:
        bad = nonestimable_columns(Z, names)
        raise RankDeficientError(
            f"Model matrix is rank deficient (rank {rank} < {p} columns, n={n}); "
            f"non-estimable / aliased parameter(s): {bad}.")
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    rss = float(resid @ resid)
    df = n - rank
    if df >= 1:
        sigma2 = rss / df
        cov = sigma2 * np.linalg.inv(Z.T @ Z)
    else:
        sigma2 = float("nan")
        cov = np.full((p, p), np.nan)
    return OLSFit(beta, rank, n, df, rss, sigma2, cov)
