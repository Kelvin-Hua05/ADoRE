"""Main-effects analysis of two-level screening designs.

Definitions (coded factors x_j in {-1,+1}; center points x_j = 0)
------------------------------------------------------------------
Model            y = b0 + sum_j b_j x_j [+ b_c * 1(center point)] + e,   e ~ N(0, sigma^2)
Effect           E_j = 2 * b_j   (OLS).  For a balanced orthogonal design this equals
                 mean(y | x_j=+1) - mean(y | x_j=-1)  over the factorial runs; both are
                 reported (`Effect`, `Mean_Diff`) and a warning is issued if they differ
                 (non-orthogonal data, e.g. missing responses).
Coefficient      b_j = E_j / 2   (regression coefficient per coded unit; NOT the effect).
SE(E_j)          2 * SE(b_j),  SE(b_j)^2 = sigma^2 [(Z'Z)^-1]_jj,  sigma^2 = RSS / df_error
t_j              E_j / SE(E_j) = b_j / SE(b_j)
p_j              two-sided,  2 P(T_df > |t_j|)
CI               E_j +- t_{1-alpha/2, df} SE(E_j)
Standardized     Effect / SE(Effect) (= t_j).  This is the scale of the t-limit / Bonferroni
effect           limit.  (Design-Expert's "standardized effect" axis on its half-normal plot
                 uses a different internal scaling; no equivalence is claimed.)
Error df         df = n_obs - rank(Z).  Pure-error df come from center points (n_c - 1), and
                 df also includes unassigned PB "dummy" columns (assumes no interactions).
                 If df < 1 NO t / p / CI are produced (NaN) and Lenth's method is offered.
Curvature        with center points a term b_c*1(center) is added (b_c = mean(center) -
                 mean(factorial) in the orthogonal case), so curvature does not inflate sigma.

Significance status here is a *single-factor* label (t-limit / Bonferroni); it is not the
iterative selection - see `adore_doe.analysis.screening`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..core.dataset import ExperimentDataset
from ..exceptions import DesignError, InferenceWarning
from . import limits as L
from .ols import fit_ols
from .views import PlotMixin

_COLS = ["Factor", "Effect", "Coefficient", "StdErr", "t", "p", "CI_low", "CI_high",
         "Std_Effect", "Abs_Effect", "Sign", "Mean_Diff", "Significance"]


@dataclass
class EffectsResult(PlotMixin):
    design: ExperimentDataset
    response: str
    alpha: float
    table: pd.DataFrame
    dummy_table: pd.DataFrame
    df_error: int
    sigma: float
    n_obs: int
    rank: int
    t_limit: float | None
    bonferroni_limit: float | None
    n_tests: int
    lenth: dict | None
    curvature: dict | None
    warnings: list = field(default_factory=list)
    # arrays reused by the selection procedure (avoid recomputation)
    _y: np.ndarray = None
    _X: np.ndarray = None
    _curv: np.ndarray | None = None

    # ------------------------------------------------------------- properties
    @property
    def inference_available(self) -> bool:
        return self.df_error >= 1

    @property
    def factors(self) -> list[str]:
        return list(self.table["Factor"])

    @property
    def _ptable(self):
        return self.table

    @property
    def _pdummy(self):
        return self.dummy_table

    @property
    def _pdf(self):
        return self.df_error

    @property
    def _palpha(self):
        return self.alpha

    @property
    def _ptlimit(self):
        return self.t_limit

    @property
    def _pbonf(self):
        return self.bonferroni_limit

    @property
    def _plenth(self):
        return self.lenth

    @property
    def _pselected(self):
        return []

    @property
    def _pdummy_label(self):
        return self.design.info.get("dummy_label", "Dummy column (error)")

    def to_dataframe(self) -> pd.DataFrame:
        return self.table.copy()

    def summary(self) -> str:
        lines = [f"Main effects of {self.response!r}  (n={self.n_obs}, error df={self.df_error}, "
                 f"sigma={self.sigma:.5g}, alpha={self.alpha:g})"]
        if self.t_limit is not None:
            lines.append(f"t-limit={self.t_limit:.5f}  Bonferroni limit={self.bonferroni_limit:.5f} "
                         f"(m={self.n_tests})")
        elif self.lenth is not None:
            lines.append(f"No error df -> Lenth PSE={self.lenth['pse']:.5g}, "
                         f"ME={self.lenth['ME']:.5g}, SME={self.lenth['SME']:.5g}")
        lines.append(self.table.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<EffectsResult {self.response!r}: {len(self.table)} factors, "
                f"error df={self.df_error}>")


def main_effects(design: ExperimentDataset, response: str, alpha: float = 0.05,
                 curvature: bool = True) -> EffectsResult:
    """Estimate main effects of a two-level design and test them (OLS).

    Raises ``RankDeficientError`` if some effects are not estimable, and returns NaN
    t/p/CI (plus a warning) if there are no error degrees of freedom.
    """
    if not isinstance(design, ExperimentDataset):
        raise TypeError("design must be a ExperimentDataset.")
    L._check(alpha)
    y_all = design.get_response(response)
    warn: list[str] = []

    def note(msg: str, category=InferenceWarning):
        warn.append(msg)
        warnings.warn(msg, category, stacklevel=3)

    ok = ~np.isnan(y_all)
    n_missing = int((~ok).sum())
    if n_missing:
        note(f"{n_missing} run(s) have no value for {response!r} and were dropped; "
             f"the remaining design may no longer be orthogonal.")
    X_all = design.coded
    ptype = np.asarray(design.point_type)
    names = design.factor_names
    k = len(names)
    y = y_all[ok]
    X = X_all[ok]
    is_center = (ptype[ok] == "center")
    use_curv = bool(curvature and is_center.any())
    cols = [np.ones(len(y)), *[X[:, j] for j in range(k)]]
    zn = ["Intercept", *names]
    if use_curv:
        cols.append(is_center.astype(float))
        zn.append("Curvature")
    Z = np.column_stack(cols)
    fit = fit_ols(Z, y, zn)                       # raises RankDeficientError if not estimable
    n = len(y)
    inf_ok = fit.df >= 1
    if not inf_ok:
        note(f"Error degrees of freedom = {fit.df} (n={n}, parameters={fit.rank}): classical "
             f"t-/p-values and confidence intervals are NOT computed. Use Lenth/half-normal "
             f"analysis, or add center points / replicates.")

    beta = fit.beta[1:1 + k]
    eff = 2.0 * beta
    se = 2.0 * fit.se[1:1 + k]
    t = fit.t[1:1 + k]
    p = fit.p[1:1 + k]
    if inf_ok:
        tcrit = stats.t.ppf(1 - alpha / 2, fit.df)
        lo, hi = eff - tcrit * se, eff + tcrit * se
        tl = L.t_limit(alpha, fit.df)
        bl = L.bonferroni_limit(alpha, fit.df, k)
        status = L.classify(t, tl, bl)
        sigma = float(np.sqrt(fit.sigma2))
    else:
        lo = hi = np.full(k, np.nan)
        tl = bl = None
        status = ["n/a (df=0)"] * k
        sigma = float("nan")

    # Mean(high) - Mean(low) over factorial runs (descriptive, definition from the brief)
    fm = ~is_center
    multilevel = bool(np.any(np.isclose(X[fm], 0.0)))      # multi-level (non two-level) columns
    if multilevel:
        md = np.full(k, np.nan)                             # two-level mean difference undefined
    else:
        md = np.array([y[fm & (X[:, j] > 0)].mean() - y[fm & (X[:, j] < 0)].mean() for j in range(k)])
        if not np.allclose(md, eff, rtol=1e-6, atol=1e-9 * max(1.0, float(np.abs(y).max()))):
            note("OLS effects (2*coefficient) differ from raw mean differences: the design is "
                 "unbalanced/non-orthogonal (missing runs?). The OLS effect is reported.")

    table = pd.DataFrame({
        "Factor": names, "Effect": eff, "Coefficient": beta, "StdErr": se, "t": t, "p": p,
        "CI_low": lo, "CI_high": hi, "Std_Effect": t, "Abs_Effect": np.abs(eff),
        "Sign": np.where(eff >= 0, "+", "-"), "Mean_Diff": md, "Significance": status})[_COLS]

    # ---- contrasts of all columns (factors + dummies) for Lenth / half-normal ------------
    dummy_table = pd.DataFrame({"Factor": [], "Effect": []})
    lenth_info = None
    full_rows = (n_missing == 0)
    Dm = design.dummy_coded
    Fm = (ptype == "factorial")
    if full_rows and Fm.sum() >= 4:
        A = np.column_stack([X_all[Fm], Dm[Fm]])
        nf = int(Fm.sum())
        orth = np.allclose(A.T @ A, nf * np.eye(A.shape[1]), atol=1e-8) and \
            np.allclose(A.sum(axis=0), 0, atol=1e-8)
        if orth:
            contr = A.T @ y_all[Fm] / (nf / 2.0)
            if Dm.shape[1]:
                pre = "" if design.info.get("dummy_kind") == "interaction" else "dummy:"
                dummy_table = pd.DataFrame({"Factor": [f"{pre}{d}" for d in design.dummy_names],
                                            "Effect": contr[k:]})
            if contr.size >= 3:
                lenth_info = L.lenth(contr, alpha)
        else:
            warn.append("Factorial columns are not orthogonal; Lenth/half-normal contrasts skipped.")
    elif n_missing:
        warn.append("Lenth/half-normal contrasts skipped because responses are missing.")

    curv = None
    if use_curv:
        j = 1 + k
        curv = {"n_center": int(is_center.sum()), "estimate": float(fit.beta[j]),
                "se": float(fit.se[j]), "t": float(fit.t[j]), "p": float(fit.p[j]),
                "pure_error_df": int(is_center.sum() - 1)}
    return EffectsResult(design=design, response=response, alpha=alpha, table=table,
                         dummy_table=dummy_table, df_error=int(fit.df), sigma=sigma, n_obs=n,
                         rank=fit.rank, t_limit=tl, bonferroni_limit=bl, n_tests=k,
                         lenth=lenth_info, curvature=curv, warnings=warn,
                         _y=y, _X=X, _curv=(is_center.astype(float) if use_curv else None))
