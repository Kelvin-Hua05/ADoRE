"""Screening API: turns an EffectsResult into a (shortlist of) selected factors.

    screening = adore_doe.screening.screen(effects, alpha=0.05)
    screening.selected_factors      # shortlist for the next (RSM) stage
    screening.iterations            # full audit trail of every iteration

Three distinct things are kept apart (see also `limits.py`):
  1. single-factor test      |t| > t-limit        (each factor on its own)
  2. multiplicity correction |t| > Bonferroni limit (family-wise error control)
  3. the stepwise procedure  (selection.py) that refits the model between steps.
`candidates` are factors that exceed the limit; `selected_factors` are what the procedure kept.
Neither is proof of a causal "key factor".

Methods: 'ols' (error df available, iterative t-based selection) or 'lenth' (no error df: one
pass with Lenth ME/SME limits; no p-values).  'auto' picks 'ols' when df_error >= 1.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import InferenceError, InferenceWarning
from . import limits as L
from .effects import EffectsResult
from .selection import (IterationRecord, SelectionResult, model_state, partial_stats,
                        stepwise_selection)
from .views import PlotMixin

_CAVEAT = ("Screening shortlist only: stepwise selection reuses the data for selection and "
           "testing (p-values/CIs are optimistic) and PB main effects are aliased with "
           "interactions. Confirm with follow-up experiments (e.g. RSM stage).")


@dataclass
class ScreeningResult(PlotMixin):
    effects: EffectsResult
    method: str
    alpha: float
    criterion: str
    selected_factors: list
    candidates: list
    bonferroni_selected: list
    iterations: list
    stop_reason: str
    final_table: pd.DataFrame
    df_error: int | None
    t_limit: float | None
    bonferroni_limit: float | None
    lenth: dict | None
    notes: list = field(default_factory=list)

    # plot-mixin interface -----------------------------------------------------------------
    @property
    def response(self):
        return self.effects.response

    @property
    def _ptable(self):
        return self.final_table

    @property
    def _pdummy(self):
        return self.effects.dummy_table

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
        return list(self.selected_factors)

    @property
    def _pdummy_label(self):
        return self.effects.design.info.get("dummy_label", "Dummy column (error)")

    # outputs ---------------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return self.final_table.copy()

    def history(self) -> pd.DataFrame:
        """One row per iteration (audit trail)."""
        rows = []
        for r in self.iterations:
            rows.append({"step": r.step, "model_terms": ", ".join(r.model_terms),
                         "df_error": r.df_error, "sigma": r.sigma, "t_limit": r.t_limit,
                         "bonferroni_limit": r.bonferroni_limit,
                         "candidates": ", ".join(r.candidates),
                         "removal_candidates": ", ".join(r.removal_candidates),
                         "action": r.action, "added": ", ".join(r.added),
                         "removed": ", ".join(r.removed), "kept": ", ".join(r.kept),
                         "stop_reason": r.stop_reason or ""})
        return pd.DataFrame(rows)

    def selected_factor_objects(self):
        """`Factor` objects of the selected factors (hand-off to the RSM stage)."""
        d = self.effects.design
        return [d.factor(n) for n in self.selected_factors]

    def export_selected(self, path) -> Path:
        """Write the selected-factor list (.json with factor definitions, else .csv)."""
        path = Path(path)
        sel = self.final_table[self.final_table["Factor"].isin(self.selected_factors)]
        if path.suffix.lower() == ".json":
            payload = {"response": self.response, "method": self.method, "alpha": self.alpha,
                       "criterion": self.criterion, "stop_reason": self.stop_reason,
                       "selected_factors": [f.to_dict() for f in self.selected_factor_objects()],
                       "note": _CAVEAT}
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        else:
            sel.to_csv(path, index=False)
        return path

    def summary(self) -> str:
        L_ = [f"Screening of {self.response!r}  method={self.method}  alpha={self.alpha:g}  "
              f"criterion={self.criterion}"]
        if self.t_limit is not None:
            L_.append(f"final model: error df={self.df_error}  t-limit={self.t_limit:.5f}  "
                      f"Bonferroni limit={self.bonferroni_limit:.5f}")
        elif self.lenth:
            L_.append(f"Lenth PSE={self.lenth['pse']:.5g}  ME={self.lenth['ME']:.5g}  "
                      f"SME={self.lenth['SME']:.5g}  (d={self.lenth['df']:.2f})")
        L_.append(f"initial candidates: {self.candidates}")
        L_.append(f"selected factors  : {self.selected_factors}")
        L_.append(f"  also beyond Bonferroni limit: {self.bonferroni_selected}")
        L_.append(f"stop reason: {self.stop_reason}")
        L_.append(self.final_table.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
        L_ += [f"NOTE: {n}" for n in self.notes]
        return "\n".join(L_)

    def __repr__(self) -> str:
        return f"<ScreeningResult {self.response!r} method={self.method} selected={self.selected_factors}>"


def screen(effects: EffectsResult, alpha: float | None = None, method: str = "auto",
              criterion: str = "t_limit", direction: str = "both", start="full",
              add: str = "best", max_steps: int = 50, min_df: int = 1) -> ScreeningResult:
    """Select screening factors from an :class:`EffectsResult`.

    Parameters
    ----------
    alpha : significance level (default: the one used in ``main_effects``; project setting 0.05).
    method : 'auto' | 'ols' | 'lenth'.
    criterion : 't_limit' (single-test limit) or 'bonferroni' (family-wise limit) as the
        enter/stay rule.
    direction, start, add, max_steps, min_df : see :func:`selection.stepwise_selection`.
    """
    if not isinstance(effects, EffectsResult):
        raise TypeError("effects must be an EffectsResult (see adore_doe.screening.main_effects).")
    alpha = effects.alpha if alpha is None else alpha
    L._check(alpha)
    if method not in ("auto", "ols", "lenth"):
        raise ValueError("method must be 'auto', 'ols' or 'lenth'.")
    notes = [_CAVEAT]
    if effects.design.info.get("design_type") == "plackett_burman":
        notes.append(effects.design.info.get("aliasing_note", ""))
    if method == "auto":
        if effects.inference_available:
            method = "ols"
        else:
            method = "lenth"
            warnings.warn("No error degrees of freedom: falling back to Lenth's method "
                          "(no p-values).", InferenceWarning, stacklevel=2)
    names = effects.factors
    k = len(names)

    if method == "lenth":
        if effects.lenth is None:
            raise InferenceError("Lenth's method is unavailable (needs >=3 orthogonal contrasts "
                                 "and no missing responses).")
        le = L.lenth(np.r_[effects.table["Effect"].to_numpy(),
                           effects.dummy_table["Effect"].to_numpy()], alpha)
        lim = le["ME"] if criterion == "t_limit" else le["SME"]
        eff = effects.table["Effect"].to_numpy()
        sel = [n for n, e in zip(names, eff) if abs(e) > lim]
        cand = [n for n, e in zip(names, eff) if abs(e) > le["ME"]]
        bsel = [n for n in sel if abs(eff[names.index(n)]) > le["SME"]]
        tab = effects.table[["Factor", "Effect", "Abs_Effect", "Sign"]].copy()
        tab["StdErr"] = np.nan
        tab["t"] = np.nan
        tab["p"] = np.nan          # Lenth gives no valid p-values
        tab["Pseudo_t (Effect/PSE)"] = eff / le["pse"] if le["pse"] > 0 else np.nan
        tab["Beyond_ME"] = np.abs(eff) > le["ME"]
        tab["Beyond_SME"] = np.abs(eff) > le["SME"]
        tab["Selected"] = tab["Factor"].isin(sel)
        notes.append("Lenth limits assume effect sparsity and approximate t_{m/3} reference; "
                     "they are guidance, not exact tests; no p-values are reported.")
        return ScreeningResult(effects, "lenth", alpha, criterion, sel, cand, bsel, [],
                               "single pass: Lenth rule (no error df)", tab, None, None, None,
                               le, notes)

    # ---- OLS / t-limit path -------------------------------------------------------------
    if not effects.inference_available:
        raise InferenceError("method='ols' needs error df >= 1; use method='lenth' or 'auto'.")
    res: SelectionResult = stepwise_selection(
        effects._X, effects._y, names, curv=effects._curv, alpha=alpha, criterion=criterion,
        direction=direction, start=start, add=add, max_steps=max_steps, min_df=min_df)
    base = np.ones((len(effects._y), 1)) if effects._curv is None else \
        np.column_stack([np.ones(len(effects._y)), effects._curv])
    S = [names.index(n) for n in res.selected]
    df, sigma, _ = model_state(base, effects._X, effects._y, S)
    eff, se, t = partial_stats(base, effects._X, effects._y, S, sigma)
    tl = L.t_limit(alpha, df)
    bl = L.bonferroni_limit(alpha, df, k)
    p = 2 * stats.t.sf(np.abs(t), df)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    tab = pd.DataFrame({
        "Factor": names, "Effect": eff, "StdErr": se, "t": t, "p": p,
        "CI_low": eff - tcrit * se, "CI_high": eff + tcrit * se, "Std_Effect": t,
        "Abs_Effect": np.abs(eff), "Sign": np.where(eff >= 0, "+", "-"),
        "In_Model": [n in res.selected for n in names],
        "Significance": L.classify(t, tl, bl)})
    tab["Selected"] = tab["In_Model"]
    first = res.iterations[0]
    cand = [n for n in names if abs(first.t_values[n]) > first.t_limit]
    bsel = [n for n in res.selected if abs(tab.loc[tab.Factor == n, "t"].iloc[0]) > bl]
    if df < 3:
        notes.append(f"Final model has only {df} error df: limits are very wide / unstable.")
    if criterion == "t_limit":
        notes.append(f"With {k} factors tested at alpha={alpha:g}, about {k * alpha:.2g} false "
                     f"flag(s) are expected under the global null at the t-limit; the "
                     f"Bonferroni limit controls the family-wise error.")
    return ScreeningResult(effects, "ols", alpha, criterion, list(res.selected), cand, bsel,
                           res.iterations, res.stop_reason, tab, int(df), tl, bl, None, notes)
