"""Iterative (stepwise) factor selection driven by t-limit / Bonferroni limit.

This module is independent of plotting and of the design container: it works on arrays.

Procedure (one *iteration* = one pass through steps 1-5)
---------------------------------------------------------
State: a model M (set of factors in the model) fitted by OLS with intercept
(+ curvature term if center points exist).  sigma_M^2 = RSS_M / df_M is the error estimate
("pooled error from everything NOT in the model").

 1. For every factor j compute the *partial* t-statistic given the other terms of M
    (for j in M: j removed from M; for j not in M: j added to M), always using sigma_M:
        r_j = residual of x_j regressed on Z_{M \\ j},  b_j = r_j'y / r_j'r_j,
        t_j = b_j / (sigma_M / sqrt(r_j'r_j)).
    Using one sigma_M and df_M for all factors puts all t's on one scale, which is what a
    Pareto chart with a single t-limit requires.
 2. limit = t_{1-alpha/2, df_M}  (criterion='t_limit')  or  t_{1-alpha/(2m), df_M}
    (criterion='bonferroni'), m = total number of candidate factors.
 3. candidates       = factors NOT in M with |t_j| > limit        (entry candidates)
    removal cands.   = factors IN M with |t_j| <= limit
 4. Rule: direction='backward' / 'forward' / 'both'.  Removal is tried first (backward /
    both): the *least* significant removal candidate leaves.  Otherwise the *most*
    significant entry candidate enters (add='best') or all of them (add='all').
 5. Refit and repeat until: no change ('converged'), `max_steps`, residual df would drop
    below `min_df`, or a previously visited model recurs ('cycle').

Default start is the FULL model (needs df_full >= min_df), because starting from the null
model inflates sigma_M with the very effects being searched for.

Caveats (documented, not hidden): the output is a *screening shortlist*.  Stepwise selection
uses the data twice (select, then test), so p-values/CIs after selection are optimistic and
the procedure gives no guarantee about causality or about the true active set.  In PB designs
main effects are aliased with interactions.  "Exceeds the limit" != "established key factor".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..exceptions import InferenceError
from . import limits as L


@dataclass
class IterationRecord:
    step: int
    model_terms: list           # factors in the model BEFORE this step's action
    df_error: int
    sigma: float
    t_limit: float
    bonferroni_limit: float
    t_values: dict              # partial t of every factor under this model's sigma
    candidates: list            # not in model, |t| > limit
    removal_candidates: list    # in model, |t| <= limit
    action: str                 # 'remove' | 'add' | 'none'
    added: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    kept: list = field(default_factory=list)   # model terms AFTER this step
    stop_reason: str | None = None


@dataclass
class SelectionResult:
    iterations: list
    selected: list
    stop_reason: str
    criterion: str
    direction: str
    alpha: float


def _resid_x(base, X, others, j):
    Zo = np.column_stack([base, X[:, others]]) if others else base
    coef, *_ = np.linalg.lstsq(Zo, X[:, j], rcond=None)
    return X[:, j] - Zo @ coef


def model_state(base, X, y, S):
    """OLS of y on [base, X[:, S]] -> (df, sigma, rss). sigma=nan if df<1."""
    Z = np.column_stack([base, X[:, S]]) if S else base
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    r = y - Z @ beta
    rss = float(r @ r)
    df = len(y) - int(np.linalg.matrix_rank(Z))
    sigma = float(np.sqrt(rss / df)) if df >= 1 else float("nan")
    return df, sigma, rss


def partial_stats(base, X, y, S, sigma):
    """Partial (effect, se_effect, t) of each factor given model S, using sigma (see module doc)."""
    k = X.shape[1]
    eff = np.full(k, np.nan)
    se = np.full(k, np.nan)
    t = np.full(k, np.nan)
    for j in range(k):
        others = [i for i in S if i != j]
        r = _resid_x(base, X, others, j)
        d = float(r @ r)
        if d <= 1e-10 * float(X[:, j] @ X[:, j]):      # aliased with the model -> not estimable
            continue
        b = float(r @ y) / d
        s = sigma / np.sqrt(d)
        eff[j], se[j] = 2 * b, 2 * s
        t[j] = (b / s) if s > 0 else (np.sign(b) * np.inf if b != 0 else 0.0)
    return eff, se, t


def stepwise_selection(X, y, names, curv=None, alpha=0.05, criterion="t_limit",
                       direction="both", start="full", add="best", max_steps=50,
                       min_df=1) -> SelectionResult:
    """Run the iterative selection described in the module docstring."""
    if criterion not in ("t_limit", "bonferroni"):
        raise ValueError("criterion must be 't_limit' or 'bonferroni'.")
    if direction not in ("backward", "forward", "both"):
        raise ValueError("direction must be 'backward', 'forward' or 'both'.")
    if add not in ("best", "all"):
        raise ValueError("add must be 'best' or 'all'.")
    L._check(alpha)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    k = X.shape[1]
    base = np.ones((len(y), 1)) if curv is None else np.column_stack([np.ones(len(y)), curv])
    if start == "full":
        S = list(range(k))
    elif start == "empty":
        S = []
    else:
        S = sorted(names.index(s) for s in start)
    df0, _, _ = model_state(base, X, y, S)
    if df0 < min_df:
        raise InferenceError(
            f"Starting model has {df0} error df (< min_df={min_df}); cannot run t-based selection. "
            f"Use start='empty', Lenth's method, or add center points / replicates.")

    iters: list[IterationRecord] = []
    seen = {frozenset(S)}
    reason = f"max_steps={max_steps} reached"
    for step in range(1, max_steps + 1):
        df, sigma, _ = model_state(base, X, y, S)
        eff, se, t = partial_stats(base, X, y, S, sigma)
        tl = L.t_limit(alpha, df)
        bl = L.bonferroni_limit(alpha, df, k)
        lim = tl if criterion == "t_limit" else bl
        absn = np.abs(np.nan_to_num(t, nan=0.0))
        cand = [j for j in range(k) if j not in S and absn[j] > lim]
        rem = [j for j in S if absn[j] <= lim]
        rec = IterationRecord(step, [names[j] for j in S], df, sigma, tl, bl,
                              {names[j]: float(t[j]) for j in range(k)},
                              [names[j] for j in sorted(cand, key=lambda j: -absn[j])],
                              [names[j] for j in sorted(rem, key=lambda j: absn[j])],
                              "none")
        new_S = list(S)
        if rem and direction in ("backward", "both"):
            j = min(rem, key=lambda j: absn[j])
            new_S.remove(j)
            rec.action, rec.removed = "remove", [names[j]]
        elif cand and direction in ("forward", "both"):
            order = sorted(cand, key=lambda j: -absn[j])
            take = order[:1] if add == "best" else order
            for j in take:
                trial = sorted(new_S + [j])
                dft, _, _ = model_state(base, X, y, trial)
                if dft < min_df:
                    break
                new_S = trial
                rec.added.append(names[j])
            if rec.added:
                rec.action = "add"
            else:
                rec.stop_reason = "adding a candidate would leave fewer than min_df error df"
        if rec.action == "none":
            rec.stop_reason = rec.stop_reason or "converged: no factor to add or remove"
            rec.kept = [names[j] for j in S]
            iters.append(rec)
            reason = rec.stop_reason
            break
        new_S = sorted(new_S)
        rec.kept = [names[j] for j in new_S]
        key = frozenset(new_S)
        if key in seen:
            rec.stop_reason = "cycle detected (model revisited); selection stopped"
            iters.append(rec)
            reason = rec.stop_reason
            S = new_S
            break
        seen.add(key)
        S = new_S
        iters.append(rec)
    else:
        iters[-1].stop_reason = reason
    return SelectionResult(iters, [names[j] for j in S], reason, criterion, direction, alpha)
