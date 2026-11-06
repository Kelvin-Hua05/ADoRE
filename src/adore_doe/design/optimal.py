"""I-optimal (primary) and D-optimal (comparison) designs and design augmentation.

Search method: candidate-exchange (Fedorov-type) over a finite candidate set, with several
starts (first = greedy, others random).  For a current information matrix A = F'F (+ tiny ridge,
used ONLY inside the search) and a candidate model row f, the change from adding f is exact:

  D:  det(A + f f') = det(A) (1 + f' A^-1 f)                         (matrix determinant lemma)
      gain  = log(1 + f' A^-1 f)                                      (maximise)
  I:  tr[(A + f f')^-1 B] = tr[A^-1 B] - f' A^-1 B A^-1 f / (1 + f' A^-1 f)   (Sherman-Morrison)
      reduction = f' A^-1 B A^-1 f / (1 + f' A^-1 f)                  (maximise)

A swap removes one point (A - f_s f_s') and adds the best candidate; sweeps repeat until no swap
improves.  This is a HEURISTIC: it is not guaranteed to find the global optimum (several starts
reduce, not remove, that risk; on small problems it is verified against exhaustive search in the
tests).  The reported criteria are recomputed from the final design WITHOUT the ridge, via
rank-checked SVD/QR routines; a singular final design raises RankDeficientError.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.dataset import ExperimentDataset
from ..core.seed import SeedManager, derive_seed
from ..exceptions import DesignError, InferenceWarning, RankDeficientError
from .candidates import CandidateSet, generate_candidates
from .criteria import check_design, d_efficiency, i_criterion, log_det
from .model import ModelSpec, as_model
from .moment import IntegrationDomain, i_criterion_convergence
from .region import Region

RIDGE = 1e-6


def candidate_scores(Minv, Fc, B, criterion: str) -> np.ndarray:
    """Exact incremental benefit (larger = better) of adding each candidate row of Fc to a design
    whose (inverse) information matrix is Minv.  criterion: 'D' (log-det gain) or 'I'."""
    G = Fc @ Minv
    q = np.einsum("ij,ij->i", G, Fc)
    if criterion == "D":
        return np.log1p(q)
    return np.einsum("ij,ij->i", G @ B, G) / (1.0 + q)


def _search(Fc, B, A_fixed, n_free, criterion, allow_rep, rng, init, max_iter):
    N, p = Fc.shape
    idx = list(init)
    A = A_fixed + RIDGE * np.eye(p)
    for i in idx:
        A = A + np.outer(Fc[i], Fc[i])
    while len(idx) < n_free:                                     # greedy fill
        s = candidate_scores(np.linalg.inv(A), Fc, B, criterion)
        if not allow_rep and idx:
            s[idx] = -np.inf
        j = int(np.argmax(s))
        idx.append(j)
        A = A + np.outer(Fc[j], Fc[j])
    it = 0
    for it in range(1, max_iter + 1):                            # exchange sweeps
        improved = False
        for slot in rng.permutation(n_free):
            f = Fc[idx[slot]]
            A0 = A - np.outer(f, f)
            s = candidate_scores(np.linalg.inv(A0), Fc, B, criterion)
            if not allow_rep:
                others = [i for t, i in enumerate(idx) if t != slot]
                if others:
                    s[others] = -np.inf
            j = int(np.argmax(s))
            if j != idx[slot] and s[j] > s[idx[slot]] * (1 + 1e-10) + 1e-12:
                idx[slot] = j
                A = A0 + np.outer(Fc[j], Fc[j])
                improved = True
            else:
                A = A0 + np.outer(f, f)
        if not improved:
            break
    return idx, it


def _value(X, B, criterion):
    """Criterion value of a design, or None if it is singular (never uses a pseudo-inverse)."""
    if check_design(X).singular:
        return None
    return log_det(X) if criterion == "D" else i_criterion(X, B)


def _better(a, b, criterion):
    if b is None:
        return a is not None
    if a is None:
        return False
    return a > b if criterion == "D" else a < b


def _as_candidates(candidates, region, sm, exclude):
    if candidates is None:
        return generate_candidates(region, "auto", seed=sm.child_seed("candidates"), exclude=exclude)
    if isinstance(candidates, CandidateSet):
        P, cs = candidates.points, candidates
    else:
        cs = generate_candidates(region, "user", points=candidates, seed=sm.child_seed("candidates"))
        P = cs.points
    if exclude is not None and len(exclude):
        E = np.atleast_2d(exclude)
        keep = ~np.any(np.max(np.abs(P[:, None, :] - E[None, :, :]), axis=2) <= 1e-9, axis=1)
        if not keep.all():
            cs = CandidateSet(P[keep], cs.method, cs.n_requested, cs.acceptance_rate, cs.seed,
                              cs.levels, int((~keep).sum()))
        if len(cs.points) == 0:
            raise DesignError("candidate set is empty after removing existing points.")
    bad = ~region.contains(cs.points)
    if bad.any():
        raise DesignError(f"{int(bad.sum())} candidate(s) lie outside the design region.")
    return cs


def _setup(factors, model, region, constraints, weight, m, tol, sm):
    factors = list(factors)
    k = len(factors)
    if k < 1:
        raise DesignError("At least one factor is required.")
    if len({f.name for f in factors}) != k:
        raise DesignError("Duplicate factor names.")
    spec = as_model(model, k)
    if region is None:
        if any(not f.is_numeric for f in factors):
            raise DesignError("optimal designs need numeric factors.")
        region = Region.from_factors(factors, constraints)
    elif constraints:
        raise DesignError("pass either `region` or `constraints`, not both.")
    if region.k != k:
        raise DesignError("region and factors have different k.")
    domain = IntegrationDomain(region, weight, m=m, tol=tol, seed=sm.child_seed("moment"))
    return factors, spec, region, domain


def _run_starts(criterion, spec, Fc, Ffix, B, n_free, allow_rep, n_starts, max_iter, sm):
    N, p = Fc.shape
    if not allow_rep and N < n_free:
        raise DesignError(f"only {N} candidate points for {n_free} runs (allow_replicates=False).")
    rng = sm.rng("starts")
    A_fixed = Ffix.T @ Ffix
    best, vals = None, []
    for s in range(max(1, int(n_starts))):
        if n_free == 0:
            idx, it = [], 0
        else:
            init = [] if s == 0 else rng.choice(N, n_free, replace=allow_rep).tolist()
            idx, it = _search(Fc, B, A_fixed, n_free, criterion, allow_rep, rng, init, max_iter)
        X = np.vstack([Ffix, Fc[idx]]) if len(idx) else Ffix
        v = _value(X, B, criterion)
        vals.append(v)
        if best is None or _better(v, best[1], criterion):
            best = (idx, v, it, s)
    if best[1] is None:
        raise RankDeficientError(f"no start produced an identifiable design for the {spec.kind} model "
                                 f"(p={spec.n_terms}); the candidate set cannot support the model or "
                                 f"runs are too few.")
    return best, vals


def _hits(vals, best):
    """How many starts reached the best value (a reliability hint for the heuristic)."""
    return int(sum(v is not None and abs(v - best) <= 1e-6 * max(1.0, abs(best)) for v in vals))


def _check_int(name, v, lo=0):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
        raise DesignError(f"{name} must be an integer >= {lo}.")
    return int(v)


def _optimal(criterion, factors, model, runs, *, region, constraints, weight, candidates,
             center_points, include, allow_replicates, n_starts, m, tol, max_iter, randomize, seed):
    sm = SeedManager(seed)
    factors, spec, region, domain = _setup(factors, model, region, constraints, weight, m, tol, sm)
    k, p = len(factors), spec.n_terms
    runs = _check_int("runs", runs, 1)
    center_points = _check_int("center_points", center_points, 0)
    inc = np.zeros((0, k)) if include is None else np.atleast_2d(np.asarray(include, float))
    if inc.shape[1] != k:
        raise DesignError(f"`include` must have {k} columns.")
    if len(inc) and not region.contains(inc).all():
        raise DesignError("some `include` points lie outside the design region.")
    cen = np.zeros((center_points, k))
    if center_points and not region.contains(cen).all():
        raise DesignError("the centre point is infeasible in this constrained region.")
    fixed = np.vstack([inc, cen])
    n_free = runs - len(fixed)
    if n_free < 0:
        raise DesignError(f"runs={runs} is smaller than the {len(fixed)} forced runs "
                          f"(include + centre points).")
    if runs < p:
        raise DesignError(f"runs={runs} < {p} parameters of the {spec.kind} model: not identifiable.")
    if runs == p:
        warnings.warn(f"runs == number of model terms ({p}): saturated design, no error df.",
                      InferenceWarning, stacklevel=3)
    cs = _as_candidates(candidates, region, sm, None)
    Fc = spec.expand(cs.points)
    Ffix = spec.expand(fixed) if len(fixed) else np.zeros((0, p))
    mom = domain.matrix(spec)
    (idx, val, it, bstart), vals = _run_starts(criterion, spec, Fc, Ffix, mom.B, n_free,
                                                allow_replicates, n_starts, max_iter, sm)
    free = cs.points[idx] if len(idx) else np.zeros((0, k))
    if len(free):
        free = free[np.lexsort(free.T[::-1])]
    coded = np.vstack([inc, free, cen])
    ptype = np.array(["included"] * len(inc) + ["optimal"] * len(free) + ["center"] * center_points,
                     dtype=object)
    X = spec.expand(coded)
    chk = check_design(X)
    conv = i_criterion_convergence(X, spec, domain)
    if not conv["converged"]:
        warnings.warn(f"I-criterion not converged over the integration points (last relative change "
                      f"{conv['rel_changes'][-1]:.3g} > tol={tol:g}); increase `m`.", InferenceWarning,
                      stacklevel=3)
    info = {"design_type": f"{criterion.lower()}_optimal", "criterion": criterion,
            "k": k, "total_runs": runs, "n_free": n_free, "n_included": len(inc),
            "n_center": center_points, "model_spec": spec.to_dict(), "n_terms": p,
            "rank": chk.rank, "cond_X": chk.cond,
            "I_criterion": i_criterion(X, mom.B), "logdet_XtX": log_det(X),
            "D_efficiency": d_efficiency(X),
            "weight": "uniform" if domain.uniform else "custom",
            "moment": {"method": mom.method, "m": mom.m, "converged": mom.converged,
                       "tol": mom.tol, "rel_changes": mom.rel_changes,
                       "acceptance_rate": mom.acceptance_rate},
            "moment_check": conv,
            "candidates": cs.info(), "region": region.to_dict(),
            "allow_replicates": bool(allow_replicates), "n_starts": int(n_starts),
            "start_values": [None if v is None else float(v) for v in vals],
            "starts_reaching_best": _hits(vals, val),
            "best_start": int(bstart), "exchange_sweeps": int(it),
            "search": "candidate-exchange heuristic (not guaranteed globally optimal)",
            "randomized": False}
    ds = ExperimentDataset(factors, coded, point_type=ptype, info=info)
    if randomize:
        ds.randomize(sm.child_seed("randomize"))
    ds.info["seed"] = sm.root
    ds.info["seeds"] = sm.as_dict()
    return ds


def i_optimal(factors, model, runs, *, region=None, constraints=None, weight=None, candidates=None,
              center_points=0, include=None, allow_replicates=False, n_starts=20, m=4096, tol=0.01,
              max_iter=100, randomize=False, seed=None) -> ExperimentDataset:
    """I-optimal design (minimises tr[(X'X)^-1 B], the average prediction variance).

    `runs` is the TOTAL number of runs (including forced `include` points and `center_points`)
    and has no default.  `region`/`constraints` (linear, actual units), `weight` (callable on
    coded points; default uniform), `candidates` (CandidateSet or coded array), `m` (integration
    points) and `tol` (convergence tolerance) are recorded in `dataset.info`."""
    return _optimal("I", factors, model, runs, region=region, constraints=constraints, weight=weight,
                    candidates=candidates, center_points=center_points, include=include,
                    allow_replicates=allow_replicates, n_starts=n_starts, m=m, tol=tol,
                    max_iter=max_iter, randomize=randomize, seed=seed)


def d_optimal(factors, model, runs, *, region=None, constraints=None, weight=None, candidates=None,
              center_points=0, include=None, allow_replicates=False, n_starts=20, m=4096, tol=0.01,
              max_iter=100, randomize=False, seed=None) -> ExperimentDataset:
    """D-optimal design (maximises log det X'X); kept as the comparison method.  Same arguments as
    :func:`i_optimal` (the integration domain is only used to REPORT the I-criterion)."""
    return _optimal("D", factors, model, runs, region=region, constraints=constraints, weight=weight,
                    candidates=candidates, center_points=center_points, include=include,
                    allow_replicates=allow_replicates, n_starts=n_starts, m=m, tol=tol,
                    max_iter=max_iter, randomize=randomize, seed=seed)


@dataclass
class Augmentation:
    """Proposed extra runs.  Nothing is simulated or added until you call `to_dataset`."""
    points: np.ndarray
    factors: list
    criterion: str
    value_before: float | None          # None = existing design not identifiable
    value_after: float
    info: dict = field(default_factory=dict)

    def actual_df(self) -> pd.DataFrame:
        return pd.DataFrame({f.name: f.from_coded(self.points[:, j]) for j, f in enumerate(self.factors)})

    def to_csv(self, path):
        df = self.actual_df()
        for j, f in enumerate(self.factors):
            df[f"{f.name}[coded]"] = self.points[:, j]
        df.to_csv(path, index=False)
        return path

    def to_dataset(self, dataset: ExperimentDataset, source: str | None = None) -> np.ndarray:
        """Append the proposed runs to `dataset` (responses NaN); returns the new run_ids."""
        return dataset.append_runs(self.points, source=source or f"augment-{self.criterion}",
                                   point_type="augment")


def augment(dataset: ExperimentDataset, n_add: int, *, criterion: str = "I", model="quadratic",
            region=None, constraints=None, weight=None, candidates=None, allow_replicates=False,
            exclude_existing=True, n_starts=5, m=4096, tol=0.01, max_iter=100, seed=None) -> Augmentation:
    """Choose `n_add` additional runs that best improve the I (or D) criterion given the existing
    runs.  Existing runs are kept fixed; candidates already used are excluded unless
    `allow_replicates=True`; validation runs are not part of the training design."""
    if criterion not in ("I", "D"):
        raise DesignError("criterion must be 'I' or 'D'.")
    n_add = _check_int("n_add", n_add, 1)
    sm = SeedManager(seed)
    ds = dataset.training() if dataset.validation_mask.any() else dataset
    factors, spec, region, domain = _setup(ds.factors, model, region, constraints, weight, m, tol, sm)
    existing = ds.coded
    cs = _as_candidates(candidates, region, sm, existing if exclude_existing and not allow_replicates else None)
    Fc, Fex = spec.expand(cs.points), spec.expand(existing)
    B = domain.matrix(spec).B
    before = _value(Fex, B, criterion)
    if before is None:
        warnings.warn(f"existing design is not identifiable for the {spec.kind} model "
                      f"(rank {check_design(Fex).rank} < {spec.n_terms}); the {criterion}-criterion before "
                      f"augmentation is undefined.", InferenceWarning, stacklevel=2)
    (idx, val, it, bstart), vals = _run_starts(criterion, spec, Fc, Fex, B, n_add, allow_replicates,
                                                n_starts, max_iter, sm)
    pts = cs.points[idx]
    info = {"criterion": criterion, "n_existing": len(existing), "n_add": n_add,
            "model_spec": spec.to_dict(), "candidates": cs.info(), "allow_replicates": allow_replicates,
            "moment": {"m": domain.matrix(spec).m, "converged": domain.matrix(spec).converged},
            "best_start": int(bstart), "starts_reaching_best": _hits(vals, val),
            "seeds": sm.as_dict(), "seed": sm.root,
            "search": "candidate-exchange heuristic (not guaranteed globally optimal)"}
    return Augmentation(pts, list(ds.factors), criterion, before, float(val), info)
