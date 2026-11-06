"""Candidate sets for optimal designs / augmentation (research_methodology §4.2 row: 候選集).

methods: 'grid' (levels^k lattice incl. vertices/midpoints), 'sobol', 'lhs', 'uniform', 'user',
'auto' (grid whose size is manageable; for constrained regions a Sobol top-up is added).
Every method is reproducible from `seed`; points outside the region are removed; the size and
acceptance rate are recorded.  Candidate generation never evaluates anything.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..core.seed import derive_seed, resolve_seed
from ..exceptions import DesignError
from .region import Region


@dataclass
class CandidateSet:
    points: np.ndarray
    method: str
    n_requested: int
    acceptance_rate: float
    seed: int
    levels: int | None = None
    n_removed_existing: int = 0

    def __len__(self):
        return len(self.points)

    def info(self) -> dict:
        return {"method": self.method, "size": len(self.points), "n_requested": self.n_requested,
                "acceptance_rate": self.acceptance_rate, "levels": self.levels, "seed": self.seed,
                "n_removed_existing": self.n_removed_existing}


def _dedupe(P, decimals=9):
    _, idx = np.unique(np.round(P, decimals), axis=0, return_index=True)
    return P[np.sort(idx)]


def generate_candidates(region, method: str = "auto", n: int = 2000, levels: int | None = None,
                        seed=None, points=None, exclude=None, exclude_tol: float = 1e-9) -> CandidateSet:
    region = Region.unit_cube(region) if isinstance(region, (int, np.integer)) else region
    k = region.k
    seed = resolve_seed(seed)
    s = derive_seed(seed, "candidates")
    rate, used_levels = 1.0, None
    if method == "user":
        if points is None:
            raise DesignError("method='user' needs `points`.")
        P = np.atleast_2d(np.asarray(points, float))
        if P.shape[1] != k:
            raise DesignError(f"user candidates must have {k} columns.")
        bad = ~region.contains(P)
        if bad.any():
            raise DesignError(f"{int(bad.sum())} user candidate(s) lie outside the design region.")
    elif method in ("grid", "auto"):
        lv = levels or (5 if k <= 4 else 3)
        if lv < 2:
            raise DesignError("levels must be >= 2.")
        if method == "auto" and lv ** k > 6000:
            lv = 3 if 3 ** k <= 6000 else 0
        if lv:
            g = np.linspace(-1, 1, lv)
            G = np.array(list(itertools.product(g, repeat=k)))
            P = G[region.contains(G)]
            used_levels = lv
            if len(P) == 0 and method == "grid":
                raise DesignError("no grid point is feasible; use method='sobol'.")
        else:
            P = np.zeros((0, k))
        if method == "auto" and (region.constrained or len(P) == 0):
            extra, rate = region.sample(n if len(P) == 0 else max(n // 4, 200), seed=s)
            P = np.vstack([P, extra]) if len(P) else extra
    elif method in ("sobol", "lhs", "uniform"):
        P, rate = region.sample(n, seed=s, method=method)
    else:
        raise DesignError("method must be one of auto, grid, sobol, lhs, uniform, user.")
    P = _dedupe(P)
    removed = 0
    if exclude is not None and len(np.atleast_2d(exclude)):
        E = np.atleast_2d(np.asarray(exclude, float))
        d = np.max(np.abs(P[:, None, :] - E[None, :, :]), axis=2)
        keep = ~np.any(d <= exclude_tol, axis=1)
        removed = int((~keep).sum())
        P = P[keep]
    if len(P) == 0:
        raise DesignError("candidate set is empty after removing existing points.")
    return CandidateSet(P, method, int(n), float(rate), seed, used_levels, removed)
