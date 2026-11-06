"""Integration domain and moment matrix B (research_methodology §4.2 rows: 權重, 數值近似).

    B = ∫_Ω f(x) f(x)' w(x) dx / ∫_Ω w(x) dx  ≈  Σ_j w_j f(u_j) f(u_j)' / Σ_j w_j

u_j are scrambled-Sobol points inside the feasible region (rejection sampling for linear
constraints); w defaults to the uniform weight (callable w(Z) -> weights >= 0 otherwise).
The number of integration points m is configurable.  Two approximation checks are recorded:

* moment_matrix(): m is doubled until the relative Frobenius change of B is below `tol`
  (default 1 %, ADR-001);
* i_criterion_convergence(X): for a given design, the I-criterion is recomputed with
  m, 2m, 4m, ... points and the relative change of the CRITERION VALUE is checked
  (this is the check that the acceptance test refers to).

For the unconstrained cube with uniform weight the exact moments are also available
(`exact_uniform_moment`): E[prod z_j^a_j] = prod_j 1/(a_j+1) for even a_j, 0 otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.seed import derive_seed, resolve_seed
from ..exceptions import DesignError
from .criteria import i_criterion
from .model import ModelSpec
from .region import Region


def exact_uniform_moment(spec: ModelSpec) -> np.ndarray:
    T = np.array(spec.terms)
    p = len(T)
    B = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            e = T[i] + T[j]
            B[i, j] = 0.0 if np.any(e % 2) else float(np.prod(1.0 / (e + 1)))
    return B


@dataclass
class MomentResult:
    B: np.ndarray
    m: int
    method: str                       # 'sobol-numeric' | 'analytic'
    converged: bool
    tol: float
    rel_changes: list = field(default_factory=list)
    acceptance_rate: float = 1.0


class IntegrationDomain:
    def __init__(self, region: Region, weight=None, m: int = 4096, tol: float = 0.01,
                 seed=None, m_max: int = 2**17):
        if m < 16:
            raise DesignError("m (integration points) must be >= 16.")
        if not 0 < tol < 1:
            raise DesignError("tol must be in (0, 1).")
        self.region, self.weight, self.m, self.tol, self.m_max = region, weight, int(m), tol, int(m_max)
        self.seed = resolve_seed(seed)
        self._pts, self._rate, self._cache = None, 1.0, {}

    @classmethod
    def for_cube(cls, k: int, **kw) -> "IntegrationDomain":
        return cls(Region.unit_cube(k), **kw)

    @property
    def k(self) -> int:
        return self.region.k

    @property
    def uniform(self) -> bool:
        return self.weight is None

    def points(self, n: int):
        """First n integration points (coded) and their weights.  Re-using the same seed makes
        every prefix identical, so larger n only EXTENDS the sample."""
        if self._pts is None or len(self._pts) < n:
            P, rate = self.region.sample(n, seed=derive_seed(self.seed, "integration"))
            self._pts, self._rate = P, rate
        P = self._pts[:n]
        if self.weight is None:
            return P, np.ones(len(P))
        w = np.asarray(self.weight(P), float)
        if w.shape != (len(P),) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise DesignError("weight(Z) must return finite non-negative weights (sum > 0).")
        return P, w

    def _B(self, spec, n):
        P, w = self.points(n)
        F = spec.expand(P)
        return (F * w[:, None]).T @ F / w.sum()

    def matrix(self, spec: ModelSpec, exact: bool = False) -> MomentResult:
        if spec.k != self.k:
            raise DesignError("model and integration domain have different k.")
        if exact:
            if self.region.constrained or not self.uniform:
                raise DesignError("exact moments exist only for the unconstrained cube with uniform weight.")
            return MomentResult(exact_uniform_moment(spec), 0, "analytic", True, 0.0)
        key = (spec.kind, spec.k)
        if key in self._cache:
            return self._cache[key]
        m, changes, converged = self.m, [], False
        Bm = self._B(spec, m)
        while True:
            B2 = self._B(spec, 2 * m)
            rel = float(np.linalg.norm(B2 - Bm) / np.linalg.norm(B2))
            changes.append(rel)
            Bm, m = B2, 2 * m
            if rel < self.tol:
                converged = True
                break
            if m * 2 > self.m_max:
                break
        res = MomentResult(Bm, m, "sobol-numeric", converged, self.tol, changes, self._rate)
        self._cache[key] = res
        return res


def i_criterion_convergence(X_design, spec: ModelSpec, domain: IntegrationDomain) -> dict:
    """Relative change of the I-criterion value as the number of integration points doubles."""
    m, vals, ms, rels = domain.m, [], [], []
    X = np.asarray(X_design, float)
    while True:
        ms.append(m)
        vals.append(i_criterion(X, domain._B(spec, m)))
        if len(vals) > 1:
            rels.append(abs(vals[-1] - vals[-2]) / abs(vals[-1]))
            if rels[-1] < domain.tol:
                break
        if m * 2 > domain.m_max:
            break
        m *= 2
    return {"m_values": ms, "I_values": vals, "rel_changes": rels, "tol": domain.tol,
            "converged": bool(rels and rels[-1] < domain.tol), "acceptance_rate": domain._rate}
