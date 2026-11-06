"""Design region in coded space: the cube [-1, 1]^k, optionally cut by LINEAR constraints
A z <= b (ADR-001: bound + linear constraints; non-linear constraints are not supported).

Sampling uses (scrambled) Sobol points and rejection sampling; the acceptance rate is
recorded.  If the feasible fraction is tiny or the region is empty an InfeasibleRegionError
is raised instead of returning a misleading sample (research_methodology §4.2, 約束區域).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

from ..core.seed import make_rng
from ..exceptions import DesignError, InfeasibleRegionError


@dataclass
class Region:
    k: int
    A: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)

    def __post_init__(self):
        self.k = int(self.k)
        self.A = np.zeros((0, self.k)) if self.A is None else np.atleast_2d(np.asarray(self.A, float))
        self.b = np.zeros(0) if self.b is None else np.atleast_1d(np.asarray(self.b, float))
        if self.A.shape[1] != self.k or self.A.shape[0] != self.b.shape[0]:
            raise DesignError("constraint matrix A must be (m x k) and b of length m.")
        if not (np.all(np.isfinite(self.A)) and np.all(np.isfinite(self.b))):
            raise DesignError("constraints must be finite.")
        if self.A.shape[0] and np.any(np.all(self.A == 0, axis=1)):
            raise DesignError("a constraint with all-zero coefficients is not allowed.")

    @classmethod
    def unit_cube(cls, k: int) -> "Region":
        return cls(k)

    @classmethod
    def from_factors(cls, factors, constraints=None) -> "Region":
        """`constraints`: list of (coeffs, op, rhs) in ACTUAL units, e.g.
        ({"Width": 1, "Spacing": 1}, "<=", 0.8)  means  Width + Spacing <= 0.8."""
        fs = list(factors)
        if any(not f.is_numeric for f in fs):
            raise DesignError("constrained regions need numeric factors.")
        k = len(fs)
        names = [f.name for f in fs]
        rows, rhs = [], []
        for c in constraints or []:
            try:
                coeffs, op, r = c
            except (TypeError, ValueError) as e:
                raise DesignError("each constraint must be (coeffs_dict, op, rhs).") from e
            if op not in ("<=", ">="):
                raise DesignError(f"constraint operator must be '<=' or '>=', got {op!r}.")
            unknown = set(coeffs) - set(names)
            if unknown:
                raise DesignError(f"constraint refers to unknown factor(s): {sorted(unknown)}.")
            a = np.array([float(coeffs.get(n, 0.0)) for n in names])
            if not np.any(a):
                raise DesignError("a constraint needs at least one non-zero coefficient.")
            # sum a_j x_j <= r, x_j = c_j + h_j z_j   ->   sum (a_j h_j) z_j <= r - sum a_j c_j
            h = np.array([f.half_range for f in fs])
            c0 = np.array([f.center for f in fs])
            row, val = a * h, float(r) - float(a @ c0)
            if op == ">=":
                row, val = -row, -val
            rows.append(row)
            rhs.append(val)
        return cls(k, np.array(rows) if rows else None, np.array(rhs) if rhs else None)

    @property
    def constrained(self) -> bool:
        return self.A.shape[0] > 0

    def contains(self, Z, tol: float = 1e-9) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, float))
        ok = np.all(np.abs(Z) <= 1 + tol, axis=1)
        if self.constrained:
            ok &= np.all(Z @ self.A.T <= self.b + tol, axis=1)
        return ok

    def sample(self, n: int, seed=None, method: str = "sobol", min_acceptance: float = 1e-3,
               max_draws: int = 2_000_000):
        """n feasible points (coded) and the acceptance rate of the rejection sampler."""
        if n < 1:
            raise DesignError("n must be >= 1.")
        if method not in ("sobol", "lhs", "uniform"):
            raise DesignError("method must be 'sobol', 'lhs' or 'uniform'.")
        rng = make_rng(seed)
        k = self.k
        sob = qmc.Sobol(d=k, scramble=True, seed=rng) if method == "sobol" else None
        lhs = qmc.LatinHypercube(d=k, seed=rng) if method == "lhs" else None

        def draw(m):
            if method == "sobol":
                e = max(1, math.ceil(math.log2(m)))
                return 2 * sob.random_base2(e) - 1
            if method == "lhs":
                return 2 * lhs.random(m) - 1
            return 2 * rng.random((m, k)) - 1

        if not self.constrained:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return draw(n)[:n], 1.0
        got, draws, acc = [], 0, 0
        chunk = 4096
        while acc < n:
            P = draw(chunk)
            draws += len(P)
            chunk = draws                      # Sobol: total points must stay a power of two
            P = P[self.contains(P)]
            got.append(P)
            acc += len(P)
            if draws >= 100_000 and acc == 0:
                raise InfeasibleRegionError("the constraints leave no feasible point "
                                            f"(0 of {draws} candidates feasible).")
            if draws >= max_draws and acc < n:
                raise InfeasibleRegionError(f"only {acc} of {n} feasible points found in "
                                            f"{draws} draws: feasible region too small.")
        rate = acc / draws
        if rate < min_acceptance:
            raise InfeasibleRegionError(f"feasible fraction {rate:.2e} is below min_acceptance="
                                        f"{min_acceptance:g}: region too small to sample reliably.")
        return np.vstack(got)[:n], rate

    def to_dict(self) -> dict:
        return {"k": self.k, "A": self.A.tolist(), "b": self.b.tolist()}
