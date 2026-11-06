"""Fraction of Design Space (FDS) of a design.

For a design X (model matrix of its runs) the prediction variance at a point x is
    v(x) = sigma^2 f(x)' (X'X)^-1 f(x)         (sigma^2 = 1 here; `scaled=True` multiplies by n)
v is evaluated at integration points drawn from the design region (Sobol, constraints respected,
weights honoured) and sorted; FDS(q) = v at the q-quantile of the (weighted) region.  The
weighted mean of v is the I-criterion; the maximum is the G-criterion (worst-case variance).

This is the CLASSICAL polynomial-model variance -- NOT a GPR predictive variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.dataset import ExperimentDataset
from ..core.seed import SeedManager
from ..exceptions import DesignError
from .criteria import prediction_variance, require_full_rank
from .model import ModelSpec, as_model
from .moment import IntegrationDomain
from .region import Region


@dataclass
class FDSResult:
    fraction: np.ndarray
    variance: np.ndarray
    n_points: int
    scaled: bool
    weighted: bool
    n_runs: int
    stats: dict

    def value_at(self, q: float) -> float:
        """Prediction variance that holds over the fraction q of the design space."""
        return float(np.interp(q, self.fraction, self.variance))


def fds(design, model=None, *, region=None, constraints=None, weight=None, m: int = 4096,
        seed=None, scaled: bool = False) -> FDSResult:
    """FDS of an ExperimentDataset (training runs) or of a coded run matrix `design` (+ `model`)."""
    if isinstance(design, ExperimentDataset):
        ds = design.training() if design.validation_mask.any() else design
        Z, factors = ds.coded, ds.factors
        if model is None:
            model = ModelSpec.from_dict(ds.info["model_spec"]) if "model_spec" in ds.info else "quadratic"
    else:
        Z, factors = np.atleast_2d(np.asarray(design, float)), None
        if model is None:
            raise DesignError("pass `model` when `design` is a coded matrix.")
    spec = as_model(model, Z.shape[1])
    if region is None:
        region = Region.from_factors(factors, constraints) if factors else Region.unit_cube(spec.k)
    sm = SeedManager(seed)
    dom = IntegrationDomain(region, weight, m=m, seed=sm.child_seed("fds"))
    X = spec.expand(Z)
    require_full_rank(X, what="design")
    P, w = dom.points(m)
    v = prediction_variance(X, spec.expand(P))
    if scaled:
        v = v * len(Z)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    frac = (np.cumsum(w) - 0.5 * w) / w.sum()
    stats = {"min": float(v[0]), "max": float(v[-1]), "mean": float(np.sum(w * v) / w.sum()),
             "median": float(np.interp(0.5, frac, v)), "q80": float(np.interp(0.8, frac, v))}
    return FDSResult(frac, v, int(m), bool(scaled), weight is not None, len(Z), stats)


def plot_fds(results, ax=None, save_path=None, title=None):
    """Plot one FDSResult or a dict {label: FDSResult} (lower curve = better)."""
    from .plots import plot_fds as _plot
    return _plot(results, ax=ax, save_path=save_path, title=title)
