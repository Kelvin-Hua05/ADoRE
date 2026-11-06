"""Central Composite Design, as fixed by ADR-010.

* full factorial portion 2^k (levels -1/+1, standard Yates order: first factor fastest),
* axial ("star") points at +-alpha with alpha = 1  (FACE-CENTERED: axial points lie on the faces
  of the cube, so every point stays inside the factor bounds),
* center points: default 1 (ADR-010); user may change.
Runs = 2^k + 2k + n_center  (k=3: 15, k=4: 25, k=5: 43 with the default 1 center point).

Not implemented on purpose (ADR-010): fractional-factorial CCD and alpha > 1 (rotatable /
circumscribed) designs, whose axial points would leave the factor bounds.

Identifiability: the quadratic model has p = (k+1)(k+2)/2 terms; the design is checked to have
rank p.  With ONE center point there is NO pure-error estimate (no replicated run), so a
lack-of-fit test is not possible; use >= 2 center points (or add replicates) if LOF is needed.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..core.dataset import ExperimentDataset
from ..core.factor import Factor
from ..core.seed import SeedManager, derive_seed
from ..exceptions import DesignError, InferenceWarning
from .criteria import check_design, d_efficiency, i_criterion, log_det, require_full_rank
from .model import ModelSpec
from .moment import exact_uniform_moment

K_MAX = 8


def ccd_run_count(k: int, center_points: int = 1) -> int:
    return 2 ** k + 2 * k + int(center_points)


def ccd(factors: list[Factor], center_points: int = 1, randomize: bool = False,
        seed: int | None = None) -> ExperimentDataset:
    """Face-centered full-factorial CCD (ADR-010).  Returns an ExperimentDataset (coded + actual)."""
    factors = list(factors)
    k = len(factors)
    if k < 2:
        raise DesignError("A CCD needs at least 2 factors.")
    if k > K_MAX:
        raise DesignError(f"Full-factorial CCD with k={k} needs {ccd_run_count(k)} runs; the limit is "
                          f"k={K_MAX}. Use an I-optimal design for many factors.")
    if not all(isinstance(f, Factor) and f.is_numeric for f in factors):
        raise DesignError("CCD needs numeric Factor objects.")
    if len({f.name for f in factors}) != k:
        raise DesignError("Duplicate factor names.")
    if isinstance(center_points, bool) or not isinstance(center_points, (int, np.integer)) or center_points < 0:
        raise DesignError("center_points must be a non-negative integer.")
    n_total = ccd_run_count(k, center_points)
    if k >= 5:
        warnings.warn(f"CCD with k={k} factors needs {n_total} runs (> the 10-30 run budget typical for "
                      f"Module 1); consider an I-optimal design.", InferenceWarning, stacklevel=2)
    n_fact = 2 ** k
    idx = np.arange(n_fact)
    fact = np.column_stack([np.where((idx >> j) & 1, 1.0, -1.0) for j in range(k)])
    axial = np.zeros((2 * k, k))
    for j in range(k):
        axial[2 * j, j], axial[2 * j + 1, j] = -1.0, 1.0
    cen = np.zeros((int(center_points), k))
    coded = np.vstack([fact, axial, cen])
    ptype = np.array(["factorial"] * n_fact + ["axial"] * (2 * k) + ["center"] * int(center_points),
                     dtype=object)

    spec = ModelSpec("quadratic", k)
    X = spec.expand(coded)
    chk = require_full_rank(X, what=f"CCD (k={k}, {n_total} runs)")
    sm = SeedManager(seed)
    info = {"design_type": "ccd_face_centered", "k": k, "alpha": 1.0, "total_runs": n_total,
            "n_factorial": n_fact, "n_axial": 2 * k, "n_center": int(center_points),
            "model_spec": spec.to_dict(), "n_terms": spec.n_terms, "rank": chk.rank,
            "cond_X": chk.cond, "I_criterion_uniform": i_criterion(X, exact_uniform_moment(spec)),
            "logdet_XtX": log_det(X), "D_efficiency": d_efficiency(X),
            "pure_error_df": max(int(center_points) - 1, 0),
            "randomized": False, "seed": sm.root,
            "lof_note": ("no replicated run: pure error and lack-of-fit are NOT available "
                         "(ADR-010); use center_points >= 2 or add replicates." if center_points < 2 else
                         f"{center_points - 1} pure-error df from center points."),
            "note": "face-centered CCD: all runs inside the factor bounds; alpha>1 and fractional CCD "
                    "are not implemented (ADR-010)."}
    ds = ExperimentDataset(factors, coded, point_type=ptype, info=info)
    if randomize:
        ds.randomize(derive_seed(sm.root, "randomize"))
        ds.info["randomize_seed"] = derive_seed(sm.root, "randomize")
        sm.child_seed("randomize")
        ds.info["seed"] = sm.root
    ds.info["seeds"] = sm.as_dict()
    return ds
