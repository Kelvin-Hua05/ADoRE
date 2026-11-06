import numpy as np
import pytest

import adore_doe as ad
from adore_doe.design import (ModelSpec, ccd, d_optimal, exact_uniform_moment, fds, i_criterion,
                              i_optimal, plot_fds)


def F(k):
    return [ad.Factor(chr(65 + i), 0.0, 10.0) for i in range(k)]


def test_fds_mean_equals_i_criterion_and_is_sorted():
    d = ccd(F(3))
    r = fds(d, m=8192, seed=1)
    exact = d.info["I_criterion_uniform"]
    assert r.stats["mean"] == pytest.approx(exact, rel=0.01)           # mean variance == I-criterion
    assert np.all(np.diff(r.variance) >= 0) and np.all(np.diff(r.fraction) > 0)
    assert 0 < r.fraction[0] < r.fraction[-1] < 1 and r.n_runs == 15
    assert r.stats["min"] <= r.stats["median"] <= r.stats["q80"] <= r.stats["max"]
    assert r.value_at(0.5) == pytest.approx(r.stats["median"])


def test_fds_scaled_is_n_times_unscaled_and_reproducible():
    d = ccd(F(2))
    a, b = fds(d, seed=3), fds(d, seed=3, scaled=True)
    assert np.allclose(b.variance, a.variance * d.n_runs) and b.scaled and not a.scaled
    assert np.array_equal(a.variance, fds(d, seed=3).variance)


def test_fds_on_coded_matrix_and_hand_value():
    # x=[-1,0,1], quadratic: average prediction variance over [-1,1] is exactly 4/5
    r = fds(np.array([[-1.0], [0.0], [1.0]]), "quadratic", m=16384, seed=1)
    assert r.stats["mean"] == pytest.approx(0.8, rel=0.01)
    with pytest.raises(ad.DesignError):
        fds(np.array([[-1.0], [0.0], [1.0]]))                           # model required for a matrix


def test_fds_compares_designs_and_respects_constraints_and_weights():
    fs = F(3)
    di, dd = i_optimal(fs, "quadratic", 15, seed=1), d_optimal(fs, "quadratic", 15, seed=1)
    ri, rd = fds(di, seed=2), fds(dd, seed=2)
    assert ri.stats["mean"] < rd.stats["mean"]                          # I-optimal has lower average variance
    cons = [({"A": 1, "B": 1}, "<=", 12)]
    dc = i_optimal(F(2), "quadratic", 9, constraints=cons, seed=1)
    rc = fds(dc, constraints=cons, seed=2)
    assert np.isfinite(rc.variance).all()
    w = lambda Z: np.exp(-3 * (Z ** 2).sum(axis=1))
    d2 = ccd(F(2))
    rw, ru = fds(d2, weight=w, m=8192, seed=2), fds(d2, seed=2)
    assert rw.weighted and not ru.weighted
    # the weighted mean variance must equal the I-criterion built from the SAME weighted moment matrix
    from adore_doe.design import IntegrationDomain, ModelSpec, i_criterion
    spec = ModelSpec("quadratic", 2)
    Bw = IntegrationDomain.for_cube(2, weight=w, m=8192, seed=5).matrix(spec).B
    assert rw.stats["mean"] == pytest.approx(i_criterion(spec.expand(d2.coded), Bw), rel=0.02)
    assert rw.stats["mean"] != pytest.approx(ru.stats["mean"], rel=0.02)         # weights matter


def test_fds_singular_design_is_reported():
    d = ad.ExperimentDataset(F(2), np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]))
    with pytest.raises(ad.RankDeficientError):
        fds(d, "quadratic")


def test_fds_plot_draws_one_curve_per_design_and_saves(tmp_path):
    fs = F(2)
    res = {"CCD": fds(ccd(fs), seed=1), "I-opt": fds(i_optimal(fs, "quadratic", 9, seed=1), seed=1)}
    out = tmp_path / "fds.png"
    ax = plot_fds(res, save_path=out)
    assert len(ax.lines) == 2 and out.exists() and out.stat().st_size > 1000
    assert ax.get_xlabel() == "Fraction of design space" and "σ²=1" in ax.get_ylabel()
    ax2 = plot_fds(fds(ccd(fs), seed=1, scaled=True))
    assert "Scaled" in ax2.get_ylabel() and len(ax2.lines) == 1
    import matplotlib.pyplot as plt
    plt.close("all")
