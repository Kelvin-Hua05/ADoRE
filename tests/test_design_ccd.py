import numpy as np
import pytest

import adore_doe as ad
from adore_doe.design.ccd import ccd, ccd_run_count
from adore_doe.design.criteria import check_design, i_criterion
from adore_doe.design.model import ModelSpec
from adore_doe.design.moment import exact_uniform_moment


def F(k, lo=0.0, hi=10.0):
    return [ad.Factor(chr(65 + i), lo, hi) for i in range(k)]


@pytest.mark.parametrize("k,expect", [(2, 9), (3, 15), (4, 25), (5, 43), (6, 77)])
def test_run_counts_match_adr010_formula(k, expect):
    d = ccd(F(k))
    assert d.n_runs == expect == 2 ** k + 2 * k + 1 == ccd_run_count(k)
    assert d.info["n_factorial"] == 2 ** k and d.info["n_axial"] == 2 * k and d.info["n_center"] == 1
    types = list(d.point_type)
    assert types.count("factorial") == 2 ** k and types.count("axial") == 2 * k and types.count("center") == 1


@pytest.mark.parametrize("k", [2, 3, 4])
def test_structure_of_point_groups(k):
    d = ccd(F(k), center_points=3)
    C = d.coded
    fact = C[d.point_type == "factorial"]
    axl = C[d.point_type == "axial"]
    cen = C[d.point_type == "center"]
    assert set(np.unique(fact)) == {-1.0, 1.0} and len({tuple(r) for r in fact}) == 2 ** k   # all corners
    assert np.allclose(fact.sum(axis=0), 0) and np.allclose(fact.T @ fact, 2 ** k * np.eye(k))
    assert np.all((axl != 0).sum(axis=1) == 1) and set(np.abs(axl[axl != 0])) == {1.0}        # alpha = 1
    assert len({tuple(r) for r in axl}) == 2 * k
    assert np.all(cen == 0) and len(cen) == 3
    assert np.abs(C).max() == 1.0                                                              # inside the cube


def test_all_runs_inside_factor_bounds_in_actual_units():
    fs = [ad.Factor("W", 10, 50), ad.Factor("S", 5, 30), ad.Factor("T", 20, 80)]
    act = ccd(fs).actual_df()
    for f in fs:
        assert act[f.name].min() == pytest.approx(f.low) and act[f.name].max() == pytest.approx(f.high)
        assert act[f.name].between(f.low, f.high).all()
    assert (act.iloc[-1] == [30, 17.5, 50]).all()                                              # centre point


def test_standard_order_first_factor_fastest():
    d = ccd(F(2))
    assert np.array_equal(d.coded[:4], [[-1, -1], [1, -1], [-1, 1], [1, 1]])
    assert np.array_equal(d.coded[4:8], [[-1, 0], [1, 0], [0, -1], [0, 1]])


def test_identifiability_and_info():
    for k in (2, 3, 4):
        d = ccd(F(k))
        p = (k + 1) * (k + 2) // 2
        assert d.info["n_terms"] == p and d.info["rank"] == p and d.info["total_runs"] > p - 1
        X = ModelSpec.from_dict(d.info["model_spec"]).expand(d.coded)
        assert check_design(X).rank == p and np.isfinite(d.info["logdet_XtX"])
    d = ccd(F(3))
    X = ModelSpec("quadratic", 3).expand(d.coded)
    assert d.info["I_criterion_uniform"] == pytest.approx(i_criterion(X, exact_uniform_moment(ModelSpec("quadratic", 3))))


def test_k2_ccd_is_the_3x3_factorial():
    d = ccd(F(2))                                        # faced CCD with 1 centre point == 3^2 grid
    g = np.array([[a, b] for a in (-1, 0, 1) for b in (-1, 0, 1)], float)
    assert {tuple(r) for r in d.coded} == {tuple(r) for r in g}
    s = ModelSpec("quadratic", 2)
    assert d.info["I_criterion_uniform"] == pytest.approx(i_criterion(s.expand(g), exact_uniform_moment(s)))


def test_lack_of_fit_limitation_is_documented_in_info():
    one = ccd(F(3), center_points=1)
    assert one.info["pure_error_df"] == 0 and "NOT available" in one.info["lof_note"]
    three = ccd(F(3), center_points=3)
    assert three.info["pure_error_df"] == 2 and three.n_replicated_runs == 2
    assert ccd(F(3), center_points=0).n_runs == 14


def test_validation_errors_and_scope_limits():
    with pytest.raises(ad.DesignError):
        ccd(F(1))
    with pytest.raises(ad.DesignError, match="limit"):
        ccd(F(9))
    with pytest.raises(ad.DesignError):
        ccd(F(3), center_points=-1)
    with pytest.raises(ad.DesignError):
        ccd(F(3), center_points=True)
    with pytest.raises(ad.DesignError):
        ccd([ad.Factor("A", 0, 1), ad.Factor("A", 0, 2)])
    with pytest.raises(ad.DesignError):
        ccd([ad.Factor("A", 0, 1), ad.Factor("B", "a", "b", factor_type="categorical")])
    import inspect
    assert "alpha" not in inspect.signature(ccd).parameters          # ADR-010: no alpha>1 / no fractional
    assert "fraction" not in inspect.signature(ccd).parameters


def test_large_k_warns_about_run_budget():
    with pytest.warns(ad.InferenceWarning, match="43 runs"):
        ccd(F(5))


def test_seed_reproducibility_and_recording(tmp_path):
    a = ccd(F(3), randomize=True, seed=5)
    b = ccd(F(3), randomize=True, seed=5)
    c = ccd(F(3), randomize=True, seed=6)
    assert np.array_equal(a.run_order, b.run_order) and not np.array_equal(a.run_order, c.run_order)
    assert np.array_equal(a.coded, c.coded) and a.info["seed"] == 5 and a.info["seeds"]["root"] == 5
    assert "randomize" in a.info["seeds"]
    u = ccd(F(3), randomize=True)                                    # seed=None -> recorded, reproducible
    v = ccd(F(3), randomize=True, seed=u.info["seed"])
    assert np.array_equal(u.run_order, v.run_order)
    p = a.to_csv(tmp_path / "ccd.csv")                               # CSV round trip keeps design metadata
    r = ad.ExperimentDataset.from_csv(p)
    assert np.allclose(r.coded, a.coded) and r.info["design_type"] == "ccd_face_centered"
    assert r.info["model_spec"]["kind"] == "quadratic" and list(r.point_type) == list(a.point_type)
