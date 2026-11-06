import itertools
import warnings

import numpy as np
import pytest

import adore_doe as ad
from adore_doe.design.candidates import CandidateSet, generate_candidates
from adore_doe.design.ccd import ccd
from adore_doe.design.criteria import check_design, i_criterion, log_det
from adore_doe.design.model import ModelSpec
from adore_doe.design.moment import exact_uniform_moment
from adore_doe.design.optimal import augment, candidate_scores, d_optimal, i_optimal
from adore_doe.design.region import Region


def F(k, lo=0.0, hi=10.0):
    return [ad.Factor(chr(65 + i), lo, hi) for i in range(k)]


def X_of(ds):
    return ModelSpec.from_dict(ds.info["model_spec"]).expand(ds.coded)


# ---------------------------------------------------------------- exact incremental formulas
def test_incremental_scores_equal_recomputed_criteria_differences():
    spec = ModelSpec("quadratic", 2)
    rng = np.random.default_rng(0)
    X = spec.expand(rng.uniform(-1, 1, (12, 2)))
    Fc = spec.expand(rng.uniform(-1, 1, (30, 2)))
    B = exact_uniform_moment(spec)
    Minv = np.linalg.inv(X.T @ X)
    sD = candidate_scores(Minv, Fc, B, "D")
    sI = candidate_scores(Minv, Fc, B, "I")
    for c in range(len(Fc)):
        Xn = np.vstack([X, Fc[c]])
        assert sD[c] == pytest.approx(log_det(Xn) - log_det(X), abs=1e-9)          # determinant lemma
        assert sI[c] == pytest.approx(i_criterion(X, B) - i_criterion(Xn, B), abs=1e-9)   # Sherman-Morrison


# ---------------------------------------------------------------- exchange vs exhaustive search
def brute(k, n, L, crit):
    spec = ModelSpec("quadratic", k)
    P = generate_candidates(k, "grid", levels=L).points
    Fm, B = spec.expand(P), exact_uniform_moment(spec)
    best = None
    for c in itertools.combinations(range(len(P)), n):
        X = Fm[list(c)]
        if check_design(X).singular:
            continue
        v = log_det(X) if crit == "D" else i_criterion(X, B)
        if best is None or (v > best if crit == "D" else v < best):
            best = v
    return best


@pytest.mark.parametrize("k,n,L", [(1, 3, 11), (1, 4, 11), (2, 6, 3), (2, 7, 3)])
@pytest.mark.parametrize("crit", ["I", "D"])
def test_exchange_finds_the_exhaustive_optimum_on_small_problems(k, n, L, crit):
    cs = generate_candidates(k, "grid", levels=L)
    fn = i_optimal if crit == "I" else d_optimal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fn(F(k), "quadratic", n, candidates=cs, seed=1)
    got = d.info["I_criterion"] if crit == "I" else d.info["logdet_XtX"]
    assert got == pytest.approx(brute(k, n, L, crit), abs=2e-5)


def test_known_optimum_one_factor_three_runs():
    d = i_optimal(F(1), "quadratic", 3, candidates=generate_candidates(1, "grid", levels=11), seed=0)
    assert sorted(d.coded.ravel()) == [-1.0, 0.0, 1.0]                  # {-1,0,1}, I = 4/5
    assert d.info["I_criterion"] == pytest.approx(0.8, abs=1e-4)
    assert d.info["logdet_XtX"] == pytest.approx(np.log(4.0))


# ---------------------------------------------------------------- comparison properties
def test_i_optimal_vs_d_optimal_vs_ccd_at_equal_runs():
    fs = F(3)
    di, dd, dc = i_optimal(fs, "quadratic", 15, seed=1), d_optimal(fs, "quadratic", 15, seed=1), ccd(fs)
    assert di.n_runs == dd.n_runs == dc.n_runs == 15
    assert di.info["I_criterion"] <= dd.info["I_criterion"] + 1e-9           # I minimises I
    assert dd.info["logdet_XtX"] >= di.info["logdet_XtX"] - 1e-9             # D maximises logdet
    assert di.info["I_criterion"] <= dc.info["I_criterion_uniform"] + 1e-6   # CCD points are candidates
    assert dd.info["logdet_XtX"] >= dc.info["logdet_XtX"] - 1e-9
    assert dd.info["I_criterion"] > di.info["I_criterion"] * 1.05            # the two criteria really differ


def test_reliability_hint_and_default_starts():
    d = i_optimal(F(3), "quadratic", 15, seed=1)
    assert d.info["n_starts"] == 20 and 1 <= d.info["starts_reaching_best"] <= 20
    assert len(d.info["start_values"]) == 20 and d.info["search"].startswith("candidate-exchange")
    assert min(v for v in d.info["start_values"] if v is not None) == pytest.approx(d.info["I_criterion"])


# ---------------------------------------------------------------- structure / forced runs
def test_center_points_include_and_counts():
    inc = np.array([[1.0, 1.0], [-1.0, -1.0]])
    d = i_optimal(F(2), "quadratic", 12, center_points=2, include=inc, seed=3)
    assert d.n_runs == 12 and d.info["n_center"] == 2 and d.info["n_included"] == 2 and d.info["n_free"] == 8
    types = list(d.point_type)
    assert types.count("center") == 2 and types.count("included") == 2 and types.count("optimal") == 8
    assert np.all(d.coded[d.point_type == "center"] == 0) and d.n_replicated_runs >= 1
    assert all(any(np.allclose(r, p) for r in d.coded) for p in inc)
    assert d.info["total_runs"] == 12 and np.abs(d.coded).max() <= 1.0


def test_runs_are_required_validated_and_saturation_flagged():
    with pytest.raises(TypeError):
        i_optimal(F(2), "quadratic")                                  # no hidden default for runs
    with pytest.raises(ad.DesignError, match="not identifiable"):
        i_optimal(F(2), "quadratic", 5)                               # 5 < 6 terms
    with pytest.raises(ad.DesignError):
        i_optimal(F(2), "quadratic", 6, center_points=4, include=np.zeros((3, 2)))   # forced > runs
    with pytest.raises(ad.DesignError):
        i_optimal(F(2), "quadratic", 0)
    with pytest.warns(ad.InferenceWarning, match="saturated"):
        i_optimal(F(2), "quadratic", 6, seed=1)
    with pytest.raises(ad.DesignError):
        i_optimal(F(2), "cubic", 10)
    with pytest.raises(ad.DesignError):
        i_optimal([ad.Factor("A", 0, 1), ad.Factor("A", 0, 2)], "linear", 4)
    with pytest.raises(ad.DesignError):
        i_optimal(F(2), "quadratic", 10, constraints=[({"A": 1}, "<=", 5)], region=Region.unit_cube(2))


def test_candidate_set_that_cannot_support_the_model_is_reported():
    bad = np.array([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])             # collinear points
    with pytest.raises(ad.RankDeficientError, match="candidate set cannot support"):
        i_optimal(F(2), "quadratic", 6, candidates=bad, allow_replicates=True, seed=1)
    with pytest.raises(ad.DesignError, match="only 3 candidate"):
        i_optimal(F(2), "quadratic", 8, candidates=bad, seed=1)       # not enough distinct candidates


def test_replicates_allowed_only_on_request():
    cand = np.array([[-1.0], [0.0], [1.0]])
    d = i_optimal(F(1), "linear", 6, candidates=cand, allow_replicates=True, seed=1)
    assert set(d.coded.ravel()) == {-1.0, 1.0} and d.n_replicated_runs == 4       # linear model: replicate ends
    with pytest.raises(ad.DesignError):
        i_optimal(F(1), "linear", 6, candidates=cand, seed=1)                      # only 3 distinct points
    d2 = i_optimal(F(1), "linear", 3, candidates=cand, seed=1)
    assert d2.n_replicated_runs == 0 and d2.info["allow_replicates"] is False


# ---------------------------------------------------------------- constrained regions & weights
def test_constrained_region_design_is_entirely_feasible():
    fs = F(3)
    cons = [({"A": 1, "B": 1, "C": 1}, "<=", 18)]
    d = i_optimal(fs, "quadratic", 14, constraints=cons, seed=2)
    r = Region.from_factors(fs, cons)
    assert r.contains(d.coded).all()
    act = d.actual_df()
    assert (act.sum(axis=1) <= 18 + 1e-9).all()
    assert d.info["region"]["b"] and 0 < d.info["moment"]["acceptance_rate"] < 1
    assert d.info["candidates"]["acceptance_rate"] <= 1
    dd = d_optimal(fs, "quadratic", 14, constraints=cons, seed=2)
    assert r.contains(dd.coded).all()
    with pytest.raises(ad.InfeasibleRegionError):
        i_optimal(fs, "quadratic", 14, constraints=[({"A": 1, "B": 1}, "<=", -3)], seed=2)


def test_weight_function_changes_the_design_and_is_recorded():
    fs = F(2)
    u = i_optimal(fs, "quadratic", 12, seed=4)
    w = i_optimal(fs, "quadratic", 12, weight=lambda Z: np.exp(-3 * (Z ** 2).sum(axis=1)), seed=4)
    assert u.info["weight"] == "uniform" and w.info["weight"] == "custom"
    assert (w.coded ** 2).sum() < (u.coded ** 2).sum()                 # centre-weighted -> points pulled inward
    assert not np.allclose(np.sort(w.coded, axis=0), np.sort(u.coded, axis=0))


# ---------------------------------------------------------------- numerical-approximation record
def test_integration_approximation_is_recorded_and_checked():
    d = i_optimal(F(3), "quadratic", 15, m=2048, tol=0.01, seed=1)
    mo, ch = d.info["moment"], d.info["moment_check"]
    assert mo["method"] == "sobol-numeric" and mo["m"] >= 2048 and mo["converged"]
    assert ch["converged"] and ch["tol"] == 0.01 and ch["m_values"][0] == 2048
    exact = i_criterion(X_of(d), exact_uniform_moment(ModelSpec("quadratic", 3)))
    assert d.info["I_criterion"] == pytest.approx(exact, rel=0.01)       # numeric vs analytic I
    # on the unconstrained cube the Sobol moments of low-degree polynomials are (nearly) exact, so the
    # criterion check passes immediately; an impossible tolerance on a CONSTRAINED region must be reported
    cons = [({"A": 1, "B": 1}, "<=", 12)]
    with pytest.warns(ad.InferenceWarning, match="not converged"):
        i_optimal(F(2), "quadratic", 9, constraints=cons, m=16, tol=1e-9, seed=1)


# ---------------------------------------------------------------- reproducibility
def test_same_seed_same_design_and_seeds_recorded(tmp_path):
    a = i_optimal(F(3), "quadratic", 15, seed=7, randomize=True)
    b = i_optimal(F(3), "quadratic", 15, seed=7, randomize=True)
    c = i_optimal(F(3), "quadratic", 15, seed=8, randomize=True)
    assert np.array_equal(a.coded, b.coded) and np.array_equal(a.run_order, b.run_order)
    assert a.info["seed"] == 7 and set(a.info["seeds"]) >= {"root", "moment", "candidates", "starts", "randomize"}
    assert not np.array_equal(a.run_order, c.run_order)
    u = d_optimal(F(2), "quadratic", 9)                                 # seed=None is recorded
    v = d_optimal(F(2), "quadratic", 9, seed=u.info["seed"])
    assert np.array_equal(u.coded, v.coded)
    rt = ad.ExperimentDataset.from_csv(a.to_csv(tmp_path / "opt.csv"))
    assert np.allclose(rt.coded, a.coded) and rt.info["design_type"] == "i_optimal"


# ---------------------------------------------------------------- augmentation
def test_augment_single_point_equals_exhaustive_best_addition():
    base = i_optimal(F(2), "quadratic", 9, seed=1)
    spec = ModelSpec("quadratic", 2)
    B = exact_uniform_moment(spec)
    cs = generate_candidates(2, "grid", levels=5, exclude=base.coded)
    for crit in ("I", "D"):
        aug = augment(base, 1, criterion=crit, candidates=cs, seed=1)
        scores = []
        for p in cs.points:
            X = spec.expand(np.vstack([base.coded, p]))
            scores.append(log_det(X) if crit == "D" else -i_criterion(X, B))
        assert np.allclose(aug.points[0], cs.points[int(np.argmax(scores))])
        assert aug.info["n_existing"] == 9 and aug.info["criterion"] == crit


def test_augment_excludes_existing_improves_and_never_simulates(tmp_path):
    base = ccd(F(2))
    aug = augment(base, 3, seed=2)
    assert len(aug.points) == 3 and len({tuple(np.round(p, 9)) for p in aug.points}) == 3
    for p in aug.points:
        assert not any(np.allclose(p, r) for r in base.coded)            # completed points excluded
    assert aug.value_after < aug.value_before                              # I decreases when runs are added
    assert aug.info["candidates"]["n_removed_existing"] >= 1
    n0 = base.n_runs
    ids = aug.to_dataset(base)
    assert list(ids) == [n0 + 1, n0 + 2, n0 + 3] and base.n_runs == n0 + 3
    assert list(base.point_type[-3:]) == ["augment"] * 3 and base.source[-1] == "augment-I"
    base.add_response("y", np.arange(base.n_runs, dtype=float))
    assert aug.actual_df().shape == (3, 2)
    aug.to_csv(tmp_path / "aug.csv")
    assert (tmp_path / "aug.csv").exists()


def test_augment_with_replicates_allowed_and_d_criterion():
    base = ccd(F(2))
    a = augment(base, 2, criterion="D", allow_replicates=True, seed=1)
    assert a.value_after > a.value_before and a.criterion == "D"
    r = augment(base, 3, allow_replicates=True, seed=1)
    assert r.info["allow_replicates"] and r.info["candidates"]["n_removed_existing"] == 0


def test_augment_unidentifiable_start_is_reported_and_validated():
    d = ad.ExperimentDataset(F(2), np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]))
    with pytest.warns(ad.InferenceWarning, match="not identifiable"):
        a = augment(d, 4, seed=1)
    assert a.value_before is None and np.isfinite(a.value_after)
    with pytest.raises(ad.RankDeficientError):
        augment(d, 1, seed=1)                                              # 4 runs < 6 terms
    for bad in (dict(criterion="A"), dict(n_add=0)):
        kw = dict(n_add=1, criterion="I"); kw.update(bad)
        with pytest.raises(ad.DesignError):
            augment(ccd(F(2)), **kw)


def test_augment_ignores_validation_runs_and_respects_constraints():
    base = ccd(F(2))
    base.append_runs(np.array([[0.3, 0.3]]), source="validation")
    a = augment(base, 1, seed=1)
    assert a.info["n_existing"] == 9                                         # validation row not in the design
    fs = F(2)
    d = i_optimal(fs, "quadratic", 8, constraints=[({"A": 1, "B": 1}, "<=", 12)], seed=1)
    aug = augment(d, 2, constraints=[({"A": 1, "B": 1}, "<=", 12)], seed=1)
    assert Region.from_factors(fs, [({"A": 1, "B": 1}, "<=", 12)]).contains(aug.points).all()
