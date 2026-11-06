import warnings

import numpy as np
import pytest

import adore_doe as ad
from adore_doe.design.candidates import generate_candidates
from adore_doe.design.criteria import (check_design, d_efficiency, i_criterion, log_det,
                                       prediction_variance, require_full_rank)
from adore_doe.design.model import ModelSpec, as_model
from adore_doe.design.moment import (IntegrationDomain, exact_uniform_moment,
                                     i_criterion_convergence)
from adore_doe.design.region import Region


def F(k, lo=0.0, hi=10.0):
    return [ad.Factor(chr(65 + i), lo, hi) for i in range(k)]


# ------------------------------------------------------------------ ModelSpec
@pytest.mark.parametrize("kind,k,p", [("linear", 3, 4), ("interaction", 3, 7), ("quadratic", 3, 10),
                                      ("quadratic", 4, 15), ("quadratic", 1, 3), ("quadratic", 5, 21)])
def test_model_term_counts(kind, k, p):
    assert ModelSpec(kind, k).n_terms == p
    if kind == "quadratic":
        assert p == (k + 1) * (k + 2) // 2


def test_model_columns_and_names():
    s = ModelSpec("quadratic", 2)
    assert s.names(["A", "B"]) == ["Intercept", "A", "B", "A*B", "A^2", "B^2"]
    Z = np.array([[0.5, -2.0], [1.0, 1.0]])
    X = s.expand(Z)
    assert np.allclose(X[0], [1, 0.5, -2.0, -1.0, 0.25, 4.0])
    assert np.allclose(X[1], [1, 1, 1, 1, 1, 1])
    assert np.allclose(ModelSpec("linear", 2).expand(Z), X[:, :3])


def test_model_validation_and_roundtrip():
    for bad in (("cubic", 2), ("linear", 0)):
        with pytest.raises(ad.DesignError):
            ModelSpec(*bad)
    s = ModelSpec("interaction", 3)
    assert ModelSpec.from_dict(s.to_dict()) == s
    with pytest.raises(ad.DesignError):
        s.expand(np.zeros((2, 2)))
    with pytest.raises(ad.DesignError):
        as_model(ModelSpec("linear", 2), 3)
    with pytest.raises(ad.DesignError):
        s.names(["A"])


# ------------------------------------------------------------------ Region
def test_region_from_factors_converts_actual_constraints_to_coded():
    fs = [ad.Factor("W", 10, 50), ad.Factor("S", 5, 30)]
    r = Region.from_factors(fs, [({"W": 1, "S": 1}, "<=", 60)])
    # check on actual values: (W=50,S=30) violates, (W=10,S=5) satisfies, boundary W+S=60 is feasible
    z = lambda w, s: np.array([[fs[0].to_coded(w), fs[1].to_coded(s)]])
    assert not r.contains(z(50, 30))[0] and r.contains(z(10, 5))[0]
    assert r.contains(z(40, 20))[0] and r.contains(z(35, 25))[0] and not r.contains(z(35, 26))[0]
    r2 = Region.from_factors(fs, [({"W": 1}, ">=", 30)])
    assert r2.contains(z(30, 5))[0] and not r2.contains(z(29, 5))[0]


def test_region_validation():
    fs = F(2)
    for c in ([({"A": 1}, "<", 3)], [({"Z": 1}, "<=", 3)], [({"A": 0}, "<=", 3)], ["bad"]):
        with pytest.raises(ad.DesignError):
            Region.from_factors(fs, c)
    with pytest.raises(ad.DesignError):
        Region.from_factors([ad.Factor("A", "a", "b", factor_type="categorical")], [])
    with pytest.raises(ad.DesignError):
        Region(2, np.zeros((1, 3)), np.zeros(1))


def test_region_samples_are_feasible_and_reproducible():
    fs = F(3)
    r = Region.from_factors(fs, [({"A": 1, "B": 1, "C": 1}, "<=", 12)])
    P, rate = r.sample(500, seed=3)
    assert P.shape == (500, 3) and r.contains(P).all() and 0 < rate < 1
    P2, rate2 = r.sample(500, seed=3)
    P3, _ = r.sample(500, seed=4)
    assert np.array_equal(P, P2) and rate == rate2 and not np.array_equal(P, P3)
    U, ru = Region.unit_cube(3).sample(100, seed=1)
    assert U.shape == (100, 3) and np.abs(U).max() <= 1 and ru == 1.0
    for method in ("sobol", "lhs", "uniform"):
        Q, _ = r.sample(200, seed=1, method=method)
        assert r.contains(Q).all()


def test_infeasible_and_tiny_regions_are_reported():
    fs = F(2)
    with pytest.raises(ad.InfeasibleRegionError):
        Region.from_factors(fs, [({"A": 1, "B": 1}, "<=", -5)]).sample(10, seed=1)   # empty
    tiny = Region.from_factors(fs, [({"A": 1, "B": 1}, "<=", 0.05)])               # ~1e-4 of cube
    with pytest.raises(ad.InfeasibleRegionError):
        tiny.sample(50, seed=1)
    assert issubclass(ad.InfeasibleRegionError, ad.DesignError)


# ------------------------------------------------------------------ criteria
def test_hand_computed_criteria_linear_one_factor():
    # f=[1,x], X=[-1,1]: X'X=diag(2,2); B=diag(1,1/3)  ->  I = 1/2 + (1/3)/2 = 2/3 ; det = 4
    s = ModelSpec("linear", 1)
    X = s.expand(np.array([[-1.0], [1.0]]))
    B = exact_uniform_moment(s)
    assert np.allclose(B, np.diag([1, 1 / 3]))
    assert i_criterion(X, B) == pytest.approx(2 / 3)
    assert np.exp(log_det(X)) == pytest.approx(4.0)
    assert d_efficiency(X) == pytest.approx(np.sqrt(4.0) / 2)            # det^(1/p)/n = 2/2


def test_hand_computed_criteria_quadratic_three_points():
    # x=[-1,0,1], f=[1,x,x^2]: X'X=[[3,0,2],[0,2,0],[2,0,2]], det=4;
    # B=[[1,0,1/3],[0,1/3,0],[1/3,0,1/5]]; tr[(X'X)^-1 B] = 1 - 2/3 + 3/10 + 1/6 = 4/5
    s = ModelSpec("quadratic", 1)
    X = s.expand(np.array([[-1.0], [0.0], [1.0]]))
    B = exact_uniform_moment(s)
    assert np.allclose(B, [[1, 0, 1 / 3], [0, 1 / 3, 0], [1 / 3, 0, 1 / 5]])
    assert i_criterion(X, B) == pytest.approx(0.8)
    assert np.exp(log_det(X)) == pytest.approx(4.0)
    # the average prediction variance equals tr[(X'X)^-1 B]: integrate numerically on a fine grid
    g = np.linspace(-1, 1, 200001)[:, None]
    pv = prediction_variance(X, s.expand(g))
    assert pv.mean() == pytest.approx(0.8, abs=1e-4)


def test_singular_designs_raise_instead_of_pseudo_inverse():
    s = ModelSpec("quadratic", 1)
    X2 = s.expand(np.array([[-1.0], [1.0]]))                              # 2 runs < 3 terms
    B = exact_uniform_moment(s)
    with pytest.raises(ad.RankDeficientError, match="not identifiable"):
        i_criterion(X2, B)
    with pytest.raises(ad.RankDeficientError):
        log_det(X2)
    with pytest.raises(ad.RankDeficientError):
        prediction_variance(X2, X2)
    Xrep = s.expand(np.array([[0.5]] * 5))                                # replicates of one point
    c = check_design(Xrep)
    assert c.singular and c.rank == 1 and c.p == 3 and c.n == 5
    with pytest.raises(ad.RankDeficientError, match="rank 1 < 3"):
        require_full_rank(Xrep)
    with pytest.raises(ad.RankDeficientError, match="2 runs < 3 model terms"):
        require_full_rank(X2)


def test_ill_conditioned_design_warns():
    s = ModelSpec("linear", 1)
    X = s.expand(np.array([[0.0], [1e-9], [2e-9]]))
    with pytest.warns(ad.InferenceWarning, match="ill-conditioned"):
        require_full_rank(X)


# ------------------------------------------------------------------ moment matrix / integration
@pytest.mark.parametrize("kind,k", [("linear", 2), ("quadratic", 2), ("quadratic", 3)])
def test_numeric_moment_matches_analytic(kind, k):
    s = ModelSpec(kind, k)
    d = IntegrationDomain.for_cube(k, seed=1)
    r = d.matrix(s)
    assert r.method == "sobol-numeric" and r.converged and r.rel_changes[-1] < 0.01
    assert np.abs(r.B - exact_uniform_moment(s)).max() < 5e-3
    assert d.matrix(s, exact=True).method == "analytic"


def test_weights_change_the_moment_matrix_and_are_validated():
    s = ModelSpec("quadratic", 1)
    centre = IntegrationDomain.for_cube(1, seed=2, weight=lambda Z: np.exp(-4 * Z[:, 0] ** 2))
    u = IntegrationDomain.for_cube(1, seed=2)
    Bc, Bu = centre.matrix(s).B, u.matrix(s).B
    assert Bc[1, 1] < Bu[1, 1]                                            # E[x^2] smaller when centre-weighted
    with pytest.raises(ad.DesignError):
        IntegrationDomain.for_cube(1, weight=lambda Z: -np.ones(len(Z))).matrix(s)
    with pytest.raises(ad.DesignError):
        IntegrationDomain.for_cube(1, weight=lambda Z: np.zeros(len(Z))).matrix(s)
    with pytest.raises(ad.DesignError):
        centre.matrix(s, exact=True)                                      # exact only for uniform cube
    with pytest.raises(ad.DesignError):
        IntegrationDomain.for_cube(1, m=4)
    with pytest.raises(ad.DesignError):
        IntegrationDomain.for_cube(2).matrix(s)


def test_integration_points_are_feasible_prefix_consistent_and_seeded():
    fs = F(3)
    r = Region.from_factors(fs, [({"A": 1, "B": 1}, "<=", 8)])
    d = IntegrationDomain(r, m=512, seed=5)
    P1, w1 = d.points(300)
    P2, _ = d.points(1000)
    assert r.contains(P2).all() and np.array_equal(P1, P2[:300]) and np.all(w1 == 1)
    d2 = IntegrationDomain(r, m=512, seed=5)
    assert np.array_equal(d2.points(300)[0], P1)
    res = d.matrix(ModelSpec("quadratic", 3))
    assert 0 < res.acceptance_rate < 1 and np.isfinite(res.B).all()


def test_i_criterion_convergence_check_is_recorded_and_passes():
    s = ModelSpec("quadratic", 2)
    g = np.array([[a, b] for a in (-1, 0, 1) for b in (-1, 0, 1)], float)        # 3^2 factorial
    X = s.expand(g)
    d = IntegrationDomain.for_cube(2, seed=1, m=1024, tol=0.01)
    chk = i_criterion_convergence(X, s, d)
    assert chk["converged"] and chk["rel_changes"][-1] < 0.01
    assert chk["m_values"][0] == 1024 and len(chk["m_values"]) >= 2
    assert all(b == 2 * a for a, b in zip(chk["m_values"], chk["m_values"][1:]))        # m doubles
    exact = i_criterion(X, exact_uniform_moment(s))
    assert chk["I_values"][-1] == pytest.approx(exact, rel=0.01)                   # approximation error check
    tight = i_criterion_convergence(X, s, IntegrationDomain.for_cube(2, seed=1, m=16, tol=1e-9, m_max=64))
    assert not tight["converged"]                                                  # impossible tol -> reported


# ------------------------------------------------------------------ candidates
def test_grid_candidates_counts_vertices_and_reproducibility():
    c = generate_candidates(3, method="grid", levels=3)
    assert len(c) == 27 and c.levels == 3
    assert {tuple(p) for p in c.points} >= {(1, 1, 1), (-1, -1, -1), (0, 0, 0), (1, 0, -1)}
    auto = generate_candidates(3)                       # auto -> 5 levels for k<=4
    assert len(auto) == 125 and auto.method == "auto"
    assert len(generate_candidates(6, method="auto")) == 729            # 3 levels when 5^6 is too big
    a = generate_candidates(3, method="sobol", n=300, seed=7)
    b = generate_candidates(3, method="sobol", n=300, seed=7)
    assert np.array_equal(a.points, b.points) and len(a) == 300 and a.info()["seed"] == 7


def test_constrained_candidates_are_feasible_and_exclusions_work():
    fs = F(2)
    r = Region.from_factors(fs, [({"A": 1, "B": 1}, "<=", 12)])
    c = generate_candidates(r, method="auto", n=400, seed=1)
    assert r.contains(c.points).all() and 0 < c.acceptance_rate <= 1
    g = generate_candidates(r, method="grid", levels=5)
    assert r.contains(g.points).all() and len(g) < 25
    done = np.array([[0.0, 0.0], [1.0, -1.0]])
    g2 = generate_candidates(r, method="grid", levels=5, exclude=done)
    assert g2.n_removed_existing == 2 and len(g2) == len(g) - 2
    assert not any(np.allclose(p, [0, 0]) for p in g2.points)


def test_user_candidates_validation():
    r = Region.from_factors(F(2), [({"A": 1, "B": 1}, "<=", 12)])
    ok = generate_candidates(r, method="user", points=[[0, 0], [-1, -1]])
    assert len(ok) == 2
    with pytest.raises(ad.DesignError):
        generate_candidates(r, method="user", points=[[1, 1]])           # infeasible
    with pytest.raises(ad.DesignError):
        generate_candidates(r, method="user")
    with pytest.raises(ad.DesignError):
        generate_candidates(r, method="user", points=[[0, 0, 0]])
    with pytest.raises(ad.DesignError):
        generate_candidates(2, method="nope")
    with pytest.raises(ad.DesignError):
        generate_candidates(2, method="grid", levels=1)
    with pytest.raises(ad.DesignError):
        generate_candidates(2, method="user", points=[[0, 0]], exclude=[[0, 0]])   # empty after exclusion
