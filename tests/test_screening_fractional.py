import numpy as np
import pytest

import adore_doe as ad
from adore_doe.screening import alias_table, fractional_factorial, main_effects, resolution_of


def F(k, lo=-1, hi=1):
    return [ad.Factor(chr(65 + i), lo, hi) for i in range(k)]


# ---------------------------------------------------------------- known design matrices
def test_full_factorial_2_3_standard_order():
    d = fractional_factorial(F(3))
    expect = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                       [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], float)
    assert np.array_equal(d.coded, expect)
    assert d.info["design_type"] == "full_factorial" and d.info["resolution"] is None
    assert d.dummy_names == ["AB", "AC", "BC", "ABC"]


def test_half_fraction_2_3_1_known_matrix():
    d = fractional_factorial(F(3), generators=["C=AB"])
    assert np.array_equal(d.coded, np.array([[-1, -1, 1], [1, -1, -1], [-1, 1, -1], [1, 1, 1]], float))
    assert d.info["defining_relation"] == ["I", "ABC"] and d.info["resolution"] == 3


def test_fraction_2_4_1_and_aliases():
    d = fractional_factorial(F(4), generators=["D=ABC"])
    assert d.n_runs == 8 and d.info["resolution"] == 4
    at = alias_table(4, ["D=ABC"], max_order=2)
    assert at["A"] == ["BCD"] and at["AB"] == ["CD"] and at["D"] == ["ABC"]


@pytest.mark.parametrize("k,p,res", [(3, 1, 3), (4, 1, 4), (5, 1, 5), (6, 1, 6), (7, 1, 7),
                                     (5, 2, 3), (6, 2, 4), (6, 3, 3), (7, 2, 4), (7, 3, 4),
                                     (7, 4, 3), (8, 2, 5), (8, 4, 4)])
def test_computed_resolution_matches_literature(k, p, res):
    d = fractional_factorial(F(k), p=p)
    assert d.info["resolution"] == res
    assert d.n_runs == 2 ** (k - p)
    assert resolution_of(k, d.info["generators"]) == res


@pytest.mark.parametrize("k,p", [(3, 0), (4, 0), (5, 1), (6, 2), (7, 4), (8, 4)])
def test_orthogonality_all_contrasts(k, p):
    d = fractional_factorial(F(k), p=p)
    n = d.n_runs
    A = np.column_stack([d.coded, d.dummy_coded]) if len(d.dummy_names) else d.coded
    assert A.shape[1] == n - 1                           # main effects + interaction contrasts
    assert np.allclose(A.T @ A, n * np.eye(n - 1))       # all distinct contrasts orthogonal
    assert np.allclose(A.sum(axis=0), 0)


def test_center_points_numeric_and_categorical():
    d = fractional_factorial(F(4), p=1, center_points=3)
    assert d.n_runs == 11 and d.center_points == 3 and np.all(d.coded[-3:] == 0)
    assert np.all(d.dummy_coded[-3:] == 0)
    cats = [ad.Factor("A", "a", "b", factor_type="categorical"), ad.Factor("B", 0, 1)]
    with pytest.raises(ad.DesignError):
        fractional_factorial(cats, center_points=1)


def test_input_validation():
    for kw in (dict(generators=["C=AB", "D=AC", "E=BC", "F=ABC"]),   # runs < k+1 impossible
               dict(generators=["C=AX"]), dict(generators=["D=AB"]), dict(generators=["C=AA"]),
               dict(generators=["nonsense"]), dict(p=9), dict(p=-1)):
        with pytest.raises(ad.DesignError):
            fractional_factorial(F(3), **kw)
    with pytest.raises(ad.DesignError):
        fractional_factorial(F(9), p=1)                    # no tabulated generators
    with pytest.raises(ad.DesignError):
        fractional_factorial([ad.Factor("A", 0, 1), ad.Factor("A", 0, 2)])
    with pytest.raises(ad.DesignError):
        fractional_factorial(F(1))


def test_randomised_run_order_reproducible():
    a = fractional_factorial(F(4), p=1, randomize=True, seed=5)
    b = fractional_factorial(F(4), p=1, randomize=True, seed=5)
    assert np.array_equal(a.run_order, b.run_order) and a.info["seed"] == 5


# ---------------------------------------------------------------- textbook example
def test_montgomery_2_4_filtration_rate_effects():
    """Montgomery, Design & Analysis of Experiments, 2^4 filtration-rate example.
    Data in standard order; known effects: A=21.625, B=3.125, C=9.875, D=14.625,
    AC=-18.125, AD=16.625 (data/effects as recalled from the book - please re-check against
    your copy; the effects are reproduced exactly)."""
    y = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96]
    d = fractional_factorial(F(4))
    d.add_response("rate", y)
    e = main_effects(d, "rate")
    eff = dict(zip(e.table["Factor"], e.table["Effect"]))
    assert eff["A"] == pytest.approx(21.625) and eff["B"] == pytest.approx(3.125)
    assert eff["C"] == pytest.approx(9.875) and eff["D"] == pytest.approx(14.625)
    inter = dict(zip(e.dummy_table["Factor"], e.dummy_table["Effect"]))
    assert inter["AC"] == pytest.approx(-18.125) and inter["AD"] == pytest.approx(16.625)
    assert inter["AB"] == pytest.approx(0.125) and inter["BC"] == pytest.approx(2.375)
    # Daniel / half-normal plot uses all 15 contrasts
    assert e.lenth is not None and e.lenth["m"] == 15
    ax = e.plot_half_normal(label="all")
    assert len(ax.texts) == 15
    assert any("Interaction contrast" in t.get_text() for t in ax.get_legend().get_texts())


def test_main_effects_with_aliasing_only_reports_main_effects():
    d = fractional_factorial(F(5), p=1, center_points=2)
    X = d.coded
    d.add_response("y", 5 + 2 * X[:, 0] + 0.01 * np.random.default_rng(1).standard_normal(len(X)))
    e = main_effects(d, "y")
    assert e.table.set_index("Factor").loc["A", "Effect"] == pytest.approx(4.0, abs=0.05)
    assert e.df_error == 18 - 1 - 5 - 1          # 16 factorial + 2 center, curvature term
