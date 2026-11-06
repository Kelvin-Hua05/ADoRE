"""End-to-end reproducible regression tests."""
import numpy as np
import pytest
from scipy import stats

import adore_doe as yp


def run(seed, true, k=8, runs=12, centers=4, sigma=0.5, **kw):
    fs = [yp.Factor(f"F{i+1}", 0, 10) for i in range(k)]
    d = yp.screening.plackett_burman(fs, runs=runs, center_points=centers, randomize=True, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    X = d.coded
    y = 10 + np.zeros(len(X))
    for n, b in true.items():
        y = y + b * X[:, d.factor_names.index(n)]
    d.add_response("Y", y + sigma * rng.standard_normal(len(y)))
    e = yp.screening.main_effects(d, "Y")
    return d, e, yp.screening.screen(e, **kw)


def test_true_factors_always_identified():
    for seed in range(40):
        _, e, s = run(seed, {"F2": 4.0, "F7": -3.0}, criterion="bonferroni")
        assert {"F2", "F7"} <= set(s.selected_factors), seed
        assert e.table.set_index("Factor").loc["F7", "Effect"] < 0 < \
            e.table.set_index("Factor").loc["F2", "Effect"]


def test_null_data_not_arbitrarily_flagged():
    """No active factor, 8 inert factors.  Measured behaviour (seeded, 300 datasets):
    Bonferroni screening flags >=1 factor in ~6 % of datasets (nominal 5 %), while the plain
    t-limit rule is liberal (~40 %: 1-(0.95)^8 = 34 % even without stepwise selection)."""
    n = 300
    bonf = sum(bool(run(s, {}, criterion="bonferroni")[2].selected_factors) for s in range(n)) / n
    tlim = sum(bool(run(s, {}, criterion="t_limit")[2].selected_factors) for s in range(n)) / n
    assert bonf <= 0.10, bonf
    assert 0.25 <= tlim <= 0.60, tlim
    for s in range(20):       # Bonferroni list is always a subset of the t-limit list
        strict = run(s, {}, criterion="bonferroni")[2].selected_factors
        loose = run(s, {}, criterion="t_limit")[2].selected_factors
        assert set(strict) <= set(loose)


def test_inert_factors_rarely_beyond_bonferroni_when_signal_present():
    n = 300
    extra = [set(run(s, {"F1": 5.0}, criterion="bonferroni")[2].selected_factors) - {"F1"}
             for s in range(n)]
    assert sum(len(x) for x in extra) / n <= 0.15
    assert sum(bool(x) for x in extra) / n <= 0.10


def test_results_match_direct_scipy_computation():
    d, e, s = run(5, {"F1": 3.0, "F4": -2.0})
    y = d.get_response("Y")
    X = d.coded
    Z = np.column_stack([np.ones(len(y)), X, (d.point_type == "center").astype(float)])
    from scipy import linalg
    beta = linalg.lstsq(Z, y)[0]
    df = len(y) - Z.shape[1]
    s2 = ((y - Z @ beta) ** 2).sum() / df
    se = np.sqrt(np.diag(s2 * linalg.inv(Z.T @ Z)))
    t = beta / se
    k = X.shape[1]
    assert e.df_error == df
    assert np.allclose(e.table["t"], t[1:1 + k])
    assert np.allclose(e.table["p"], 2 * stats.t.sf(np.abs(t[1:1 + k]), df))
    assert np.allclose(e.table["Effect"], 2 * beta[1:1 + k])
    # the final model of the screening equals a direct fit on the selected terms
    cols = [d.factor_names.index(n) for n in s.selected_factors]
    Zs = np.column_stack([np.ones(len(y)), X[:, cols], (d.point_type == "center").astype(float)])
    bs = linalg.lstsq(Zs, y)[0]
    dfs = len(y) - Zs.shape[1]
    ses = np.sqrt(((y - Zs @ bs) ** 2).sum() / dfs * np.diag(linalg.inv(Zs.T @ Zs)))
    got = s.final_table.set_index("Factor").loc[s.selected_factors]
    assert s.df_error == dfs
    assert np.allclose(got["t"], (bs / ses)[1:1 + len(cols)])
    assert np.allclose(got["Effect"], 2 * bs[1:1 + len(cols)])


def test_same_seed_same_everything():
    a = run(9, {"F3": 2.0})
    b = run(9, {"F3": 2.0})
    assert np.array_equal(a[0].run_order, b[0].run_order)
    assert a[1].table.equals(b[1].table)
    assert a[2].selected_factors == b[2].selected_factors
    assert a[2].history().equals(b[2].history())


def test_pdf_page1_configuration_reproduces_reference_limits():
    """19 factors, PB-24 + 7 center points -> error df = 31-1-19-1 = 10, as in the PDF figure:
    t-limit 2.22814 and Bonferroni limit 3.97276 at alpha = 0.05."""
    fs = [yp.Factor(f"X{i+1}", -1, 1) for i in range(19)]
    d = yp.screening.plackett_burman(fs, runs=24, center_points=7, randomize=True, seed=0)
    rng = np.random.default_rng(0)
    X = d.coded
    y = 5 + 3.0 * X[:, 1] - 2.0 * X[:, 9] + 0.3 * rng.standard_normal(len(X))
    d.add_response("Vth", y)
    e = yp.screening.main_effects(d, "Vth", alpha=0.05)
    assert e.df_error == 10 and e.n_tests == 19
    assert e.t_limit == pytest.approx(2.22814, abs=5e-6)
    assert e.bonferroni_limit == pytest.approx(3.97276, abs=5e-6)
    s = yp.screening.screen(e, alpha=0.05, criterion="bonferroni")
    assert {"X2", "X10"} <= set(s.selected_factors)
    ax = e.plot_pareto()                          # full model: all 19 effects, df = 10
    assert len(ax.patches) == 19
    ys = [float(l.get_ydata()[0]) for l in ax.lines if len(set(l.get_ydata())) == 1]
    assert any(np.isclose(v, 2.22814, atol=5e-6) for v in ys)
    assert any(np.isclose(v, 3.97276, atol=5e-6) for v in ys)
    assert ax.get_ylabel() == "|t-value of effect|"


def test_export_selected(tmp_path):
    d, e, s = run(2, {"F1": 5.0, "F2": -4.0}, criterion="bonferroni")
    pj = s.export_selected(tmp_path / "sel.json")
    import json
    payload = json.loads(pj.read_text())
    assert [f["name"] for f in payload["selected_factors"]] == s.selected_factors
    assert payload["selected_factors"][0]["low"] == 0.0
    pc = s.export_selected(tmp_path / "sel.csv")
    import pandas as pd
    assert list(pd.read_csv(pc)["Factor"]) == s.selected_factors
    assert "Screening" in s.summary() and "selected factors" in s.summary()


def test_readme_style_api_smoke():
    fs = [yp.Factor("A", low=0, high=10), yp.Factor("B", low=20, high=50),
          yp.Factor("C", low=1, high=5)]
    design = yp.screening.plackett_burman(factors=fs, runs=8, center_points=3)
    design.add_response("Yield", values=[10, 12, 11, 15, 30, 28, 33, 35, 20, 21, 19])
    effects = yp.screening.main_effects(design, response="Yield")
    screening = yp.screening.screen(effects, alpha=0.05)
    screening.plot_pareto()
    screening.plot_half_normal()
    assert isinstance(screening.selected_factors, list)
