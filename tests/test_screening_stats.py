import numpy as np
import pytest
from scipy import stats

import adore_doe as yp
from adore_doe.screening import halfnormal, limits as L
from adore_doe.screening.ols import fit_ols
from adore_doe.screening.selection import model_state, partial_stats, stepwise_selection


def pb(k, runs, centers=0, seed=0):
    fs = [yp.Factor(f"F{i+1}", 0, 10) for i in range(k)]
    return yp.screening.plackett_burman(fs, runs=runs, center_points=centers,
                                     randomize=True, seed=seed)


def synth(d, true, sigma=0.3, seed=0, intercept=20.0, name="Y"):
    """y = intercept + sum(true_j * x_j) + noise;  true_j are COEFFICIENTS per coded unit."""
    rng = np.random.default_rng(seed)
    X = d.coded
    y = intercept + sum(b * X[:, d.factor_names.index(n)] for n, b in true.items())
    y = y + sigma * rng.standard_normal(len(y))
    d.add_response(name, y)
    return y


# ------------------------------------------------------------------ limits
def test_limits_match_pdf_page1():
    # PDF p.1: df=10, 19 effects, alpha=0.05 -> t-limit 2.22814, Bonferroni 3.97276
    assert L.t_limit(0.05, 10) == pytest.approx(2.22814, abs=5e-6)
    assert L.bonferroni_limit(0.05, 10, 19) == pytest.approx(3.97276, abs=5e-6)


def test_limit_formulas_and_relations():
    for df in (1, 5, 10, 40):
        assert L.t_limit(0.05, df) == pytest.approx(stats.t.ppf(0.975, df))
        for m in (1, 5, 19):
            b = L.bonferroni_limit(0.05, df, m)
            assert b == pytest.approx(stats.t.ppf(1 - 0.05 / (2 * m), df))
            assert b >= L.t_limit(0.05, df) - 1e-12
            # union bound: m two-sided tests at level alpha/m -> FWER <= alpha (exact tail check)
            assert m * 2 * stats.t.sf(b, df) == pytest.approx(0.05)
    assert L.bonferroni_limit(0.05, 7, 1) == pytest.approx(L.t_limit(0.05, 7))
    assert L.t_limit(0.05, 3) > L.t_limit(0.05, 30)           # fewer df -> wider
    assert L.t_limit(0.01, 10) > L.t_limit(0.05, 10)


def test_limits_reject_bad_input():
    with pytest.raises(yp.InferenceError):
        L.t_limit(0.05, 0)
    with pytest.raises(yp.InferenceError):
        L.bonferroni_limit(0.05, 0, 5)
    with pytest.raises(ValueError):
        L.t_limit(1.5, 10)
    with pytest.raises(ValueError):
        L.bonferroni_limit(0.05, 5, 0)


def test_classify_and_exceeds():
    tl, bl = 2.2, 4.0
    assert L.classify([5, -3, 1, np.nan], tl, bl) == ["Bonferroni", "t-limit", "not significant", "n/a"]
    assert list(L.exceeds([-3, 2, np.nan], 2.5)) == [True, False, False]


def test_lenth_known_values():
    c = np.array([0.1, -0.2, 0.15, 0.05, -0.1, 0.12, 0.08, 5.0, -4.0, 0.0, 0.2, -0.15, 0.1, 0.07, 0.11])
    le = L.lenth(c)
    a = np.abs(c)
    s0 = 1.5 * np.median(a)
    pse = 1.5 * np.median(a[a < 2.5 * s0])
    assert le["pse"] == pytest.approx(pse) and le["s0"] == pytest.approx(s0)
    d = len(c) / 3
    assert le["ME"] == pytest.approx(stats.t.ppf(0.975, d) * pse)
    gamma = (1 + 0.95 ** (1 / len(c))) / 2
    assert le["SME"] == pytest.approx(stats.t.ppf(gamma, d) * pse)
    assert le["SME"] > le["ME"]
    assert pse < 0.2                              # the two huge effects are excluded from PSE
    with pytest.raises(yp.InferenceError):
        L.lenth([1.0, 2.0])


# ------------------------------------------------------------------ main effects: definitions
def test_effect_is_mean_difference_and_twice_coefficient():
    d = pb(5, 12, seed=3)
    y = synth(d, {"F1": 4.0, "F3": -2.5}, seed=1)
    e = yp.screening.main_effects(d, "Y")
    X = d.coded
    for j, n in enumerate(d.factor_names):
        md = y[X[:, j] > 0].mean() - y[X[:, j] < 0].mean()
        row = e.table[e.table.Factor == n].iloc[0]
        assert row.Effect == pytest.approx(md)
        assert row.Mean_Diff == pytest.approx(md)
        assert row.Coefficient == pytest.approx(md / 2)       # effect = 2 * coefficient
        assert row.Std_Effect == pytest.approx(row.t)         # standardized = Effect / SE
        assert row.Effect / row.StdErr == pytest.approx(row.t)


def test_known_main_effects_signs_and_sizes():
    d = pb(7, 12, centers=4, seed=5)
    synth(d, {"F2": 3.0, "F5": -1.5}, sigma=1e-3, seed=2)
    e = yp.screening.main_effects(d, "Y")
    t = e.table.set_index("Factor")
    assert t.loc["F2", "Effect"] == pytest.approx(6.0, abs=0.01)
    assert t.loc["F5", "Effect"] == pytest.approx(-3.0, abs=0.01)
    assert t.loc["F2", "Sign"] == "+" and t.loc["F5", "Sign"] == "-"
    for n in ("F1", "F3", "F4", "F6", "F7"):
        assert abs(t.loc[n, "Effect"]) < 0.01
    assert t.loc["F2", "Significance"] == "Bonferroni"


def test_se_t_p_ci_against_independent_formulas():
    """Independent check without regression code: for a PB design with dummy columns and no
    center points, Var(effect) is estimated by the mean squared dummy contrast."""
    d = pb(5, 12, seed=11)                                    # 6 dummy columns -> df=6
    y = synth(d, {"F1": 3.0, "F4": -2.0}, sigma=0.5, seed=4)
    e = yp.screening.main_effects(d, "Y")
    Xd = d.dummy_coded
    contrasts = Xd.T @ y / 6.0                                # mean(high) - mean(low) of dummies
    se_ind = np.sqrt(np.mean(contrasts ** 2))
    assert e.df_error == 6
    assert np.allclose(e.table["StdErr"], se_ind)
    t_ind = e.table["Effect"] / se_ind
    assert np.allclose(e.table["t"], t_ind)
    assert np.allclose(e.table["p"], 2 * stats.t.sf(np.abs(t_ind), 6))
    tc = stats.t.ppf(0.975, 6)
    assert np.allclose(e.table["CI_low"], e.table["Effect"] - tc * se_ind)
    assert np.allclose(e.table["CI_high"], e.table["Effect"] + tc * se_ind)
    assert e.sigma == pytest.approx(se_ind * np.sqrt(12) / 2)  # SE(effect)=2 sigma/sqrt(N)


def test_ols_matches_scipy_lstsq_and_optional_statsmodels():
    d = pb(6, 12, centers=3, seed=2)
    y = synth(d, {"F1": 2.0, "F2": -1.0, "F6": 0.7}, sigma=0.4, seed=9)
    e = yp.screening.main_effects(d, "Y")
    center = (d.point_type == "center").astype(float)
    Z = np.column_stack([np.ones(len(y)), d.coded, center])
    from scipy import linalg
    beta, rss, rank, _ = linalg.lstsq(Z, y)
    df = len(y) - rank
    s2 = float(((y - Z @ beta) ** 2).sum() / df)
    cov = s2 * linalg.inv(Z.T @ Z)
    k = 6
    assert e.df_error == df
    assert np.allclose(e.table["Coefficient"], beta[1:1 + k])
    assert np.allclose(e.table["StdErr"], 2 * np.sqrt(np.diag(cov))[1:1 + k])
    assert e.curvature["estimate"] == pytest.approx(beta[-1])
    sm = pytest.importorskip("statsmodels.api")
    res = sm.OLS(y, Z).fit()
    assert np.allclose(e.table["t"], res.tvalues[1:1 + k])
    assert np.allclose(e.table["p"], res.pvalues[1:1 + k])
    assert np.allclose(e.table["CI_low"], 2 * res.conf_int(0.05)[1:1 + k, 0])


def test_curvature_term_protects_error_estimate():
    d = pb(4, 8, centers=4, seed=1)
    y = synth(d, {"F1": 2.0}, sigma=0.2, seed=3)
    is_c = d.point_type == "center"
    d.add_response("Yc", y + 6.0 * is_c)                      # strong curvature at the center
    e1 = yp.screening.main_effects(d, "Y")
    e2 = yp.screening.main_effects(d, "Yc")
    assert np.allclose(e1.table["Effect"], e2.table["Effect"], atol=1e-9)
    assert e2.sigma == pytest.approx(e1.sigma, rel=1e-6)      # curvature absorbed by its own term
    assert e2.curvature["p"] < 1e-3 < e1.curvature["p"]
    assert e2.curvature["pure_error_df"] == 3
    e3 = yp.screening.main_effects(d, "Yc", curvature=False)   # not recommended: inflates sigma
    assert e3.sigma > 5 * e2.sigma


def test_error_df_accounting():
    assert yp.screening.main_effects(_filled(pb(5, 12)), "Y").df_error == 6       # 12-1-5
    assert yp.screening.main_effects(_filled(pb(5, 12, centers=3)), "Y").df_error == 8   # 15-1-5-1
    assert yp.screening.main_effects(_filled(pb(11, 12, centers=3)), "Y").df_error == 2  # pure error 3-1


def _filled(d, true=None):
    synth(d, true or {"F1": 2.0}, seed=1)
    return d


# ------------------------------------------------------------------ insufficient df / rank
def test_saturated_design_gives_no_pvalues():
    d = _filled(pb(7, 8))
    with pytest.warns(yp.InferenceWarning, match="NOT computed"):
        e = yp.screening.main_effects(d, "Y")
    assert e.df_error == 0 and not e.inference_available
    for c in ("StdErr", "t", "p", "CI_low", "CI_high"):
        assert e.table[c].isna().all()
    assert e.t_limit is None and e.bonferroni_limit is None
    assert set(e.table["Significance"]) == {"n/a (df=0)"}
    assert e.table["Effect"].notna().all()                    # effects themselves are estimable
    assert e.lenth is not None
    with pytest.raises(yp.InferenceError):
        yp.screening.screen(e, method="ols")
    with pytest.raises(yp.InferenceError):
        e.plot_pareto(scale="t")


def test_saturated_with_center_points_recovers_pure_error_df():
    d = _filled(pb(7, 8, centers=4))
    e = yp.screening.main_effects(d, "Y")
    assert e.df_error == 3 and e.inference_available            # = n_center - 1


def test_rank_deficient_design_raises_with_names():
    d = pb(3, 4)
    C = d.coded
    X = np.column_stack([C[:, 0], C[:, 1], C[:, 0]])           # duplicate column
    fs = [yp.Factor(n, 0, 1) for n in ("A", "B", "Cdup")]
    bad = yp.ExperimentDataset(fs, X)
    bad.add_response("Y", [1.0, 2.0, 3.0, 5.0])
    with pytest.raises(yp.RankDeficientError, match="Cdup|A"):
        yp.screening.main_effects(bad, "Y")


def test_missing_responses_warn_and_can_break_estimability():
    d = pb(5, 12, centers=3)
    y = synth(d, {"F1": 3.0})
    v = d.get_response("Y"); v[0] = np.nan
    d.add_response("Y", v)
    with pytest.warns(yp.InferenceWarning, match="dropped"):
        e = yp.screening.main_effects(d, "Y")
    assert e.n_obs == 14 and e.lenth is None                   # contrasts need a complete design
    d8 = pb(7, 8)
    synth(d8, {"F1": 3.0})
    v = d8.get_response("Y"); v[2] = np.nan
    d8.add_response("Y", v)
    with pytest.raises(yp.RankDeficientError):
        yp.screening.main_effects(d8, "Y")
    with pytest.raises(KeyError):
        yp.screening.main_effects(d8, "nope")


def test_noise_free_data_does_not_crash():
    d = pb(4, 12, seed=0)
    X = d.coded
    d.add_response("Y", 10 + 2 * X[:, 0])
    e = yp.screening.main_effects(d, "Y")
    assert e.sigma == pytest.approx(0, abs=1e-9)
    s = yp.screening.screen(e)
    assert s.selected_factors == ["F1"]


# ------------------------------------------------------------------ Lenth path
def test_lenth_fallback_for_saturated_design():
    d = pb(7, 8, seed=0)
    synth(d, {"F1": 8.0, "F4": -6.0}, sigma=0.3, seed=0)
    e = yp.screening.main_effects(d, "Y")
    with pytest.warns(yp.InferenceWarning):
        s = yp.screening.screen(e)                             # auto -> lenth
    assert s.method == "lenth" and s.t_limit is None
    assert {"F1", "F4"} <= set(s.selected_factors)
    assert s.final_table["p"].isna().all()                       # no p-values from PSE
    ax = s.plot_pareto()                                         # auto -> effect scale
    assert ax.get_ylabel() == "|Effect|"
    assert len(ax.get_legend().get_texts()) >= 5


# ------------------------------------------------------------------ selection procedure
def test_partial_stats_consistent_with_full_ols():
    d = pb(6, 12, centers=3, seed=7)
    y = synth(d, {"F1": 2.0, "F3": -1.2}, sigma=0.5, seed=8)
    e = yp.screening.main_effects(d, "Y")
    base = np.column_stack([np.ones(len(e._y)), e._curv])
    S = list(range(6))
    df, sigma, _ = model_state(base, e._X, e._y, S)
    eff, se, t = partial_stats(base, e._X, e._y, S, sigma)
    assert df == e.df_error and sigma == pytest.approx(e.sigma)
    assert np.allclose(t, e.table["t"]) and np.allclose(eff, e.table["Effect"])
    # sub-model: t of members equals direct OLS t of that sub-model
    S2 = [0, 2]
    df2, s2, _ = model_state(base, e._X, e._y, S2)
    eff2, se2, t2 = partial_stats(base, e._X, e._y, S2, s2)
    Z = np.column_stack([base, e._X[:, S2]])
    f = fit_ols(Z, e._y)
    assert np.allclose(t2[S2], f.t[2:]) and df2 == f.df
    assert np.allclose(se2[S2], 2 * f.se[2:])


def test_selection_identifies_true_factors_and_audit_trail():
    d = pb(8, 12, centers=4, seed=21)
    synth(d, {"F2": 5.0, "F6": -3.0}, sigma=0.3, seed=21)
    e = yp.screening.main_effects(d, "Y")
    s = yp.screening.screen(e, alpha=0.05, criterion="bonferroni")
    assert s.selected_factors == ["F2", "F6"]
    assert s.stop_reason.startswith("converged")
    assert s.iterations[0].model_terms == d.factor_names         # starts from the full model
    for r in s.iterations:                                       # everything is recorded
        assert r.t_limit == pytest.approx(stats.t.ppf(0.975, r.df_error))
        assert r.bonferroni_limit == pytest.approx(stats.t.ppf(1 - 0.05 / 16, r.df_error))
        assert set(r.t_values) == set(d.factor_names)
        assert r.action in ("remove", "add", "none")
    assert s.iterations[-1].action == "none" and s.iterations[-1].stop_reason
    removed = [n for r in s.iterations for n in r.removed]
    assert set(removed) == set(d.factor_names) - {"F2", "F6"}
    assert set(s.candidates) >= {"F2", "F6"}
    h = s.history()
    assert len(h) == len(s.iterations) and "stop_reason" in h
    # final model t-values: members equal direct OLS on the final model
    fm = fit_ols(np.column_stack([np.ones(len(e._y)), e._X[:, [1, 5]], e._curv]), e._y)
    row = s.final_table.set_index("Factor")
    assert row.loc["F2", "t"] == pytest.approx(fm.t[1]) and s.df_error == fm.df
    assert [f.name for f in s.selected_factor_objects()] == ["F2", "F6"]


def test_criteria_and_directions():
    d = pb(6, 12, centers=3, seed=2)
    synth(d, {"F1": 2.0, "F3": -0.35}, sigma=0.2, seed=2)        # F3 modest effect
    e = yp.screening.main_effects(d, "Y")
    loose = yp.screening.screen(e, criterion="t_limit")
    strict = yp.screening.screen(e, criterion="bonferroni")
    assert set(strict.selected_factors) <= set(loose.selected_factors)
    assert "F1" in strict.selected_factors
    fwd = yp.screening.screen(e, direction="forward", start="empty")
    assert "F1" in fwd.selected_factors
    allin = yp.screening.screen(e, direction="forward", start="empty", add="all")
    assert "F1" in allin.selected_factors
    mx = yp.screening.screen(e, max_steps=1)
    assert len(mx.iterations) == 1 and "max_steps" in mx.stop_reason


def test_selection_df_floor_and_start_errors():
    d = _filled(pb(11, 12, centers=3))                           # full model: df = 2
    e = yp.screening.main_effects(d, "Y")
    with pytest.raises(yp.InferenceError):
        yp.screening.screen(e, min_df=5)
    X, y = e._X, e._y
    with pytest.raises(ValueError):
        stepwise_selection(X, y, d.factor_names, criterion="nope")
    with pytest.raises(ValueError):
        yp.screening.screen(e, method="nope")


def test_alpha_override_changes_limits():
    d = pb(6, 12, seed=1)
    synth(d, {"F1": 2.0}, seed=1)
    e = yp.screening.main_effects(d, "Y")
    s1 = yp.screening.screen(e, alpha=0.05)
    s2 = yp.screening.screen(e, alpha=0.01)
    assert s2.t_limit > s1.t_limit and s2.alpha == 0.01
    with pytest.raises(ValueError):
        yp.screening.screen(e, alpha=0)


# ------------------------------------------------------------------ half-normal statistics
def test_half_normal_quantiles():
    q = halfnormal.half_normal_quantiles(4)
    expect = stats.norm.ppf(0.5 + 0.5 * (np.arange(1, 5) - 0.5) / 4)
    assert np.allclose(q, expect) and np.all(np.diff(q) > 0) and q[0] > 0
    # these are quantiles of |Z|: P(|Z| <= q_i) = (i - 0.5)/m
    assert np.allclose(2 * stats.norm.cdf(q) - 1, (np.arange(1, 5) - 0.5) / 4)


def test_half_normal_points_and_reference_line():
    pts = halfnormal.half_normal_points(list("ABCDE"), [3.0, -0.1, 0.2, -0.3, 0.15],
                                        ["factor"] * 5)
    assert list(pts["Label"]) == ["B", "E", "C", "D", "A"]       # ascending |effect|
    assert list(pts["Sign"]) == ["-", "+", "+", "-", "+"]
    assert np.allclose(pts["AbsEffect"], [0.1, 0.15, 0.2, 0.3, 3.0])
    slope = halfnormal.reference_slope(pts, fraction=0.6)
    n = 3
    q, a = pts["Quantile"].to_numpy()[:n], pts["AbsEffect"].to_numpy()[:n]
    assert slope == pytest.approx((q * a).sum() / (q * q).sum())
    assert slope < 0.5                       # the effect of size 3 does not tilt the line
    # pure noise: half-normal points of N(0, s^2) effects follow slope ~ s
    rng = np.random.default_rng(0)
    big = halfnormal.half_normal_points([str(i) for i in range(400)], 2.0 * rng.standard_normal(400))
    assert halfnormal.reference_slope(big, 1.0) == pytest.approx(2.0, rel=0.08)
    with pytest.raises(ValueError):
        halfnormal.reference_slope(pts, 0)
