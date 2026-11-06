import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

import adore_doe as yp
from adore_doe.screening import halfnormal


def screened(seed=0, k=8, runs=12, centers=4, true=None, **kw):
    fs = [yp.Factor(f"F{i+1}", 0, 10) for i in range(k)]
    d = yp.screening.plackett_burman(fs, runs=runs, center_points=centers, randomize=True, seed=seed)
    rng = np.random.default_rng(seed)
    true = true or {"F1": 6.0, "F3": -4.0, "F5": 1.5}
    X = d.coded
    y = 30 + sum(b * X[:, d.factor_names.index(n)] for n, b in true.items())
    d.add_response("Y", y + 0.4 * rng.standard_normal(len(y)))
    e = yp.screening.main_effects(d, "Y")
    return d, e, yp.screening.screen(e, **kw)


def line_ys(ax):
    return [float(l.get_ydata()[0]) for l in ax.lines if len(set(l.get_ydata())) == 1]


def test_pareto_sorted_and_scale_matches_thresholds():
    d, e, s = screened()
    ax = s.plot_pareto()
    heights = [p.get_height() for p in ax.patches]
    assert heights == sorted(heights, reverse=True)
    t = s.final_table.set_index("Factor")["t"].abs()
    labels = [x.get_text() for x in ax.get_xticklabels()]
    assert labels == list(t.sort_values(ascending=False).index)
    assert np.allclose(heights, t.sort_values(ascending=False).to_numpy())
    ys = line_ys(ax)
    assert any(np.isclose(y, s.t_limit) for y in ys)
    assert any(np.isclose(y, s.bonferroni_limit) for y in ys)
    assert ax.get_ylabel() == "|t-value of effect|" and ax.get_xlabel() == "Factor"
    leg = [x.get_text() for x in ax.get_legend().get_texts()]
    assert any("Bonferroni" in x for x in leg) and any("t-limit" in x for x in leg)
    assert any("Selected" in x for x in leg)
    plt.close("all")


def test_pareto_colors_by_sign_and_selected_marking():
    d, e, s = screened()
    ax = s.plot_pareto(colors=("#ff0000", "#0000ff"))
    names = [x.get_text() for x in ax.get_xticklabels()]
    eff = s.final_table.set_index("Factor")["Effect"]
    for patch, n in zip(ax.patches, names):
        expect = (1, 0, 0) if eff[n] > 0 else (0, 0, 1)
        assert np.allclose(patch.get_facecolor()[:3], expect)
        assert (patch.get_linewidth() > 2) == (n in s.selected_factors)
    plt.close("all")


def test_pareto_options_top_n_title_save_and_no_limits(tmp_path):
    d, e, s = screened()
    out = tmp_path / "pareto.png"
    ax = s.plot_pareto(top_n=3, title="My title", show_limits=False, save_path=out)
    assert len(ax.patches) == 3 and ax.get_title() == "My title"
    assert line_ys(ax) == []
    assert out.exists() and out.stat().st_size > 1000
    ax2 = e.plot_pareto()                                # from EffectsResult: nothing 'selected'
    assert len(ax2.patches) == 8
    plt.close("all")


def test_pareto_effect_scale_threshold_conversion():
    d, e, s = screened()
    ax = e.plot_pareto(scale="effect")
    se = e.table["StdErr"].iloc[0]
    ys = line_ys(ax)
    assert any(np.isclose(y, e.t_limit * se) for y in ys)
    assert any(np.isclose(y, e.bonferroni_limit * se) for y in ys)
    heights = [p.get_height() for p in ax.patches]
    assert np.allclose(heights, np.sort(e.table["Abs_Effect"])[::-1])
    # an effect whose |t| exceeds the t-limit also exceeds the converted effect-scale limit
    for _, r in e.table.iterrows():
        assert (abs(r.t) > e.t_limit) == (r.Abs_Effect > e.t_limit * se)
    with pytest.raises(ValueError):
        e.plot_pareto(scale="bad")
    plt.close("all")


def test_half_normal_plot_uses_actual_quantiles_and_labels():
    d, e, s = screened()
    ax = s.plot_half_normal()
    pts_obj = [c for c in ax.collections if len(c.get_offsets()) > 0]
    xs = np.concatenate([np.asarray(c.get_offsets())[:, 0] for c in pts_obj])
    ys = np.concatenate([np.asarray(c.get_offsets())[:, 1] for c in pts_obj])
    contr = np.r_[e.table["Effect"], e.dummy_table["Effect"]]
    m = len(contr)
    assert len(xs) == m == 8 + 3                                   # factors + 3 dummy columns
    assert np.allclose(np.sort(xs), halfnormal.half_normal_quantiles(m))
    assert np.allclose(np.sort(ys), np.sort(np.abs(contr)))
    texts = {t.get_text() for t in ax.texts}
    assert texts == set(s.selected_factors)                        # labels = selected factors
    assert ax.get_xlabel() == "Half-Normal Quantile" and ax.get_ylabel() == "Absolute effects"
    ref = [l for l in ax.lines if l.get_label().startswith("Reference")][0]
    xr, yr = ref.get_data()
    assert yr[0] == 0 and np.isclose(yr[1] / xr[1], yr[1] / xr[1])
    pts = halfnormal.half_normal_points(list(range(m)), contr)
    assert np.isclose(yr[1] / xr[1], halfnormal.reference_slope(pts, 0.5))
    # selected factors really sit above the noise line
    for n in s.selected_factors:
        r = e.table.set_index("Factor").loc[n, "Abs_Effect"]
        assert r > yr[1] / xr[1] * 1.0
    plt.close("all")


def test_half_normal_variants(tmp_path):
    d, e, s = screened()
    ax = e.plot_half_normal(label="all", include_dummy=False)
    assert len(ax.texts) == 8
    ax = e.plot_half_normal(label="none")
    assert len(ax.texts) == 0
    ax = e.plot_half_normal(style="probability", save_path=tmp_path / "hn.png")
    assert ax.get_ylabel() == "Half-normal probability" and (tmp_path / "hn.png").exists()
    ax = s.plot_half_normal(show_limits=True)
    assert len(line_ys(ax)) == 2
    for bad in (dict(style="x"), dict(label="x")):
        with pytest.raises(ValueError):
            e.plot_half_normal(**bad)
    plt.close("all")


def test_abs_effects_plot():
    d, e, s = screened()
    ax = s.plot_abs_effects()
    ys = [float(c.get_paths()[0].vertices[1, 1]) for c in ax.collections
          if hasattr(c, "get_paths") and len(c.get_paths()) == 1 and c.get_paths()[0].vertices.shape[0] == 2]
    assert len(ys) == 8 and ys == sorted(ys, reverse=True)
    assert ax.get_ylabel() == "Absolute effects"
    se = s.final_table["StdErr"].iloc[0]
    assert any(np.isclose(y, s.t_limit * se) for y in line_ys(ax))
    plt.close("all")
