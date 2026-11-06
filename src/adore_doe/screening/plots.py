"""Matplotlib drawing functions. They only draw numbers computed elsewhere
(effects tables, limits, half-normal coordinates); no statistics are computed here
except sorting/slicing for display."""
from __future__ import annotations

try:
    import matplotlib.pyplot as plt
except ImportError as e:   # matplotlib is an optional extra (ADR-008: minimal core dependencies)
    raise ImportError("Plotting needs matplotlib: pip install 'adore-doe[plot]'") from e
import numpy as np
from scipy import stats

_LIMIT_STYLES = [("#222222", "--"), ("#C0392B", "-")]


def _finish(fig, ax, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def _axes(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _draw_limits(ax, limits: dict, horizontal=True):
    for (label, val), (c, ls) in zip(limits.items(), _LIMIT_STYLES):
        ax.axhline(val, color=c, linestyle=ls, linewidth=1.3, label=label)


def _legend_handles(ax, colors, has_selected):
    from matplotlib.patches import Patch
    h, lab = ax.get_legend_handles_labels()
    extra = [Patch(facecolor=colors[0], label="Positive effect"),
             Patch(facecolor=colors[1], label="Negative effect")]
    if has_selected:
        extra.append(Patch(facecolor="white", edgecolor="black", linewidth=2, label="Selected"))
    ax.legend(handles=extra + h, loc="upper right", fontsize=8, framealpha=0.9)


def _rank(table, scale, top_n):
    vals = (np.abs(table["t"].to_numpy(float)) if scale == "t"
            else np.abs(table["Effect"].to_numpy(float)))
    order = np.argsort(-np.nan_to_num(vals, nan=-1.0), kind="stable")
    if top_n is not None:
        order = order[:top_n]
    return order, vals


def pareto(table, scale, limits, selected, subtitle, response, top_n=None, title=None,
           colors=("#E8871E", "#1F77D0"), ax=None, save_path=None):
    order, vals = _rank(table, scale, top_n)
    names = table["Factor"].to_numpy()[order]
    v = vals[order]
    pos = table["Effect"].to_numpy(float)[order] >= 0
    fig, ax = _axes(ax, (max(6, 0.45 * len(order) + 3), 4.5))
    x = np.arange(len(order))
    bars = ax.bar(x, v, color=[colors[0] if p else colors[1] for p in pos], edgecolor="k",
                  linewidth=0.6, width=0.7)
    for b, n in zip(bars, names):
        if n in selected:
            b.set_edgecolor("black")
            b.set_linewidth(2.4)
    _draw_limits(ax, limits)
    top = max([*v, *limits.values()] or [1.0])
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Factor")
    ax.set_ylabel("|t-value of effect|" if scale == "t" else "|Effect|")
    ax.set_title(title or f"Pareto Chart - {response}\n{subtitle}", fontsize=10)
    _legend_handles(ax, colors, bool(selected))
    return _finish(fig, ax, save_path)


def abs_effects(table, limits, selected, subtitle, response, top_n=None, title=None,
                colors=("#E8871E", "#1F77D0"), ax=None, save_path=None):
    order, vals = _rank(table, "effect", top_n)
    names = table["Factor"].to_numpy()[order]
    v = vals[order]
    pos = table["Effect"].to_numpy(float)[order] >= 0
    fig, ax = _axes(ax, (max(6, 0.45 * len(order) + 3), 4.5))
    x = np.arange(len(order))
    for xi, vi, p, n in zip(x, v, pos, names):
        c = colors[0] if p else colors[1]
        ax.vlines(xi, 0, vi, color=c, linewidth=2)
        ax.plot(xi, vi, "o", color=c, markeredgecolor="black" if n in selected else c,
                markeredgewidth=2 if n in selected else 0.5, markersize=8)
    _draw_limits(ax, limits)
    ax.set_ylim(0, max([*v, *limits.values()] or [1.0]) * 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Factor (ranked)")
    ax.set_ylabel("Absolute effects")
    ax.set_title(title or f"Absolute effects - {response}\n{subtitle}", fontsize=10)
    _legend_handles(ax, colors, bool(selected))
    return _finish(fig, ax, save_path)


def half_normal(points, slope, selected, label, style, limits, subtitle, response, title=None,
                colors=("#E8871E", "#1F77D0"), ax=None, save_path=None,
                dummy_label="Dummy column (error)"):
    """points: DataFrame from halfnormal.half_normal_points (ascending |effect|)."""
    fig, ax = _axes(ax, (6.5, 5))
    q = points["Quantile"].to_numpy()
    a = points["AbsEffect"].to_numpy()
    pos = points["Sign"].to_numpy() == "+"
    dummy = points["Kind"].to_numpy() == "dummy"
    lab = points["Label"].to_numpy()
    if style == "quantile":
        X, Y = q, a
        xl, yl = "Half-Normal Quantile", "Absolute effects"
    elif style == "probability":
        X, Y = a, stats.norm.cdf(q) * 2 - 1      # P(|Z| <= q): half-normal cumulative prob.
        xl, yl = "|Effect|", "Half-normal probability"
    else:
        raise ValueError("style must be 'quantile' or 'probability'.")
    col = np.where(pos, colors[0], colors[1])
    ax.scatter(X[~dummy], Y[~dummy], c=col[~dummy], edgecolors="k", s=36, zorder=3)
    if dummy.any():
        ax.scatter(X[dummy], Y[dummy], facecolors="none", edgecolors="gray", s=36, zorder=3,
                   label=dummy_label)
    # reference line through the origin (slope from halfnormal.reference_slope)
    if style == "quantile":
        xs = np.array([0, q.max() * 1.05])
        ax.plot(xs, slope * xs, color="#E57373", linewidth=1.5, label="Reference line (noise)")
        for lbl, val in limits.items():
            c, ls = _LIMIT_STYLES[list(limits).index(lbl) % 2]
            ax.axhline(val, color=c, linestyle=ls, linewidth=1.1, label=lbl)
    else:
        qs = np.linspace(0, q.max() * 1.05, 50)
        ax.plot(slope * qs, stats.norm.cdf(qs) * 2 - 1, color="#E57373", linewidth=1.5,
                label="Reference line (noise)")
    # labels
    if label == "auto":
        idx = [i for i, n in enumerate(lab) if n in selected] or list(range(max(0, len(lab) - 5), len(lab)))
    elif label == "all":
        idx = list(range(len(lab)))
    elif label == "selected":
        idx = [i for i, n in enumerate(lab) if n in selected]
    elif label == "none":
        idx = []
    else:
        raise ValueError("label must be 'auto', 'all', 'selected' or 'none'.")
    for i in idx:
        ax.annotate(lab[i], (X[i], Y[i]), xytext=(5, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold" if lab[i] in selected else "normal")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title or f"Half-Normal Plot - {response}\n{subtitle}", fontsize=10)
    from matplotlib.lines import Line2D
    h, _ = ax.get_legend_handles_labels()
    extra = [Line2D([], [], marker="o", ls="", markerfacecolor=colors[0], markeredgecolor="k",
                    label="Positive effect"),
             Line2D([], [], marker="o", ls="", markerfacecolor=colors[1], markeredgecolor="k",
                    label="Negative effect")]
    ax.legend(handles=extra + h, loc="upper left", fontsize=8, framealpha=0.9)
    return _finish(fig, ax, save_path)
