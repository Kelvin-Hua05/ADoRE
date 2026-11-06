"""Plot-facing API shared by EffectsResult and ScreeningResult.

The mixin only *selects which already-computed numbers* are drawn; every statistic comes from
the result objects.  Required attributes on the host class:
``_ptable`` (DataFrame: Factor, Effect, StdErr, t), ``_pdummy`` (DataFrame: Factor, Effect),
``_pdf``, ``_palpha``, ``_ptlimit``, ``_pbonf`` (None if no inference), ``_plenth`` (dict|None),
``_pselected`` (list), ``response``.
"""
from __future__ import annotations

import numpy as np

from ..exceptions import InferenceError
from . import halfnormal


class PlotMixin:
    def _uniform_se(self):
        se = self._ptable["StdErr"].to_numpy(dtype=float)
        if np.all(np.isfinite(se)) and np.allclose(se, se[0], rtol=1e-8):
            return float(se[0])
        return None

    def _scale(self, scale: str) -> str:
        if scale == "auto":
            return "t" if self._ptlimit is not None else "effect"
        if scale not in ("t", "effect"):
            raise ValueError("scale must be 'auto', 't' or 'effect'.")
        if scale == "t" and self._ptlimit is None:
            raise InferenceError(
                "No error degrees of freedom: t-scale Pareto is unavailable. "
                "Use scale='effect' (Lenth limits) or add center points / replicates.")
        return scale

    def _thresholds(self, scale: str) -> dict:
        """Thresholds on the SAME scale as the plotted values."""
        if scale == "t":
            return {f"t-limit = {self._ptlimit:.5g}": self._ptlimit,
                    f"Bonferroni limit = {self._pbonf:.5g}": self._pbonf}
        se = self._uniform_se()
        if self._ptlimit is not None and se is not None:
            return {f"t-limit = {self._ptlimit * se:.4g}": self._ptlimit * se,
                    f"Bonferroni limit = {self._pbonf * se:.4g}": self._pbonf * se}
        if self._plenth is not None:
            return {f"Lenth ME = {self._plenth['ME']:.4g}": self._plenth["ME"],
                    f"Lenth SME = {self._plenth['SME']:.4g}": self._plenth["SME"]}
        return {}

    def _subtitle(self, scale: str) -> str:
        if scale == "t" or self._ptlimit is not None:
            return f"alpha={self._palpha:g}, error df={self._pdf}"
        return f"alpha={self._palpha:g}, Lenth PSE (no error df)"

    def plot_pareto(self, scale: str = "auto", top_n: int | None = None, show_limits: bool = True,
                    title: str | None = None, colors=("#E8871E", "#1F77D0"), ax=None,
                    save_path=None):
        """Pareto chart of |t| (scale='t') or |effect| (scale='effect'), sorted descending.
        Orange = positive, blue = negative effect; thick edge = selected factor."""
        from . import plots
        sc = self._scale(scale)
        return plots.pareto(self._ptable, scale=sc,
                            limits=self._thresholds(sc) if show_limits else {},
                            selected=self._pselected, subtitle=self._subtitle(sc),
                            response=self.response, top_n=top_n, title=title, colors=colors,
                            ax=ax, save_path=save_path)

    def plot_half_normal(self, include_dummy: bool = True, label: str = "auto", fraction: float = 0.5,
                         style: str = "quantile", show_limits: bool = False,
                         title: str | None = None, colors=("#E8871E", "#1F77D0"), ax=None,
                         save_path=None):
        """Half-normal plot: x = half-normal quantile, y = |effect| (style='quantile', as in the
        project PDF) or Design-Expert-like axes (style='probability')."""
        from . import plots
        t = self._ptable
        labels = list(t["Factor"])
        eff = list(t["Effect"])
        kinds = ["factor"] * len(labels)
        if include_dummy and len(self._pdummy):
            labels += list(self._pdummy["Factor"])
            eff += list(self._pdummy["Effect"])
            kinds += ["dummy"] * len(self._pdummy)
        pts = halfnormal.half_normal_points(labels, eff, kinds)
        slope = halfnormal.reference_slope(pts, fraction)
        lim = self._thresholds(self._scale("effect")) if show_limits else {}
        return plots.half_normal(pts, slope, selected=self._pselected, label=label, style=style,
                                 limits=lim, subtitle=self._subtitle("effect"),
                                 response=self.response, title=title, colors=colors, ax=ax,
                                 save_path=save_path, dummy_label=self._pdummy_label)

    def plot_abs_effects(self, top_n: int | None = None, show_limits: bool = True,
                         title: str | None = None, colors=("#E8871E", "#1F77D0"), ax=None,
                         save_path=None):
        """Absolute-effect plot (raw effect units), ranked, with thresholds on the effect scale."""
        from . import plots
        sc = "effect"
        return plots.abs_effects(self._ptable,
                                 limits=self._thresholds(sc) if show_limits else {},
                                 selected=self._pselected, subtitle=self._subtitle(sc),
                                 response=self.response, top_n=top_n, title=title, colors=colors,
                                 ax=ax, save_path=save_path)
