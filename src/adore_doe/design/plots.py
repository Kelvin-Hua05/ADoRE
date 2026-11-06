"""FDS plot (matplotlib is an optional extra: pip install 'adore-doe[plot]')."""
from __future__ import annotations

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("Plotting needs matplotlib: pip install 'adore-doe[plot]'") from e


def plot_fds(results, ax=None, save_path=None, title=None):
    items = results if isinstance(results, dict) else {"design": results}
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    fig = ax.figure
    scaled = False
    for label, r in items.items():
        ax.plot(r.fraction, r.variance, label=f"{label} (n={r.n_runs}, mean={r.stats['mean']:.3g}, "
                                              f"max={r.stats['max']:.3g})", linewidth=1.8)
        scaled = scaled or r.scaled
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of design space")
    ax.set_ylabel("Scaled prediction variance  n·f'(X'X)⁻¹f" if scaled else "Prediction variance  f'(X'X)⁻¹f  (σ²=1)")
    ax.set_title(title or "FDS plot (polynomial-model variance; not a GPR variance)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
