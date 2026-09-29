"""Histograms of where particles leave the plasma: theta and phi side by
side (data from ashen.diagnostics.particle_exits).

Weighted to the fraction of exiting particles per bin, like
ashen.plotting.theta_histogram, so a case with more particles reads on the
same scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ashen.diagnostics.particle_exits import ExitResult
from ashen.plotting import DEFAULT_DPI, style

__all__ = ["exit_caption", "plot_exit_histograms"]

_THETA_TICKS = ([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
_PHI_TICKS = ([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
              ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])


def exit_caption(result: ExitResult, *, psi_n: float, boundary: bool = False) -> str:
    """How many particles the histograms stand for, and how each exited."""
    where = f"psi_n = {psi_n:g}"
    if boundary:
        where += " or the plasma boundary before extension"
    text = f"{result.n_exited} of {result.n_considered} particles exit past {where}"
    if boundary:
        text += (
            f" ({result.n_outside_boundary} at the boundary, "
            f"{result.n_crossed} at psi_n)"
        )
    if result.n_left_grid:
        text += (
            f" ({result.n_left_grid} left the grid first -- "
            "drawn at their last position on it)"
        )
    return text


def _draw(ax, angles: np.ndarray, *, bins: int, lo: float, hi: float, ticks) -> None:
    if angles.size == 0:
        ax.text(0.5, 0.5, "No exits", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.hist(
            angles, bins=bins, range=(lo, hi), weights=np.full(angles.shape, 1.0 / angles.size),
            color="tab:blue", edgecolor="tab:blue", alpha=0.7, linewidth=0.5,
        )
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks[0])
    ax.set_xticklabels(ticks[1])
    ax.tick_params(direction="in", which="both", top=True, right=True)


def plot_exit_histograms(
    result: ExitResult,
    out_path: Path | str,
    *,
    bins: int = 72,
    caption: str | None = None,
    figsize: tuple[float, float] = (8.0, 3.2),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save the theta (left) and phi (right) exit histograms."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with style():
        fig, (ax_theta, ax_phi) = plt.subplots(
            1, 2, figsize=figsize, layout="constrained", sharey=True,
        )
        _draw(ax_theta, result.theta, bins=bins, lo=-np.pi, hi=np.pi, ticks=_THETA_TICKS)
        _draw(ax_phi, result.phi, bins=bins, lo=0.0, hi=2 * np.pi, ticks=_PHI_TICKS)
        ax_theta.set_xlabel(r"poloidal angle $\theta$ at exit")
        ax_phi.set_xlabel(r"toroidal angle $\phi$ at exit")
        ax_theta.set_ylabel("fraction of exits")
        if caption:
            fig.suptitle(caption, fontsize=9)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
