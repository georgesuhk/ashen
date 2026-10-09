"""Runaway and thermal current: totals against time, density against psi_N.

``draw_*`` onto a given ``ax``, a figure builder that owns no file, and a
file-owning ``plot_*`` wrapper, as in :mod:`ashen.plotting.four_modes`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ashen.diagnostics.re_current import CurrentDensity, CurrentTotals
from ashen.plotting import DEFAULT_DPI, style

__all__ = [
    "COLORS", "draw_current_density", "draw_current_totals", "plot_re_current",
    "re_current_figure",
]

#: One colour per species, the same in both panels.
COLORS = {"total": "#52514e", "re": "#c8442f", "thermal": "#2a78d6"}
_LABELS = {"total": "total", "re": "runaway", "thermal": "thermal (total $-$ runaway)"}


def draw_current_totals(ax, totals: CurrentTotals, *, mark_step: int | None = None) -> None:
    """Total, RE and thermal current against time (ms), or against step
    when any step lacks its time."""
    by_time = bool(np.isfinite(totals.time).all()) and len(totals.steps) > 0
    x = totals.time * 1e3 if by_time else np.asarray(totals.steps, dtype=float)
    for name in ("total", "re", "thermal"):
        ax.plot(x, getattr(totals, name) / 1e3, color=COLORS[name], marker="o", markersize=3,
                label=_LABELS[name])
    if mark_step is not None and mark_step in totals.steps:
        ax.axvline(x[totals.steps.index(mark_step)], color="#898781", lw=1, ls="--")
    ax.axhline(0.0, color="#898781", lw=0.6)
    ax.set_xlabel("t [ms]" if by_time else "Time step")
    ax.set_ylabel("toroidal current [kA]")
    ax.set_title("Current through the poloidal plane", loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=8)


def draw_current_density(
    ax, density: CurrentDensity, *, step: int | None = None, real_psi_edge: float = 1.0,
) -> None:
    """The three current densities against psi_N on the outer midplane."""
    for name in ("total", "re", "thermal"):
        ax.plot(density.psi_n, getattr(density, name) / 1e6, color=COLORS[name], lw=1.8,
                label=_LABELS[name])
    if real_psi_edge < 1.0:
        ax.axvline(real_psi_edge, color="#898781", lw=1, ls="--")
    ax.axhline(0.0, color="#898781", lw=0.6)
    ax.set_xlabel(r"$\psi_N$ (JOREK grid), outer midplane")
    ax.set_ylabel(r"toroidal current density [MA/m$^2$]")
    where = "" if step is None else f", step {step}"
    ax.set_title(f"Current density{where}", loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=8)


def re_current_figure(
    fig,
    totals: CurrentTotals,
    densities: dict[int, CurrentDensity],
    *,
    step: int | None = None,
    real_psi_edge: float = 1.0,
):
    """Both panels on ``fig``. ``step`` picks the density profile; default:
    the last step that has one."""
    ax_t, ax_j = fig.subplots(1, 2)
    if step is None and densities:
        step = max(densities)
    if np.isfinite(totals.total).any():
        draw_current_totals(ax_t, totals, mark_step=step)
        if not np.isfinite(totals.re).any():
            ax_t.text(0.5, 0.04, "runaway share not shown: vpar_re_sign not found in in_main,\n"
                      "or the zeroD files have no Ipre_tot", ha="center", va="bottom",
                      transform=ax_t.transAxes, fontsize=8)
    else:
        ax_t.text(0.5, 0.5, "no zeroD cache (analyse --diag re_current)", ha="center",
                  transform=ax_t.transAxes)
    if step in densities:
        draw_current_density(ax_j, densities[step], step=step, real_psi_edge=real_psi_edge)
    else:
        ax_j.text(0.5, 0.5, "no currdens / recurrdens profile for this step\n"
                  "(analyse --diag re_current)", ha="center", transform=ax_j.transAxes)
    return fig


def plot_re_current(
    totals: CurrentTotals,
    densities: dict[int, CurrentDensity],
    out_path: Path | str,
    *,
    step: int | None = None,
    real_psi_edge: float = 1.0,
    figsize: tuple[float, float] = (13, 4.6),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save the two-panel figure."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with style():
        fig = plt.figure(figsize=figsize)
        re_current_figure(fig, totals, densities, step=step, real_psi_edge=real_psi_edge)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
