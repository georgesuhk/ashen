"""Runaway and thermal current: totals against time, density against psi_N.

``draw_*`` onto a given ``ax``, a figure builder that owns no file, and a
file-owning ``plot_*`` wrapper, as in :mod:`ashen.plotting.four_modes`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ashen.diagnostics.re_current import CurrentDensity, CurrentTotals, RatioMap
from ashen.plotting import DEFAULT_DPI, style

__all__ = [
    "COLORS", "RATIO_RANGE", "SPECIES", "draw_current_density", "draw_current_ratio_map", "draw_li",
    "draw_current_totals", "plot_current_ratio_map", "plot_re_current", "ratio_map_figure", "re_current_figure",
]

#: Where the ratio map's log colour scale saturates: 100 times more runaway
#: current than thermal (deep red) to 100 times more thermal (deep green).
RATIO_RANGE = (1e-2, 1e2)

#: Axis text for each radial coordinate the profiles can be on.
COORD_LABELS = {
    "Psi_N": r"$\psi_N$ (JOREK grid), outer midplane",
    "R": r"$R$ [m], outer midplane",
}

#: One colour per species, the same in both panels.
COLORS = {"total": "#52514e", "re": "#c8442f", "thermal": "#2a78d6"}
_LABELS = {"total": "total", "re": "runaway", "thermal": "thermal (total $-$ runaway)"}
#: The three currents, in drawing order. ``show`` below is a subset of these.
SPECIES = ("total", "re", "thermal")


def _zero_line(ax, show) -> None:
    """The line at zero. With all of SPECIES drawn it is part of the y
    range, as before; with a selection the range follows the drawn curves
    alone, so a curve far from zero fills the panel."""
    if all(name in show for name in SPECIES):
        ax.axhline(0.0, color="#898781", lw=0.6)
        return
    ylim = ax.get_ylim()
    ax.axhline(0.0, color="#898781", lw=0.6)
    ax.set_ylim(ylim)


def draw_current_totals(
    ax, totals: CurrentTotals, *, mark_step: int | None = None, show=SPECIES,
) -> None:
    """Total, RE and thermal current against time (ms), or against step
    when any step lacks its time. ``show``: which of SPECIES to draw."""
    by_time = bool(np.isfinite(totals.time).all()) and len(totals.steps) > 0
    x = totals.time * 1e3 if by_time else np.asarray(totals.steps, dtype=float)
    for name in (name for name in SPECIES if name in show):
        ax.plot(x, getattr(totals, name) / 1e3, color=COLORS[name], marker="o", markersize=3,
                label=_LABELS[name])
    if mark_step is not None and mark_step in totals.steps:
        ax.axvline(x[totals.steps.index(mark_step)], color="#898781", lw=1, ls="--")
    _zero_line(ax, show)
    ax.set_xlabel("t [ms]" if by_time else "Time step")
    ax.set_ylabel("toroidal current [kA]")
    ax.set_title("Current through the poloidal plane", loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)


def draw_current_density(
    ax, density: CurrentDensity, *, step: int | None = None, real_psi_edge: float = 1.0,
    show=SPECIES,
) -> None:
    """The current densities against psi_N or R on the outer midplane.
    ``show``: which of SPECIES to draw. The plasma edge is marked on a
    psi_N axis only (real_psi_edge is a psi_N)."""
    for name in (name for name in SPECIES if name in show):
        ax.plot(density.x, getattr(density, name) / 1e6, color=COLORS[name], lw=1.8,
                label=_LABELS[name])
    if density.coords_var == "Psi_N" and real_psi_edge < 1.0:
        ax.axvline(real_psi_edge, color="#898781", lw=1, ls="--")
    _zero_line(ax, show)
    ax.set_xlabel(COORD_LABELS[density.coords_var])
    ax.set_ylabel(r"toroidal current density [MA/m$^2$]")
    where = "" if step is None else f", step {step}"
    ax.set_title(f"Current density{where}", loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)


def re_current_figure(
    fig,
    totals: CurrentTotals,
    densities: dict[int, CurrentDensity],
    *,
    step: int | None = None,
    real_psi_edge: float = 1.0,
    show=SPECIES,
):
    """Both panels on ``fig``. ``step`` picks the density profile; default:
    the last step that has one. ``show``: which of SPECIES both panels draw."""
    ax_t, ax_j = fig.subplots(1, 2)
    if step is None and densities:
        step = max(densities)
    if np.isfinite(totals.total).any():
        draw_current_totals(ax_t, totals, mark_step=step, show=show)
        if not np.isfinite(totals.re).any():
            ax_t.text(0.5, 0.04, "runaway share not shown: vpar_re_sign not found in in_main,\n"
                      "or the zeroD files have no Ipre_tot", ha="center", va="bottom",
                      transform=ax_t.transAxes, fontsize=8)
    else:
        ax_t.text(0.5, 0.5, "no zeroD cache (analyse --diag re_current)", ha="center",
                  transform=ax_t.transAxes)
    if step in densities:
        draw_current_density(ax_j, densities[step], step=step, real_psi_edge=real_psi_edge,
                             show=show)
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


def draw_current_ratio_map(
    ax, ratio_map: RatioMap, x, *, xlabel: str = "", real_psi_edge: float = 1.0,
):
    """|j_thermal / j_RE| against time (or step) and psi_N or R as a colour map,
    as plotting.connection_length draws LC. Returns the mappable.

    Red: more runaway current; green: more thermal; yellow: equal. The
    scale is logarithmic and saturates at RATIO_RANGE. Hatched where the
    thermal current runs against the runaway current (a negative ratio,
    drawn by its size). Grey where there is no current to take a ratio of.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    Z = ratio_map.ratio.T                       # (n_psi, n_steps)
    size = np.ma.masked_invalid(np.abs(np.where(np.isinf(Z), np.nan, Z)))
    # no runaway current at all: all thermal, off the top of the scale
    size = np.ma.where(np.isinf(Z), RATIO_RANGE[1] * 10, size)
    size = np.ma.masked_where(np.isnan(Z), size)
    X, Y = np.meshgrid(np.asarray(x, dtype=float), ratio_map.x)
    cmap = plt.get_cmap("RdYlGn").with_extremes(bad="#d9d7d2")
    pcm = ax.pcolormesh(X, Y, size, cmap=cmap, shading="auto",
                        norm=LogNorm(vmin=RATIO_RANGE[0], vmax=RATIO_RANGE[1]))
    against = np.where(np.isfinite(Z), Z < 0, False)
    if against.any() and against.shape[1] > 1:
        ax.contourf(X, Y, against.astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=["///"])
    if ratio_map.coords_var == "Psi_N" and real_psi_edge < 1.0:
        ax.axhline(real_psi_edge, color="black", lw=1, ls="--")
    ax.set_ylabel(COORD_LABELS[ratio_map.coords_var])
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title("Thermal / runaway current density", loc="left", fontsize=11, fontweight="bold")
    return pcm


def draw_li(ax, li, x) -> None:
    """l_i against the map's x axis: JOREK's li3 and, where there are
    q-profile caches, the plasma's own (diagnostics.equilibrium.LiSeries)."""
    drawn = False
    if np.isfinite(li.li_plasma).any():
        ax.plot(x, li.li_plasma, color="#2a78d6", marker="o", markersize=3,
                label=r"$l_i$, plasma only (from q, cylinder definition)")
        drawn = True
    if np.isfinite(li.li3).any():
        ax.plot(x, li.li3, color="#52514e", marker="o", markersize=3,
                label=r"$l_i(3)$, JOREK zeroD (whole domain)")
        drawn = True
    if drawn:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no l_i: no zeroD cache for these steps", ha="center",
                va="center", transform=ax.transAxes, fontsize=8)
    ax.set_ylabel(r"$l_i$")
    ax.grid(True, linestyle=":", alpha=0.4)


def ratio_map_figure(
    fig, ratio_map: RatioMap, time=None, *, real_psi_edge: float = 1.0, li=None,
):
    """The ratio map with its colour bar on ``fig``. ``time`` (seconds, one
    per step) is the x axis when every step has one, else the step index.
    ``li`` (an equilibrium.LiSeries for the map's steps) adds l_i against
    the same axis in a panel underneath."""
    if len(ratio_map.steps) < 2:
        ax = fig.subplots()
        ax.text(0.5, 0.5, "the ratio map needs currdens / recurrdens profiles at two steps "
                "or more\n(analyse --diag re_current)", ha="center", transform=ax.transAxes)
        return fig
    by_time = time is not None and bool(np.isfinite(np.asarray(time, dtype=float)).all())
    x = np.asarray(time) * 1e3 if by_time else np.asarray(ratio_map.steps, dtype=float)
    xlabel = "t [ms]" if by_time else "Time step"
    if li is None:
        ax, axes = fig.subplots(), None
    else:
        ax, ax_li = axes = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    pcm = draw_current_ratio_map(ax, ratio_map, x, xlabel="" if axes is not None else xlabel,
                                 real_psi_edge=real_psi_edge)
    if axes is not None:
        draw_li(ax_li, li, x)
        ax_li.set_xlabel(xlabel)
    # on both panels, so the colour bar takes the same width from each and
    # their time axes stay lined up
    bar = fig.colorbar(pcm, ax=ax if axes is None else list(axes), extend="both")
    bar.set_label(r"$|j_\mathrm{thermal}\,/\,j_\mathrm{RE}|$   "
                  "(red: more runaway, green: more thermal)")
    return fig


def plot_current_ratio_map(
    ratio_map: RatioMap,
    out_path: Path | str,
    *,
    time=None,
    real_psi_edge: float = 1.0,
    li=None,
    figsize: tuple[float, float] | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save the ratio map."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if figsize is None:
        figsize = (8, 5) if li is None else (8, 6.4)
    with style():
        fig = plt.figure(figsize=figsize, layout="constrained")
        ratio_map_figure(fig, ratio_map, time, real_psi_edge=real_psi_edge, li=li)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
