"""The wetted-area figure: where particles hit the wall, on the wall's own
(phi, l) coordinates, with its toroidal and poloidal profiles alongside
(data from ashen.diagnostics.particle_wetted).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ashen.diagnostics.particle_wetted import WallHits, WettedResult
from ashen.plotting import DEFAULT_DPI, style

__all__ = [
    "COUNT_LABEL","plot_wetted_area", "wetted_caption"]

_PHI_TICKS = ([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
              ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])


def _pm(value: float, error: float) -> str:
    if np.isnan(error):
        return f"{value:.3f}"
    return f"{value:.3f} ± {error:.3f}"


def wetted_caption(
    result: WettedResult, hits: WallHits, *, duration: float | None = None,
    note: str | None = None,
) -> str:
    """The four measures, how many hits they rest on and, given duration
    [s], how much of the trace they were collected over. note, e.g. which
    markers were counted, goes on a line of its own."""
    placed = (f", {hits.n_left_grid} left the grid first (at the nearest wall point)"
              if hits.n_left_grid else "")
    if duration is not None:
        placed += f", over {duration * 1e6:.4g} µs of trace"
    return (
        f"f_pol = {_pm(result.f_pol, result.f_pol_err)}   "
        f"f_tor = {_pm(result.f_tor, result.f_tor_err)}   "
        f"f_tot = {_pm(result.f_tot, result.f_tot_err)} ({result.area:.3g} m²)   "
        f"s = {_pm(result.s, result.s_err)}\n"
        f"{hits.n} of {hits.n_considered} particles hit the wall{placed}"
    ) + (f"\n{note}" if note else "")


#: The map's colour scale: a density, so not bounded by 1.
DENSITY_LABEL = r"(fraction of total hits) / m$^2$"
#: ...and in counts mode: how many particles hit each cell.
COUNT_LABEL = "particles hitting each cell"


def plot_wetted_area(
    result: WettedResult,
    hits: WallHits,
    out_path: Path | str,
    *,
    figsize: tuple[float, float] = (8.0, 6.0),
    density_range: tuple[float, float] | None = None,
    duration: float | None = None,
    note: str | None = None,
    counts: bool = False,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save the (phi, l) hit-density map -- the share of all hits
    per m^2 of wall, a density in 1/m^2 that integrates to 1 over the wall
    (so it passes 1 wherever a cell much smaller than 1 m^2 holds a few
    percent of the hits) -- with the toroidal profile above it and the
    poloidal one to its right, each as the fraction of hits per bin.

    With counts, every panel shows numbers of particles instead: the map
    how many hit each cell (so an inboard cell, smaller, shows fewer for
    the same density), the profiles how many hit each bin.

    density_range, (min, max) in the map's own units -- 1/m^2, or particles
    with counts -- fixes the colour scale; cells beyond it take the end
    colours, and the colourbar says so."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = max(result.counts.sum(), 1)
    if counts:
        density, label, scale, profile_label = result.counts, COUNT_LABEL, 1, "particles"
    else:
        density, label = result.counts / result.cell_area / total, DENSITY_LABEL
        scale, profile_label = total, "fraction"

    with style():
        fig = plt.figure(figsize=figsize, layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4))
        ax_map = fig.add_subplot(grid[1, 0])
        ax_tor = fig.add_subplot(grid[0, 0], sharex=ax_map)
        ax_pol = fig.add_subplot(grid[1, 1], sharey=ax_map)

        mesh = ax_map.pcolormesh(
            result.phi_edges, result.l_edges, np.ma.masked_equal(density, 0),
            cmap="viridis", shading="flat",
            vmin=density_range[0] if density_range else None,
            vmax=density_range[1] if density_range else None,
        )
        extend = "neither"
        if density_range is not None and density.max() > density_range[1]:
            extend = "max"
        fig.colorbar(mesh, ax=ax_pol, extend=extend, label=label)
        ax_map.set_xlim(0, 2 * np.pi)
        ax_map.set_ylim(result.l_edges[0], result.l_edges[-1])
        ax_map.set_xticks(_PHI_TICKS[0])
        ax_map.set_xticklabels(_PHI_TICKS[1])
        ax_map.set_xlabel(r"toroidal angle $\phi$")
        ax_map.set_ylabel(r"poloidal arc length $l$ [m] (0: outboard midplane, ccw)")

        ax_tor.stairs(result.counts.sum(axis=0) / scale, result.phi_edges,
                      fill=True, color="tab:blue", alpha=0.7)
        ax_tor.set_ylabel(profile_label)
        ax_tor.tick_params(labelbottom=False)
        ax_pol.stairs(result.counts.sum(axis=1) / scale, result.l_edges,
                      fill=True, color="tab:blue", alpha=0.7, orientation="horizontal")
        ax_pol.set_xlabel(profile_label)
        ax_pol.tick_params(labelleft=False)

        fig.suptitle(wetted_caption(result, hits, duration=duration, note=note), fontsize=9)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
