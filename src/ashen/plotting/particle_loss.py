"""The particle-loss map: time against the psi_n the particles started at,
coloured by the fraction of them that has left -- the connection-length
map's layout (ashen.plotting.connection_length), with what the traced
particles did in place of how long the field lines are.

Drawn to be read beside it: the same time axis (simulation time in
microseconds, as LCTT's), and the same colours for the same thing -- where
field lines are short and particles leave, both are green; where field
lines are long and particles stay, both are red. Black is no data: a psi_n
no particle started at here, an untraced field line there.

The data is ashen.diagnostics.particle_exits.loss_map; this module only draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ashen.diagnostics.particle_exits import LossMap
from ashen.plotting import DEFAULT_DPI, style

__all__ = ["LOSS_LABEL", "draw_loss_map", "plot_loss_map"]

LOSS_LABEL = "fraction of particles lost"


def draw_loss_map(ax, result: LossMap) -> "object":
    """Draw the loss fraction as a pcolormesh on ax, time [us] against
    starting psi_n. Returns the mappable, for a colourbar."""
    import matplotlib.pyplot as plt

    # The connection-length map's colours run green (short) to red (long);
    # lost is the short end, so the scale is the same one unreversed.
    cmap = plt.get_cmap("RdYlGn").with_extremes(bad="black")
    centres = 0.5 * (result.psi_edges[:-1] + result.psi_edges[1:])
    mesh = ax.pcolormesh(
        result.time * 1e6, centres, np.ma.masked_invalid(result.fraction),
        cmap=cmap, shading="nearest", vmin=0.0, vmax=1.0,
    )
    ax.set_ylim(result.psi_edges[0], result.psi_edges[-1])
    ax.set_xlabel(r"t [$\mu s$]")
    ax.set_ylabel(r"starting $\Psi_N$")
    return mesh


def plot_loss_map(
    result: LossMap,
    out_path: Path | str,
    *,
    caption: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save the loss map, with how many particles started in each
    psi_n bin alongside -- a bin with a handful moves in big steps, and one
    with none is black."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with style():
        fig = plt.figure(figsize=figsize, layout="constrained")
        grid = fig.add_gridspec(1, 2, width_ratios=(5, 1))
        ax = fig.add_subplot(grid[0, 0])
        ax_n = fig.add_subplot(grid[0, 1], sharey=ax)
        mesh = draw_loss_map(ax, result)
        fig.colorbar(mesh, ax=ax, label=LOSS_LABEL)
        ax_n.stairs(result.counts, result.psi_edges, fill=True, color="tab:blue",
                    alpha=0.7, orientation="horizontal")
        ax_n.set_xlabel("particles")
        ax_n.tick_params(labelleft=False)
        if caption:
            fig.suptitle(caption, fontsize=9)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
