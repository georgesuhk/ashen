"""LC / LCTT connection-length colour maps.

Ports ``castor3d/util/data_jorek.py:597 color_con_length_plot``. The physics
(the matrix itself) has moved to :mod:`ashen.diagnostics.connection_length`,
which has no matplotlib import -- this module is drawing only.

Legacy filename convention preserved: ``LCTT_`` for the true-time x-axis,
``LC_`` for the step-index x-axis (``data_jorek.py:663,666``) -- the *other*
sub-plotter, ``connection_length_line_plot`` (the ``L2_``/``L2TT_`` line
plots), is out of scope for this pass; see ``KNOWN_ISSUES.md`` #4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ashen.diagnostics.connection_length import smooth_ignoring_inf
from ashen.plotting import style

__all__ = ["draw_connection_length_map", "plot_connection_length_map"]

#: Legacy hardcoded bounds (data_jorek.py:605-606,655) -- connection lengths
#: below/above these saturate the colour scale. Kept as-is; not a physics
#: change to the matrix, only where the log colour scale clips.
_LOG_VMIN = 10.0
_LOG_VMAX = 60000.0
_LINEAR_VMAX = 50000.0


def draw_connection_length_map(
    ax,
    matrix: np.ndarray,
    x: np.ndarray,
    psi_n: np.ndarray,
    *,
    log: bool = True,
    smooth: bool = False,
    xlabel: str = "",
) -> "object":
    """Draw one ``(n_steps, n_psi)`` connection-length matrix as a
    ``pcolormesh`` onto ``ax``. Returns the mappable, for an external
    colourbar -- unlike the legacy version, which always drew its own.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    Z = matrix.T  # (n_psi, n_steps), matching data_jorek.py:610 `L_matrix.T`
    if smooth:
        Z = smooth_ignoring_inf(Z, window=3)

    X, Y = np.meshgrid(x, psi_n)
    Z_masked = np.ma.masked_where(np.isinf(Z), Z)

    cmap = plt.get_cmap("RdYlGn_r").with_extremes(bad="black")

    if log:
        pcm = ax.pcolormesh(
            X, Y, Z_masked, cmap=cmap, shading="auto",
            norm=LogNorm(vmin=_LOG_VMIN, vmax=_LOG_VMAX),
        )
    else:
        pcm = ax.pcolormesh(X, Y, Z_masked, cmap=cmap, shading="auto", vmax=_LINEAR_VMAX)

    ax.set_ylabel(r"$\Psi_N$")
    if xlabel:
        ax.set_xlabel(xlabel)
    return pcm


def plot_connection_length_map(
    matrix: np.ndarray,
    steps: list[int],
    psi_n: np.ndarray,
    out_dir: Path | str,
    *,
    true_times: list[float] | None = None,
    plot_true_times: bool = True,
    log: bool = True,
    smooth: bool = False,
    figsize: tuple[float, float] = (7, 5),
    dpi: int = 150,
) -> Path:
    """Draw and save one LC/LCTT figure. ``true_times`` is required when
    ``plot_true_times=True`` -- ports ``data_jorek.py:598-600``'s
    ``x = np.array(true_times)*1e6`` vs ``x = restart_times`` branch.
    """
    import matplotlib.pyplot as plt

    if plot_true_times:
        if true_times is None:
            raise ValueError("plot_true_times=True needs true_times")
        x = np.asarray(true_times) * 1e6
        prefix, xlabel = "LCTT", r"t [$\mu s$]"
    else:
        x = np.asarray(steps)
        prefix, xlabel = "LC", "Time step"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_{min(steps)}_{max(steps)}.png"

    with style():
        fig, ax = plt.subplots(figsize=figsize)
        pcm = draw_connection_length_map(ax, matrix, x, psi_n, log=log, smooth=smooth, xlabel=xlabel)
        fig.colorbar(pcm, ax=ax, label="Connection Length [m]")
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
