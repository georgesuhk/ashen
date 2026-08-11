"""Radial profile families -- one curve per restart step, coloured by time.

Draws the .npz caches diagnostics.profiles writes, read back through
read_profile_series. Ports the plotting half of gather_profiles.py::
plot_postproc_profs (KNOWN_ISSUES.md #8), minus its derived q-profile,
which stays unported.

Colour is by step or time, not radial coordinate -- the one difference
from plotting.poincare: here the radial coordinate is the x-axis, so the
free dimension a colourmap can carry is when the profile was taken.
PsiColorer is reused as-is for that -- despite the name it's a plain
float->colour map over a fixed [vmin, vmax] with a scalar_mappable for
the colourbar, so a near-duplicate class would buy nothing.

Several tor_modes are drawn as separate panels sharing a y-axis, not
overlaid: a flux-surface average and a midplane outer cut of the same
variable are different quantities that happen to share units, and the
interesting comparison is where one stops having curves at all
(KNOWN_ISSUES.md #9), not a point-by-point difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ashen.plotting import style
from ashen.plotting.colors import PsiColorer, colorer

__all__ = ["draw_profile_family", "plot_profile_comparison"]


def draw_profile_family(
    ax,
    series: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    color_by: Mapping[int, float] | None = None,
    colors: PsiColorer | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> PsiColorer:
    """Draw one line per step in series onto ax, in step order.

    color_by maps a step to the scalar it should be coloured by (true
    time, say); defaults to the step index itself. colors lets a caller
    share one colourer -- and therefore one colour scale/colourbar --
    across several axes; when omitted, one is built spanning just this
    axes' own values. Returns whichever colourer was used, so a
    single-panel caller can attach a colourbar without rebuilding it.
    """
    steps = sorted(series)
    values = {step: float(step) for step in steps} if color_by is None else color_by

    if colors is None:
        colors = colorer([values[s] for s in steps if s in values])

    for step in steps:
        x, y = series[step]
        ax.plot(x, y, color=colors(values.get(step, float(step))), linewidth=1.0)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return colors


def plot_profile_comparison(
    series_by_mode: Mapping[str, Mapping[int, tuple[np.ndarray, np.ndarray]]],
    var: str,
    out_path: Path | str,
    *,
    color_by: Mapping[int, float] | None = None,
    color_label: str = "Time step",
    xlabel: str = "",
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Path:
    """One panel per tor_mode, sharing the y-axis, with a shared colourbar.

    Modes are drawn in the order given. A mode whose series is empty
    still gets its (empty) panel, labelled as such -- that a mode
    produced nothing is the result worth seeing, not a reason to
    silently renumber the panels.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    modes = list(series_by_mode)
    if figsize is None:
        figsize = (1 + 4.5 * max(len(modes), 1), 4.5)

    # One colourer across every panel, spanning every step any mode has, so
    # the same colour means the same time in all of them.
    all_steps = sorted({step for series in series_by_mode.values() for step in series})
    values = {s: float(s) for s in all_steps} if color_by is None else color_by
    colors = colorer([values[s] for s in all_steps if s in values])

    with style():
        fig, axes = plt.subplots(
            1, max(len(modes), 1), figsize=figsize, sharey=True, squeeze=False
        )
        row = axes[0]
        for ax, mode in zip(row, modes):
            series = series_by_mode[mode]
            draw_profile_family(
                ax, series, color_by=values, colors=colors,
                xlabel=xlabel, title=mode if series else f"{mode} (no data)",
            )
        row[0].set_ylabel(var)

        if all_steps:
            fig.colorbar(colors.scalar_mappable(), ax=list(row), label=color_label)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
