"""One scalar per case, plotted against a scan parameter.

Ports the core of the notebook's ``eta_plot`` (``Columbia/NL_kinks/
prod_plots_draft0.ipynb``, cell 0), stripped to what George asked to keep:
a single series, log-x by default, markers connected by a line. The
notebook's dual y-axis, highlight-point circles, vertical fading band,
``\\textbf{}`` figure-corner label and vline/annotation machinery are not
ported -- add them back here if a future comparison actually needs one, not
speculatively.

Deliberately generic over what ``y`` is, not specific to wetted fraction --
:mod:`ashen.diagnostics.theta_histogram`'s ``wetted_fraction`` is the first
consumer, not the only intended one (see the "scan-vs-x plots" pattern George
flagged as recurring). ``draw_*``/``plot_*`` split, matching every other
module in this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ashen.plotting import style

__all__ = ["draw_wetted_fraction_vs_x", "plot_wetted_fraction_vs_x"]


def draw_wetted_fraction_vs_x(
    ax,
    x: Sequence[float],
    y: Sequence[float],
    *,
    xlabel: str = "",
    ylabel: str = "",
    log_x: bool = True,
) -> None:
    """Draw one ``y`` vs. ``x`` series onto ``ax``, one marker per case."""
    ax.plot(x, y, marker="o", linestyle="-", color="tab:blue")
    if log_x:
        ax.set_xscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.4)


def plot_wetted_fraction_vs_x(
    x: Sequence[float],
    y: Sequence[float],
    out_path: Path | str,
    *,
    xlabel: str = "",
    ylabel: str = "Wetted fraction",
    log_x: bool = True,
    figsize: tuple[float, float] = (6, 3.5),
    dpi: int = 200,
) -> Path:
    """Draw and save one figure -- the file-owning counterpart to
    :func:`draw_wetted_fraction_vs_x`."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with style():
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        draw_wetted_fraction_vs_x(ax, x, y, xlabel=xlabel, ylabel=ylabel, log_x=log_x)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
