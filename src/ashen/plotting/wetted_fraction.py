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

:func:`plot_wetted_fraction_vs_x` draws one series (one comparison's flat
``cases``). :func:`plot_wetted_fraction_datasets` draws several named series
on the same axes with a legend -- for a ``datasets``-style comparison
(:class:`ashen.comparisons.Comparison`), e.g. the same resistivity scan
repeated under two profile assumptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ashen.plotting import style
from ashen.plotting.colors import DISCRETE_PALETTE

__all__ = [
    "draw_wetted_fraction_vs_x",
    "plot_wetted_fraction_vs_x",
    "plot_wetted_fraction_datasets",
]


def draw_wetted_fraction_vs_x(
    ax,
    x: Sequence[float],
    y: Sequence[float],
    *,
    xlabel: str = "",
    ylabel: str = "",
    log_x: bool = True,
    label: str | None = None,
    color: str = "tab:blue",
) -> None:
    """Draw one ``y`` vs. ``x`` series onto ``ax``, one marker per case.

    ``label``/``color`` let a caller draw more than one series onto the same
    ``ax`` (e.g. :func:`plot_wetted_fraction_datasets`, one call per
    dataset) and tell them apart -- ``label`` is only passed to ``ax.plot``
    when given, so a single-series caller doesn't pick up an unwanted legend
    entry.
    """
    ax.plot(x, y, marker="o", linestyle="-", color=color, **({"label": label} if label else {}))
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


def plot_wetted_fraction_datasets(
    series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    out_path: Path | str,
    *,
    xlabel: str = "",
    ylabel: str = "Wetted fraction",
    log_x: bool = True,
    figsize: tuple[float, float] = (6, 3.5),
    dpi: int = 200,
    colors: Sequence[str | None] | None = None,
) -> Path:
    """Overlay several named series on one axes -- e.g. the same resistivity
    scan repeated under two profile assumptions ("normal" vs "rho19") -- with
    a legend distinguishing them.

    ``series`` is ``[(dataset_label, x, y), ...]``, one entry per dataset,
    drawn via :func:`draw_wetted_fraction_vs_x` in order. ``colors`` is
    parallel to ``series``; a ``None`` entry (or ``colors`` omitted entirely)
    assigns from :data:`ashen.plotting.colors.DISCRETE_PALETTE`, cycling by
    that series' position -- the same discrete categorical palette used
    elsewhere in this package, so a dataset's colour is at least consistent
    with the rest of a figure set even without an explicit choice.

    The single-series :func:`plot_wetted_fraction_vs_x` is unchanged and
    still the right call for a plain (non-dataset) comparison -- this is the
    ``datasets``-mode counterpart, not a replacement.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with style():
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        for idx, (label, x, y) in enumerate(series):
            color = None
            if colors is not None and idx < len(colors):
                color = colors[idx]
            if color is None:
                color = DISCRETE_PALETTE[idx % len(DISCRETE_PALETTE)]
            draw_wetted_fraction_vs_x(
                ax, x, y, xlabel=xlabel, ylabel=ylabel, log_x=log_x,
                label=label, color=color,
            )
        ax.legend()
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
