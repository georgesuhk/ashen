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

__all__ = ["animate_profile_comparison", "draw_profile_family", "plot_profile_comparison"]


def _profile_frame_label(step: int, time_by_step: Mapping[int, float] | None) -> str:
    """"step <n>[, t = <value> us]" -- an animate_profile_comparison frame's
    title. Always states the step; appends the true time (seconds in
    time_by_step, displayed in microseconds) only if time_by_step covers
    this step -- independent of whatever the colourbar itself is keyed on,
    so the step and the time are both always readable as text when known.
    """
    label = f"step {step}"
    if time_by_step is not None and step in time_by_step:
        label += f", t = {time_by_step[step] * 1e6:.4g} \N{GREEK SMALL LETTER MU}s"
    return label


def draw_profile_family(
    ax,
    series: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    color_by: Mapping[int, float] | None = None,
    colors: PsiColorer | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    rational_lines: list[tuple[float, str, str]] | None = None,
) -> PsiColorer:
    """Draw one line per step in series onto ax, in step order.

    color_by maps a step to the scalar it should be coloured by (true
    time, say); defaults to the step index itself. colors lets a caller
    share one colourer -- and therefore one colour scale/colourbar --
    across several axes; when omitted, one is built spanning just this
    axes' own values. Returns whichever colourer was used, so a
    single-panel caller can attach a colourbar without rebuilding it.

    rational_lines, if given, is a [(psi_n, color, label), ...] list drawn
    as dashed vertical lines -- q=m/n rational-surface positions, same
    shape cli/plot.py's mark_rational computes. A reversed-shear q-profile
    can produce several crossings (several lines) sharing one mode's label;
    only the first per label is added to the legend, so the legend has one
    entry per mode, not one per crossing.
    """
    steps = sorted(series)
    values = {step: float(step) for step in steps} if color_by is None else color_by

    if colors is None:
        colors = colorer([values[s] for s in steps if s in values])

    for step in steps:
        x, y = series[step]
        ax.plot(x, y, color=colors(values.get(step, float(step))), linewidth=1.0)

    seen_labels: set[str] = set()
    for psi_n, color, label in rational_lines or []:
        ax.axvline(
            psi_n, color=color, linestyle="--", linewidth=1.0, alpha=0.7,
            label=None if label in seen_labels else label,
        )
        seen_labels.add(label)
    if seen_labels:
        ax.legend(fontsize=8, loc="best")

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
    rational_lines: list[tuple[float, str, str]] | None = None,
    cmap: str = "turbo",
) -> Path:
    """One panel per tor_mode, sharing the y-axis, with a shared colourbar.

    Modes are drawn in the order given. A mode whose series is empty
    still gets its (empty) panel, labelled as such -- that a mode
    produced nothing is the result worth seeing, not a reason to
    silently renumber the panels.

    rational_lines, if given, is drawn on every panel -- see
    draw_profile_family.

    cmap is any matplotlib colormap name -- passed straight through to
    colorer, unvalidated here (an invalid name surfaces as matplotlib's own
    error at draw time).
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
    colors = colorer([values[s] for s in all_steps if s in values], cmap=cmap)

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
                rational_lines=rational_lines,
            )
        row[0].set_ylabel(var)

        if all_steps:
            fig.colorbar(colors.scalar_mappable(), ax=list(row), label=color_label)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def animate_profile_comparison(
    series_by_mode: Mapping[str, Mapping[int, tuple[np.ndarray, np.ndarray]]],
    var: str,
    out_path: Path | str,
    *,
    color_by: Mapping[int, float] | None = None,
    color_label: str = "Time step",
    time_by_step: Mapping[int, float] | None = None,
    xlabel: str = "",
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    rational_lines: list[tuple[float, str, str]] | None = None,
    cmap: str = "turbo",
    fps: float = 2.0,
) -> Path | None:
    """The animated counterpart to plot_profile_comparison -- a GIF with one
    frame per restart step, each panel showing that step's curve alone
    (not the whole family at once), so the profile's time evolution is
    watched directly rather than read off a static colourbar.

    Same panel-per-tor_mode layout and colour convention as
    plot_profile_comparison, but axis limits are fixed up front to the full
    data range across every step, so panels don't rescale frame to frame.
    rational_lines (mark_rational's vertical lines + one legend entry per
    mode) are drawn once and held static across every frame, same as on the
    static figure.

    Every frame's title always states the restart step, and the true time
    (seconds, formatted in microseconds) if time_by_step covers that step --
    independent of what color_by/color_label are colouring the curve by.
    So even when the colourbar is by step index (color_by is None or a
    zeroD-incomplete case fell back to it), a frame's real time is still
    readable in text, and even when the colourbar is by time, the step
    number stays visible too -- the two are complementary labels, not one
    substituting for the other.

    Returns None (no file written) if there are fewer than two steps across
    every mode -- a one-frame "animation" isn't one. Uses matplotlib's
    Pillow-backed GIF writer; Pillow is already a hard matplotlib dependency
    (image I/O), not an extra one ashen introduces.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    modes = list(series_by_mode)
    all_steps = sorted({step for series in series_by_mode.values() for step in series})
    if len(all_steps) < 2:
        return None

    if figsize is None:
        figsize = (1 + 4.5 * max(len(modes), 1), 4.5)

    values = {s: float(s) for s in all_steps} if color_by is None else color_by
    colors = colorer([values[s] for s in all_steps if s in values], cmap=cmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with style():
        fig, axes = plt.subplots(
            1, max(len(modes), 1), figsize=figsize, sharey=True, squeeze=False
        )
        row = axes[0]
        lines = []
        for ax, mode in zip(row, modes):
            series = series_by_mode[mode]
            ax.set_title(mode if series else f"{mode} (no data)")
            if xlabel:
                ax.set_xlabel(xlabel)
            if series:
                xs = np.concatenate([x for x, _ in series.values()])
                ys = np.concatenate([y for _, y in series.values()])
                y_pad = (float(ys.max()) - float(ys.min())) * 0.05 or 1.0
                ax.set_xlim(float(xs.min()), float(xs.max()))
                ax.set_ylim(float(ys.min()) - y_pad, float(ys.max()) + y_pad)
            (line,) = ax.plot([], [], linewidth=1.5)
            lines.append(line)

            seen_labels: set[str] = set()
            for psi_n, color, label in rational_lines or []:
                ax.axvline(
                    psi_n, color=color, linestyle="--", linewidth=1.0, alpha=0.7,
                    label=None if label in seen_labels else label,
                )
                seen_labels.add(label)
            if seen_labels:
                ax.legend(fontsize=8, loc="best")
        row[0].set_ylabel(var)

        fig.colorbar(colors.scalar_mappable(), ax=list(row), label=color_label)
        suptitle = fig.suptitle("")

        def _update(step):
            for line, mode in zip(lines, modes):
                series = series_by_mode[mode]
                if step in series:
                    x, y = series[step]
                    line.set_data(x, y)
                    line.set_color(colors(values.get(step, float(step))))
                else:
                    line.set_data([], [])
            suptitle.set_text(_profile_frame_label(step, time_by_step))
            return [*lines, suptitle]

        anim = animation.FuncAnimation(fig, _update, frames=all_steps, blit=False)
        anim.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path
