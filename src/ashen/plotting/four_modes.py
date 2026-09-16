"""Time evolution of jorek2_four mode amplitudes.

One figure per variable, one distinctly-coloured line per ``(n, m)`` --
mirrors :mod:`ashen.plotting.poincare` and :mod:`ashen.plotting.
connection_length`'s split between a pure ``draw_*`` (onto a given ``ax``)
and a file-owning ``plot_*`` wrapper.

Colour comes from :data:`ashen.plotting.colors.DISCRETE_PALETTE` rather than
:class:`~ashen.plotting.colors.PsiColorer`: a mode is identified by a
discrete ``(n, m)`` pair, not a continuous physical quantity like ``psi_n``,
so a categorical palette is the right tool here, not a colormap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from ashen.diagnostics.four_modes import GrowthFit, ModeKey
from ashen.plotting import DEFAULT_DPI, style
from ashen.plotting.colors import DISCRETE_PALETTE

if TYPE_CHECKING:
    from ashen.plotting.profiles import RationalBand

__all__ = [
    "draw_mode_amplitudes", "plot_mode_amplitudes",
    "mode_panel_label", "plot_mode_radial",
]


def draw_mode_amplitudes(
    ax,
    x: Sequence[float],
    series: Mapping[ModeKey, "object"],
    *,
    variable: str,
    rational_series: Mapping[ModeKey, "object"] | None = None,
    growth_fits: Mapping[ModeKey, GrowthFit] | None = None,
    log: bool = True,
    xlabel: str = "",
    ylabel: str | None = None,
    label_suffix: str = "",
    caption: str | None = None,
    title: str = "",
    grid: bool = True,
    ylim: tuple[float, float] | None = None,
    vline: tuple[float, str] | None = None,
    colors: Mapping[tuple[int, int], str] | None = None,
) -> None:
    """Draw every ``(n, m)`` mode of ``variable`` present in ``series`` onto
    ``ax``, each a differently-coloured line.

    Modes are sorted by ``(n, m)`` before colour assignment, so the same mode
    gets the same colour and legend position across figures/re-runs rather
    than depending on dict iteration order.

    ``rational_series``, if given (see :func:`ashen.diagnostics.four_modes.
    rational_surface_series`), overlays each mode's amplitude pinned to its
    ``q = m/n`` resonant surface as a dashed line in the same colour --
    directly comparable to the solid ``series`` line for the same mode, so a
    reader can see whether the mode's growth is actually localised at the
    surface it resonates on.

    ``growth_fits``, if given (see :func:`ashen.diagnostics.four_modes.
    growth_rate_series`), appends each mode's fitted growth rate (1/s) to
    its legend label -- ``gamma`` is a single physical number, not a
    function of the x-axis, so it's shown the same way regardless of
    whether ``x`` is step index or time.

    ``ylabel``, if given, overrides the default ``f"max |{variable}|"`` --
    for a caller drawing the rational-surface amplitude as the *primary*
    (not overlaid) series, where "max" is the wrong description.
    ``label_suffix`` is appended to every mode's legend label (before the
    growth-rate suffix) -- same use: a primary series that's the
    rational-surface value, not the domain-wide max, wants its own label
    without needing an overlay.

    ``caption``, if given, is drawn as a small boxed annotation in the
    lower-right corner (axes fraction, so it holds its position regardless
    of scale) -- for a figure-level summary number (e.g. peak delta-B) that
    isn't tied to any one mode's line/legend entry.

    ``title`` is empty by default: the y-label already names the quantity, so
    a title repeating it is redundant chrome on a figure that's usually
    embedded next to its own caption. Pass one explicitly to restore it --
    same convention as :func:`ashen.plotting.poincare.draw_poincare`.

    ``ylim``, if given, is a ``(min, max)`` pair applied via ``ax.set_ylim``
    -- for pinning a variable's axis to a fixed range across a comparison of
    figures, instead of matplotlib auto-scaling each one independently.

    ``vline``, if given, is an ``(x, label)`` pair drawn as a vertical dashed
    line spanning the axes, with ``label`` in the legend -- e.g. marking a
    manually-determined deconfinement time. ``x`` must already be in the
    same units as ``x`` above (seconds vs. microseconds vs. step index is the
    caller's problem, not this function's).

    ``colors``, if given, is ``{(n, m): colour}`` -- a mode it names is drawn
    in that colour (solid line and its rational-surface overlay alike), so
    the CLI can match the colours ``poincare_highlight``/``mark_rational``
    use for the same mode. Modes it doesn't name keep their sorted-index
    ``DISCRETE_PALETTE`` colour.
    """
    modes = sorted((n, m) for (var, n, m) in series if var == variable)

    for i, (n, m) in enumerate(modes):
        y = series[(variable, n, m)]
        color = DISCRETE_PALETTE[i % len(DISCRETE_PALETTE)]
        if colors is not None and (n, m) in colors:
            color = colors[(n, m)]
        key = (variable, n, m)
        label = f"n={n}, m={m}{label_suffix}"
        if growth_fits is not None and key in growth_fits:
            label += f" (\N{GREEK SMALL LETTER GAMMA}={growth_fits[key].gamma:.3g} /s)"
        ax.plot(x, y, color=color, marker="o", markersize=4, label=label)

        if rational_series is not None and key in rational_series and n != 0:
            ax.plot(
                x, rational_series[key], color=color, linestyle="--", marker="o",
                markersize=4, alpha=0.6, label=f"n={n}, m={m} @ q={m / n:g} surface",
            )

    if vline is not None:
        vx, vlabel = vline
        ax.axvline(vx, color="blue", linestyle="--", linewidth=1.5, label=vlabel)

    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel or f"max |{variable}|")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if modes or vline is not None:
        ax.legend()
    if title:
        ax.set_title(title)
    if grid:
        # Same weight as ashen.plotting.wetted_fraction's -- readable against
        # a log y-axis's minor ticks without competing with the data lines.
        ax.grid(True, linestyle=":", alpha=0.4)

    if caption:
        ax.text(
            0.98, 0.02, caption, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.7"),
        )


def plot_mode_amplitudes(
    x: Sequence[float],
    series: Mapping[ModeKey, "object"],
    variable: str,
    out_path: Path | str,
    *,
    rational_series: Mapping[ModeKey, "object"] | None = None,
    growth_fits: Mapping[ModeKey, GrowthFit] | None = None,
    log: bool = True,
    xlabel: str = "",
    ylabel: str | None = None,
    label_suffix: str = "",
    caption: str | None = None,
    title: str = "",
    grid: bool = True,
    ylim: tuple[float, float] | None = None,
    vline: tuple[float, str] | None = None,
    colors: Mapping[tuple[int, int], str] | None = None,
    figsize: tuple[float, float] = (7, 5),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save one variable's mode-amplitude time series."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_mode_amplitudes(
            ax, x, series, variable=variable, rational_series=rational_series,
            growth_fits=growth_fits, log=log, xlabel=xlabel, ylabel=ylabel,
            label_suffix=label_suffix, caption=caption, title=title, grid=grid,
            ylim=ylim, vline=vline, colors=colors,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def mode_panel_label(key: ModeKey) -> str:
    """A mode's panel title -- ``n=2, m=3``. Drops the variable, which
    ``plot_mode_radial`` already carries on the y-axis and in the filename."""
    _, n, m = key
    return f"n={n}, m={m}"


def plot_mode_radial(
    series: Mapping[ModeKey, Mapping[int, "object"]],
    variable: str,
    out_path: Path | str,
    *,
    color_by: Mapping[int, float] | None = None,
    color_label: str = "Time step",
    xlabel: str = r"$\psi_N$",
    ylabel: str | None = None,
    rational_lines: list[tuple[float, str, str]] | None = None,
    rational_bands: Sequence[RationalBand] | None = None,
    log: bool = True,
    ylim: tuple[float, float] | None = None,
    cmap: str = "turbo",
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save one variable's radial mode eigenfunctions: one panel per
    ``(n, m)``, one colour-graded line per restart step.

    A thin adapter over :func:`ashen.plotting.profiles.plot_profile_comparison`
    rather than a drawing routine of its own -- that function already takes
    exactly this shape ({panel: {step: (x, y)}}), grades lines by step across
    one shared colourbar, and draws ``rational_lines``. The only translation
    needed is ModeKey -> panel label.

    Panels are ordered by ``(n, m)``, so a mode keeps its position across
    figures and re-runs instead of following dict order -- the same stability
    :func:`draw_mode_amplitudes` gives its colours.

    ``series`` values are {step: (psi_n, abs)} as
    :func:`ashen.diagnostics.four_modes.radial_amplitude_series` returns them;
    a step absent from a mode's inner dict simply isn't drawn in that panel.
    """
    from ashen.plotting.profiles import plot_profile_comparison

    ordered = sorted(series, key=lambda key: (key[1], key[2]))
    series_by_panel = {mode_panel_label(key): series[key] for key in ordered}

    return plot_profile_comparison(
        series_by_panel,
        ylabel if ylabel is not None else f"|{variable}| amplitude",
        out_path,
        color_by=color_by,
        color_label=color_label,
        xlabel=xlabel,
        rational_lines=rational_lines,
        rational_bands=rational_bands,
        cmap=cmap,
        ylim=ylim,
        logy=log,
    )
