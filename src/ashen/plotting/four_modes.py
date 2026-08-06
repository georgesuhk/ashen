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
from typing import Mapping, Sequence

from ashen.diagnostics.four_modes import GrowthFit, ModeKey
from ashen.plotting import style
from ashen.plotting.colors import DISCRETE_PALETTE

__all__ = ["draw_mode_amplitudes", "plot_mode_amplitudes"]


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
) -> None:
    """Draw every ``(n, m)`` mode of ``variable`` present in ``series`` onto
    ``ax``, each a differently-coloured line.

    Modes are sorted by ``(n, m)`` before colour assignment, so the same mode
    gets the same colour and legend position across figures/re-runs rather
    than depending on dict iteration order.

    ``rational_series``, if given (see :func:`ashen.diagnostics.four_modes.
    rational_surface_series`), overlays each mode's amplitude pinned to its
    ``q = m/n`` resonant surface as a dashed line in the same colour --
    directly comparable to the solid whole-domain-max line for the same
    mode, so a reader can see whether the mode's growth is actually
    localised at the surface it resonates on.

    ``growth_fits``, if given (see :func:`ashen.diagnostics.four_modes.
    growth_rate_series`), appends each mode's fitted growth rate (1/s) to
    its legend label -- ``gamma`` is a single physical number, not a
    function of the x-axis, so it's shown the same way regardless of
    whether ``x`` is step index or time.
    """
    modes = sorted((n, m) for (var, n, m) in series if var == variable)

    for i, (n, m) in enumerate(modes):
        y = series[(variable, n, m)]
        color = DISCRETE_PALETTE[i % len(DISCRETE_PALETTE)]
        key = (variable, n, m)
        label = f"n={n}, m={m}"
        if growth_fits is not None and key in growth_fits:
            label += f" (\N{GREEK SMALL LETTER GAMMA}={growth_fits[key].gamma:.3g} /s)"
        ax.plot(x, y, color=color, marker="o", markersize=4, label=label)

        if rational_series is not None and key in rational_series and n != 0:
            ax.plot(
                x, rational_series[key], color=color, linestyle="--", marker="o",
                markersize=4, alpha=0.6, label=f"n={n}, m={m} @ q={m / n:g} surface",
            )

    if log:
        ax.set_yscale("log")
    ax.set_ylabel(f"max |{variable}|")
    if xlabel:
        ax.set_xlabel(xlabel)
    if modes:
        ax.legend()
    ax.set_title(variable)


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
    figsize: tuple[float, float] = (7, 5),
    dpi: int = 150,
) -> Path:
    """Draw and save one variable's mode-amplitude time series."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_mode_amplitudes(
            ax, x, series, variable=variable, rational_series=rational_series,
            growth_fits=growth_fits, log=log, xlabel=xlabel,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
