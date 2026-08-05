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

from ashen.diagnostics.four_modes import ModeKey
from ashen.plotting import style
from ashen.plotting.colors import DISCRETE_PALETTE

__all__ = ["draw_mode_amplitudes", "plot_mode_amplitudes"]


def draw_mode_amplitudes(
    ax,
    x: Sequence[float],
    series: Mapping[ModeKey, "object"],
    *,
    variable: str,
    log: bool = True,
    xlabel: str = "",
) -> None:
    """Draw every ``(n, m)`` mode of ``variable`` present in ``series`` onto
    ``ax``, each a differently-coloured line.

    Modes are sorted by ``(n, m)`` before colour assignment, so the same mode
    gets the same colour and legend position across figures/re-runs rather
    than depending on dict iteration order.
    """
    modes = sorted((n, m) for (var, n, m) in series if var == variable)

    for i, (n, m) in enumerate(modes):
        y = series[(variable, n, m)]
        color = DISCRETE_PALETTE[i % len(DISCRETE_PALETTE)]
        ax.plot(x, y, color=color, label=f"n={n}, m={m}")

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
        draw_mode_amplitudes(ax, x, series, variable=variable, log=log, xlabel=xlabel)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
