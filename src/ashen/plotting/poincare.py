"""Poincare puncture plots.

Ports data_jorek.py:354 plot_poincare, redrawn against the per-line cache
from Phase 4b (poincare_cache) instead of the three dense .npz files it
used to load. Legacy indexed colour by scan position
(contrast_colors[i]); here colour comes from each line's actual psi_n via
plotting.colors, surviving a scan being extended/widened without
re-indexing.

draw_poincare takes an ax and draws onto it -- unlike every legacy
plotting function (data_jorek.py inventory: zero ax= parameters
anywhere). That's what lets a case's punctures be composed into a grid,
or reused from a notebook, instead of each call owning its own figure
and file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ashen.diagnostics.poincare_cache import LineKey, LineRecord
from ashen.plotting import style
from ashen.plotting.colors import colorer

__all__ = ["draw_poincare", "plot_poincare_step"]


def draw_poincare(
    ax,
    records: Mapping[LineKey, LineRecord],
    *,
    s: float = 0.1,
    alpha: float = 1.0,
    cmap: str = "viridis",
    discrete: bool = False,
    highlight: Mapping[float, str] | None = None,
    title: str | None = None,
) -> None:
    """Scatter every cached puncture of records onto ax, coloured by each
    line's starting psi_n.

    highlight maps a psi_n value to the colour it draws at full opacity,
    everything else dimmed to light grey -- ports the legacy highlight/
    highlight_idx pair (data_jorek.py:354,376-380), extended so each
    highlighted surface can carry its own colour (e.g. one per rational
    surface) instead of a single shared one.
    """
    grouped: dict[float, list[LineRecord]] = {}
    for key, record in records.items():
        grouped.setdefault(key.psi_n, []).append(record)

    color_of = colorer(grouped.keys(), cmap=cmap, discrete=discrete)

    for psi_n in sorted(grouped, reverse=True):
        group = grouped[psi_n]
        R = np.concatenate([r.R for r in group]) if group else np.empty(0)
        Z = np.concatenate([r.Z for r in group]) if group else np.empty(0)
        if R.size == 0:
            continue
        if highlight:
            hl_color = highlight.get(psi_n)
            color = hl_color if hl_color is not None else "lightgray"
            point_alpha = alpha if hl_color is not None else 0.1
        else:
            color = color_of(psi_n)
            point_alpha = alpha
        ax.scatter(R, Z, s=s, color=color, alpha=point_alpha)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$Z$")
    if title:
        ax.set_title(title)


def plot_poincare_step(
    records: Mapping[LineKey, LineRecord],
    out_path: Path | str,
    *,
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 200,
    title: str | None = None,
    **draw_kwargs,
) -> Path:
    """Draw and save one step's Poincare plot. The file-owning counterpart to
    :func:`draw_poincare`, for the CLI and for quick notebook use."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_poincare(ax, records, title=title, **draw_kwargs)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
