"""Poincare puncture plots.

Ports ``castor3d/util/data_jorek.py:354 plot_poincare``, redrawn against the
per-line cache from Phase 4b (:mod:`ashen.diagnostics.poincare_cache`) instead
of the three dense ``.npz`` files it used to load. The legacy version indexed
colour by scan position (``contrast_colors[i]``); here colour comes from each
line's actual ``psi_n`` via :mod:`ashen.plotting.colors`, which survives a
scan being extended or widened without needing re-indexing.

``draw_poincare`` takes an ``ax`` and draws onto it -- unlike every legacy
plotting function, none of which accepted one (``castor3d/util/data_jorek.py``
inventory: zero `ax=` parameters anywhere). That is what lets a case's
punctures be composed into a grid, or reused from a notebook, instead of each
call owning its own figure and file.
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
    highlight: set[float] | None = None,
    title: str | None = None,
) -> None:
    """Scatter every cached puncture of ``records`` onto ``ax``, coloured by
    each line's starting ``psi_n``.

    ``highlight`` is a set of ``psi_n`` values to draw at full opacity with
    everything else dimmed to light grey -- ports the legacy ``highlight`` /
    ``highlight_idx`` pair (``data_jorek.py:354,376-380``), now selecting by
    the physical quantity instead of a scan index that no longer exists.
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
            color = color_of(psi_n) if psi_n in highlight else "lightgray"
            point_alpha = alpha if psi_n in highlight else 0.1
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
