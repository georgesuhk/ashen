"""Grid of poloidal-angle exit-crossing histograms.

Ports the drawing half of the notebook function ``plot_theta_histogram_matrix``
(``Columbia/NL_kinks/prod_plots_draft0.ipynb``, cell 5) -- the physics moved to
:mod:`ashen.diagnostics.theta_histogram`, which has no matplotlib import; this
module is drawing only, mirroring the ``draw_*``/``plot_*`` split every other
module in this package uses (see ``plotting/four_modes.py`` docstring).

Kept from the notebook: the grid layout itself (a shared x/y appearance,
x-ticks only at ``[-pi, 0, pi]`` with the first/last inward-aligned so they
don't overhang, y-label only on the left column, a shared ``supxlabel``).

Changed:
- ``sharex``/``sharey`` are applied by hand (same ``xlim``/``ylim`` set on
  every axis) instead of via ``plt.subplots(sharex=True, sharey=True)``. A
  true shared axis suppresses tick *labels* on non-bottom-row members keyed
  off grid position, not "last panel actually present in this column" -- for
  a partially-filled last row (e.g. 5 panels over 3 columns) that leaves one
  column's real bottom axis one row up with its labels silently empty
  (``set_xticklabels`` on it returns ``[]``, not the 3 requested). The bottom
  axis of each column is computed explicitly here instead.
- No import-time or call-time ``rcParams`` mutation -- drawn inside
  :func:`ashen.plotting.style`, which deliberately keeps ``text.usetex=False``
  (an HPC/Windows-clone LaTeX install is not assumed) with
  ``mathtext.fontset="cm"``, so ``$\\theta$``/``$\\eta$``-style labels still
  render.
- The y-limit is auto-scaled from the tallest bin across every panel instead
  of the notebook's hardcoded ``0.06`` -- George's call: panels stay
  comparable to each other but the limit no longer needs retuning by hand
  when ``bins`` or the underlying data changes. ``y_max`` can still be pinned
  explicitly to reproduce an older figure.
- The ``show_threshold``/``threshold_percentile`` horizontal-line overlay and
  its ``counts_compare`` companion are dropped, not ported (George, this
  session) -- not carried forward at all.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from ashen.diagnostics.theta_histogram import theta_histogram
from ashen.plotting import style

__all__ = ["draw_theta_histogram", "plot_theta_histogram_grid"]


def draw_theta_histogram(ax, angles: np.ndarray, *, bins: int) -> np.ndarray:
    """Draw one panel's histogram onto ``ax``. Returns the bin counts, so a
    caller can compute a shared y-limit across several panels before drawing
    any of them.

    Draws a centred "No data" note instead of an empty axes when ``angles``
    is empty -- distinguishable from a genuinely flat/empty distribution
    drawn at zero height.
    """
    counts, _ = theta_histogram(angles, bins=bins)
    if angles.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=12)
        return counts

    ax.hist(
        angles, bins=bins, range=(-np.pi, np.pi), weights=np.full(angles.shape, 1.0 / angles.size),
        color="tab:blue", edgecolor="tab:blue", alpha=0.7, linewidth=0.5,
    )
    return counts


def plot_theta_histogram_grid(
    panels: Sequence[tuple[str, np.ndarray]],
    out_path: Path | str,
    *,
    bins: int = 500,
    n_cols: int = 4,
    figsize_per_panel: tuple[float, float] = (1.8, 1.8),
    y_max: float | None = None,
    dpi: int = 200,
) -> Path:
    """Draw and save a grid of theta-crossing histograms, one panel per
    ``(label, angles)`` pair in ``panels`` -- the file-owning counterpart to
    :func:`draw_theta_histogram`.

    ``panels`` order fixes grid order (row-major). What each panel represents
    is the caller's choice: one panel per step for a single case (``ashen.cli.
    plot``'s per-case mode), or one panel per case for a cross-case comparison
    (``ashen.cli.plot``'s ``--compare`` mode) -- this function draws the grid
    either way, agnostic to which.
    """
    import math

    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(panels)
    n_rows = math.ceil(n_panels / n_cols) if n_panels else 1
    figsize = (figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows)

    # The bottom-of-column axis for column c is the last visible (idx <
    # n_panels) axis with idx % n_cols == c -- NOT simply the last row of the
    # n_rows x n_cols grid, since a partially-filled last row leaves some
    # columns' true bottom in the row above. sharex/sharey are deliberately
    # off (unlike every other grid in this package): matplotlib suppresses
    # tick labels on a shared axis's non-bottom-row members via its own
    # `label_outer`-style bookkeeping, keyed off grid position rather than
    # "last panel present in this column" -- calling set_xticklabels on one
    # of those returns an empty list, not the 3 labels requested. x/y limits
    # are instead applied uniformly by hand below, which is what sharex/
    # sharey would have done anyway.
    bottom_of_column = {}
    for idx in range(n_panels):
        bottom_of_column[idx % n_cols] = idx

    with style():
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize,
            gridspec_kw={"wspace": 0.1, "hspace": 0.45}, squeeze=False,
        )
        axes = axes.flatten()

        all_counts = []
        for idx, ax in enumerate(axes):
            if idx >= n_panels:
                ax.set_visible(False)
                continue
            label, angles = panels[idx]
            counts = draw_theta_histogram(ax, angles, bins=bins)
            all_counts.append(counts)

            ax.set_title(label, loc="left", fontsize=10, pad=5)
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks([-np.pi, 0, np.pi])

            if bottom_of_column.get(idx % n_cols) == idx:
                labels = ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
                labels[0].set_ha("left")
                labels[-1].set_ha("right")
            else:
                ax.tick_params(labelbottom=False)

            if idx % n_cols == 0:
                ax.set_ylabel("# field lines [a.u.]", labelpad=12)
            else:
                ax.tick_params(labelleft=False)

        limit = y_max
        if limit is None:
            limit = max((float(np.max(c)) for c in all_counts if c.size), default=0.0)
            limit = limit * 1.1 if limit > 0 else 1.0
        for ax in axes:
            if ax.get_visible():
                ax.set_ylim(0, limit)

        fig.supxlabel(r"$\theta$ [rad]")
        # tight_layout warns when the grid has a hidden trailing axis (an
        # unfilled last row) -- cosmetic only, the layout it produces is
        # fine; every other module in this package never hides an axis, so
        # this warning is unique to this one.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This figure includes Axes")
            fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
