"""Poloidal-angle distribution of field-line exit crossings.

Ports the data-reduction half of the notebook function
plot_theta_histogram_matrix (prod_plots_draft0.ipynb, cell 5): for each
traced field line, find the first puncture whose flux surface exceeds a
target psi_n and record the poloidal angle theta there -- where the line
first crosses out past a chosen surface. Histogramming those angles across
many lines gives a strike-position/loss-angle distribution.

No matplotlib import, matching diagnostics.connection_length -- usable
from a notebook/test without a plotting backend.

real_psi_edge is applied exactly once, to the threshold, not the data.
The notebook computed psi_n_ij/psi_edge > target_psi, algebraically
identical to psi_n_ij > target_psi*real_psi_edge. Since target_psi is a
user-facing plasma-fraction quantity and record.psi_n is already
JOREK-grid (KNOWN_ISSUES.md #7), scaling the threshold is the
correct-by-construction form of "apply real_psi_edge exactly once" --
unlike connection_length.line_connection_length's legacy double-division,
which this deliberately does NOT replicate. Nothing to fix here.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ashen.diagnostics.poincare_cache import LineKey, LineRecord

__all__ = [
    "CrossingResult",
    "crossing_angles",
    "pooled_crossing_angles",
    "theta_histogram",
    "wetted_fraction",
]


class CrossingResult:
    """Angles at first exit crossing, plus how many lines were considered.
    The notebook silently discarded the crossed/considered counts -- this
    is the fastest way to spot a target_psi set so high (or psi_n_range
    so narrow) that the histogram has little or no data behind it.
    """

    __slots__ = ("angles", "n_crossed", "n_considered")

    def __init__(self, angles: np.ndarray, n_crossed: int, n_considered: int) -> None:
        self.angles = angles
        self.n_crossed = n_crossed
        self.n_considered = n_considered


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap into ``(-pi, pi]``, matching the notebook's ``a %= 2*pi; a[a >
    pi] -= 2*pi`` exactly."""
    wrapped = np.mod(angles, 2 * np.pi)
    wrapped[wrapped > np.pi] -= 2 * np.pi
    return wrapped


def crossing_angles(
    records: Mapping[LineKey, LineRecord],
    *,
    target_psi: float,
    real_psi_edge: float,
    psi_n_range: tuple[float, float] | None = None,
) -> CrossingResult:
    """One step's crossing angles.

    psi_n_range (if given) is a [min, max] filter on each line's starting
    psi_n (LineKey.psi_n) -- user-facing plasma-fraction bounds, scaled by
    real_psi_edge the same way target_psi is, since LineKey.psi_n is also
    JOREK-grid. Replaces the notebook's i_lim (a positional index into
    scan order, silently changing meaning whenever psi_n_in is widened or
    reordered -- the exact hazard the per-line cache was keyed by starting
    position to remove).

    A line whose punctures never cross the threshold (confined)
    contributes no angle but is still counted in n_considered. A line's
    own starting psi_n can never already exceed the threshold:
    jorek2_postproc's fluxsurface command rejects any request outside
    [0, 1] (exec_commands.f90:3063-3068), so every start is
    <= real_psi_edge while the threshold is target_psi*real_psi_edge with
    target_psi > 1 -- no "already past threshold at start" case to guard.
    """
    threshold = target_psi * real_psi_edge

    range_lo = range_hi = None
    if psi_n_range is not None:
        lo, hi = psi_n_range
        range_lo, range_hi = lo * real_psi_edge, hi * real_psi_edge

    angles: list[float] = []
    n_considered = 0
    for key, record in records.items():
        if range_lo is not None and not (range_lo <= key.psi_n <= range_hi):
            continue
        n_considered += 1

        psi_n = record.psi_n
        mask = psi_n > threshold
        idx = np.argmax(mask) if np.any(mask) else None
        if idx is not None:
            angles.append(float(record.theta[idx]))

    result = np.asarray(angles, dtype=np.float64)
    if result.size:
        result = _wrap_to_pi(result)
    return CrossingResult(angles=result, n_crossed=result.size, n_considered=n_considered)


def pooled_crossing_angles(
    records_by_step: Mapping[int, Mapping[LineKey, LineRecord]],
    steps: Sequence[int],
    *,
    target_psi: float,
    real_psi_edge: float,
    psi_n_range: tuple[float, float] | None = None,
) -> CrossingResult:
    """Concatenate :func:`crossing_angles` over several steps into one panel
    -- what a comparison's per-case panel, or a multi-step per-case panel,
    pools."""
    all_angles: list[np.ndarray] = []
    n_crossed = 0
    n_considered = 0
    for step in steps:
        result = crossing_angles(
            records_by_step.get(step, {}),
            target_psi=target_psi, real_psi_edge=real_psi_edge, psi_n_range=psi_n_range,
        )
        all_angles.append(result.angles)
        n_crossed += result.n_crossed
        n_considered += result.n_considered

    angles = np.concatenate(all_angles) if all_angles else np.empty(0, dtype=np.float64)
    return CrossingResult(angles=angles, n_crossed=n_crossed, n_considered=n_considered)


def theta_histogram(angles: np.ndarray, *, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin angles over (-pi, pi), weighted to the fraction of lines per
    bin -- ports the notebook's weights = np.ones_like(data)/len(data)
    exactly (George's call: keep legacy fraction normalisation; y-limit
    auto-scales at draw time instead of the notebook's hardcoded 0.06).

    Returns (counts, bin_edges), matching np.histogram's convention. An
    empty `angles` returns all-zero counts over the full range rather than
    raising, so a panel with no crossings still shares bin edges/x-axis
    with its neighbours.
    """
    if angles.size == 0:
        edges = np.linspace(-np.pi, np.pi, bins + 1)
        return np.zeros(bins, dtype=np.float64), edges

    weights = np.full(angles.shape, 1.0 / angles.size)
    counts, edges = np.histogram(angles, bins=bins, range=(-np.pi, np.pi), weights=weights)
    return counts, edges


def wetted_fraction(counts: np.ndarray, *, threshold: float) -> float:
    """Fraction of histogram bins whose count exceeds `threshold` -- ports
    the notebook's wetted_A/total_bins (prod_plots_draft0.ipynb, cell 8):
    a rough measure of how broadly (not how narrowly) an exit population
    spreads over the poloidal angle -- one scalar per case, meant to be
    plotted against a scan parameter (e.g. eta) via ashen.comparisons.

    `threshold` is on the same scale as theta_histogram's output
    (fraction-of-lines-per-bin, bin weights sum to 1) -- caller typically
    passes 1/bins, what a perfectly uniform distribution puts in every
    bin, so "wetted" = "above what uniform spreading would give this bin".
    NaN for an empty `counts` (no bins to judge), distinguishable from a
    genuine 0.0 (every bin at or below threshold).
    """
    if counts.size == 0:
        return math.nan
    return float(np.count_nonzero(counts > threshold)) / counts.size
