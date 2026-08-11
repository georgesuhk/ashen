"""Field-line connection length -- the physics behind the LC/LCTT maps.

Ports the computation trapped inside data_jorek.py's
plot_connection_length (:528) and its near-duplicate, unused,
get_connection_lengths (:491): the two disagreed on how psi_n_in was
normalised and re-derived the same harmonic mean independently. One
implementation here, and pure -- no matplotlib import, usable from a
notebook/test without a plotting backend.

Changed: R0 now read from the run's log (logfile.r_axis) instead of
hardcoded R0=1.36 at data_jorek.py:497,531 (unused duplicate at
:409,452). This DOES change every connection length's absolute value by
whatever the hardcoded constant was off by -- KNOWN_ISSUES.md #6 has the
number and George's sign-off.

Per-line n_turns is read from each poincare_cache.LineRecord rather than
passed as one scalar for the whole step (data_jorek.py:533's n_turns),
since Phase 4b's incremental cache lets lines within one step be extended
to different totals.

Preserved exactly as plot_connection_length computed it: the psi_n<1
inside/outside split, the harmonic mean over angular samples with a
zero-length line contributing 0 (not undefined), and dividing every
psi_n value (requested surfaces AND every traced point) by real_psi_edge
a SECOND time even though jorek2_poincare's own output is already
boundary-normalised (jorek2_poincare.f90's get_psi_n:
psi_n=(psi-psi_axis)/(psi_bnd-psi_axis)). That second division looks like
double-normalising, but changing it changes the numbers -- left alone and
flagged, not silently resolved. KNOWN_ISSUES.md #7.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ashen.diagnostics.poincare_cache import LineKey, LineRecord

__all__ = [
    "group_by_psi_n",
    "line_connection_length",
    "harmonic_connection_length",
    "connection_lengths_for_step",
    "connection_length_matrix",
    "smooth_ignoring_inf",
]


def group_by_psi_n(
    records: Mapping[LineKey, LineRecord]
) -> dict[float, list[LineRecord]]:
    """Bucket a step's cache by the flux surface each line started from.
    Replaces the legacy positional index i (psi_n_out[i]) -- the new cache
    has no such index, keyed by starting position, not scan order.
    """
    grouped: dict[float, list[LineRecord]] = {}
    for key, record in records.items():
        grouped.setdefault(key.psi_n, []).append(record)
    return grouped


def line_connection_length(record: LineRecord, R0: float, *, real_psi_edge: float) -> float:
    """One field line's connection length. Ports ``calc_single_connection_length``
    (``data_jorek.py:688``).

    A line that never left the ``psi_n < 1`` region for its whole traced
    length (confined) returns ``inf``, matching the legacy convention.
    """
    psi_n = record.psi_n / real_psi_edge
    inside = psi_n[psi_n < 1]
    if len(inside) == record.n_turns:
        return math.inf
    return len(inside) * 2.0 * math.pi * R0


def harmonic_connection_length(
    records: Sequence[LineRecord], R0: float, *, real_psi_edge: float
) -> float:
    """The connection length reported for one flux surface: harmonic mean
    of its angular samples' individual lengths. Ports
    calc_connection_length_2 (data_jorek.py:670).

    A zero-length sample (left the mesh immediately) contributes 0 to the
    sum, not a divide-by-zero -- effectively excluded, as legacy did
    explicitly. An empty group (requested surface with no cached samples)
    returns nan, not inf, so it's visibly missing, not indistinguishable
    from "every sample confined".
    """
    if not records:
        return math.nan
    lengths = [line_connection_length(r, R0, real_psi_edge=real_psi_edge) for r in records]
    denom = 0.0
    for length in lengths:
        if length != 0:
            denom += 1.0 / length  # 1/inf == 0.0, no special-casing needed
    if denom == 0:
        return math.inf
    return len(lengths) / denom


def connection_lengths_for_step(
    records: Mapping[LineKey, LineRecord],
    psi_n_targets: Sequence[float],
    *,
    real_psi_edge: float,
    R0: float,
) -> np.ndarray:
    """One row of the LC/LCTT matrix: connection length per requested
    surface, for one step's cache.

    psi_n_targets are the SCALED values lines were traced at (what
    diagnostics.poincare calls with -- case.psi_n_in[k] * real_psi_edge,
    matching LineKey.psi_n), not the raw cases.toml fractions.
    """
    # Re-key by the same quantisation LineKey applies, so a target computed
    # as case.psi_n_in[k] * real_psi_edge matches a stored key even if the
    # two float computations differ in the last bit.
    grouped = {
        LineKey(psi_n=psi_n, R=0.0, Z=0.0, phi=0.0).quantised().psi_n: group
        for psi_n, group in group_by_psi_n(records).items()
    }
    out = np.full(len(psi_n_targets), np.nan)
    for i, target in enumerate(psi_n_targets):
        target_q = LineKey(psi_n=target, R=0.0, Z=0.0, phi=0.0).quantised().psi_n
        matches = grouped.get(target_q)
        if matches:
            out[i] = harmonic_connection_length(matches, R0, real_psi_edge=real_psi_edge)
    return out


def connection_length_matrix(
    records_by_step: Mapping[int, Mapping[LineKey, LineRecord]],
    steps: Sequence[int],
    psi_n_targets: Sequence[float],
    *,
    real_psi_edge: float,
    R0: float,
) -> np.ndarray:
    """The full (n_steps, n_psi) matrix color_con_length_plot draws.
    `steps` fixes row order explicitly rather than relying on dict
    iteration order, since records_by_step may come from an unordered
    gather.
    """
    matrix = np.empty((len(steps), len(psi_n_targets)))
    for row, step in enumerate(steps):
        matrix[row] = connection_lengths_for_step(
            records_by_step.get(step, {}), psi_n_targets,
            real_psi_edge=real_psi_edge, R0=R0,
        )
    return matrix


def smooth_ignoring_inf(Z: np.ndarray, window: int = 3) -> np.ndarray:
    """Time-axis moving average treating inf (a confined line) as missing
    rather than letting it swamp the average.

    Ports the inline smoother in color_con_length_plot
    (data_jorek.py:611-639): averages only finite values in each window,
    then restores inf at cells that were inf to begin with. Z is
    (n_psi, n_steps) -- smoothing runs along the time axis (columns) only,
    matching the legacy (1, window) kernel.
    """
    from scipy.signal import convolve2d

    valid = ~np.isinf(Z)
    Z_calc = np.where(valid, Z, 0.0)
    kernel = np.ones((1, window))

    total = convolve2d(Z_calc, kernel, mode="same", boundary="symm")
    count = convolve2d(valid.astype(float), kernel, mode="same", boundary="symm")

    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = total / count
    smoothed[~valid] = np.inf
    return smoothed
