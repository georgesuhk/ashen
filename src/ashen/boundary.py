"""Plasma boundary geometry and psi-grid extension.

Ports, from ``castor3d/util/run_jorek_util.py``: ``downsample_boundary:106``,
``extend_psi:214``, ``extend_prof:232``, ``get_bnd_from_castor:293``; and from
``castor3d/util/data.py``: ``expand_boundary:54``.

**Extension, not just the boundary.** ``extend_psi``/``extend_prof`` live here
rather than in :mod:`ashen.profiles` because the psi grid they build is shared
by the boundary write (the `psi_boundary` values) and every profile -- the
boundary and the extended grid are one design, not two.

**Suffix threaded explicitly.** The old code hardcoded the machine suffix
``"DIIID"`` into five filenames across this module and `profiles.py`. Every
function here takes ``suffix`` as a required argument instead.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit

from ashen.castor_io import load_two_col_data

__all__ = [
    "ExtendedPsi",
    "expand_boundary",
    "extend_prof",
    "extend_psi",
    "downsample_boundary",
    "read_boundary_from_castor",
]


def expand_boundary(
    bnd: np.ndarray, R0: float, Z0: float, scale: float
) -> np.ndarray:
    """Radially scale boundary points ``(R, Z)`` about a center."""
    center = np.array([R0, Z0])
    return center + scale * (np.asarray(bnd) - center)


def downsample_boundary(
    points: np.ndarray, n_points: int, closed: bool = True
) -> np.ndarray:
    """Arc-length resample a 2D boundary to ``n_points``."""
    pts = np.asarray(points, dtype=float)
    if closed:
        pts = np.vstack([pts, pts[0]])

    d = np.diff(pts, axis=0)
    ds = np.linalg.norm(d, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    length = s[-1]

    s_new = np.linspace(0.0, length, n_points + int(closed))
    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])
    out = np.column_stack((x_new, y_new))

    return out[:-1] if closed else out


def _saturating_fit(index: np.ndarray, amplitude: float, tau: float) -> np.ndarray:
    """``A * (1 - exp(-index / tau))`` -- the shape psi(index) is fit to."""
    return amplitude * (1 - np.exp(-index / tau))


def _inverse_saturating_fit(value: float, amplitude: float, tau: float) -> float:
    return -tau * np.log(1 - value / amplitude)


class ExtendedPsi(NamedTuple):
    """The extended psi grid, plus what's needed to extend anything onto it."""

    psi: np.ndarray
    psi_n: np.ndarray
    extended_idx_range: np.ndarray
    real_psi_edge: float


def extend_psi(psi: np.ndarray, extend_ratio: float, extend_reso: int) -> ExtendedPsi:
    """Extend a psi grid past the plasma edge by fitting its saturating shape.

    JOREK's free-boundary solver wants vacuum region beyond the last closed
    flux surface. This fits ``psi(index)`` as a saturating exponential, then
    extrapolates ``extend_reso`` points out to where the fit reaches
    ``psi[-1] * extend_ratio``.

    ``real_psi_edge`` is the ratio of the true plasma edge to the new
    (extended) edge -- callers rescale physical ``psi_n`` values against it
    (e.g. `Columbia/NL_kinks/analysis.py:143`,
    `psi_n_in_adjusted = psi_n_in * real_psi_edge`).
    """
    psi = np.asarray(psi, dtype=float)
    psi_target = psi[-1] * extend_ratio

    (amplitude, tau), _ = curve_fit(
        _saturating_fit, np.arange(len(psi)), psi, p0=[1, 200]
    )

    idx_max = int(np.round(_inverse_saturating_fit(psi_target, amplitude, tau)))
    extended_idx_range = np.linspace(len(psi), idx_max, extend_reso)

    new_indices = np.concatenate([np.arange(len(psi)), extended_idx_range])
    new_psi = _saturating_fit(new_indices, amplitude, tau)
    new_psi_n = new_psi / new_psi[-1]
    real_psi_edge = psi[-1] / new_psi[-1]

    return ExtendedPsi(new_psi, new_psi_n, extended_idx_range, real_psi_edge)


def extend_prof(prof: np.ndarray, extended_idx_range: np.ndarray) -> np.ndarray:
    """Extend a profile flat (at its edge value) onto an extended psi grid.

    Pairs with :func:`extend_psi`: ``extended_idx_range`` gives the point
    count to append, not the grid itself -- the profile has no opinion on
    where in psi those points sit, only how many there are.
    """
    prof = np.asarray(prof)
    tail = np.full(len(extended_idx_range), prof[-1])
    return np.concatenate([prof, tail])


def read_boundary_from_castor(castor_dir: str, suffix: str) -> np.ndarray:
    """Read the plasma boundary CASTOR3D wrote for one (qa, g) point.

    Returns an (N, 2) array of (R, Z). Extension/expansion is the caller's
    concern (see :func:`expand_boundary`) -- this only reads the raw boundary,
    unlike the old ``get_bnd_from_castor`` which conflated the two.
    """
    return load_two_col_data(f"{castor_dir}/xm_plasma_0_{suffix}_n1")


def boundary_center(bnd: np.ndarray) -> tuple[float, float]:
    """The center `expand_boundary` scales about.

    Faithful to the old convention: R0 is R at the point of maximum Z, Z0 is Z
    at the point of maximum R. Not a centroid -- preserved as-is since it is
    what every existing boundary was generated with.
    """
    bnd = np.asarray(bnd)
    R0 = bnd[np.argmax(bnd[:, 1]), 0]
    Z0 = bnd[np.argmax(bnd[:, 0]), 1]
    return float(R0), float(Z0)
