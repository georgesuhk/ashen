"""CASTOR3D -> JOREK profile translation: psi, ffprime, T, and resampling.

Ports, from ``castor3d/util/run_jorek_util.py``: ``get_psi_from_castor:206``,
``get_ffprime_prof_from_castor:238``, ``get_T_prof_from_castor:257``,
``create_profiles_from_castor:281``; and from ``castor3d/util/data.py``:
``resample_profile:17``, ``smooth_profile_preserve_center:76``.

**Faithfully replicated, not yet fixed -- read before changing anything here.**

``get_t_profile_from_castor`` recomputes its own psi grid from file rather
than using the caller's, so with an extended boundary T sits on the
*unextended* grid while ffprime sits on the extended one. This is almost
certainly a bug, but fixing it changes numerical output for every
extended-boundary run already published. **Do not fix without confirming with
George first** -- see the "T profile grid" item in the refactor plan. This
module intentionally reproduces the old behaviour so golden tests can compare
against it; the fix, once approved, lands as its own isolated commit.

The ``jpol[1::]`` slice in ``get_ffprime_profile_from_castor`` was flagged in
an earlier pass as a length-mismatch bug; verified false against the real
`qa2.1_g2.300` fixtures (both arrays are length 1000 after slicing). The
`xn_h*` vs `xn_f*` naming is the standard VMEC/NEMEC half-mesh/full-mesh
convention, so the slice plausibly aligns two grids on purpose. Preserved
as-is.

**Suffix threaded explicitly** rather than hardcoded, same as `boundary.py`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from ashen.castor_io import load_two_col_data
from ashen.physics import E_CHARGE, MU_0

__all__ = [
    "CastorPsi",
    "create_profile_from_castor",
    "get_ffprime_profile_from_castor",
    "get_psi_from_castor",
    "get_t_profile_from_castor",
    "resample_profile",
    "smooth_profile_preserve_center",
]


class CastorPsi(NamedTuple):
    psi: np.ndarray
    psi_n: np.ndarray
    psi_edge: float


def get_psi_from_castor(cotrans_dir: str, suffix: str) -> CastorPsi:
    """Read the psi grid CASTOR3D computed for one (qa, g) point."""
    psi = np.abs(load_two_col_data(f"{cotrans_dir}/xn_fpol_stor0_{suffix}")[:, 1])
    psi_edge = psi[-1]
    return CastorPsi(psi=psi, psi_n=psi / psi_edge, psi_edge=psi_edge)


def smooth_profile_preserve_center(
    x: np.ndarray, y: np.ndarray, sigma: float = 3, preserve_width: float = 0.05
) -> np.ndarray:
    """Gaussian-smooth ``y(x)``, leaving the region near ``x=0`` untouched."""
    y = np.asarray(y, dtype=float)
    y_smooth = gaussian_filter1d(y, sigma=sigma, mode="nearest")
    mask = np.abs(x) < preserve_width
    y_smooth[mask] = y[mask]
    return y_smooth


def resample_profile(
    x: np.ndarray,
    y: np.ndarray,
    n_points: int | None = None,
    x_new: np.ndarray | None = None,
    kind: str = "cubic",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a 1D profile onto a new grid (default: evenly spaced)."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x_new is None:
        if n_points is None:
            raise ValueError("Either n_points or x_new must be provided")
        x_new = np.linspace(x.min(), x.max(), n_points)
    f = interp1d(x, y, kind=kind, fill_value="extrapolate")
    return x_new, f(x_new)


def get_ffprime_profile_from_castor(
    psi: np.ndarray, cotrans_dir: str, suffix: str, sigma: float = 10
) -> np.ndarray:
    """The ff' profile JOREK's Grad-Shafranov solver needs.

    ``psi_n`` is not an argument here (unlike the old
    ``get_ffprime_prof_from_castor(psi_n, psi, ...)``) because it was only
    ever used for the smoothing call below, computed as ``psi / psi[-1]``;
    passing ``psi`` alone removes the chance of it disagreeing with the grid
    ``psi`` describes -- a tighter version of the "wrong grid" problem
    documented on `get_t_profile_from_castor` below, fixed here because it
    does not change any numerical output (it is the exact same computation,
    just with one fewer redundant argument).
    """
    jpol = load_two_col_data(f"{cotrans_dir}/xn_hjpol_stor0_{suffix}")[:, 1][1:]
    Ipol = jpol

    dIpol_dpsi = np.gradient(Ipol, psi)
    ffprime = (MU_0 / (2 * np.pi)) ** 2 * Ipol * dIpol_dpsi
    ffprime_jorek = ffprime * 2 * np.pi

    psi_n = psi / psi[-1]
    return smooth_profile_preserve_center(psi_n, ffprime_jorek, sigma=sigma, preserve_width=0)


def get_t_profile_from_castor(
    rho_profile: np.ndarray, cotrans_dir: str, suffix: str, sigma: float = 5
) -> np.ndarray:
    """The temperature profile JOREK needs.

    KNOWN GRID BUG, PRESERVED INTENTIONALLY: this recomputes its own psi grid
    from ``xn_fpol_stor0_{suffix}`` rather than accepting the caller's,
    exactly as the old ``get_T_prof_from_castor`` did. When the boundary is
    extended, this means T ends up on the unextended grid while ffprime (which
    does take the caller's ``psi``, see above) sits on the extended one. Do
    not fix without confirming with George -- it changes published numbers.

    SEPARATE FINDING, ALSO UNCONFIRMED: ``rho_profile`` has **no effect on the
    output**. ``T_eV = pres / (rho*e)`` followed by
    ``T_jorek = T_eV * (rho*e*MU_0)`` cancels ``rho`` and ``e`` exactly, so
    ``T_jorek`` reduces algebraically to ``pres * MU_0`` (plus the near-zero
    floor) regardless of what density is passed in. Verified numerically
    against the real `qa2.1_g2.300` fixture across several very different
    ``rho_profile`` values (scalar and full-length array) -- output was
    bit-for-bit identical every time. This is preserved as-is because it is
    not clear whether it is a bug (rho was meant to matter and doesn't due to
    a formula error) or intentional (the "T" JOREK receives may be a
    normalised quantity where this cancellation is by design). Ask George.
    """
    rho_profile = np.asarray(rho_profile, dtype=float)
    psi = np.abs(load_two_col_data(f"{cotrans_dir}/xn_fpol_stor0_{suffix}")[:, 1])
    psi_edge = psi[-1]
    psi_n = psi / psi_edge  # NOT the caller's grid -- see docstring.

    pres = load_two_col_data(f"{cotrans_dir}/xn_hpres_stor0_{suffix}")[:, 1][1:]

    if len(rho_profile) == 1:
        rho_profile = np.full(len(pres), rho_profile[0])

    T_eV = pres / (rho_profile * E_CHARGE)
    T_jorek = T_eV * (rho_profile * E_CHARGE * MU_0)

    atol = 1e-8
    T_jorek[np.isclose(T_jorek, 0.0, atol=atol)] = 1e-8

    return smooth_profile_preserve_center(psi_n, T_jorek, sigma=sigma, preserve_width=0.05)


def create_profile_from_castor(
    prof_name: str,
    psi: np.ndarray,
    cotrans_dir: str,
    suffix: str,
    rho_profile: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to the right profile builder by name.

    Unlike the old ``create_profiles_from_castor``, an unknown ``prof_name``
    raises immediately rather than returning an unbound variable.
    """
    if prof_name == "ffprime":
        return get_ffprime_profile_from_castor(psi, cotrans_dir, suffix)
    if prof_name == "T":
        if rho_profile is None:
            raise ValueError("rho_profile cannot be None for the T profile")
        return get_t_profile_from_castor(rho_profile, cotrans_dir, suffix)
    raise ValueError(f"unknown profile {prof_name!r}; expected 'ffprime' or 'T'")
