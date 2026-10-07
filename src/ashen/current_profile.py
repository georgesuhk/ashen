"""A toroidal current-density profile -> the FF' profile JOREK reads.

JOREK's Grad-Shafranov source is (``models/current.f90``,
``models/equilibrium.f90``)::

    Delta* psi = zj = FFprime - R^2 d(rho T)/dpsi

with psi in SI [T m^2 = Wb/rad], and from the wiki's normalization page::

    j_phi,SI [A/m^2] = -zj / (R mu_0)        p_SI = rho T / mu_0
    T_SI [K] = T / (k_B mu_0 n_0)            n_0 = central_density * 1e20

So the ``FFprime`` in ``ffprime_file`` is **minus** the textbook F dF/dpsi
(``ffprime.f90``: "JOREK uses a negative FF' in the GS-equation"), in plain
SI [T^2 m^2 / (Wb/rad) = T], against psi_N. Solving for it::

    FFprime = -mu_0 R j_phi + R^2 dp/dpsi                  (p = rho T, JOREK units)

On one flux surface FFprime is a single number while R varies, so j_phi is
not constant on a surface: without a pressure gradient it goes as 1/R. The
profile this module takes is therefore j(psi_N) = j_phi at R = R0 on each
surface, i.e. ``R j_phi / R0``. With dp/dpsi = 0 the conversion is exact;
with a pressure gradient pass ``dp_dpsi`` or accept an error of order beta.

Sign: positive j gives negative FFprime, the sign the CASTOR3D-sourced
profiles of every existing run have.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from ashen.castor_io import load_two_col_data
from ashen.physics import ELEMENTARY_CHARGE, MU_0

__all__ = [
    "current_from_ffprime",
    "extend_psi_n",
    "ffprime_from_current",
    "load_current_profile",
    "psi_n_from_rho",
    "resample_monotone",
    "temperature_to_jorek",
]


def ffprime_from_current(j, R0: float, dp_dpsi=0.0) -> np.ndarray:
    """JOREK's FFprime from j = j_phi at R = R0 [A/m^2].

    ``dp_dpsi`` is d(rho T)/dpsi in JOREK units (psi in Wb/rad, not psi_N);
    leave it 0 for a flat or negligible pressure.
    """
    return -MU_0 * R0 * np.asarray(j, dtype=float) + R0**2 * np.asarray(dp_dpsi, dtype=float)


def current_from_ffprime(ffprime, R0: float, dp_dpsi=0.0) -> np.ndarray:
    """Inverse of :func:`ffprime_from_current`: j_phi at R = R0 [A/m^2]."""
    return -(np.asarray(ffprime, dtype=float) - R0**2 * np.asarray(dp_dpsi, dtype=float)) / (MU_0 * R0)


def psi_n_from_rho(rho, j) -> np.ndarray:
    """psi_N at each r/a of a profile j(r/a), for a circular cylinder.

    dpsi/dr = R0 B_theta ~ I(r)/r, so
    psi_N(r) = int_0^r I/r' dr' / int_0^1 I/r' dr' with I(r) = int_0^r j r' dr'.
    No shaping or toroidal correction: the profile lands on the right flux
    surfaces only as far as the plasma is a large-aspect-ratio circle.
    """
    rho = np.asarray(rho, dtype=float)
    j = np.asarray(j, dtype=float)
    if rho[0] != 0.0 or np.any(np.diff(rho) <= 0):
        raise ValueError("r/a must start at 0 and increase strictly")
    enclosed = _cumtrapz(j * rho, rho)
    # I/r -> j(0) r / 2 at the axis
    b_theta = np.empty_like(rho)
    b_theta[1:] = enclosed[1:] / rho[1:]
    b_theta[0] = 0.0
    psi = _cumtrapz(b_theta, rho)
    if psi[-1] <= 0:
        raise ValueError("the profile carries no net current, so r/a has no psi_N")
    return psi / psi[-1]


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))))


def load_current_profile(path: Path | str, coord: str = "psi_n") -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column (x, j [A/m^2]) file and return (psi_N, j).

    ``coord`` says what x is: ``"psi_n"`` (0 at the axis, 1 at the plasma
    edge) or ``"rho"`` (r/a, mapped by :func:`psi_n_from_rho`).
    """
    data = load_two_col_data(path)
    if len(data) < 2:
        raise ValueError(f"{path}: needs at least two (x, j) rows")
    x, j = data[:, 0], data[:, 1]
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"{path}: the first column must increase strictly")
    if abs(x[0]) > 1e-12 or abs(x[-1] - 1.0) > 1e-9:
        raise ValueError(
            f"{path}: the first column must run from 0 to 1, got {x[0]:g} to {x[-1]:g}"
        )
    if coord == "psi_n":
        return x, j
    if coord == "rho":
        return psi_n_from_rho(x, j), j
    raise ValueError(f"current_coord={coord!r}; expected 'psi_n' or 'rho'")


def resample_monotone(x, y, x_new) -> np.ndarray:
    """Interpolate without overshoot (PCHIP), so an edge kink does not ring."""
    return PchipInterpolator(np.asarray(x, dtype=float), np.asarray(y, dtype=float))(x_new)


def extend_psi_n(psi_n, extend_ratio: float, extend_reso: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Put a plasma psi_N grid on a domain that reaches ``extend_ratio`` in psi.

    Returns (grid on the extended domain, its vacuum part, real_psi_edge).
    The plasma edge sits at real_psi_edge = 1/extend_ratio of the new grid,
    the same number :func:`ashen.boundary.extend_psi` gives a CASTOR3D grid.
    """
    real_psi_edge = 1.0 / extend_ratio
    vacuum = np.linspace(real_psi_edge, 1.0, extend_reso + 1)[1:]
    grid = np.concatenate([np.asarray(psi_n, dtype=float) * real_psi_edge, vacuum])
    return grid, vacuum, real_psi_edge


def temperature_to_jorek(T_eV, central_density: float) -> np.ndarray:
    """Te + Ti in eV -> JOREK's T, for n_0 = central_density * 1e20 m^-3."""
    return np.asarray(T_eV, dtype=float) * ELEMENTARY_CHARGE * MU_0 * central_density * 1e20
