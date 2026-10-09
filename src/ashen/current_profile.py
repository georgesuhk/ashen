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

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from ashen.castor_io import load_two_col_data
from ashen.physics import ELEMENTARY_CHARGE, MU_0

__all__ = [
    "ALPHA_RANGE",
    "PlasmaGeometry",
    "QLiProfile",
    "boundary_geometry",
    "current_from_ffprime",
    "current_from_q_li",
    "q0_window",
    "shape_profile",
    "solve_shape",
    "extend_psi_n",
    "ffprime_from_current",
    "load_current_profile",
    "load_temperature_profile",
    "profile_from_ffprime",
    "psi_n_from_rho",
    "resample_monotone",
    "parametric_temperature",
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


def load_temperature_profile(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column (psi_N, Te + Ti [eV]) file.

    psi_N must run from 0 to 1, increasing, and T must be positive
    everywhere: JOREK rejects a profile that is not.
    """
    data = load_two_col_data(path)
    if len(data) < 2:
        raise ValueError(f"{path}: needs at least two (psi_N, T) rows")
    x, T = data[:, 0], data[:, 1]
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"{path}: the first column must increase strictly")
    if abs(x[0]) > 1e-12 or abs(x[-1] - 1.0) > 1e-9:
        raise ValueError(
            f"{path}: the first column must run from 0 to 1, got {x[0]:g} to {x[-1]:g}"
        )
    if np.any(T <= 0):
        raise ValueError(f"{path}: T must be positive everywhere, lowest is {T.min():g} eV")
    return x, T


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


def parametric_temperature(
    psi_n, core: float, edge: float, alpha: float = 1.0, beta: float = 2.0
) -> np.ndarray:
    """A temperature profile from four numbers [eV]:

        T(psi_N) = T_edge + (T_core - T_edge) (1 - psi_N^alpha)^beta

    for psi_N from 0 (axis) to 1 (plasma edge); T_edge beyond. ``alpha``
    flattens the core as it grows, ``beta`` sets how the profile meets the
    edge: with beta >= 2 it arrives with zero gradient, so there is no kink
    where the plasma meets the vacuum of an extended boundary. The default
    (1, 2) is the (1 - psi_N)^2 parabola. A guess function, not a fit: with
    a fixed density it is the pressure profile's shape too.
    """
    x = np.clip(np.asarray(psi_n, dtype=float), 0.0, 1.0)
    return edge + (core - edge) * (1.0 - x**alpha) ** beta


def temperature_to_jorek(T_eV, central_density: float) -> np.ndarray:
    """Te + Ti in eV -> JOREK's T, for n_0 = central_density * 1e20 m^-3."""
    return np.asarray(T_eV, dtype=float) * ELEMENTARY_CHARGE * MU_0 * central_density * 1e20


# --- a current profile from q0, l_i and q_edge --------------------------------
#
# Circular cylinder, radius normalised to 1 (ported from q0_li_playground.ipynb):
#
#     j(rho) = j0 (1 - rho^alpha)^nu        I(rho) = int_0^rho j rho' drho'
#     q(rho) = q_edge rho^2 I(1) / I(rho)   l_i = 2 / I(1)^2 int_0^1 I^2 / rho drho
#
# q_edge / q0 = j0 / <j>, so (alpha, nu) are fixed by l_i and q0 / q_edge; q0
# then sets the size of j. alpha = 2 is the Wesson profile; larger alpha
# flattens the core and steepens the edge.

#: Range of the shape exponent alpha that solve_shape searches. The q0 window
#: at a given l_i and q_edge belongs to this family and this range, not to
#: physics. alpha < 2 gives a pointed peak on axis.
ALPHA_RANGE = (1.0, 30.0)

_RHO = np.linspace(0.0, 1.0, 2001)


def shape_profile(alpha: float, nu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(rho, j / j0, q / q_edge, l_i) of j = j0 (1 - rho^alpha)^nu."""
    rho = _RHO
    j = np.exp(nu * np.log1p(-rho[:-1] ** alpha))
    j = np.append(j, 0.0 if nu > 0 else 1.0)
    enclosed = _cumtrapz(j * rho, rho)
    q = np.empty_like(rho)
    q[1:] = rho[1:] ** 2 * enclosed[-1] / enclosed[1:]
    q[0] = 2.0 * enclosed[-1]                     # <j> / j0
    g = np.zeros_like(rho)
    g[1:] = enclosed[1:] ** 2 / rho[1:]
    li = 2.0 * _cumtrapz(g, rho)[-1] / enclosed[-1] ** 2
    return rho, j, q, float(li)


def _nu_for_li(alpha: float, li: float) -> float:
    # l_i rises from 0.5 (flat current, nu = 0) without limit as nu grows
    def miss(log_nu: float) -> float:
        return shape_profile(alpha, np.exp(log_nu))[3] - li

    hi = 0.0
    while miss(hi) < 0:
        hi += 1.0
    return float(np.exp(brentq(miss, np.log(1e-8), hi, xtol=1e-12)))


def _q0_ratio(alpha: float, li: float) -> float:
    """q0 / q_edge of the profile with this alpha and l_i."""
    return float(shape_profile(alpha, _nu_for_li(alpha, li))[2][0])


@lru_cache(maxsize=256)
def _q0_curve(li: float) -> tuple[np.ndarray, np.ndarray]:
    alphas = np.geomspace(*ALPHA_RANGE, 60)
    return alphas, np.array([_q0_ratio(a, li) for a in alphas])


def q0_window(li: float, q_edge: float) -> tuple[float, float]:
    """The q0 this family can reach at this l_i and q_edge."""
    if li <= 0.5:
        raise ValueError("l_i must exceed 0.5, the value for a flat current")
    _, ratio = _q0_curve(round(float(li), 6))
    return float(q_edge * ratio.min()), float(q_edge * ratio.max())


def solve_shape(q0: float, li: float, q_edge: float) -> tuple[float, float]:
    """(alpha, nu) giving this l_i and q0 / q_edge.

    Raises ValueError, naming the reachable q0 window, if there is none.
    """
    lo, hi = q0_window(li, q_edge)
    alphas, ratio = _q0_curve(round(float(li), 6))
    d = ratio - q0 / q_edge
    crossings = np.nonzero(d[:-1] * d[1:] <= 0)[0]
    if len(crossings) == 0:
        raise ValueError(
            f"q0 = {q0:g} is outside what l_i = {li:g} and qa = {q_edge:g} "
            f"allow for this profile family: q0 from {lo:.3f} to {hi:.3f}"
        )
    k = crossings[0]
    alpha = brentq(
        lambda a: _q0_ratio(a, li) - q0 / q_edge, alphas[k], alphas[k + 1], xtol=1e-10
    )
    return float(alpha), _nu_for_li(alpha, li)


@dataclass(frozen=True)
class PlasmaGeometry:
    """What the q0/l_i/q_edge model needs to know about the plasma."""

    R0: float      #: major radius of the boundary's centre [m]
    a: float       #: minor radius, half the boundary's width in R [m]
    kappa: float   #: elongation, height / width
    B0: float      #: toroidal field at R0 [T]


def boundary_geometry(bnd: np.ndarray, F0: float) -> PlasmaGeometry:
    """R0, a and kappa from a plasma boundary's extents; B0 = |F0| / R0."""
    bnd = np.asarray(bnd, dtype=float)
    r_min, r_max = bnd[:, 0].min(), bnd[:, 0].max()
    R0 = 0.5 * (r_min + r_max)
    a = 0.5 * (r_max - r_min)
    kappa = (bnd[:, 1].max() - bnd[:, 1].min()) / (2.0 * a)
    return PlasmaGeometry(R0=float(R0), a=float(a), kappa=float(kappa), B0=abs(F0) / float(R0))


@dataclass(frozen=True)
class QLiProfile:
    """A current profile made from q0, l_i and q_edge."""

    rho: np.ndarray     #: r / a
    psi_n: np.ndarray   #: psi_N at each rho (circular cylinder)
    j: np.ndarray       #: j_phi at R = R0 [A/m^2]
    q: np.ndarray       #: the cylinder's q at each rho
    alpha: float | None  #: None for a profile that is not of the family
    nu: float | None
    li: float           #: l_i of the profile as built
    j0: float           #: j on axis [A/m^2]
    Ip: float           #: total current [A], for an ellipse of this a, kappa


def current_from_q_li(q0: float, li: float, q_edge: float, geometry: PlasmaGeometry) -> QLiProfile:
    """The current profile with this q0, l_i and q_edge, in A/m^2.

    The shape is the cylinder's. The size is set by q0 on the axis of an
    ellipse: q0 = B0 (1 + kappa^2) / (mu_0 R0 kappa j0), which for a circle is
    2 B0 / (mu_0 R0 j0). The same factor relates q_edge to the mean current,
    so q_edge / q0 keeps its cylinder value. No triangularity, Shafranov shift
    or toroidicity: JOREK's equilibrium will miss the targets by some percent.
    """
    alpha, nu = solve_shape(q0, li, q_edge)
    rho, shape, q_norm, li_built = shape_profile(alpha, nu)
    g = geometry
    j0 = g.B0 * (1.0 + g.kappa**2) / (MU_0 * g.R0 * g.kappa * q0)
    j = j0 * shape
    mean_j = j0 * q_norm[0]
    return QLiProfile(
        rho=rho, psi_n=psi_n_from_rho(rho, j), j=j, q=q_edge * q_norm,
        alpha=alpha, nu=nu, li=li_built, j0=float(j0),
        Ip=float(mean_j * np.pi * g.a**2 * g.kappa),
    )


def profile_from_ffprime(psi_n, ffprime, geometry: PlasmaGeometry) -> QLiProfile:
    """The cylinder picture of an FF' profile JOREK reads: j, q and l_i.

    The inverse of the "q_li" route, for looking at a profile that came from
    elsewhere (CASTOR3D, a file). ``psi_n`` is the plasma's own psi_N (0 at
    the axis, 1 at its edge) and ``ffprime`` JOREK's FFprime there.

    j = -FFprime / (mu_0 R0). r/a is found from psi_N the way psi_n_from_rho
    goes the other way (dpsi/drho ~ I / rho), by iterating to consistency. q
    then follows from q0 = B0 (1 + kappa^2) / (mu_0 R0 kappa j0), as in
    current_from_q_li. Same limits: a cylinder with elongation, no pressure
    gradient current. A profile of the q0/l_i/qa family comes back with the
    q0, l_i and qa it was made from.
    """
    psi_n = np.asarray(psi_n, dtype=float)
    j_of_psi = current_from_ffprime(ffprime, geometry.R0)
    if j_of_psi[0] <= 0:
        raise ValueError("the profile carries no current on axis in the direction expected")
    rho = _RHO
    psi_of_rho = rho**2
    for _ in range(200):
        j = np.interp(psi_of_rho, psi_n, j_of_psi)
        updated = psi_n_from_rho(rho, j)
        change = np.max(np.abs(updated - psi_of_rho))
        psi_of_rho = updated
        if change < 1e-12:
            break
    j = np.interp(psi_of_rho, psi_n, j_of_psi)
    enclosed = _cumtrapz(j * rho, rho)
    g = geometry
    q0 = g.B0 * (1.0 + g.kappa**2) / (MU_0 * g.R0 * g.kappa * j[0])
    q = np.empty_like(rho)
    q[1:] = q0 * j[0] * rho[1:] ** 2 / (2.0 * enclosed[1:])
    q[0] = q0
    weight = np.zeros_like(rho)
    weight[1:] = enclosed[1:] ** 2 / rho[1:]
    li = 2.0 * _cumtrapz(weight, rho)[-1] / enclosed[-1] ** 2
    return QLiProfile(
        rho=rho, psi_n=psi_of_rho, j=j, q=q, alpha=None, nu=None, li=float(li),
        j0=float(j[0]), Ip=float(2.0 * enclosed[-1] * np.pi * g.a**2 * g.kappa),
    )
