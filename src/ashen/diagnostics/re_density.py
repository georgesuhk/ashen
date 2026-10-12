"""The runaway-electron density of JOREK's RE fluid, read from a restart.

No JOREK tool is run: the node values of every toroidal harmonic are in the
restart file. For model600 built with the RE fluid, n_RE is the last
variable (``var_nre``, models/model600/mod_model_settings.f90:67).

SI units as jorek2_postproc's ``nre`` expression gives them
(diagnostics/new_diag/mod_expression.f90:2021, :2053)::

    n_RE [m^-3] = nre * sqrt(rho_0 / mu_0) / (e R),   rho_0 = central_density 1e20 central_mass amu

Toroidal basis (elements/basis_at_gaussian.f90:78-88): harmonic 1 is n = 0,
harmonic 2i is cos(i n_period phi), harmonic 2i+1 is sin(i n_period phi).

What is here is the field **on the grid nodes**. Between nodes a plot
interpolates linearly, where JOREK uses cubic Bezier elements: good for
looking at where the runaways are, not for measuring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ashen.physics import ELEMENTARY_CHARGE, MU_0

__all__ = ["ReDensity", "read_re_density"]

_SPEED_OF_LIGHT = 299792458.0
_ATOMIC_MASS_UNIT = 1.66053906660e-27


@dataclass(frozen=True)
class ReDensity:
    """n_RE on the nodes of one restart, every toroidal harmonic."""

    #: JOREK's index_now for this restart.
    index: int
    #: Seconds; nan if the restart does not say.
    time: float
    R: np.ndarray
    Z: np.ndarray
    #: (n_triangles, 3) node indices: each grid element as two triangles.
    triangles: np.ndarray
    #: Axisymmetric (n = 0) psi and zj on the nodes, JOREK units.
    psi: np.ndarray
    zj: np.ndarray
    #: (n_tor, n_nodes), m^-3. Signed so that the n = 0 density is positive
    #: (JOREK stores it with the sign of the runaways' parallel velocity).
    harmonics: np.ndarray
    n_period: int
    #: |Ipre_tot| JOREK itself recorded for this step, A; nan if not in the file.
    ipre_tot: float

    @property
    def n0(self) -> np.ndarray:
        """The axisymmetric part."""
        return self.harmonics[0]

    def at(self, phi: float, *, perturbation: bool = False) -> np.ndarray:
        """n_RE on the nodes at toroidal angle ``phi`` (radians).
        ``perturbation``: the n > 0 part alone."""
        out = np.zeros_like(self.n0) if perturbation else self.n0.copy()
        for i in range(1, (self.harmonics.shape[0] - 1) // 2 + 1):
            angle = i * self.n_period * phi
            out += self.harmonics[2 * i - 1] * np.cos(angle) + self.harmonics[2 * i] * np.sin(angle)
        return out

    def mode_amplitudes(self) -> dict[int, float]:
        """{toroidal mode number n: largest amplitude of that mode on any
        node}, m^-3. n = 0 is the largest |n_RE,0|."""
        out = {0: float(np.abs(self.n0).max())}
        for i in range(1, (self.harmonics.shape[0] - 1) // 2 + 1):
            out[i * self.n_period] = float(
                np.hypot(self.harmonics[2 * i - 1], self.harmonics[2 * i]).max()
            )
        return out

    def current(self) -> float:
        """e c times the integral of the n = 0 density over the poloidal
        plane, A: the runaway current if every runaway moved toroidally at
        c. Linear over each triangle, and without JOREK's B_phi / |B|, so it
        sits a little off Ipre_tot (0.06% on a 60 x 60 grid); what it is for
        is seeing whether runaways have left."""
        t = self.triangles
        r, z = self.R[t], self.Z[t]
        area = 0.5 * np.abs(
            (r[:, 1] - r[:, 0]) * (z[:, 2] - z[:, 0]) - (r[:, 2] - r[:, 0]) * (z[:, 1] - z[:, 0])
        )
        return float(ELEMENTARY_CHARGE * _SPEED_OF_LIGHT * (self.n0[t].mean(axis=1) * area).sum())


def read_re_density(path: Path | str) -> ReDensity:
    """n_RE from one restart. Raises ValueError, saying what is missing, for
    a file without the datasets the conversion to m^-3 needs.

    The caller must know the run has the RE fluid: nothing in a restart
    says which variable is which, and without it the last variable is
    something else.
    """
    import h5py  # lazy, as elsewhere: the package imports without it

    path = Path(path)
    with h5py.File(path, "r") as f:
        missing = [k for k in ("values", "x", "central_density", "central_mass") if k not in f]
        if missing:
            raise ValueError(f"{path.name}: no {', '.join(missing)} in this restart")
        values, x = f["values"][()], f["x"][()]
        rho_0 = float(f["central_density"][0]) * 1e20 * float(f["central_mass"][0]) * _ATOMIC_MASS_UNIT
        index = int(f["index_now"][0]) if "index_now" in f else 0
        n_period = int(f["n_period"][0]) if "n_period" in f else 1
        time = float("nan")
        if "t_now" in f and "sqrt_mu0_rho0" in f:
            time = float(f["t_now"][0]) * float(f["sqrt_mu0_rho0"][0])
        ipre = float("nan")
        if "Ipre_tot_t" in f and 0 < index <= f["Ipre_tot_t"].shape[0]:
            ipre = abs(float(f["Ipre_tot_t"][index - 1]))
        quads = None
        if "vertex" in f:
            n_elements = int(f["n_elements"][0]) if "n_elements" in f else f["vertex"].shape[1]
            quads = f["vertex"][:, :n_elements].T - 1

    R, Z = x[0, 0, 0, :], x[1, 0, 0, :]
    if quads is None:
        # no element list (a file not written by JOREK): any triangulation of the nodes
        from matplotlib.tri import Triangulation

        triangles = Triangulation(R, Z).triangles
    else:
        triangles = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])

    harmonics = values[-1, 0, :, :] * np.sqrt(rho_0 / MU_0) / (ELEMENTARY_CHARGE * R)
    n0 = harmonics[0]
    if n0.size and n0[np.argmax(np.abs(n0))] < 0:
        harmonics = -harmonics
    return ReDensity(
        index=index, time=time, R=R, Z=Z, triangles=triangles,
        psi=values[0, 0, 0, :], zj=values[2, 0, 0, :],
        harmonics=harmonics, n_period=n_period, ipre_tot=ipre,
    )
