"""Reading particle positions from JOREK particle files (part_restart*.h5).

A particle file is what JOREK's ``write_simulation_hdf5``
(``particles/mod_particle_io.f90``) writes: the simulation time in SI
seconds at ``/time``, and one HDF5 group per particle group under
``/groups/<id>/``, each holding at least

- ``x``     -- position (R [m], Z [m], phi [rad]) per particle,
- ``i_elm`` -- the grid element the particle is in; <= 0 once it is lost,
- ``weight`` -- how many physical particles it stands for.

A ptrace folder (ashen.ptracing) holds one file per snapshot the program
wrote: e.g. re_gc_current_density_initialisation's
``part_restart<time>.h5`` every ``write_step`` plus ``part_restart.h5`` at
the end, or ptrace_gc's ``part_restart_s<step>_t<time>.h5``, whose
``<step>`` :func:`named_step` reads back. :func:`find_snapshots` collects
them in time order.

Pure data: no matplotlib here (see ashen.plotting.particles).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import re

import numpy as np

from ashen.diagnostics.hdf5 import require_h5py

__all__ = [
    "PARTICLE_FILE_GLOB",
    "ParticleFileError",
    "ParticleSnapshot",
    "find_snapshots",
    "inside_polygon",
    "named_step",
    "read_snapshot",
]

#: Every particle file a JOREK particle program writes into its folder.
PARTICLE_FILE_GLOB = "part_restart*.h5"

#: ptrace_gc's snapshot names: part_restart_s<step>_t<time>.h5.
_NAMED_STEP = re.compile(r"^part_restart_s(\d+)_t")


class ParticleFileError(RuntimeError):
    """A particle file that is missing, unreadable, or not laid out as
    write_simulation_hdf5 writes it."""


@dataclass(frozen=True)
class ParticleSnapshot:
    """Every particle in one particle file, all groups concatenated."""

    path: Path
    #: Simulation time [s].
    time: float
    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    #: True for a particle that has left the grid (i_elm <= 0). Its position
    #: is where it was when it was lost.
    lost: np.ndarray
    weight: np.ndarray

    @property
    def n(self) -> int:
        return int(self.R.size)

    @property
    def n_lost(self) -> int:
        return int(np.count_nonzero(self.lost))


def _positions(x: np.ndarray, n_elm: int, where: str) -> np.ndarray:
    """``x`` as (n_particles, 3).

    Fortran writes x_arr(3, n); read back through HDF5's row-major view
    that is (n, 3). Transposed data (3, n) is accepted too, told apart by
    i_elm's length, so a file written some other way still reads.
    """
    if x.ndim == 2 and x.shape == (n_elm, 3):
        return x
    if x.ndim == 2 and x.shape == (3, n_elm):
        return x.T
    raise ParticleFileError(
        f"{where}: x has shape {x.shape}, expected ({n_elm}, 3) -- one "
        "(R, Z, phi) per particle, matching i_elm"
    )


def read_snapshot(path: Path | str) -> ParticleSnapshot:
    """Read every particle's position from one particle file."""
    h5py = require_h5py("reading particle files", ParticleFileError)
    path = Path(path)
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        raise ParticleFileError(f"{path}: cannot open -- {exc}") from exc

    with f:
        if "time" not in f:
            raise ParticleFileError(f"{path}: no /time -- not a JOREK particle file")
        time = float(np.asarray(f["time"]).reshape(-1)[0])
        groups = f.get("groups")
        if groups is None or not len(groups):
            raise ParticleFileError(f"{path}: no particle groups under /groups")

        R, Z, phi, lost, weight = [], [], [], [], []
        for name in sorted(groups):
            group = groups[name]
            where = f"{path}: group {name!r}"
            for dataset in ("x", "i_elm"):
                if dataset not in group:
                    raise ParticleFileError(f"{where} has no {dataset!r}")
            i_elm = np.asarray(group["i_elm"]).reshape(-1)
            x = _positions(np.asarray(group["x"]), i_elm.size, where)
            R.append(x[:, 0])
            Z.append(x[:, 1])
            phi.append(x[:, 2])
            lost.append(i_elm <= 0)
            weight.append(
                np.asarray(group["weight"], dtype=float).reshape(-1)
                if "weight" in group else np.ones(i_elm.size)
            )

    return ParticleSnapshot(
        path=path, time=time,
        R=np.concatenate(R).astype(float), Z=np.concatenate(Z).astype(float),
        phi=np.concatenate(phi).astype(float), lost=np.concatenate(lost),
        weight=np.concatenate(weight),
    )


def find_snapshots(folder: Path | str) -> list[ParticleSnapshot]:
    """Every particle file in `folder`, read and sorted by simulation time.

    Two files at the same time -- e.g. a snapshot written on the final step
    alongside part_restart.h5 -- are the same state; the first by name is
    kept.
    """
    folder = Path(folder)
    snapshots: dict[float, ParticleSnapshot] = {}
    for path in sorted(folder.glob(PARTICLE_FILE_GLOB)):
        snapshot = read_snapshot(path)
        snapshots.setdefault(snapshot.time, snapshot)
    return [snapshots[t] for t in sorted(snapshots)]


def named_step(path: Path | str) -> int | None:
    """The JOREK step ptrace_gc put in a snapshot's name -- that of the
    restart closest in time -- or None for a name without one."""
    match = _NAMED_STEP.match(Path(path).name)
    return int(match.group(1)) if match else None


def inside_polygon(R: np.ndarray, Z: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Whether each point (R, Z) is inside the closed polygon, an (N, 2)
    array of (R, Z) vertices (first and last need not repeat). Even-odd ray
    casting, so points exactly on an edge may go either way."""
    R, Z = np.asarray(R, dtype=float), np.asarray(Z, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    r0, z0 = poly[:, 0], poly[:, 1]
    r1, z1 = np.roll(r0, -1), np.roll(z0, -1)
    inside = np.zeros(R.shape, dtype=bool)
    for a_r, a_z, b_r, b_z in zip(r0, z0, r1, z1):
        crosses = (a_z > Z) != (b_z > Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_cross = a_r + (Z - a_z) * (b_r - a_r) / (b_z - a_z)
        inside ^= crosses & (R < r_cross)
    return inside
