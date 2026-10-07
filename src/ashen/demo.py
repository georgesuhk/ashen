"""A made-up campaign and run folder, for trying ashen.viewer without JOREK.

``make_demo_campaign(root)`` writes a small campaign (site.toml, template,
empty exe) and one run folder in it. The run's inputs are prepared by the
real ``prepare_run`` from a ``ffprime_method = "q_li"`` shotfile. Its
"JOREK output" (restarts, q-profile, zeroD, Fourier and profile caches) is
invented: an equilibrium that is the cylinder model with q0, l_i and q_edge
a little off the requested ones, and modes that grow and saturate. None of
it is physics; it only has the shapes and file formats the viewer reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ashen import current_profile as cur
from ashen.config import load_site
from ashen.diagnostics import four_cache as fc
from ashen.paths import RunPaths
from ashen.physics import MU_0
from ashen.runner import prepare_run
from ashen.shotfile import load_shotfile

__all__ = ["DEMO_STEPS", "make_demo_campaign"]

#: Restart steps the demo run pretends to have.
DEMO_STEPS = tuple(range(0, 2001, 200))

_F0 = 3.7
_R0, _A, _KAPPA = 1.40, 0.36, 1.10
#: How far the pretend equilibrium lands from the request (q0, l_i, q_edge).
_MISS = (0.05, 0.06, -0.12)

_NAMELIST = """\
 &in1
 restart = {restart}
 eta = 1.d-6
 central_density = 1.d-2
 central_mass = 2.0
 freeboundary = .f.
 tstep_n = 1
 nstep_n = 1200
 nout = 200
 F0 = {f0}
 R_geo = {r0}
 Z_geo = 0.0
 ffprime_file = 'ffprime_prof.dat'
 T_file       = 'T_prof.dat'
 rho_file     = 'rho_prof.dat'
&end
"""

_STARWALL = """\
&PARAMS
  i_response = 2,
  n_harm     = 1,
  n_tor      = 1,
/
&PARAMS_WALL
  eta_thin_w = 1.d-4
  mn_w       = 2,
  n_w        =    0,      0,
  m_w        =    0,      1,
  rc_w       =    {r0},   0.70,
  rs_w       =    0.,     0.,
  zc_w       =    0.,     0.,
  zs_w       =    0.,     0.75,
/
"""

_SITE = """\
# A made-up campaign written by ashen.demo. Nothing here can run JOREK.
[paths]
exe         = "./exe"
template    = "./template"
jobscripts  = "./jobscripts"
jorek       = "./jorek"
jorek_re    = "./jorek_RE"
castor_root = "./castor3d"

[launch]
interactive_prelude = ""
batch_prelude       = ""
mpirun              = "mpirun -n {n}"
n_jorek             = 1
n_starwall          = 1
"""

_SHOTFILE = """\
# A made-up shot written by ashen.demo, for trying the viewer.
qa = 3.3
g = 3.2
n0 = 1e18
eta = 1e-3
tstep_n = [0.03]
nstep_n = [2000]
nout = 200

exe = "jorek_model600_demo"
jobscript = "2h"
freeboundary = False
extend_bnd = True

# --- current profile: tune these three in the viewer -----------------
ffprime_method = "q_li"
current_q0 = 1.05
current_li = 1.2
current_q_edge = 3.3

T_method = "const"
T_const = 100.0
rho_method = "const"
rho_const = n0

bnd_method = "file"
bnd_file = "plasma_bnd.dat"
bnd_file_is_plasma = True
"""


def _plasma_boundary(n: int = 180) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # a little triangularity, so it is not just an ellipse
    R = _R0 + _A * np.cos(theta + 0.15 * np.sin(theta))
    return np.column_stack((R, _A * _KAPPA * np.sin(theta)))


def _write_campaign(root: Path) -> Path:
    for sub in ("exe", "jobscripts", "jorek/util", "jorek_RE/util", "castor3d",
                "template/copy", "template/symlink/base", "template/symlink/RE",
                "template/symlink/standard", "template/symlink/starwall"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "site.toml").write_text(_SITE, encoding="utf-8")
    (root / "exe" / "jorek_model600_demo").write_text("#!/bin/sh\necho 'demo: not JOREK'\n")
    (root / "jobscripts" / "2h").write_text("#!/bin/sh\n")
    copy = root / "template" / "copy"
    for name, restart in (("in_eq", ".f."), ("in_main", ".f."), ("in_main_r", ".t.")):
        (copy / name).write_text(_NAMELIST.format(restart=restart, f0=_F0, r0=_R0))
    (copy / "input_starwall").write_text(_STARWALL.format(r0=_R0))
    (root / "template" / "symlink" / "base" / "submit_jorek.sh").write_text("#!/bin/sh\n")
    return root


def _pretend_equilibrium(params, geometry: cur.PlasmaGeometry, extend_ratio: float):
    """The cylinder model a little off the request, laid over the domain:
    plasma for rho <= 1, vacuum (no current, q ~ rho^2) out to extend_ratio."""
    made = cur.current_from_q_li(
        params.current_q0 + _MISS[0], params.current_li + _MISS[1],
        params.current_q_edge + _MISS[2], geometry,
    )
    r = made.rho * geometry.a
    psi = np.concatenate(([0.0], np.cumsum(
        0.5 * (r[1:] / made.q[1:] + r[:-1] / made.q[:-1]) * np.diff(r)
    ))) * geometry.B0                                  # Wb/rad from the axis
    rho_vac = np.linspace(1.0, extend_ratio, 60)[1:]
    # prepare_run puts the plasma edge at 1/extend_ratio of the domain's psi
    psi_vac = psi[-1] * rho_vac
    rho = np.concatenate((made.rho, rho_vac))
    return {
        "rho": rho,
        "psi": np.concatenate((psi, psi_vac)),
        "q": np.concatenate((made.q, made.q[-1] * rho_vac**2)),
        "j": np.concatenate((made.j, np.zeros_like(rho_vac))),
        "delta_psi": float(psi_vac[-1]),
        "made": made,
    }


def _write_restart(path: Path, eq, geometry, bnd: np.ndarray, extend_ratio: float) -> None:
    import h5py

    n_rad, n_pol = 28, 48
    s = np.linspace(0.0, extend_ratio, n_rad)                # rho of each ring
    theta = np.linspace(0.0, 2.0 * np.pi, n_pol, endpoint=False)
    centre = np.array([geometry.R0, 0.0])
    ring = np.column_stack((np.interp(theta, np.linspace(0, 2 * np.pi, len(bnd), endpoint=False),
                                      bnd[:, 0], period=2 * np.pi),
                            np.interp(theta, np.linspace(0, 2 * np.pi, len(bnd), endpoint=False),
                                      bnd[:, 1], period=2 * np.pi))) - centre
    R = (centre[0] + np.outer(s, ring[:, 0])).ravel()
    Z = (centre[1] + np.outer(s, ring[:, 1])).ravel()
    rho = np.repeat(s, n_pol)
    values = np.zeros((8, 4, 1, R.size))
    values[0, 0, 0] = np.interp(rho, eq["rho"], eq["psi"]) - eq["delta_psi"]   # 0 at the domain edge
    values[2, 0, 0] = -MU_0 * geometry.R0 * np.interp(rho, eq["rho"], eq["j"])
    values[4, 0, 0] = 1.0
    x = np.zeros((2, 4, 1, R.size))
    x[0, 0, 0], x[1, 0, 0] = R, Z
    with h5py.File(path, "w") as f:
        f["values"], f["x"] = values, x
        f["F0"], f["central_density"] = [_F0], [0.01]


def _write_outputs(run_dir: Path, params, geometry, bnd: np.ndarray) -> None:
    paths = RunPaths(run_dir, pad_width=6)
    eq = _pretend_equilibrium(params, geometry, params.extend_ratio)
    psi_n_of_rho = eq["psi"] / eq["delta_psi"]               # JOREK-grid psi_N
    grid = np.linspace(0.01, 0.99, 120)
    q_on_grid = np.interp(grid, psi_n_of_rho, eq["q"])
    j_on_grid = np.interp(grid, psi_n_of_rho, eq["j"])
    t_norm = 6.48e-8

    domain = np.array([geometry.R0, 0.0]) + params.extend_ratio * (bnd - [geometry.R0, 0.0])
    rows = ["     60     60      1      1      2"]
    rows += [f"  {i:5d}  {i:5d}  {i + 1:5d}  {r:.8E}  {z:.8E}  0.0 0.0 1.0 1.0"
             for i, (r, z) in enumerate(domain[::3], 1)]
    (run_dir / "boundary.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    paths.postproc_dir.mkdir(exist_ok=True)

    modes = {(1, 1): 1.0, (1, 2): 2.3, (1, 3): 3.1, (2, 3): 1.5, (2, 5): 2.6}   # (n, m): q
    for step in DEMO_STEPS:
        _write_restart(paths.restart(step), eq, geometry, bnd, params.extend_ratio)

        lines = ["# ", f"# time step #{step:06d}"]
        lines += [f"  {p:.15E}  {v:.15E}" for p, v in zip(grid, q_on_grid)]
        paths.qprofile(step).write_text("\n".join(lines) + "\n", encoding="utf-8")
        zero_d = {
            "Time": step * 0.03 * t_norm, "index_now": float(step),
            "psi_axis": -eq["delta_psi"], "R_axis": geometry.R0 + 0.03, "Z_axis": 0.0,
            "psi_bnd": 0.0, "Ip_tot": eq["made"].Ip, "li3": eq["made"].li + 0.25,
            "q95": float(np.interp(0.95, grid, q_on_grid)),
        }
        paths.zero_d(step).write_text(
            "  ".join(f"{k:>22s}" for k in zero_d) + "\n"
            + "  ".join(f"{v:22.15E}" for v in zero_d.values()) + "\n", encoding="utf-8",
        )

        # modes: grow exponentially, then saturate; peaked at their q = m/n surface
        growth = 1e-9 * np.exp(0.012 * min(step, 1400))
        records = []
        for (n, m), q_res in modes.items():
            at = float(np.interp(q_res, q_on_grid, grid)) if q_on_grid[0] < q_res else 0.0
            shape = np.exp(-((grid - at) / 0.18) ** 2) * grid ** (0.5 * m)
            amp = growth * shape / (n * m) * (1.0 + 0.1 * np.sin(0.002 * step * m))
            phase = 0.004 * step * n
            for variable, scale in (("Psi", 1.0), ("T", 3e-3)):
                records.append(fc.FourRecord(
                    variable, n, m, grid, scale * amp * np.cos(phase), scale * amp * np.sin(phase)
                ))
        records.append(fc.FourRecord("Psi", 0, 0, grid, eq["delta_psi"] * (1 - grid), 0 * grid))
        fc.write_cache(paths.four_cache(step), step=step, pad_width=6, records=records)

        # current density flattening around the 2/1 surface as the mode grows
        island = float(np.interp(2.3, q_on_grid, grid))
        flatten = 1.0 - 0.35 * min(step, 1400) / 1400 * np.exp(-((grid - island) / 0.08) ** 2)
        np.savez(paths.profile_cache("Psi_N", "currdens", step, "midplane outer"),
                 x=grid, y=j_on_grid * flatten)
        np.savez(paths.profile_cache("Psi_N", "Btor", step, "midplane outer"),
                 x=grid, y=_F0 / (geometry.R0 + geometry.a * np.sqrt(grid)))


def make_demo_campaign(root: Path | str) -> Path:
    """Write the demo campaign under ``root`` and return its run folder.

    Safe to call again: the campaign files and the pretend outputs are
    rewritten, but an existing ``shotfile.py`` is kept, so values saved from
    the viewer survive. Delete the folder to start over.
    """
    root = _write_campaign(Path(root).resolve())
    run_dir = root / "demo_shot" / "q_li_demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    shotfile = run_dir / "shotfile.py"
    if not shotfile.is_file():
        shotfile.write_text(_SHOTFILE, encoding="utf-8")
    bnd = _plasma_boundary()
    np.savetxt(run_dir / "plasma_bnd.dat", bnd)

    params = load_shotfile(shotfile)
    prepare_run(params, load_site(root / "site.toml"), run_dir)
    geometry = cur.boundary_geometry(bnd, _F0)
    _write_outputs(run_dir, params, geometry, bnd)
    return run_dir
