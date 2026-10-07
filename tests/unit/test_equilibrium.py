"""ashen.diagnostics.equilibrium -- what q0, edge q and l_i an equilibrium has."""

from __future__ import annotations

import numpy as np
import pytest

from ashen import current_profile as cur
from ashen.diagnostics.equilibrium import achieved_q_li, cylinder_li
from ashen.paths import RunPaths, write_float
from ashen.physics import MU_0


def test_cylinder_li_returns_the_li_a_profile_was_built_with():
    alpha, nu = cur.solve_shape(1.0, 1.3, 3.5)
    rho, _, q, li = cur.shape_profile(alpha, nu)
    assert cylinder_li(rho, q, 1.0) == pytest.approx(li, rel=1e-4)
    assert cylinder_li(rho * 0.4, 3.5 * q, 0.4) == pytest.approx(li, rel=1e-4)   # scale-free


def test_cylinder_li_of_a_flat_current_is_one_half():
    r = np.linspace(0, 1, 501)
    assert cylinder_li(r, np.full_like(r, 2.0), 1.0) == pytest.approx(0.5, rel=1e-5)


def _write_equilibrium(paths, made, geometry, *, real_psi_edge, step=0):
    """Caches a JOREK run would have if its equilibrium were exactly the
    cylinder model `made`, with the plasma edge at real_psi_edge of the domain."""
    # psi per radian: dpsi/dr = r B0 / q  (r in metres), plasma only
    r = made.rho * geometry.a
    dpsi = np.concatenate(([0.0], np.cumsum(
        0.5 * (r[1:] / made.q[1:] + r[:-1] / made.q[:-1]) * np.diff(r)
    ))) * geometry.B0
    psi_plasma = dpsi[-1]
    psi_n = dpsi / psi_plasma * real_psi_edge          # JOREK-grid psi_N
    keep = psi_n > 0
    q_path = paths.qprofile(step)
    q_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Psi_n q", f"# time step #{step:06d}"]
    lines += [f"{p:.12e} {v:.12e}" for p, v in zip(psi_n[keep], made.q[keep])]
    q_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.zero_d(step).write_text(
        "psi_axis psi_bnd R_axis li3 q95\n"
        f"0.0 {psi_plasma / real_psi_edge:.12e} {geometry.R0:.6f} 1.9 4.4\n",
        encoding="utf-8",
    )
    write_float(paths.real_psi_edge, real_psi_edge)


@pytest.mark.parametrize("real_psi_edge", [1.0, 1 / 1.2])
def test_achieved_q_li_recovers_the_model_it_was_given(tmp_path, real_psi_edge):
    paths = RunPaths(tmp_path / "run", pad_width=6)
    paths.run_dir.mkdir()
    geometry = cur.PlasmaGeometry(R0=1.5, a=0.4, kappa=1.0, B0=2.0)
    made = cur.current_from_q_li(1.05, 1.25, 3.3, geometry)
    _write_equilibrium(paths, made, geometry, real_psi_edge=real_psi_edge)

    got = achieved_q_li(paths, 0, f0=geometry.B0 * geometry.R0)

    assert got.q0 == pytest.approx(1.05, abs=2e-3)
    assert got.q_edge == pytest.approx(3.3, rel=2e-3)
    assert got.li == pytest.approx(1.25, rel=5e-3)
    assert got.a == pytest.approx(0.4, rel=2e-3)
    assert got.zero_d["li3"] == 1.9


def test_achieved_q_li_is_none_without_the_caches(tmp_path):
    paths = RunPaths(tmp_path / "run", pad_width=6)
    paths.run_dir.mkdir()
    assert achieved_q_li(paths, 0, f0=3.0) is None
