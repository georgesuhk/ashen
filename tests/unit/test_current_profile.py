"""Tests for the current-density -> FF' conversion and its helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ashen import current_profile as cur
from ashen.physics import MU_0


def test_ffprime_is_minus_mu0_R0_j():
    """Delta* psi = FFprime (no pressure) and j_phi = -Delta* psi / (R mu_0)."""
    assert cur.ffprime_from_current(1.0e6, 1.5) == pytest.approx(-MU_0 * 1.5 * 1.0e6)


def test_ffprime_matches_the_gs_source_with_pressure():
    """zj = FFprime - R^2 p' must give back j_phi = -zj / (R mu_0) at R0."""
    R0, j, dp_dpsi = 1.7, 2.5e6, -0.3
    ffprime = cur.ffprime_from_current(j, R0, dp_dpsi)
    zj = ffprime - R0**2 * dp_dpsi
    assert -zj / (R0 * MU_0) == pytest.approx(j)


def test_current_from_ffprime_inverts():
    j = np.array([3.0e6, 1.0e6, 0.0])
    back = cur.current_from_ffprime(cur.ffprime_from_current(j, 1.4, 0.2), 1.4, 0.2)
    np.testing.assert_allclose(back, j, rtol=1e-12, atol=1e-6)


def test_existing_run_scale():
    """FFprime = -5.70 at R = 1.37 m is about 3.3 MA/m^2, a DIII-D-like q0 ~ 1."""
    assert cur.current_from_ffprime(-5.70, 1.37) == pytest.approx(3.31e6, rel=1e-2)


def test_psi_n_from_rho_flat_current_is_r_squared():
    rho = np.linspace(0, 1, 2001)
    np.testing.assert_allclose(cur.psi_n_from_rho(rho, np.ones_like(rho)), rho**2, atol=1e-6)


def test_psi_n_from_rho_parabolic_current():
    """j = 1 - r^2: I ~ r^2/2 - r^4/4, psi ~ r^2/4 - r^4/16."""
    rho = np.linspace(0, 1, 4001)
    exact = (rho**2 / 4 - rho**4 / 16) / (1 / 4 - 1 / 16)
    np.testing.assert_allclose(cur.psi_n_from_rho(rho, 1 - rho**2), exact, atol=1e-6)


def test_psi_n_from_rho_rejects_bad_grids():
    with pytest.raises(ValueError):
        cur.psi_n_from_rho([0.1, 0.5, 1.0], [1, 1, 1])
    with pytest.raises(ValueError):
        cur.psi_n_from_rho([0.0, 0.5, 1.0], [0, 0, 0])


def test_load_current_profile(tmp_path):
    path = tmp_path / "j.dat"
    np.savetxt(path, np.column_stack(([0.0, 0.5, 1.0], [2e6, 1e6, 0.0])))
    x, j = cur.load_current_profile(path)
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(j, [2e6, 1e6, 0.0])

    x_rho, _ = cur.load_current_profile(path, "rho")
    assert x_rho[0] == 0.0 and x_rho[-1] == pytest.approx(1.0)
    assert x_rho[1] > 0.25  # peaked current: psi_N(r) runs ahead of r^2

    with pytest.raises(ValueError, match="psi_n"):
        cur.load_current_profile(path, "r")


def test_load_current_profile_rejects_a_grid_not_on_0_1(tmp_path):
    path = tmp_path / "j.dat"
    np.savetxt(path, np.column_stack(([0.0, 0.5, 0.9], [2e6, 1e6, 0.0])))
    with pytest.raises(ValueError, match="0 to 1"):
        cur.load_current_profile(path)


def test_extend_psi_n():
    grid, vacuum, real_psi_edge = cur.extend_psi_n(np.linspace(0, 1, 11), 1.25, 4)
    assert real_psi_edge == pytest.approx(0.8)
    assert len(grid) == 15 and len(vacuum) == 4
    assert grid[10] == pytest.approx(0.8) and grid[-1] == 1.0
    assert np.all(np.diff(grid) > 0)


def test_temperature_to_jorek():
    """Wiki: T_SI[K] = T / (k_B mu_0 n_0), so T = T[eV] e mu_0 n_0."""
    assert cur.temperature_to_jorek(1000.0, 0.5) == pytest.approx(
        1000.0 * 1.602176634e-19 * MU_0 * 0.5e20
    )


# --- q0, l_i, q_edge ---------------------------------------------------------------


def test_wesson_profile_has_its_known_q_ratio_and_li():
    """j = j0 (1 - rho^2)^nu: q_edge / q0 = nu + 1; nu = 2 gives l_i = 1.2167."""
    for nu in (1.0, 2.0, 3.0):
        _, _, q, _ = cur.shape_profile(2.0, nu)
        assert 1 / q[0] == pytest.approx(nu + 1, rel=1e-5)
    assert cur.shape_profile(2.0, 2.0)[3] == pytest.approx(1.21667, abs=1e-4)


def test_flat_current_has_li_one_half_and_flat_q():
    _, _, q, li = cur.shape_profile(2.0, 1e-9)
    assert li == pytest.approx(0.5, abs=1e-3)
    assert q[0] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("q0, li, q_edge", [(1.0, 1.2, 3.0), (1.6, 0.9, 3.3), (1.05, 1.5, 4.5)])
def test_solve_shape_round_trips(q0, li, q_edge):
    alpha, nu = cur.solve_shape(q0, li, q_edge)
    _, _, q, li_built = cur.shape_profile(alpha, nu)
    assert li_built == pytest.approx(li, abs=1e-6)
    assert q_edge * q[0] == pytest.approx(q0, abs=1e-6)
    assert q[-1] == pytest.approx(1.0)
    assert np.all(np.diff(q) >= -1e-12)          # q rises monotonically


def test_solve_shape_names_the_window_when_q0_is_out_of_reach():
    lo, hi = cur.q0_window(1.2, 3.0)
    assert lo < 1.0 < hi
    with pytest.raises(ValueError, match=f"{lo:.3f} to {hi:.3f}"):
        cur.solve_shape(0.2, 1.2, 3.0)
    with pytest.raises(ValueError, match="0.5"):
        cur.solve_shape(1.0, 0.4, 3.0)


def test_boundary_geometry_of_an_ellipse():
    theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    bnd = np.column_stack((1.7 + 0.5 * np.cos(theta), 0.1 + 0.8 * np.sin(theta)))
    g = cur.boundary_geometry(bnd, F0=-3.4)
    assert (g.R0, g.a, g.kappa, g.B0) == pytest.approx((1.7, 0.5, 1.6, 2.0), rel=1e-4)


def test_current_from_q_li_sets_the_size_from_q0():
    g = cur.PlasmaGeometry(R0=1.5, a=0.5, kappa=1.0, B0=2.0)
    made = cur.current_from_q_li(1.0, 1.2, 3.0, g)
    assert made.j0 == pytest.approx(2 * 2.0 / (MU_0 * 1.5 * 1.0))     # circle
    assert made.j[0] == made.j0 and made.j[-1] == 0.0
    assert made.q[0] == pytest.approx(1.0, abs=1e-6) and made.q[-1] == pytest.approx(3.0)
    # the cylinder's edge q from the total current: q_a = 2 pi a^2 B0 / (mu_0 R0 Ip)
    assert 2 * np.pi * 0.5**2 * 2.0 / (MU_0 * 1.5 * made.Ip) == pytest.approx(3.0, rel=1e-5)
    assert made.psi_n[0] == 0.0 and made.psi_n[-1] == pytest.approx(1.0)


def test_current_from_q_li_elongation_raises_the_current():
    circle = cur.current_from_q_li(1.0, 1.2, 3.0, cur.PlasmaGeometry(1.5, 0.5, 1.0, 2.0))
    ellipse = cur.current_from_q_li(1.0, 1.2, 3.0, cur.PlasmaGeometry(1.5, 0.5, 1.6, 2.0))
    assert ellipse.j0 / circle.j0 == pytest.approx((1 + 1.6**2) / (2 * 1.6))
    np.testing.assert_allclose(ellipse.q, circle.q)


# --- the cylinder picture of any FF' profile --------------------------------------------


@pytest.mark.parametrize("q0, li, q_edge", [(1.05, 1.2, 3.3), (0.95, 1.26, 2.96), (1.6, 0.9, 3.3)])
def test_profile_from_ffprime_returns_what_a_family_member_was_made_from(q0, li, q_edge):
    g = cur.PlasmaGeometry(R0=1.4, a=0.36, kappa=1.1, B0=2.6)
    made = cur.current_from_q_li(q0, li, q_edge, g)
    back = cur.profile_from_ffprime(made.psi_n, cur.ffprime_from_current(made.j, g.R0), g)
    assert (back.q[0], back.li, back.q[-1]) == pytest.approx((q0, li, q_edge), abs=1e-6)
    assert back.Ip == pytest.approx(made.Ip, rel=1e-8) and back.j0 == pytest.approx(made.j0)
    np.testing.assert_allclose(back.psi_n, made.psi_n, atol=1e-9)
    assert back.alpha is None and back.nu is None


def test_profile_from_ffprime_works_on_a_coarse_uneven_grid():
    g = cur.PlasmaGeometry(R0=1.4, a=0.36, kappa=1.1, B0=2.6)
    made = cur.current_from_q_li(1.05, 1.2, 3.3, g)
    psi_n = np.linspace(0, 1, 200) ** 1.3
    ffprime = cur.ffprime_from_current(np.interp(psi_n, made.psi_n, made.j), g.R0)
    back = cur.profile_from_ffprime(psi_n, ffprime, g)
    assert (back.q[0], back.li, back.q[-1]) == pytest.approx((1.05, 1.2, 3.3), rel=2e-3)


def test_profile_from_ffprime_needs_current_on_axis():
    g = cur.PlasmaGeometry(R0=1.4, a=0.36, kappa=1.1, B0=2.6)
    with pytest.raises(ValueError, match="no current on axis"):
        cur.profile_from_ffprime([0.0, 1.0], [0.0, 0.0], g)


def test_parametric_temperature():
    psi_n = np.linspace(0, 1.2, 25)
    t = cur.parametric_temperature(psi_n, 500.0, 20.0)
    assert t[0] == 500.0 and np.all(t[psi_n >= 1.0] == 20.0)
    np.testing.assert_allclose(t[psi_n <= 1], 20.0 + 480.0 * (1 - psi_n[psi_n <= 1]) ** 2)
    assert abs(t[20] - t[19]) < 0.003 * 480.0             # meets the edge with no gradient
    flat = cur.parametric_temperature(psi_n, 500.0, 20.0, alpha=8.0, beta=2.0)
    assert flat[10] > 0.99 * 500.0 > t[10]                # psi_N = 0.5: still core
    np.testing.assert_allclose(cur.parametric_temperature(psi_n, 50.0, 50.0), 50.0)
