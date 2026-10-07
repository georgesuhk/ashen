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
