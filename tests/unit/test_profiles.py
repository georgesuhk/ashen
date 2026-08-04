"""Tests for CASTOR3D -> JOREK profile translation.

ffprime and T are ported faithfully (including a known grid bug in T, see
profiles.py's module docstring), so most tests check shape and known-value
properties against synthetic cotrans-format files rather than real physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashen.profiles import (
    create_profile_from_castor,
    get_ffprime_profile_from_castor,
    get_psi_from_castor,
    get_t_profile_from_castor,
    resample_profile,
    smooth_profile_preserve_center,
)


def write_two_col(path, x, y):
    with open(path, "w", encoding="utf-8") as f:
        for xi, yi in zip(x, y):
            f.write(f"   {xi:.8E}     {yi:.8E}\n")


@pytest.fixture
def cotrans_dir(tmp_path):
    """A synthetic cotrans directory in the real xn_*_stor0_<suffix> layout."""
    n = 200
    index = np.arange(n)

    # xn_fpol: psi grid, saturating shape like the real fixture
    psi = 1.0 * (1 - np.exp(-index / 150.0))
    write_two_col(tmp_path / "xn_fpol_stor0_TEST", index, psi)

    # xn_hjpol: poloidal current, one extra half-mesh point (dropped via [1:])
    jpol = 1.0e7 * np.exp(-index / 300.0)
    jpol_full = np.concatenate([[jpol[0]], jpol])  # n+1 points
    write_two_col(tmp_path / "xn_hjpol_stor0_TEST", np.arange(n + 1), jpol_full)

    # xn_hpres: pressure, same half-mesh convention
    pres = 100.0 * (1 - index / n)
    pres_full = np.concatenate([[pres[0]], pres])
    write_two_col(tmp_path / "xn_hpres_stor0_TEST", np.arange(n + 1), pres_full)

    return tmp_path, psi


# --- get_psi_from_castor -------------------------------------------------------


def test_reads_psi_and_normalises(cotrans_dir):
    directory, psi = cotrans_dir

    result = get_psi_from_castor(str(directory), suffix="TEST")

    assert len(result.psi) == len(psi)
    assert result.psi_edge == pytest.approx(psi[-1])
    assert result.psi_n[-1] == pytest.approx(1.0)
    assert result.psi_n[0] == pytest.approx(0.0)


def test_psi_is_absolute_valued(tmp_path):
    """The old code wraps the column in abs() -- sign is not physical here."""
    write_two_col(tmp_path / "xn_fpol_stor0_TEST", [0, 1, 2], [-0.1, -0.5, -1.0])

    result = get_psi_from_castor(str(tmp_path), suffix="TEST")

    assert np.all(result.psi >= 0)


def test_suffix_selects_the_file(tmp_path):
    write_two_col(tmp_path / "xn_fpol_stor0_JET", [0, 1], [0.0, 1.0])

    with pytest.raises(OSError):
        get_psi_from_castor(str(tmp_path), suffix="DIIID")


# --- smoothing / resampling -----------------------------------------------------


def test_smooth_preserves_the_protected_region():
    x = np.linspace(-1, 1, 101)
    y = np.where(np.abs(x) < 0.05, 999.0, np.sin(5 * x))

    smoothed = smooth_profile_preserve_center(x, y, sigma=5, preserve_width=0.05)

    mask = np.abs(x) < 0.05
    assert np.array_equal(smoothed[mask], y[mask])
    assert not np.array_equal(smoothed[~mask], y[~mask])


def test_resample_endpoints_are_preserved():
    x = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * x)

    x_new, y_new = resample_profile(x, y, n_points=200)

    assert x_new[0] == pytest.approx(0.0)
    assert x_new[-1] == pytest.approx(1.0)
    assert len(x_new) == 200


def test_resample_onto_explicit_grid():
    x = np.linspace(0, 1, 50)
    y = x**2
    x_new = np.array([0.25, 0.5, 0.75])

    _, y_new = resample_profile(x, y, x_new=x_new, kind="linear")

    assert y_new == pytest.approx(x_new**2, abs=1e-3)


def test_resample_requires_a_target():
    with pytest.raises(ValueError, match="n_points or x_new"):
        resample_profile([0, 1], [0, 1])


# --- ffprime ---------------------------------------------------------------------


def test_ffprime_output_matches_psi_length(cotrans_dir):
    directory, psi = cotrans_dir

    ffprime = get_ffprime_profile_from_castor(psi, str(directory), suffix="TEST")

    assert len(ffprime) == len(psi)


def test_ffprime_drops_the_half_mesh_leading_point(tmp_path):
    """The [1:] slice is what aligns the half-mesh hjpol grid to psi's."""
    psi = np.linspace(0, 1, 10)
    write_two_col(tmp_path / "xn_fpol_stor0_TEST", psi, psi)
    # a jpol file with one more point than psi, as in the real half-mesh data
    jpol_full = np.arange(11, dtype=float)
    write_two_col(tmp_path / "xn_hjpol_stor0_TEST", np.arange(11), jpol_full)

    # must not raise despite the length mismatch before slicing
    result = get_ffprime_profile_from_castor(psi, str(tmp_path), suffix="TEST")

    assert len(result) == len(psi)


# --- T profile, including the known grid bug ------------------------------------


def test_t_profile_output_length_matches_its_own_psi_grid(cotrans_dir):
    """T's length is governed by its *own* internally-read psi, not the caller's."""
    directory, psi = cotrans_dir

    t_profile = get_t_profile_from_castor([1e19], str(directory), suffix="TEST")

    assert len(t_profile) == len(psi)


def test_t_profile_ignores_the_shape_of_an_unrelated_grid(cotrans_dir):
    """Regression guard for the known bug: T never looks at a caller-supplied grid.

    If a caller-supplied psi were ever threaded through, the result length
    would follow it. It doesn't -- T always comes out at the length of its own
    xn_fpol_stor0_<suffix> read, regardless of what else is going on. This
    test exists so a future edit that starts respecting an extended grid
    changes this test's outcome, making the fix visible rather than silent.
    """
    directory, psi = cotrans_dir

    t_profile = get_t_profile_from_castor([1e19], str(directory), suffix="TEST")

    assert len(t_profile) == len(psi)  # not len(psi) + anything


def test_t_profile_requires_a_sequence_not_a_bare_scalar(cotrans_dir):
    """Documents the expected call shape: rho_profile=[value], not a bare float.

    np.asarray(scalar) is 0-d, and len() on that raises -- matching how
    run_jorek.py always calls this (rho_prof = [rho_const_jorek]).
    """
    directory, _ = cotrans_dir

    with pytest.raises(TypeError):
        get_t_profile_from_castor(1e19, str(directory), suffix="TEST")


def test_t_profile_output_is_independent_of_rho(cotrans_dir):
    """T_eV = pres/(rho*e), then T_jorek = T_eV*(rho*e*MU_0) -- rho and e cancel
    exactly, so T_jorek reduces to pres*MU_0 regardless of rho_profile.

    This was an unexpected finding verified against the real qa2.1_g2.300
    fixture (see profiles.py docstring) -- pinned here as a regression test so
    a future edit that changes this is visible rather than silent. Not
    asserted to be correct physics, only that today's implementation behaves
    this way.
    """
    directory, _ = cotrans_dir

    result_small = get_t_profile_from_castor([1e-6], str(directory), suffix="TEST")
    result_large = get_t_profile_from_castor([1e6], str(directory), suffix="TEST")

    assert np.allclose(result_small, result_large)


def test_t_profile_floors_near_zero_values(tmp_path):
    write_two_col(tmp_path / "xn_fpol_stor0_TEST", [0, 1, 2], [0.0, 0.5, 1.0])
    write_two_col(tmp_path / "xn_hpres_stor0_TEST", [0, 1, 2, 3], [0, 0.0, 0.0, 0.0])

    t_profile = get_t_profile_from_castor([1e19], str(tmp_path), suffix="TEST")

    assert np.all(t_profile >= 1e-8)


# --- dispatch ----------------------------------------------------------------


def test_dispatch_ffprime(cotrans_dir):
    directory, psi = cotrans_dir

    result = create_profile_from_castor("ffprime", psi, str(directory), suffix="TEST")

    assert len(result) == len(psi)


def test_dispatch_t_requires_rho(cotrans_dir):
    directory, psi = cotrans_dir

    with pytest.raises(ValueError, match="rho_profile"):
        create_profile_from_castor("T", psi, str(directory), suffix="TEST")


def test_dispatch_t_with_rho(cotrans_dir):
    directory, psi = cotrans_dir

    result = create_profile_from_castor(
        "T", psi, str(directory), suffix="TEST", rho_profile=[1e19]
    )

    assert len(result) == len(psi)


def test_dispatch_unknown_profile_raises_rather_than_returning_unbound():
    """The old create_profiles_from_castor returned an unbound `prof` here."""
    with pytest.raises(ValueError, match="unknown profile"):
        create_profile_from_castor("nonsense", [0, 1], "unused", suffix="TEST")
