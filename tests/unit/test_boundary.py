"""Tests for boundary geometry and psi-grid extension.

extend_psi/extend_prof are ported faithfully from run_jorek_util.py, so these
tests check shape and known-fixed-point properties of the saturating fit
rather than re-deriving the physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashen.boundary import (
    boundary_center,
    downsample_boundary,
    expand_boundary,
    extend_prof,
    extend_psi,
)


def synthetic_circle(n=100, R0=1.5, a=0.5, Z0=0.0):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = R0 + a * np.cos(theta)
    Z = Z0 + a * np.sin(theta)
    return np.column_stack([R, Z])


# --- expand_boundary ----------------------------------------------------------


def test_expand_boundary_scale_one_is_identity():
    bnd = synthetic_circle()
    out = expand_boundary(bnd, R0=1.5, Z0=0.0, scale=1.0)
    assert np.allclose(out, bnd)


def test_expand_boundary_scales_distance_from_center():
    bnd = synthetic_circle(R0=1.5, a=0.5, Z0=0.0)
    out = expand_boundary(bnd, R0=1.5, Z0=0.0, scale=2.0)

    orig_radius = np.linalg.norm(bnd - [1.5, 0.0], axis=1)
    new_radius = np.linalg.norm(out - [1.5, 0.0], axis=1)
    assert np.allclose(new_radius, 2 * orig_radius)


def test_expand_boundary_center_is_fixed():
    bnd = synthetic_circle()
    center = np.array([1.5, 0.0])
    out = expand_boundary(bnd - center + center, R0=1.5, Z0=0.0, scale=3.0)
    # the center itself, if it were a boundary point, would not move
    assert np.allclose(expand_boundary(center[None, :], 1.5, 0.0, 3.0), center)


# --- boundary_center ------------------------------------------------------------


def test_boundary_center_matches_legacy_indexing():
    """R0 = R at max Z; Z0 = Z at max R -- not a centroid, preserved as-is."""
    bnd = np.array([[1.0, 0.0], [2.0, 5.0], [0.5, 3.0]])

    R0, Z0 = boundary_center(bnd)

    assert R0 == 2.0  # R at the point of maximum Z
    assert Z0 == 5.0  # Z at the point of maximum R


# --- downsample_boundary --------------------------------------------------------


def test_downsample_preserves_point_count():
    bnd = synthetic_circle(n=200)
    out = downsample_boundary(bnd, n_points=50)
    assert out.shape == (50, 2)


def test_downsample_preserves_closed_shape_arc_length():
    bnd = synthetic_circle(n=200, R0=1.5, a=0.5)
    out = downsample_boundary(bnd, n_points=64, closed=True)

    def perimeter(pts, closed):
        p = np.vstack([pts, pts[0]]) if closed else pts
        return np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))

    assert perimeter(out, closed=True) == pytest.approx(
        2 * np.pi * 0.5, rel=1e-2
    )


def test_downsample_open_curve_does_not_wrap():
    line = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
    out = downsample_boundary(line, n_points=5, closed=False)

    assert out.shape == (5, 2)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] == pytest.approx(1.0)


# --- extend_psi / extend_prof ---------------------------------------------------


@pytest.fixture
def saturating_psi():
    """A grid whose shape extend_psi's fit model matches exactly.

    Must stay well short of the asymptote (amplitude=1.0): extend_ratio=1.2
    asks the fit to reach psi[-1]*1.2, and the inverse fit is undefined once
    that target reaches or exceeds the amplitude.
    """
    amplitude, tau = 1.0, 150.0
    index = np.arange(200)
    return amplitude * (1 - np.exp(-index / tau))


def test_extend_psi_grid_grows_by_extend_reso(saturating_psi):
    result = extend_psi(saturating_psi, extend_ratio=1.2, extend_reso=20)

    assert len(result.psi) == len(saturating_psi) + 20
    assert len(result.extended_idx_range) == 20


def test_extend_psi_original_points_are_preserved(saturating_psi):
    """The fit must reproduce the input it was fit to, not just extrapolate."""
    result = extend_psi(saturating_psi, extend_ratio=1.2, extend_reso=20)

    assert np.allclose(
        result.psi[: len(saturating_psi)], saturating_psi, rtol=1e-3
    )


def test_extend_psi_new_edge_is_normalised_to_one(saturating_psi):
    result = extend_psi(saturating_psi, extend_ratio=1.2, extend_reso=20)

    assert result.psi_n[-1] == pytest.approx(1.0)


def test_extend_psi_real_psi_edge_is_the_old_edge_fraction(saturating_psi):
    """real_psi_edge locates the true plasma edge inside the extended grid.

    It is the ratio of the true (old) edge to the far extended edge -- so
    callers can rescale a normalised psi_n back to where the real plasma
    boundary sits inside the new, larger grid.
    """
    result = extend_psi(saturating_psi, extend_ratio=1.2, extend_reso=20)

    assert 0 < result.real_psi_edge < 1
    assert result.real_psi_edge == pytest.approx(
        saturating_psi[-1] / result.psi[-1], rel=1e-2
    )
    # equivalently: the true edge's normalised position in the new grid
    assert result.psi_n[len(saturating_psi) - 1] == pytest.approx(
        result.real_psi_edge, rel=1e-2
    )


def test_extend_psi_ratio_one_barely_extends(saturating_psi):
    """extend_ratio=1 asks the fit to reach almost exactly the current edge."""
    result = extend_psi(saturating_psi, extend_ratio=1.0, extend_reso=5)

    assert result.real_psi_edge == pytest.approx(1.0, abs=0.05)


def test_extend_prof_appends_flat_tail(saturating_psi):
    result = extend_psi(saturating_psi, extend_ratio=1.2, extend_reso=20)
    prof = np.linspace(0, 5, len(saturating_psi))

    extended = extend_prof(prof, result.extended_idx_range)

    assert len(extended) == len(prof) + len(result.extended_idx_range)
    assert np.all(extended[len(prof):] == prof[-1])


def test_extend_prof_leaves_original_values_untouched(saturating_psi):
    result = extend_psi(saturating_psi, extend_ratio=1.2, extend_reso=20)
    prof = np.linspace(0, 5, len(saturating_psi))

    extended = extend_prof(prof, result.extended_idx_range)

    assert np.array_equal(extended[: len(prof)], prof)


def test_extend_prof_tail_length_matches_extended_idx_range_not_psi():
    """extend_prof only needs the count, not the psi grid itself."""
    fake_range = np.linspace(100, 150, 7)
    prof = np.array([1.0, 2.0, 3.0])

    extended = extend_prof(prof, fake_range)

    assert len(extended) == 3 + 7
