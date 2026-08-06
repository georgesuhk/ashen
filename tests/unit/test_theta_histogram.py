"""ashen.diagnostics.theta_histogram -- ports the data-reduction half of the
notebook function plot_theta_histogram_matrix (Columbia/NL_kinks/
prod_plots_draft0.ipynb, cell 5): find each field line's first puncture past
a target flux surface and record its poloidal angle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ashen.diagnostics.poincare_cache import LineKey, LineRecord
from ashen.diagnostics.theta_histogram import (
    crossing_angles,
    pooled_crossing_angles,
    theta_histogram,
)


def record(psi_n_start, rho_values, theta_values, R=1.8, Z=0.0, phi=0.0) -> LineRecord:
    rho = np.asarray(rho_values, dtype=np.float32)
    theta = np.asarray(theta_values, dtype=np.float32)
    key = LineKey(psi_n=psi_n_start, R=R, Z=Z, phi=phi).quantised()
    return LineRecord(
        key=key, n_turns=len(rho), terminated=False, n_segments=1,
        R=np.full(len(rho), R, dtype=np.float32),
        Z=np.full(len(rho), Z, dtype=np.float32),
        rho=rho, theta=theta,
    )


# --- first-crossing semantics ---------------------------------------------------


def test_first_crossing_is_used_not_a_later_one():
    """A line crossing the threshold three times contributes only the angle
    at its first crossing in trace order."""
    r = record(
        0.2,
        rho_values=np.sqrt([0.5, 1.5, 0.6, 1.6, 0.4]),  # psi_n: .5,1.5,.6,1.6,.4
        theta_values=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    result = crossing_angles(
        {r.key: r}, target_psi=1.0, real_psi_edge=1.0,
    )
    assert result.angles.size == 1
    assert result.angles[0] == pytest.approx(0.2)


def test_confined_line_contributes_no_angle_but_is_considered():
    r = record(0.2, rho_values=np.sqrt([0.1, 0.2, 0.3]), theta_values=[0.1, 0.2, 0.3])
    result = crossing_angles({r.key: r}, target_psi=1.0, real_psi_edge=1.0)
    assert result.angles.size == 0
    assert result.n_crossed == 0
    assert result.n_considered == 1


def test_n_crossed_and_n_considered_across_multiple_lines():
    crossed = record(0.2, rho_values=np.sqrt([1.5]), theta_values=[0.7], R=1.7)
    confined = record(0.2, rho_values=np.sqrt([0.1]), theta_values=[0.1], R=1.9)
    result = crossing_angles(
        {crossed.key: crossed, confined.key: confined},
        target_psi=1.0, real_psi_edge=1.0,
    )
    assert result.n_crossed == 1
    assert result.n_considered == 2


def test_a_line_with_no_punctures_is_considered_but_confined():
    r = record(0.2, rho_values=[], theta_values=[])
    result = crossing_angles({r.key: r}, target_psi=1.0, real_psi_edge=1.0)
    assert result.angles.size == 0
    assert result.n_considered == 1


# --- real_psi_edge is applied once, to the threshold, not the data ---------------


def test_real_psi_edge_scales_the_threshold_against_raw_psi_n():
    """KNOWN_ISSUES.md #7's settled rule: real_psi_edge converts a
    user-facing plasma-fraction target_psi into JOREK-grid units by scaling
    the threshold -- record.psi_n itself (already JOREK-grid) must never be
    divided. Regression: a future 'fix' that instead divides the data would
    change this result."""
    # psi_n = 0.8 (JOREK-grid). target_psi=1.0 is a plasma-fraction request.
    r = record(0.2, rho_values=np.sqrt([0.8]), theta_values=[0.5])

    # real_psi_edge=1.0 -> threshold=1.0 -> 0.8 does not cross.
    result = crossing_angles({r.key: r}, target_psi=1.0, real_psi_edge=1.0)
    assert result.angles.size == 0

    # real_psi_edge=0.5 -> threshold=0.5 -> 0.8 crosses.
    result = crossing_angles({r.key: r}, target_psi=1.0, real_psi_edge=0.5)
    assert result.angles.size == 1
    assert result.angles[0] == pytest.approx(0.5)


# --- psi_n_range filters the starting surface, scaled the same way ---------------


def test_psi_n_range_filters_by_starting_psi_n():
    inside = record(0.3, rho_values=np.sqrt([1.5]), theta_values=[0.1], R=1.7)
    outside = record(0.8, rho_values=np.sqrt([1.5]), theta_values=[0.2], R=1.9)
    result = crossing_angles(
        {inside.key: inside, outside.key: outside},
        target_psi=1.0, real_psi_edge=1.0, psi_n_range=(0.1, 0.5),
    )
    assert result.n_considered == 1
    assert result.angles.size == 1
    assert result.angles[0] == pytest.approx(0.1)


def test_psi_n_range_is_scaled_by_real_psi_edge_like_the_threshold():
    # Starting psi_n = 0.4 (JOREK-grid). User-facing range [0.5, 0.9].
    r = record(0.4, rho_values=np.sqrt([1.5]), theta_values=[0.1])

    # real_psi_edge=1.0 -> scaled range [0.5, 0.9] -> 0.4 outside -> excluded.
    result = crossing_angles(
        {r.key: r}, target_psi=1.0, real_psi_edge=1.0, psi_n_range=(0.5, 0.9),
    )
    assert result.n_considered == 0

    # real_psi_edge=0.5 -> scaled range [0.25, 0.45] -> 0.4 inside -> included.
    result = crossing_angles(
        {r.key: r}, target_psi=1.0, real_psi_edge=0.5, psi_n_range=(0.5, 0.9),
    )
    assert result.n_considered == 1


# --- wrapping to (-pi, pi] -------------------------------------------------------


def test_angles_are_wrapped_into_minus_pi_to_pi():
    r = record(0.2, rho_values=np.sqrt([1.5]), theta_values=[2 * np.pi + 0.1])
    result = crossing_angles({r.key: r}, target_psi=1.0, real_psi_edge=1.0)
    assert result.angles[0] == pytest.approx(0.1)


def test_angle_just_above_pi_wraps_negative():
    r = record(0.2, rho_values=np.sqrt([1.5]), theta_values=[np.pi + 0.2])
    result = crossing_angles({r.key: r}, target_psi=1.0, real_psi_edge=1.0)
    assert result.angles[0] == pytest.approx(-np.pi + 0.2)


# --- pooling over steps -----------------------------------------------------------


def test_pooled_crossing_angles_concatenates_steps():
    r1 = record(0.2, rho_values=np.sqrt([1.5]), theta_values=[0.1], R=1.7)
    r2 = record(0.2, rho_values=np.sqrt([1.5]), theta_values=[0.2], R=1.8)
    records_by_step = {100: {r1.key: r1}, 200: {r2.key: r2}}
    result = pooled_crossing_angles(
        records_by_step, [100, 200], target_psi=1.0, real_psi_edge=1.0,
    )
    assert sorted(result.angles.tolist()) == pytest.approx([0.1, 0.2])
    assert result.n_crossed == 2
    assert result.n_considered == 2


def test_pooled_crossing_angles_missing_step_contributes_nothing():
    r1 = record(0.2, rho_values=np.sqrt([1.5]), theta_values=[0.1])
    records_by_step = {100: {r1.key: r1}}
    result = pooled_crossing_angles(
        records_by_step, [100, 200], target_psi=1.0, real_psi_edge=1.0,
    )
    assert result.n_considered == 1
    assert result.angles.size == 1


# --- histogram ----------------------------------------------------------------


def test_histogram_weights_sum_to_one():
    angles = np.array([-1.0, 0.0, 0.5, 0.5, 1.0])
    counts, edges = theta_histogram(angles, bins=10)
    assert counts.sum() == pytest.approx(1.0)
    assert edges[0] == pytest.approx(-np.pi)
    assert edges[-1] == pytest.approx(np.pi)


def test_histogram_bin_height_is_fraction_of_lines():
    """Ports the notebook's weights = np.ones_like(data)/len(data) exactly."""
    angles = np.array([0.0, 0.0, 0.0, 1.0])  # 3/4 in one bin, 1/4 in another
    counts, edges = theta_histogram(angles, bins=2)  # bins: [-pi,0), [0,pi]
    assert counts[1] == pytest.approx(1.0)  # both 0.0 and 1.0 in the second half
    assert counts[0] == pytest.approx(0.0)


def test_empty_angles_returns_zero_counts_not_an_error():
    counts, edges = theta_histogram(np.empty(0), bins=5)
    assert counts.shape == (5,)
    assert np.all(counts == 0)
    assert edges.shape == (6,)
