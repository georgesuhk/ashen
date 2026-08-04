"""ashen.diagnostics.connection_length -- ports the connection-length math
trapped inside castor3d/util/data_jorek.py's plot_connection_length (:528)
and its unused near-duplicate get_connection_lengths (:491).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ashen.diagnostics.connection_length import (
    connection_length_matrix,
    connection_lengths_for_step,
    group_by_psi_n,
    harmonic_connection_length,
    line_connection_length,
    smooth_ignoring_inf,
)
from ashen.diagnostics.poincare_cache import LineKey, LineRecord

R0 = 1.363245


def record(psi_n_start, rho_values, n_turns=None, R=1.8, Z=0.0, phi=0.0) -> LineRecord:
    """Build a LineRecord whose .psi_n property equals rho_values**2."""
    rho = np.asarray(rho_values, dtype=np.float32)
    key = LineKey(psi_n=psi_n_start, R=R, Z=Z, phi=phi).quantised()
    return LineRecord(
        key=key,
        n_turns=n_turns if n_turns is not None else len(rho),
        terminated=False,
        n_segments=1,
        R=np.full(len(rho), R, dtype=np.float32),
        Z=np.full(len(rho), Z, dtype=np.float32),
        rho=rho,
        theta=np.zeros(len(rho), dtype=np.float32),
    )


# --- one line ------------------------------------------------------------------


def test_a_confined_line_is_infinite():
    """Every puncture stayed inside psi_n=1 for the whole requested length --
    ports calc_single_connection_length's `len(inside) == n_turns -> inf`."""
    r = record(0.2, rho_values=np.sqrt([0.1, 0.2, 0.3, 0.15]))  # all < 1
    assert math.isinf(line_connection_length(r, R0, real_psi_edge=1.0))


def test_an_escaped_line_has_a_finite_length():
    """Punctures with psi_n >= 1 are excluded from the inside count; the
    remaining inside-count times 2*pi*R0 is the reported length."""
    r = record(0.2, rho_values=np.sqrt([0.1, 0.2, 1.5, 0.3]), n_turns=4)
    got = line_connection_length(r, R0, real_psi_edge=1.0)
    assert got == pytest.approx(3 * 2 * math.pi * R0)


def test_real_psi_edge_normalises_the_trace_before_the_threshold():
    """Ports the extra division by psi_edge in plot_connection_length
    (data_jorek.py:546) -- preserved exactly, not resolved. See
    KNOWN_ISSUES.md #7."""
    r = record(0.2, rho_values=np.sqrt([0.4, 0.6]), n_turns=2)  # psi_n = 0.4, 0.6
    # Undivided: both < 1 -> confined (inf).
    assert math.isinf(line_connection_length(r, R0, real_psi_edge=1.0))
    # Divided by 0.5: psi_n becomes 0.8, 1.2 -> one escapes.
    got = line_connection_length(r, R0, real_psi_edge=0.5)
    assert got == pytest.approx(1 * 2 * math.pi * R0)


def test_zero_punctures_is_a_zero_length_not_confined():
    """n_turns=0 makes len(inside)==n_turns==0 trivially true -- matches the
    legacy formula exactly (an edge case the original code also hits)."""
    r = record(0.2, rho_values=[], n_turns=0)
    assert math.isinf(line_connection_length(r, R0, real_psi_edge=1.0))


# --- harmonic mean over a psi_n's angular samples -----------------------------


def test_harmonic_mean_of_finite_lengths():
    lines = [
        record(0.2, rho_values=np.sqrt([0.1, 1.5]), n_turns=2),  # 1 inside -> 2*pi*R0
        record(0.2, rho_values=np.sqrt([0.1, 0.2, 1.5]), n_turns=3, R=1.81),  # 2 inside
    ]
    L = [line_connection_length(r, R0, real_psi_edge=1.0) for r in lines]
    got = harmonic_connection_length(lines, R0, real_psi_edge=1.0)
    expected = 2 / (1 / L[0] + 1 / L[1])
    assert got == pytest.approx(expected)


def test_all_confined_is_infinite():
    lines = [record(0.2, rho_values=np.sqrt([0.1, 0.2])) for _ in range(3)]
    assert math.isinf(harmonic_connection_length(lines, R0, real_psi_edge=1.0))


def test_a_zero_length_line_contributes_zero_not_a_division_error():
    """Ports the `if con_length_single == 0: denom += 0` branch
    (data_jorek.py:558-561) -- a line that escaped on its very first turn
    does not blow up the harmonic mean."""
    escaped_immediately = record(0.2, rho_values=np.sqrt([1.5]), n_turns=1)
    confined = record(0.2, rho_values=np.sqrt([0.1, 0.2]), R=1.81)
    got = harmonic_connection_length(
        [escaped_immediately, confined], R0, real_psi_edge=1.0
    )
    # Only the confined line contributes -> harmonic_connection_length of one
    # infinite value among two lines is still infinite (1/inf == 0 contributes
    # nothing to the denominator either), matching the legacy behaviour.
    assert math.isinf(got)


def test_empty_group_is_nan_not_inf():
    """A requested surface with no cached samples is missing data, which must
    be visually distinguishable from 'every sample was confined'."""
    assert math.isnan(harmonic_connection_length([], R0, real_psi_edge=1.0))


# --- grouping and the per-step row --------------------------------------------


def test_group_by_psi_n_buckets_angular_samples():
    records = {
        record(0.2, [0.3], R=1.7).key: record(0.2, [0.3], R=1.7),
        record(0.2, [0.4], R=1.8).key: record(0.2, [0.4], R=1.8),
        record(0.5, [0.3], R=1.9).key: record(0.5, [0.3], R=1.9),
    }
    grouped = group_by_psi_n(records)
    assert sorted(grouped) == [0.2, 0.5]
    assert len(grouped[0.2]) == 2
    assert len(grouped[0.5]) == 1


def test_connection_lengths_for_step_matches_by_target_and_reports_missing():
    records = {
        record(0.2, np.sqrt([0.1, 0.2]), R=1.7).key: record(0.2, np.sqrt([0.1, 0.2]), R=1.7),
    }
    row = connection_lengths_for_step(records, [0.2, 0.9], real_psi_edge=1.0, R0=R0)
    assert math.isinf(row[0])
    assert math.isnan(row[1])  # 0.9 was never traced this step


def test_connection_length_matrix_shape_and_row_order():
    r1 = record(0.2, np.sqrt([0.1]), R=1.7)
    r2 = record(0.2, np.sqrt([1.5]), n_turns=1, R=1.7)
    records_by_step = {100: {r1.key: r1}, 200: {r2.key: r2}}

    matrix = connection_length_matrix(
        records_by_step, [100, 200], [0.2], real_psi_edge=1.0, R0=R0
    )

    assert matrix.shape == (2, 1)
    assert math.isinf(matrix[0, 0])
    # A single escaped-on-turn-1 line has line-length 0, which the harmonic
    # mean excludes from its denominator entirely (test_a_zero_length_line_
    # contributes_zero_not_a_division_error above) -- so with only one sample
    # the group-level result is also inf, matching the legacy formula exactly.
    assert math.isinf(matrix[1, 0])


def test_matrix_row_for_an_absent_step_is_all_nan():
    matrix = connection_length_matrix({}, [100], [0.2], real_psi_edge=1.0, R0=R0)
    assert np.all(np.isnan(matrix))


# --- smoothing ------------------------------------------------------------------


def test_smoothing_averages_only_finite_neighbours():
    Z = np.array([[10.0, 20.0, np.inf, 40.0, 50.0]])
    smoothed = smooth_ignoring_inf(Z, window=3)
    # Middle-of-window at index 2 is inf and must stay inf.
    assert math.isinf(smoothed[0, 2])
    # index 1's window is {10,20,inf}; inf is excluded from both sum and count.
    assert smoothed[0, 1] == pytest.approx((10.0 + 20.0) / 2)


def test_smoothing_preserves_shape():
    Z = np.random.default_rng(0).random((3, 7))
    assert smooth_ignoring_inf(Z, window=3).shape == Z.shape


def test_an_all_finite_row_is_unaffected_at_the_center():
    Z = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    smoothed = smooth_ignoring_inf(Z, window=3)
    assert smoothed[0, 2] == pytest.approx((2.0 + 3.0 + 4.0) / 3)
