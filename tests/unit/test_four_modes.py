"""ashen.diagnostics.four_modes -- max-amplitude time series from the
per-step jorek2_four caches."""

from __future__ import annotations

import numpy as np
import pytest

from ashen.diagnostics import four_cache as fc
from ashen.diagnostics.four_modes import (
    fit_growth_rate,
    format_growth_rates,
    growth_rate_series,
    max_amplitude_series,
    rational_surface_series,
)
from ashen.paths import RunPaths

pytest.importorskip("h5py")


def _record(variable, n, m, *, real_peak: float) -> fc.FourRecord:
    """A record whose max |amplitude| is exactly ``real_peak`` (imag=0)."""
    psi_n = np.linspace(0.0, 1.0, 4)
    real = np.array([0.1, real_peak, 0.2, 0.05])
    return fc.FourRecord(variable=variable, n=n, m=m, psi_n=psi_n, real=real, imag=np.zeros(4))


@pytest.fixture
def paths(tmp_path) -> RunPaths:
    return RunPaths(tmp_path / "run", pad_width=6)


def test_series_tracks_one_mode_across_steps(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_record("Psi", 0, 1, real_peak=1.0)],
    )
    fc.write_cache(
        paths.four_cache(200), step=200, pad_width=6,
        records=[_record("Psi", 0, 1, real_peak=2.5)],
    )

    series = max_amplitude_series(paths, [100, 200])
    np.testing.assert_allclose(series[("Psi", 0, 1)], [1.0, 2.5])


def test_missing_step_cache_is_nan(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_record("Psi", 0, 1, real_peak=1.0)],
    )
    # 200 never gathered.
    series = max_amplitude_series(paths, [100, 200])
    values = series[("Psi", 0, 1)]
    assert values[0] == pytest.approx(1.0)
    assert np.isnan(values[1])


def test_mode_absent_from_one_step_is_nan_there(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_record("Psi", 0, 1, real_peak=1.0), _record("Psi", 1, 0, real_peak=3.0)],
    )
    fc.write_cache(
        paths.four_cache(200), step=200, pad_width=6,
        records=[_record("Psi", 0, 1, real_peak=2.0)],  # n=1,m=0 missing this step
    )
    series = max_amplitude_series(paths, [100, 200])
    assert series[("Psi", 0, 1)] == pytest.approx([1.0, 2.0])
    n1m0 = series[("Psi", 1, 0)]
    assert n1m0[0] == pytest.approx(3.0)
    assert np.isnan(n1m0[1])


def test_variables_filter(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_record("Psi", 0, 0, real_peak=1.0), _record("u", 0, 0, real_peak=2.0)],
    )
    series = max_amplitude_series(paths, [100], variables=["u"])
    assert set(series) == {("u", 0, 0)}


def test_modes_filter(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[
            _record("Psi", 0, 0, real_peak=1.0),
            _record("Psi", 1, 1, real_peak=2.0),
        ],
    )
    series = max_amplitude_series(paths, [100], modes=[(1, 1)])
    assert set(series) == {("Psi", 1, 1)}


def test_no_caches_returns_empty(paths):
    assert max_amplitude_series(paths, [100, 200]) == {}


# --- rational_surface_series ---------------------------------------------------


def _write_qprofile(paths, step, psi_n, q):
    text = ["# Psi_n q", f"# time step #{step:06d}"]
    text += [f"{p} {v}" for p, v in zip(psi_n, q)]
    text.append("")
    paths.qprofile(step).parent.mkdir(parents=True, exist_ok=True)
    paths.qprofile(step).write_text("\n".join(text), encoding="utf-8")


def test_rational_surface_pins_amplitude_to_the_q_crossing(paths):
    # n=1, m=2 -> resonant at q=2.0, which this profile crosses exactly at
    # psi_n=0.5.
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_record("Psi", 1, 2, real_peak=1.0)],
    )
    _write_qprofile(paths, 100, [0.0, 0.5, 1.0], [1.0, 2.0, 3.0])

    series = rational_surface_series(paths, [100], [(1, 2)])
    record = fc.read_cache(paths.four_cache(100))[("Psi", 1, 2)]
    expected = float(np.interp(0.5, record.psi_n, record.abs))

    assert series[("Psi", 1, 2)] == pytest.approx([expected])


def test_n_zero_modes_are_skipped(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_record("Psi", 0, 1, real_peak=1.0)],
    )
    _write_qprofile(paths, 100, [0.0, 0.5, 1.0], [1.0, 2.0, 3.0])

    series = rational_surface_series(paths, [100], [(0, 1)])
    assert series == {}


def test_missing_qprofile_cache_is_nan(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_record("Psi", 1, 2, real_peak=1.0)],
    )
    # No qprofile written for step 100.
    series = rational_surface_series(paths, [100], [(1, 2)])
    assert np.isnan(series[("Psi", 1, 2)][0])


def test_no_crossing_is_nan(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_record("Psi", 1, 2, real_peak=1.0)],
    )
    # q never reaches 2.0 in this profile.
    _write_qprofile(paths, 100, [0.0, 0.5, 1.0], [1.0, 1.2, 1.4])

    series = rational_surface_series(paths, [100], [(1, 2)])
    assert np.isnan(series[("Psi", 1, 2)][0])


def test_variables_filter_applies(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[
            _record("Psi", 1, 2, real_peak=1.0),
            _record("u", 1, 2, real_peak=2.0),
        ],
    )
    _write_qprofile(paths, 100, [0.0, 0.5, 1.0], [1.0, 2.0, 3.0])

    series = rational_surface_series(paths, [100], [(1, 2)], variables=["u"])
    assert set(series) == {("u", 1, 2)}


def test_reversed_shear_takes_the_strongest_crossing(paths):
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_record("Psi", 1, 2, real_peak=5.0)],
    )
    # q=2.0 crossed twice: once near psi_n=0.5 (near the real_peak=5.0
    # sample) and once near psi_n=0.9 (near a small-amplitude sample).
    _write_qprofile(paths, 100, [0.0, 0.5, 0.8, 1.0], [1.0, 2.0, 1.8, 2.2])

    series = rational_surface_series(paths, [100], [(1, 2)])
    record = fc.read_cache(paths.four_cache(100))[("Psi", 1, 2)]
    # Second crossing lies between psi_n=0.8 (q=1.8) and 1.0 (q=2.2).
    from ashen.diagnostics.qprofile import find_rational_surfaces

    crossings = find_rational_surfaces(
        np.array([0.0, 0.5, 0.8, 1.0]), np.array([1.0, 2.0, 1.8, 2.2]), 2.0
    )
    expected = float(np.max(np.interp(crossings, record.psi_n, record.abs)))

    assert series[("Psi", 1, 2)] == pytest.approx([expected])


# --- fit_growth_rate / growth_rate_series / format_growth_rates ------------------


def test_fit_growth_rate_recovers_a_known_gamma():
    t = np.linspace(0.0, 1.0, 20)
    gamma_true, intercept_true = 3.5, -1.2
    y = np.exp(intercept_true + gamma_true * t)

    fit = fit_growth_rate(t, y)

    assert fit is not None
    assert fit.gamma == pytest.approx(gamma_true, rel=1e-6)
    assert fit.intercept == pytest.approx(intercept_true, rel=1e-6)
    assert fit.n_points == 20


def test_fit_growth_rate_ignores_nan_and_nonpositive_points():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, np.nan, 2.0, -1.0, 4.0])  # 3 usable points

    fit = fit_growth_rate(t, y)

    assert fit is not None
    assert fit.n_points == 3


def test_fit_growth_rate_none_with_fewer_than_two_points():
    assert fit_growth_rate([0.0], [1.0]) is None
    assert fit_growth_rate([], []) is None
    assert fit_growth_rate([0.0, 1.0], [1.0, np.nan]) is None


def test_growth_rate_series_one_fit_per_mode():
    gamma_a, gamma_b = 2.0, 5.0
    t = np.linspace(0.0, 1.0, 10)
    series = {
        ("Psi", 1, 2): np.exp(gamma_a * t),
        ("Psi", 2, 3): np.exp(gamma_b * t),
    }
    steps = list(range(100, 100 + 10 * 100, 100))

    fits = growth_rate_series(series, t, steps)

    assert fits[("Psi", 1, 2)].gamma == pytest.approx(gamma_a, rel=1e-6)
    assert fits[("Psi", 2, 3)].gamma == pytest.approx(gamma_b, rel=1e-6)


def test_growth_rate_series_step_range_restricts_the_fit_window():
    steps = [100, 200, 300, 400, 500]
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # Linear growth (gamma=1) only over steps 200-400; noise elsewhere would
    # bias a whole-range fit away from 1.0.
    y = np.array([50.0, np.exp(1.0), np.exp(2.0), np.exp(3.0), 0.001])
    series = {("Psi", 1, 1): y}

    fits = growth_rate_series(series, t, steps, step_range=(200, 400))

    assert fits[("Psi", 1, 1)].gamma == pytest.approx(1.0, rel=1e-6)
    assert fits[("Psi", 1, 1)].n_points == 3


def test_growth_rate_series_skips_modes_with_insufficient_points():
    steps = [100, 200]
    t = np.array([0.0, 1.0])
    series = {("Psi", 1, 1): np.array([1.0, np.nan])}

    fits = growth_rate_series(series, t, steps)

    assert fits == {}


def test_format_growth_rates_sorted_by_variable_then_m_then_n():
    from ashen.diagnostics.four_modes import GrowthFit

    fits = {
        ("Psi", 2, 3): GrowthFit(gamma=5.0, intercept=0.0, n_points=10),
        ("Psi", 1, 2): GrowthFit(gamma=2.0, intercept=0.0, n_points=8),
        ("u", 1, 1): GrowthFit(gamma=1.0, intercept=0.0, n_points=5),
    }
    text = format_growth_rates(fits)
    lines = text.splitlines()

    # header + 3 rows, Psi's m=2 row before its m=3 row, u after Psi.
    assert len(lines) == 4
    assert "Psi" in lines[1] and lines[1].split()[1] == "2"  # m=2 first
    assert "Psi" in lines[2] and lines[2].split()[1] == "3"  # m=3 second
    assert lines[3].startswith("u")
