"""ashen.diagnostics.four_modes -- max-amplitude time series from the
per-step jorek2_four caches."""

from __future__ import annotations

import numpy as np
import pytest

from ashen.diagnostics import four_cache as fc
from ashen.diagnostics.four_modes import max_amplitude_series, rational_surface_series
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
