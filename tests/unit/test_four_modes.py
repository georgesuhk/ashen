"""ashen.diagnostics.four_modes -- max-amplitude time series from the
per-step jorek2_four caches."""

from __future__ import annotations

import numpy as np
import pytest

from ashen.diagnostics import four_cache as fc
from ashen.diagnostics.four_modes import max_amplitude_series
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
