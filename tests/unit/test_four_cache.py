"""ashen.diagnostics.four_cache -- the jorek2_four HDF5 cache."""

from __future__ import annotations

import numpy as np
import pytest

from ashen.diagnostics import four_cache as fc

pytest.importorskip("h5py")


def _record(variable="Psi", n=0, m=1) -> fc.FourRecord:
    psi_n = np.linspace(0.0, 1.0, 5)
    return fc.FourRecord(
        variable=variable, n=n, m=m,
        psi_n=psi_n, real=psi_n * 2, imag=psi_n * 3,
    )


def test_missing_cache_reads_as_empty(tmp_path):
    assert fc.read_cache(tmp_path / "nope.h5") == {}
    assert fc.count_records(tmp_path / "nope.h5") == 0


def test_write_read_roundtrip(tmp_path):
    path = tmp_path / "four_s000100.h5"
    records = [_record("Psi", 0, 0), _record("Psi", 0, 1), _record("u", 1, 0)]
    fc.write_cache(path, step=100, pad_width=6, records=records)

    loaded = fc.read_cache(path)
    assert set(loaded) == {("Psi", 0, 0), ("Psi", 0, 1), ("u", 1, 0)}
    got = loaded[("Psi", 0, 1)]
    np.testing.assert_allclose(got.psi_n, records[1].psi_n)
    np.testing.assert_allclose(got.real, records[1].real)
    np.testing.assert_allclose(got.imag, records[1].imag)


def test_abs_and_phase_are_derived(tmp_path):
    record = fc.FourRecord(
        variable="Psi", n=0, m=0,
        psi_n=np.array([0.5]), real=np.array([3.0]), imag=np.array([4.0]),
    )
    np.testing.assert_allclose(record.abs, [5.0])
    np.testing.assert_allclose(record.phase, [np.arctan2(4.0, 3.0)])


def test_count_records_matches_write(tmp_path):
    path = tmp_path / "four_s000100.h5"
    records = [_record("Psi", 0, m) for m in range(4)]
    fc.write_cache(path, step=100, pad_width=6, records=records)
    assert fc.count_records(path) == 4


def test_write_is_atomic_no_leftover_tmp(tmp_path):
    path = tmp_path / "four_s000100.h5"
    fc.write_cache(path, step=100, pad_width=6, records=[_record()])
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_wrong_schema_raises(tmp_path, monkeypatch):
    path = tmp_path / "four_s000100.h5"
    fc.write_cache(path, step=100, pad_width=6, records=[_record()])
    monkeypatch.setattr(fc, "SCHEMA_VERSION", fc.SCHEMA_VERSION + 1)
    with pytest.raises(fc.FourCacheError, match="schema"):
        fc.read_cache(path)
    with pytest.raises(fc.FourCacheError, match="schema"):
        fc.count_records(path)
