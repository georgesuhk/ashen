"""ashen.diagnostics.poincare_cache -- the incremental Poincare cache.

The properties worth protecting are the ones the legacy dense-``.npz`` format
could not express at all: adding a starting position must leave every existing
line byte-for-byte alone, and extending a trace must append in place rather
than rewrite. Both are asserted directly here, not inferred.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashen.diagnostics import poincare_cache as pc

pytest.importorskip("h5py")


def arrays(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.random(n).astype(np.float32) for name in ("R", "Z", "rho", "theta")}


def key(psi_n=0.5, R=1.8, Z=0.0, phi=0.0) -> pc.LineKey:
    return pc.LineKey(psi_n=psi_n, R=R, Z=Z, phi=phi).quantised()


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "poinc_s000200.h5"


# --- round trip -------------------------------------------------------------------


def test_round_trip(cache_path):
    k, data = key(), arrays(50)
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, k, data, n_turns=50, terminated=False)

    records = pc.read_cache(cache_path)
    assert list(records) == [k]
    record = records[k]
    assert record.n_turns == 50
    assert record.n_segments == 1
    assert record.terminated is False
    assert record.n_points == 50
    for name in ("R", "Z", "rho", "theta"):
        np.testing.assert_allclose(getattr(record, name), data[name])


def test_psi_n_is_derived_from_rho(cache_path):
    """The legacy cache stored only rho**2, discarding the raw quantity the
    tool actually writes."""
    k = key()
    data = arrays(10)
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, k, data, n_turns=10, terminated=False)

    record = pc.read_cache(cache_path)[k]
    np.testing.assert_allclose(record.psi_n, record.rho.astype(float) ** 2)


def test_reading_a_missing_file_is_empty_not_an_error(tmp_path):
    assert pc.read_cache(tmp_path / "nope.h5") == {}


def test_ragged_lines_coexist(cache_path):
    """Field lines leave the mesh at different times, so lengths differ --
    the whole reason the legacy cache needed pickled object arrays."""
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, key(R=1.7), arrays(10), n_turns=100, terminated=True)
        pc.append_line(h, key(R=1.8), arrays(100), n_turns=100, terminated=False)

    records = pc.read_cache(cache_path)
    assert sorted(r.n_points for r in records.values()) == [10, 100]


def test_mismatched_array_lengths_are_refused(cache_path):
    bad = arrays(10)
    bad["theta"] = np.zeros(9, dtype=np.float32)
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        with pytest.raises(pc.PoincareCacheError, match="length mismatch"):
            pc.append_line(h, key(), bad, n_turns=10, terminated=False)


# --- adding a line must not disturb the others ------------------------------------


def test_appending_a_new_line_leaves_existing_ones_identical(cache_path):
    """The property that makes widening psi_n_in cheap *and* safe."""
    first, second = key(psi_n=0.2, R=1.7), key(psi_n=0.3, R=1.9)
    original = arrays(40, seed=1)
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, first, original, n_turns=40, terminated=False)
    before = pc.read_cache(cache_path)[first]

    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, second, arrays(40, seed=2), n_turns=40, terminated=False)

    after = pc.read_cache(cache_path)
    assert set(after) == {first, second}
    assert after[first].n_turns == before.n_turns
    assert after[first].n_segments == before.n_segments
    for name in ("R", "Z", "rho", "theta"):
        np.testing.assert_array_equal(getattr(after[first], name), getattr(before, name))


def test_appending_a_duplicate_is_refused(cache_path):
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, key(), arrays(10), n_turns=10, terminated=False)
        with pytest.raises(pc.PoincareCacheError, match="already cached"):
            pc.append_line(h, key(), arrays(10), n_turns=10, terminated=False)


def test_replace_overwrites_instead_of_raising(cache_path):
    k = key()
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, k, arrays(0), n_turns=10, terminated=False)
        pc.append_line(h, k, arrays(10, seed=5), n_turns=10, terminated=False, replace=True)

    record = pc.read_cache(cache_path)[k]
    assert record.n_points == 10
    assert record.n_segments == 1


# --- extending in place ------------------------------------------------------------


def test_extend_appends_and_leaves_the_head_untouched(cache_path):
    """Raising n_turns 40 -> 100 must cost 60 turns and preserve the first 40
    exactly."""
    k = key()
    head = arrays(40, seed=1)
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, k, head, n_turns=40, terminated=False)

    tail = arrays(60, seed=2)
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.extend_line(h, k, tail, added_turns=60, terminated=False)

    record = pc.read_cache(cache_path)[k]
    assert record.n_turns == 100
    assert record.n_points == 100
    assert record.n_segments == 2  # visibly stitched
    for name in ("R", "Z", "rho", "theta"):
        got = getattr(record, name)
        np.testing.assert_array_equal(got[:40], head[name])
        np.testing.assert_array_equal(got[40:], tail[name])


def test_extending_a_terminated_line_is_refused(cache_path):
    """The line left the mesh; there is no trajectory left to resume."""
    k = key()
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, k, arrays(12), n_turns=100, terminated=True)
        with pytest.raises(pc.PoincareCacheError, match="nothing to resume"):
            pc.extend_line(h, k, arrays(5), added_turns=5, terminated=False)


def test_extending_an_uncached_line_is_refused(cache_path):
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        with pytest.raises(pc.PoincareCacheError, match="not cached"):
            pc.extend_line(h, key(), arrays(5), added_turns=5, terminated=False)


def test_extending_repeatedly_keeps_counting_segments(cache_path):
    k = key()
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, k, arrays(10), n_turns=10, terminated=False)
        pc.extend_line(h, k, arrays(10), added_turns=10, terminated=False)
        pc.extend_line(h, k, arrays(10), added_turns=10, terminated=False)

    record = pc.read_cache(cache_path)[k]
    assert (record.n_turns, record.n_points, record.n_segments) == (30, 30, 3)


# --- work planning -----------------------------------------------------------------


def test_nothing_to_do_when_the_request_is_already_satisfied():
    cached = {key(): _record(key(), n_turns=1000, n_points=1000)}
    new, ext = pc.plan_work(cached, [key()], 1000)
    assert (new, ext) == ([], [])


def test_a_request_below_what_is_cached_is_also_satisfied():
    cached = {key(): _record(key(), n_turns=2000, n_points=2000)}
    new, ext = pc.plan_work(cached, [key()], 1000)
    assert (new, ext) == ([], [])


def test_only_the_added_psi_n_is_new():
    """Widening psi_n_in from [0.2] to [0.2, 0.3] must cost one line."""
    old, added = key(psi_n=0.2, R=1.7), key(psi_n=0.3, R=1.9)
    cached = {old: _record(old, n_turns=1000, n_points=1000)}

    new, ext = pc.plan_work(cached, [old, added], 1000)

    assert [w.key for w in new] == [added]
    assert ext == []
    assert new[0].n_turns == 1000
    assert new[0].resume_from is None


def test_only_the_shortfall_is_traced_when_turns_rise():
    k = key()
    cached = {k: _record(k, n_turns=1000, n_points=1000, last=(1.75, 0.02))}

    new, ext = pc.plan_work(cached, [k], 2000)

    assert new == []
    assert len(ext) == 1
    assert ext[0].n_turns == 1000  # the delta, not the total
    assert ext[0].is_extension
    # The last puncture, read back through float32 storage.
    assert ext[0].resume_from == pytest.approx((1.75, 0.02), rel=1e-6)
    assert ext[0].start == ext[0].resume_from


def test_a_terminated_line_is_never_extended():
    k = key()
    cached = {k: _record(k, n_turns=1000, n_points=12, terminated=True)}
    new, ext = pc.plan_work(cached, [k], 5000)
    assert (new, ext) == ([], [])


def test_a_line_with_no_punctures_is_retraced_not_resumed():
    """Nothing to resume from, so it becomes new work -- and the writer is
    told to replace rather than append."""
    k = key()
    cached = {k: _record(k, n_turns=100, n_points=0)}
    new, ext = pc.plan_work(cached, [k], 1000)
    assert [w.key for w in new] == [k]
    assert ext == []
    assert new[0].resume_from is None


def test_new_work_starts_from_the_key_position():
    k = key(R=1.83, Z=0.11)
    new, _ = pc.plan_work({}, [k], 500)
    assert new[0].start == (1.83, 0.11)


# --- legacy caches ------------------------------------------------------------------


def test_legacy_npz_is_readable(tmp_path):
    psi_in = np.array([0.2, 0.3])
    out = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            out[i, j] = np.arange(3 + i + j, dtype=float) / 10.0
    for kind in ("psi_n", "theta", "R", "Z"):
        np.savez(tmp_path / f"poinc_t000200_{kind}.npz", in_val=psi_in, out_val=out)

    records = pc.read_legacy_cache(tmp_path, "000200")

    assert len(records) == 4
    assert {k.psi_n for k in records} == {0.2, 0.3}


def test_legacy_records_are_not_extendable(tmp_path):
    """They record no starting position, so there is no trajectory to resume
    and no key to match new work against -- such a step gets retraced once."""
    psi_in = np.array([0.2])
    out = np.empty((1, 1), dtype=object)
    out[0, 0] = np.arange(5, dtype=float)
    for kind in ("psi_n", "theta", "R", "Z"):
        np.savez(tmp_path / f"poinc_t000200_{kind}.npz", in_val=psi_in, out_val=out)

    record = next(iter(pc.read_legacy_cache(tmp_path, "000200").values()))
    assert not record.extendable

    # It is never resumed. It is scheduled as a full retrace instead -- the
    # once-only cost of migrating a legacy step.
    new, ext = pc.plan_work({record.key: record}, [record.key], 1000)
    assert ext == []
    assert len(new) == 1
    assert new[0].resume_from is None


def test_absent_legacy_files_are_empty(tmp_path):
    assert pc.read_legacy_cache(tmp_path, "000200") == {}


# --- schema -----------------------------------------------------------------------


def test_wrong_schema_is_rejected(cache_path):
    import h5py

    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, key(), arrays(5), n_turns=5, terminated=False)
    with h5py.File(cache_path, "a") as h:
        h.attrs["schema"] = 99

    with pytest.raises(pc.PoincareCacheError, match="schema 99"):
        pc.read_cache(cache_path)


def test_keys_survive_the_round_trip_exactly(cache_path):
    """Start points are matched by value, so a float that changes in the
    write/read round trip would orphan its own cache entry."""
    k = key(psi_n=0.123456789, R=1.8234567, Z=-0.0412345, phi=0.5)
    with pc.open_cache(cache_path, step=200, pad_width=6) as h:
        pc.append_line(h, k, arrays(5), n_turns=5, terminated=False)

    assert k in pc.read_cache(cache_path)


def _record(k, *, n_turns, n_points, terminated=False, last=(1.7, 0.0)):
    R = np.full(n_points, last[0], dtype=np.float32)
    Z = np.full(n_points, last[1], dtype=np.float32)
    return pc.LineRecord(
        key=k, n_turns=n_turns, terminated=terminated, n_segments=1,
        R=R, Z=Z,
        rho=np.zeros(n_points, dtype=np.float32),
        theta=np.zeros(n_points, dtype=np.float32),
    )
