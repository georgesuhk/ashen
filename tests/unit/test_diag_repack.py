"""Repacking particle diagnostics files to their data's size: automatically
when a trace ends, and util --func compress_traces for the ones before."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from ashen.cli import util as util_cli
from ashen.diag_repack import data_size, diag_files, is_compressed, repack
from ashen.diagnostics.particle_exits import read_particle_diag

N, ROWS = 300, 40


def write_like_jorek(path, n=N, rows=ROWS):
    """A write_particle_diagnostics file as JOREK lays it out: every dataset
    chunked 50000 particles x 1 time, uncompressed, grown a row at a time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("groups").create_group("001")
        g.attrs["mass"] = 5.48579909065e-4
        t = g.create_dataset("t", shape=(0,), maxshape=(None,), chunks=(1000,), dtype="f4")
        sets = {}
        for name, dtype in (("psi_n", "f4"), ("R", "f4"), ("Z", "f4"), ("phi", "f4"),
                            ("theta", "f4"), ("e", "f8"), ("lost", "i4")):
            sets[name] = g.create_dataset(name, shape=(0, n), maxshape=(None, None),
                                          chunks=(1, 50000), dtype=dtype)
        for name in ("psi_axis", "psi_sep"):
            f.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(1000,), dtype="f4")
        for r in range(rows):
            t.resize((r + 1,))
            t[r] = r * 1e-8
            for name, d in sets.items():
                d.resize((r + 1, n))
                d[r] = (rng.integers(0, 2, n) if name == "lost"
                        else rng.normal(1.0, 0.2, n))
            for name in ("psi_axis", "psi_sep"):
                f[name].resize((r + 1,))
                f[name][r] = -0.3 if name == "psi_axis" else 0.0
    return path


def test_repack_keeps_every_value_and_frees_the_padding(tmp_path):
    original = write_like_jorek(tmp_path / "a.h5")
    copy = tmp_path / "b.h5"
    copy.write_bytes(original.read_bytes())
    data = data_size(copy)
    assert copy.stat().st_size > 20 * data         # JOREK's chunks: mostly padding

    result = repack(copy)
    assert result.skipped is None and result.after == copy.stat().st_size
    assert result.after < data                      # fitted, then compressed
    with h5py.File(original) as a, h5py.File(copy) as b:
        for name in ("groups/001/psi_n", "groups/001/e", "groups/001/lost", "groups/001/t",
                     "psi_axis", "psi_sep"):
            assert a[name].dtype == b[name].dtype
            np.testing.assert_array_equal(a[name][()], b[name][()])
            assert b[name].compression == "gzip"
            assert all(m is None for m in b[name].maxshape)   # still extensible
        assert b["groups/001"].attrs["mass"] == a["groups/001"].attrs["mass"]
    # what the plots read is the same
    before, after = read_particle_diag(original), read_particle_diag(copy)
    for name in ("time", "psi_n", "R", "Z", "phi", "theta", "lost"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))


def test_a_compressed_file_is_left_alone(tmp_path):
    path = write_like_jorek(tmp_path / "a.h5")
    repack(path)
    assert is_compressed(path)
    assert repack(path).skipped == "already compressed"


def test_a_file_that_is_not_hdf5_is_left_alone(tmp_path):
    path = tmp_path / "ptrace_diag.h5"
    path.write_bytes(b"")
    assert repack(path).skipped == "not an HDF5 file"


def test_a_copy_that_does_not_match_is_never_put_in_place(tmp_path, monkeypatch):
    import ashen.diag_repack as module

    path = write_like_jorek(tmp_path / "a.h5")
    size = path.stat().st_size
    monkeypatch.setattr(module, "_same", lambda copied, path, pool: "R: values differ")
    with pytest.raises(ValueError, match="not identical"):
        repack(path)
    assert path.stat().st_size == size and not is_compressed(path)
    assert not list(tmp_path.glob(".*repack"))


@pytest.mark.parametrize("workers", [1, 4])
def test_many_chunks_and_blocks_with_a_padded_edge_keep_every_value(tmp_path, monkeypatch, workers):
    """Small chunks and blocks: 40 rows in chunks of 3 (the last one
    padded) and blocks of 6, compressed on one thread or several."""
    import ashen.diag_repack as module

    original = write_like_jorek(tmp_path / "a.h5")
    copy = tmp_path / "b.h5"
    copy.write_bytes(original.read_bytes())
    monkeypatch.setattr(module, "_CHUNK_BYTES", 3 * N * 4)     # 3 rows of f4
    monkeypatch.setattr(module, "_BLOCK_BYTES", 7 * N * 4)     # 7 rows: 6, a whole 2 chunks
    monkeypatch.setattr(module, "_MAX_WORKERS", workers)
    assert repack(copy).skipped is None
    with h5py.File(original) as a, h5py.File(copy) as b:
        assert b["groups/001/R"].chunks == (3, N)
        assert b["groups/001/e"].chunks == (1, N)              # f8: a row is over the size
        for name in ("groups/001/psi_n", "groups/001/R", "groups/001/e", "groups/001/lost",
                     "groups/001/t", "psi_axis"):
            assert a[name][()].tobytes() == b[name][()].tobytes()
            assert b[name].compression == "gzip" and b[name].shuffle
        # still extensible, and what is appended reads back
        b.close()
    with h5py.File(copy, "a") as b:
        d = b["groups/001/R"]
        d.resize((ROWS + 1, N))
        d[ROWS] = 7.0
    with h5py.File(original) as a, h5py.File(copy) as b:
        np.testing.assert_array_equal(b["groups/001/R"][:ROWS], a["groups/001/R"][()])
        assert (b["groups/001/R"][ROWS] == 7.0).all()


@pytest.mark.parametrize("flip", ["first", "last"])
def test_a_chunk_that_went_wrong_is_caught_by_the_read_back(tmp_path, monkeypatch, flip):
    """The check is of the finished file, read back through HDF5: one value
    off in one chunk -- here as it is packed -- and the original stays."""
    import ashen.diag_repack as module

    path = write_like_jorek(tmp_path / "a.h5")
    size = path.stat().st_size
    monkeypatch.setattr(module, "_CHUNK_BYTES", 3 * N * 4)
    real, calls = module._packed, []

    def packed(rows, chunk):
        calls.append(1)
        if rows.dtype == np.float32 and rows.ndim == 2 and len(calls) == 5:
            rows = rows.copy()
            rows[0 if flip == "first" else -1, 0 if flip == "first" else -1] += 1e-3
        return real(rows, chunk)

    monkeypatch.setattr(module, "_MAX_WORKERS", 1)             # so the fifth call is one chunk
    monkeypatch.setattr(module, "_packed", packed)
    with pytest.raises(ValueError, match="not identical .*values differ"):
        repack(path)
    assert path.stat().st_size == size and not is_compressed(path)
    assert not list(tmp_path.glob(".*repack"))


def test_the_original_is_read_once(tmp_path, monkeypatch):
    import ashen.diag_repack as module

    path = write_like_jorek(tmp_path / "a.h5")
    opened = []
    real = h5py.File

    def file(name, mode="r", *args, **kwargs):
        opened.append((Path(name).name, mode))
        return real(name, mode, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", file)
    assert repack(path).skipped is None
    # once to see whether it is compressed (no data read), once to copy
    assert opened.count(("a.h5", "r")) == 2


# --- when a trace ends --------------------------------------------------------------


def test_a_finished_trace_is_repacked(tmp_path):
    """A stand-in program that writes its diagnostics as JOREK does: ptrace
    repacks the file once it exits, and says so."""
    from ashen.cases import Case
    from ashen.config import Diagnostics, Launch, Site
    from ashen.ptracing import plan_ptrace, run_ptrace

    run = tmp_path / "run"
    (run / "exe").mkdir(parents=True)
    (run / "jorek03000.h5").write_bytes(b"h5")
    (run / "in_main").write_text("&in1\n&end\n")
    exe = run / "exe" / "tracer"
    exe.write_text(f"""#!{sys.executable}
import sys; sys.path[:0] = [{str(os.path.dirname(__file__))!r},
                            {str(Path(__file__).resolve().parents[2] / "src")!r}]
from pathlib import Path
from test_diag_repack import write_like_jorek
write_like_jorek(Path("ptrace_diag.h5"))
Path("part_restart.h5").write_bytes(b"p")
""")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    site = Site(source=tmp_path / "site.toml", root=tmp_path,
                paths={k: tmp_path / k for k in ("exe", "template", "jobscripts", "jorek",
                                                 "jorek_re", "castor_root")},
                launch=Launch(mpirun=""), diagnostics=Diagnostics())
    case = Case(name="run", steps=[3000], ptrace_exe="./exe/tracer")
    plan = plan_ptrace(case, run, site, omp_threads=1)
    result = run_ptrace(plan)
    diag = plan.work_dir / "ptrace_diag.h5"
    assert is_compressed(diag)
    assert any(note.startswith("ptrace_diag.h5 repacked from") and note.endswith("losslessly")
               for note in result.notes)


# --- util --func compress_traces -------------------------------------------------------


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    run = tmp_path / "run"
    old = 1_000_000_000
    for folder in ("ptrace/ptrace_gc/E10000000eV_n300", "ptrace/ex7_jorek", "trace/trace_gc"):
        name = "trace_diag.h5" if folder.startswith("trace/") else (
            "diag.h5" if "ex7" in folder else "ptrace_diag.h5")
        path = write_like_jorek(run / folder / name)
        os.utime(path, (old, old))
    (tmp_path / "cases.toml").write_text("[cases.run]\nsteps = [3000]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return run


def test_every_layout_is_found(campaign):
    assert [p.relative_to(campaign).as_posix() for p in diag_files(campaign)] == [
        "ptrace/ex7_jorek/diag.h5", "ptrace/ptrace_gc/E10000000eV_n300/ptrace_diag.h5",
        "trace/trace_gc/trace_diag.h5"]


def test_without_apply_it_only_says_what_it_would_free(campaign, capsys):
    sizes = [p.stat().st_size for p in diag_files(campaign)]
    assert util_cli.main(["--func", "compress_traces"]) == 0
    out = capsys.readouterr().out
    assert "would repack 3 file(s)" in out and "add --apply to repack" in out
    assert [p.stat().st_size for p in diag_files(campaign)] == sizes


def test_apply_repacks_them(campaign, capsys):
    assert util_cli.main(["--func", "compress_traces", "--apply"]) == 0
    out = capsys.readouterr().out
    assert "repacked 3 file(s)" in out and "every value checked identical" in out
    assert all(is_compressed(p) for p in diag_files(campaign))
    # again: nothing left to do, and not taken for a trace still running
    assert util_cli.main(["--func", "compress_traces", "--apply"]) == 0
    again = capsys.readouterr().out
    assert again.count("already compressed") == 3 and "may still be running" not in again


def test_a_file_still_being_written_is_left_alone(campaign, capsys):
    busy = campaign / "ptrace" / "ex7_jorek" / "diag.h5"
    os.utime(busy, None)                             # just now
    assert util_cli.main(["--func", "compress_traces", "--apply"]) == 0
    assert not is_compressed(busy)
    assert "its trace may still be running" in capsys.readouterr().out
