"""ashen.log_archive -- each JOREK job's log kept once, gzipped, in logs/."""

from __future__ import annotations

import gzip
import json
import os
import time

import pytest

from ashen.log_archive import (
    ARCHIVE_DIR, INDEX_FILE, apply_archive, archive_current_log, archive_name, job_logs,
    open_log, plan_archive, scan_span,
)
from support import jorek_log_text


def _steps(n, dt=0.5, iters=10):
    return [(dt, iters, k == 0) for k in range(n)]


def _old(path, age=3600):
    """Make a file look written an hour ago: not a running job's."""
    then = time.time() - age
    os.utime(path, (then, then))


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "jorek00000.h5").touch()          # restarts padded to 5 digits
    return run


def test_span_and_ending_of_a_log(tmp_path):
    path = tmp_path / "log"
    path.write_text(jorek_log_text(_steps(5), first_index=961, t0=960.0, ending="gmresfail"))
    span = scan_span(path)
    assert (span.first, span.last, span.n_steps, span.outcome, span.mixed) == \
        (961, 965, 5, "gmresfail", False)
    path.write_text(jorek_log_text(_steps(3), ending="nan"))
    assert scan_span(path).outcome == "nan"
    path.write_text(jorek_log_text(_steps(3), ending="sigterm"))
    assert scan_span(path).outcome == "sigterm"
    path.write_text(jorek_log_text(_steps(3)))
    assert scan_span(path).outcome is None


def test_walltime_is_read_from_slurms_err(tmp_path):
    out = tmp_path / "3.2_0.001.4933024.out"
    out.write_text(jorek_log_text(_steps(4)))
    (tmp_path / "3.2_0.001.4933024.err").write_text(
        "error: *** JOB 4933024 ON tok368 CANCELLED AT 2026-09-18T03:06:10 DUE TO TIME LIMIT ***\n")
    assert scan_span(out).outcome == "walltime"


def test_two_jobs_writing_one_log_are_cut_where_the_second_begins(tmp_path):
    """A job launched while another runs in the same folder: its tee
    truncates log, the first keeps writing at its own offset."""
    path = tmp_path / "log"
    first = jorek_log_text(_steps(4), first_index=3401, t0=2180.0)
    other = jorek_log_text(_steps(3), first_index=3856, ending="gmresfail")
    other = other[other.index(" *   time step"):].replace("      1   3856", "   2896   3856")
    path.write_text(first + other)
    span = scan_span(path)
    assert (span.first, span.last, span.mixed) == (3401, 3404, True)
    assert span.outcome is None                         # the abort is the other job's
    assert archive_name(span, 5, 0).endswith("_mixed.gz")


def test_archive_names(run_dir):
    span = scan_span_of(run_dir, _steps(986, dt=1.0), ending="gmresfail")
    assert archive_name(span, 5, 0) == "log_s00001-s00986_gmresfail.gz"
    assert archive_name(span, 6, 0) == "log_s000001-s000986_gmresfail.gz"


def scan_span_of(run_dir, steps, **kw):
    path = run_dir / "scratch.log"
    path.write_text(jorek_log_text(steps, **kw))
    try:
        return scan_span(path)
    finally:
        path.unlink()


def test_job_logs_are_log_and_slurm_outs_of_jorek(run_dir):
    (run_dir / "log").write_text(jorek_log_text(_steps(2)))
    (run_dir / "1.8_1e-05.5229995.out").write_text(jorek_log_text(_steps(2)))
    (run_dir / "trace0000.out").write_text("ptrace output\n")          # no job id
    (run_dir / "postproc.1234.out").write_text("not a JOREK main run\n")
    assert sorted(p.name for p in job_logs(run_dir)) == ["1.8_1e-05.5229995.out", "log"]


def test_archiving_keeps_each_log_once_gzipped_and_indexed(run_dir):
    text = jorek_log_text(_steps(3), first_index=961, t0=960.0)
    (run_dir / "log").write_text(text)
    (run_dir / "a.5230509.out").write_text(text)                      # the same job's copy
    (run_dir / "a.5229995.out").write_text(jorek_log_text(_steps(2, dt=1.0), ending="gmresfail"))
    for path in run_dir.iterdir():
        _old(path)

    actions = plan_archive(run_dir)
    kinds = sorted(a.kind for a in actions)
    assert kinds == ["archive", "archive", "duplicate"]
    for action in actions:
        apply_archive(action)

    names = sorted(p.name for p in (run_dir / ARCHIVE_DIR).glob("*.gz"))
    assert names == ["log_s00001-s00002_gmresfail.gz", "log_s00961-s00963.gz"]
    with open_log(run_dir / ARCHIVE_DIR / "log_s00961-s00963.gz") as f:
        assert f.read() == text
    index = json.loads((run_dir / ARCHIVE_DIR / INDEX_FILE).read_text())
    assert index["log_s00001-s00002_gmresfail.gz"]["outcome"] == "gmresfail"
    # found again later: nothing more to archive
    assert {a.kind for a in plan_archive(run_dir)} == {"duplicate"}


def test_same_steps_different_content_get_their_own_archive(run_dir):
    (run_dir / "log").write_text(jorek_log_text(_steps(3, iters=10)))
    (action,) = plan_archive(run_dir, [run_dir / "log"])
    apply_archive(action)
    (run_dir / "log").write_text(jorek_log_text(_steps(3, iters=11)))
    (action,) = plan_archive(run_dir, [run_dir / "log"])
    assert action.kind == "archive" and action.target.name == "log_s00001-s00003_2.gz"


def test_archive_current_log_copies_and_leaves_log(run_dir):
    assert archive_current_log(run_dir) is None                      # no log yet
    text = jorek_log_text(_steps(3))
    (run_dir / "log").write_text(text)

    note = archive_current_log(run_dir, dry_run=True)
    assert "would keep" in note and not (run_dir / ARCHIVE_DIR).exists()

    note = archive_current_log(run_dir)
    assert "logs/log_s00001-s00003.gz" in note
    assert (run_dir / "log").read_text() == text                      # copied, not moved
    assert "kept already" in archive_current_log(run_dir)
    with gzip.open(run_dir / ARCHIVE_DIR / "log_s00001-s00003.gz", "rt") as f:
        assert f.read() == text


def test_a_log_with_no_time_step_is_named_by_when_it_was_written(run_dir):
    (run_dir / "log").write_text("  GIT revision        : v2\n ERROR: something\n")
    (action,) = plan_archive(run_dir, [run_dir / "log"])
    assert action.target.name.startswith("log_nosteps_")


def test_no_restarts_yet_uses_the_default_width(tmp_path):
    (tmp_path / "log").write_text(jorek_log_text(_steps(1)))
    (action,) = plan_archive(tmp_path, [tmp_path / "log"])
    assert action.target.name == "log_s000001-s000001.gz"
