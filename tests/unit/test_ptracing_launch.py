"""ashen.ptracing -- default start/end steps, and queuing a trace with a
jobscript (`ptrace --run`) against stand-ins for sbatch, squeue and sacct."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from ashen.ptracing import (
    LOG_FILE,
    META_FILE,
    PARTICLES_FILE,
    PtraceError,
    is_current,
    plan_ptrace,
    poll_job,
    run_ptrace,
    traced_start,
    traced_steps,
)
from test_ptracing import RE_GC, _case, make_run, run_dir, site  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="stand-ins are POSIX shell scripts")

JOBSCRIPT = "#!/bin/bash -l\n#SBATCH --time=02:00:00\nsrun ./\"$1\" < ./\"$2\" | tee \"$3\"\n"

#: Stand-ins for SLURM: sbatch records its arguments and prints a job id;
#: squeue and sacct print whatever state the test sets.
SBATCH = """#!/bin/sh
echo "$@" > sbatch_args.txt
echo "${FAKE_SBATCH_OUT:-4242}"
exit "${FAKE_SBATCH_EXIT:-0}"
"""
SQUEUE = """#!/bin/sh
[ -n "$FAKE_SQUEUE" ] && echo "$FAKE_SQUEUE"
exit 0
"""
SACCT = """#!/bin/sh
[ -n "$FAKE_SACCT" ] && echo "$FAKE_SACCT"
exit 0
"""


@pytest.fixture
def slurm(tmp_path, monkeypatch, site):  # noqa: F811
    bin_dir = tmp_path / "fake_bin"
    bin_dir.mkdir()
    for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE), ("sacct", SACCT)):
        script = bin_dir / name
        script.write_text(text, encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    site.jobscripts.mkdir(parents=True, exist_ok=True)
    for job in ("2h", "23h"):
        (site.jobscripts / job).write_text(JOBSCRIPT, encoding="utf-8")
    return site


def _queue(run_dir, site, **overrides):  # noqa: F811
    plan = plan_ptrace(_case(**overrides), run_dir, site, omp_threads=4, job="2h")
    return plan, run_ptrace(plan)


def _job_runs(plan, **env):
    """Do what the jobscript would: run the program in the trace folder."""
    exe = os.path.relpath(plan.exe, plan.work_dir)
    with open(plan.work_dir / plan.namelist, "rb") as stdin, \
            open(plan.work_dir / LOG_FILE, "w") as log:
        subprocess.run(["./" + exe], cwd=plan.work_dir, stdin=stdin, stdout=log,
                       env={**os.environ, **env}, check=False)


# --- default start and end steps ----------------------------------------------


def test_steps_default_to_the_first_and_last_restart(run_dir, site):  # noqa: F811
    plan = plan_ptrace(_case(start_step=None), run_dir, site, omp_threads=1)
    assert plan.steps == [3000, 3200, 3400]
    assert dict(plan.links)["jorek_restart.h5"].name == "jorek03000.h5"
    assert dict(plan.links)["jorek_pdf.h5"].name == "jorek03000.h5"


def test_default_start_with_an_end_step(run_dir, site):  # noqa: F811
    plan = plan_ptrace(_case(start_step=None, end_step=3200), run_dir, site, omp_threads=1)
    assert plan.steps == [3000, 3200]


def test_a_run_without_restarts(tmp_path, site):  # noqa: F811
    run = make_run(tmp_path, steps=())
    with pytest.raises(PtraceError, match="the run has no restarts"):
        plan_ptrace(_case(start_step=None), run, site, omp_threads=1)


def test_traced_steps_is_the_one_rule():
    assert traced_steps(_case(start_step=None), [0, 100, 200]) == [0, 100, 200]
    assert traced_steps(_case(start_step=100), [0, 100, 200]) == [100, 200]
    assert traced_steps(_case(start_step=None, end_step=150), [0, 100, 200]) == [0, 100]


def test_traced_start_for_the_plots(run_dir, site):  # noqa: F811
    assert traced_start(_case(start_step=3200), run_dir) == 3200
    # not traced yet: the run's first restart
    assert traced_start(_case(start_step=None), run_dir) == 3000
    # traced: the step it actually started from, even after earlier restarts appear
    plan = plan_ptrace(_case(start_step=None), run_dir, site, omp_threads=1)
    run_ptrace(plan)
    (run_dir / "jorek02800.h5").write_bytes(b"h5")
    assert traced_start(_case(start_step=None), run_dir) == 3000


def test_a_new_restart_makes_a_default_end_trace_out_of_date(run_dir, site):  # noqa: F811
    plan = plan_ptrace(_case(start_step=None), run_dir, site, omp_threads=1)
    run_ptrace(plan)
    (run_dir / "jorek03600.h5").write_bytes(b"h5")
    later = plan_ptrace(_case(start_step=None), run_dir, site, omp_threads=1)
    assert later.steps[-1] == 3600
    assert not is_current(later)


# --- queuing with a jobscript ---------------------------------------------------


def test_queued_command_uses_the_jobscript(run_dir, slurm):
    plan = plan_ptrace(_case(), run_dir, slurm, omp_threads=4, job="23h")
    assert plan.command.splitlines()[-1] == (
        f"sbatch --parsable {slurm.jobscripts / '23h'} "
        f"../../exe/{RE_GC} in_main ptrace.log pt_run"
    )
    assert "mpirun" not in plan.command
    assert any(line == "queue:" for line in plan.describe())


def test_absolute_exe_is_passed_relative(run_dir, slurm, tmp_path):
    """The jobscript runs ./"$exe": an absolute path would break."""
    plan = plan_ptrace(_case(exe=str(run_dir / "exe" / RE_GC)), run_dir, slurm,
                       omp_threads=4, job="2h")
    assert f" ../../exe/{RE_GC} " in plan.command


def test_unknown_jobscript_lists_the_ones_there(run_dir, slurm):
    with pytest.raises(PtraceError, match=r"no jobscript '5h' .*there: 23h, 2h"):
        plan_ptrace(_case(), run_dir, slurm, omp_threads=4, job="5h")


def test_queuing_leaves_the_restart_links_for_the_job(run_dir, slurm):
    plan, result = _queue(run_dir, slurm)
    assert result.job_id == "4242"
    assert (plan.work_dir / "jorek000000.h5").is_symlink()
    assert (plan.work_dir / "sbatch_args.txt").read_text().split()[-4:] == [
        f"../../exe/{RE_GC}", "in_main", "ptrace.log", "pt_run",
    ]
    meta = json.loads((plan.work_dir / META_FILE).read_text())
    assert (meta["job_id"], meta["jobscript"], meta["complete"]) == ("4242", "2h", False)
    assert not is_current(plan)


def test_a_job_still_queued_is_left_alone(run_dir, slurm, monkeypatch):
    plan, _ = _queue(run_dir, slurm)
    monkeypatch.setenv("FAKE_SQUEUE", "PENDING")
    queued = poll_job(plan.work_dir)
    assert (queued.running, queued.state) == (True, "PENDING")
    assert (plan.work_dir / "jorek000000.h5").is_symlink()
    with pytest.raises(PtraceError, match=r"job 4242 \(PENDING\) is still queued"):
        run_ptrace(plan, force=True)


def test_an_ended_job_is_concluded_and_then_cached(run_dir, slurm, monkeypatch):
    plan, _ = _queue(run_dir, slurm)
    _job_runs(plan)
    monkeypatch.setenv("FAKE_SACCT", "COMPLETED")
    queued = poll_job(plan.work_dir)
    assert (queued.running, queued.error) == (False, None)
    assert queued.result.job_id == "4242"
    assert not list(plan.work_dir.glob("jorek*"))
    assert is_current(plan)
    assert poll_job(plan.work_dir) is None
    assert run_ptrace(plan).ran is False


def test_a_job_that_timed_out_is_failed(run_dir, slurm, monkeypatch):
    plan, _ = _queue(run_dir, slurm)
    _job_runs(plan, STUB_NO_OUTPUT="1")
    monkeypatch.setenv("FAKE_SACCT", "TIMEOUT")
    queued = poll_job(plan.work_dir)
    assert "job 4242 ended TIMEOUT" in queued.error and "-job 23h" in queued.error
    assert not is_current(plan)
    assert not list(plan.work_dir.glob("jorek*"))


def test_a_job_without_its_particle_file_is_failed(run_dir, slurm):
    """The jobscripts tee the program's output, losing its exit status; no
    part_restart.h5 is how a crash shows. No sacct either: still judged."""
    plan, _ = _queue(run_dir, slurm)
    _job_runs(plan, STUB_EXIT="3")
    queued = poll_job(plan.work_dir)
    assert f"without the program writing {PARTICLES_FILE}" in queued.error
    assert "program failed" in queued.error
    assert not is_current(plan)


def test_a_copied_particle_file_the_job_never_touched_is_not_its_output(
    run_dir, slurm, tmp_path,
):
    seeds = tmp_path / "seeds.h5"
    seeds.write_bytes(b"seed particles")
    os.utime(seeds, (1_000_000_000, 1_000_000_000))
    plan, _ = _queue(run_dir, slurm, particles=str(seeds))
    _job_runs(plan, STUB_EXIT="3")
    assert "without the program writing" in poll_job(plan.work_dir).error


def test_a_lost_particle_stop_is_a_result(run_dir, slurm, monkeypatch):
    plan, _ = _queue(run_dir, slurm)
    _job_runs(plan, STUB_LOST="1")
    monkeypatch.setenv("FAKE_SACCT", "COMPLETED")
    queued = poll_job(plan.work_dir)
    assert queued.error is None and queued.result.lost


def test_sbatch_failure_is_reported_and_links_removed(run_dir, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_OUT", "sbatch: error: invalid qos")
    monkeypatch.setenv("FAKE_SBATCH_EXIT", "1")
    plan = plan_ptrace(_case(), run_dir, slurm, omp_threads=4, job="2h")
    with pytest.raises(PtraceError, match="(?s)sbatch failed .*invalid qos"):
        run_ptrace(plan)
    assert not list(plan.work_dir.glob("jorek*"))


def test_queued_fingerprint_follows_the_jobscript_not_n_mpi(run_dir, slurm):
    def fingerprint(job, **overrides):
        return plan_ptrace(_case(**overrides), run_dir, slurm, omp_threads=4, job=job).fingerprint

    assert fingerprint("2h") == fingerprint("2h", n_mpi=8)
    assert fingerprint("2h") != fingerprint("23h")
    assert fingerprint("2h") != fingerprint(None)
    before = fingerprint("2h")
    (slurm.jobscripts / "2h").write_text(JOBSCRIPT.replace("02:00", "03:00"), encoding="utf-8")
    assert fingerprint("2h") != before


def test_squeue_missing_is_an_error_not_a_guess(run_dir, slurm, monkeypatch):
    plan, _ = _queue(run_dir, slurm)
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(PtraceError, match="squeue not found"):
        poll_job(plan.work_dir)
    assert (plan.work_dir / "jorek000000.h5").is_symlink()


def test_relative_paths_in_queued_plan(run_dir, slurm):
    plan = plan_ptrace(_case(), run_dir, slurm, omp_threads=4, job="2h")
    assert Path(plan.work_dir, os.path.relpath(plan.exe, plan.work_dir)).resolve() == plan.exe.resolve()
