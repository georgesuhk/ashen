"""ashen.tracing -- planning, staging and running a trace, against a stub
JOREK particle program (no JOREK, no MPI)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ashen.cases import Case
from ashen.config import Diagnostics, Launch, Site
from ashen.tracing import (
    LOG_FILE,
    META_FILE,
    PARTICLES_FILE,
    TraceError,
    is_current,
    plan_trace,
    run_trace,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="stub program is a POSIX shell script")

RE_GC = "re_gc_current_density_initialisation"

#: Records what it saw -- stdin, the folder, OMP_NUM_THREADS, and the step
#: numbers ex6/ex7's last_file_before_time would bisect over (its own shell
#: pipeline, verbatim) -- then writes the outputs of whichever program it
#: was installed as.
STUB_PROGRAM = r"""#!/bin/sh
cat > stdin_seen.txt
ls -1 > listing.txt
echo "$OMP_NUM_THREADS" > omp_seen.txt
ls jorek[0-9]*.h5 | grep -o '[0-9]\{5\}' > filenums_seen.txt
[ -f part_restart.h5 ] && cp part_restart.h5 particles_seen.h5
echo "program running"
[ -n "$STUB_EXIT" ] && { echo "program failed"; exit "$STUB_EXIT"; }
[ -n "$STUB_LOST" ] && { echo " PARTICLE IS LOST, STOPPING"; exit 0; }
[ -n "$STUB_NO_OUTPUT" ] && exit 0
case "$(basename "$0")" in
  re_gc*) touch part_diag.h5 part_restart.h5 part_restart_0001.h5 ;;
  trace_gc) touch trace_diag.h5 part_restart.h5 part_restart000.00250000.h5 ;;
  *)      touch diag.h5 part_restart.h5 ;;
esac
"""


@pytest.fixture
def site(tmp_path):
    paths = {
        key: tmp_path / key
        for key in ("exe", "template", "jobscripts", "jorek", "jorek_re", "castor_root")
    }
    return Site(
        source=tmp_path / "site.toml", root=tmp_path, paths=paths,
        launch=Launch(interactive_prelude="export OMP_NUM_THREADS=10", mpirun=""),
        diagnostics=Diagnostics(),
    )


def install_program(run: Path, name: str) -> Path:
    exe = run / "exe" / name
    exe.write_text(STUB_PROGRAM, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def make_run(root: Path, steps=(3000, 3200, 3400), name: str = "run") -> Path:
    run = root / name
    (run / "exe").mkdir(parents=True)
    for step in steps:
        (run / f"jorek{step:05d}.h5").write_bytes(b"h5")
    (run / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    (run / "T_prof.dat").write_text("1 2\n", encoding="utf-8")
    for program in (RE_GC, "ex6_jorek", "ex7_jorek", "trace_gc"):
        install_program(run, program)
    return run


@pytest.fixture
def run_dir(tmp_path):
    return make_run(tmp_path / "campaign")


def _case(**overrides) -> Case:
    """A case named "run" that traces ./exe/RE_GC from step 3000. Overrides
    are trace_* fields without their prefix (start_step=, exe=, ...), plus
    program=NAME as shorthand for exe="./exe/NAME"."""
    settings = dict(name="run", steps=[3000], trace_exe=f"./exe/{RE_GC}", trace_start_step=3000)
    if "program" in overrides:
        overrides["exe"] = f"./exe/{overrides.pop('program')}"
    for key, value in overrides.items():
        settings[key if key == "note" else f"trace_{key}"] = value
    return Case(**settings)


def _plan(run_dir, site, **overrides):
    return plan_trace(_case(**overrides), run_dir, site, omp_threads=4)


def _write_zero_d(run_dir: Path, step: int, time: float) -> None:
    postproc = run_dir / "postproc"
    postproc.mkdir(exist_ok=True)
    (postproc / f"zeroD_quantities_s{step:05d}.dat").write_text(
        f"Time Ip\n{time:.6e} 1.0\n", encoding="utf-8"
    )


# --- restart links -------------------------------------------------------------


def test_restarts_linked_consecutively_under_both_widths(run_dir, site):
    """The run's restarts are 200 steps apart, beyond the reader's 20-file
    look-ahead; renumbered 0, 1, 2 they are found. Both widths, since the
    reader and last_file_before_time each accept only one."""
    plan = _plan(run_dir, site)
    assert plan.steps == [3000, 3200, 3400]
    restarts = [(name, target.name) for name, target in plan.links if name.startswith("jorek")]
    assert restarts == [
        ("jorek000000.h5", "jorek03000.h5"), ("jorek00000.h5", "jorek03000.h5"),
        ("jorek000001.h5", "jorek03200.h5"), ("jorek00001.h5", "jorek03200.h5"),
        ("jorek000002.h5", "jorek03400.h5"), ("jorek00002.h5", "jorek03400.h5"),
        # the start restart again, as the frozen-field reader names it
        ("jorek_restart.h5", "jorek03000.h5"),
    ]


def test_last_file_before_time_sees_ascending_step_numbers(tmp_path, site):
    """Its `ls | grep -o '[0-9]\\{5\\}'` over both widths must stay in
    ascending order for its bisection to work -- checked on a real staged
    folder with more restarts than one decade of indices."""
    run = make_run(tmp_path, steps=range(0, 2600, 100))
    plan = plan_trace(_case(program="ex7_jorek", start_step=0), run, site, omp_threads=1)
    run_trace(plan)
    seen = [int(n) for n in (plan.work_dir / "filenums_seen.txt").read_text().split()]
    assert seen == sorted(seen)
    assert set(seen) == set(range(len(plan.steps)))


def test_a_case_without_trace_exe_cannot_be_planned(run_dir, site):
    with pytest.raises(TraceError, match="has no trace_exe"):
        plan_trace(Case(name="run", steps=[3000]), run_dir, site, omp_threads=1)


def test_end_step_limits_the_restarts(run_dir, site):
    assert _plan(run_dir, site, end_step=3200).steps == [3000, 3200]


def test_missing_start_step_names_nearby_steps(run_dir, site):
    with pytest.raises(TraceError, match=r"no restart for trace_start_step 3100; nearby: \[3000, 3200, 3400\]"):
        _plan(run_dir, site, start_step=3100)


def test_re_gc_runs_from_a_single_restart(run_dir, site):
    """It keeps the last field frozen rather than aborting."""
    assert _plan(run_dir, site, start_step=3400).steps == [3400]


@pytest.mark.parametrize("program", ["ex6_jorek", "ex7_jorek"])
def test_ex_programs_need_two_restarts(run_dir, site, program):
    with pytest.raises(TraceError, match="at least two"):
        _plan(run_dir, site, program=program, start_step=3400)


def test_namelist_and_present_profiles_are_linked(run_dir, site):
    names = [name for name, _ in _plan(run_dir, site).links]
    assert "in_main" in names
    assert "T_prof.dat" in names
    assert "rho_prof.dat" not in names  # absent from the run: skipped, not dangling


# --- executable and command ---------------------------------------------------


def test_exe_is_relative_to_the_run_folder(run_dir, site):
    plan = _plan(run_dir, site)
    assert plan.exe == run_dir / "exe" / RE_GC
    assert plan.program.known and plan.program.name == RE_GC
    assert plan.work_dir == run_dir / "trace" / RE_GC


def test_exe_outside_exe_folder(run_dir, site):
    """Any path, not only under exe/ -- normalised, so ../ works too."""
    plan = _plan(run_dir, site, exe="../tools/re_gc_current_density_initialisation")
    assert plan.exe == run_dir.parent / "tools" / RE_GC
    assert plan.program.known


def test_absolute_exe(run_dir, site, tmp_path):
    exe = tmp_path / "bin" / "ex7_jorek"
    assert _plan(run_dir, site, exe=str(exe)).exe == exe


def test_renamed_known_program_is_not_recognised(run_dir, site):
    """Recognition is by filename; a renamed binary runs as-is."""
    plan = _plan(run_dir, site, exe="./exe/re_gc_model600")
    assert not plan.program.known
    assert plan.work_dir == run_dir / "trace" / "re_gc_model600"


def test_omp_threads_are_exported_after_the_prelude(run_dir, site):
    """site.example.toml's prelude exports its own OMP_NUM_THREADS."""
    lines = _plan(run_dir, site).command.splitlines()
    assert lines[0] == "export OMP_NUM_THREADS=10"
    assert lines[1] == "export OMP_NUM_THREADS=4"
    assert lines[2].endswith(f"{RE_GC} < in_main")


def test_trace_own_omp_threads_override_the_default(run_dir, site):
    assert _plan(run_dir, site, omp_threads=2).omp_threads == 2


# --- the hard-coded start time of ex6/ex7 ---------------------------------------


def test_no_warning_when_restarts_cover_the_fixed_start(run_dir, site):
    _write_zero_d(run_dir, 3000, 2.0e-3)
    _write_zero_d(run_dir, 3400, 3.0e-3)
    assert _plan(run_dir, site, program="ex7_jorek").warnings == []


def test_warning_when_restarts_miss_the_fixed_start(run_dir, site):
    _write_zero_d(run_dir, 3000, 3.0e-3)
    _write_zero_d(run_dir, 3400, 4.0e-3)
    [warning] = _plan(run_dir, site, program="ex6_jorek").warnings
    assert "always starts at t = 0.0025 s" in warning
    assert "span 0.003 .. 0.004 s" in warning


def test_warning_when_the_start_cannot_be_checked(run_dir, site):
    [warning] = _plan(run_dir, site, program="ex7_jorek").warnings
    assert "analyse --diag zerod" in warning


def test_no_start_time_check_for_re_gc(run_dir, site):
    assert _plan(run_dir, site).warnings == []


# --- particles --------------------------------------------------------------------


def test_particles_are_copied_not_linked(run_dir, site, tmp_path):
    """The program overwrites part_restart.h5 at the end; through a link
    that would clobber the original."""
    source = tmp_path / "seed.h5"
    source.write_bytes(b"seed particles")
    plan = _plan(run_dir, site, particles=source)
    run_trace(plan)
    staged = plan.work_dir / "particles_seen.h5"
    assert staged.read_bytes() == b"seed particles"
    assert source.read_bytes() == b"seed particles"
    assert not (plan.work_dir / PARTICLES_FILE).is_symlink()


def test_missing_particles_file(run_dir, site, tmp_path):
    with pytest.raises(TraceError, match="trace_particles .* not found"):
        _plan(run_dir, site, particles=tmp_path / "nope.h5")


def test_leftover_particles_are_cleared_before_a_fresh_run(run_dir, site):
    """re_gc starts from part_restart.h5 if it exists -- the previous
    trace's final state must not become this one's start."""
    plan = _plan(run_dir, site)
    run_trace(plan)
    run_trace(plan, force=True)
    assert not (plan.work_dir / "particles_seen.h5").exists()


# --- running and caching ------------------------------------------------------------


def test_run_stages_relative_links_and_runs_the_program(run_dir, site):
    plan = _plan(run_dir, site)
    result = run_trace(plan)
    assert result.ran and not result.lost

    work = plan.work_dir
    assert work == run_dir / "trace" / RE_GC
    link = work / "jorek000001.h5"
    assert link.is_symlink()
    assert not Path(os.readlink(link)).is_absolute()
    assert link.resolve() == (run_dir / "jorek03200.h5").resolve()
    assert (work / "stdin_seen.txt").read_text(encoding="utf-8") == "&in1\n&end\n"
    assert (work / "omp_seen.txt").read_text(encoding="utf-8").strip() == "4"
    assert "program running" in (work / LOG_FILE).read_text(encoding="utf-8")
    assert is_current(plan)


def test_a_completed_trace_is_cached(run_dir, site):
    plan = _plan(run_dir, site)
    run_trace(plan)
    (plan.work_dir / "listing.txt").unlink()
    assert run_trace(plan).ran is False
    assert not (plan.work_dir / "listing.txt").exists()


def test_force_reruns(run_dir, site):
    plan = _plan(run_dir, site)
    run_trace(plan)
    assert run_trace(plan, force=True).ran is True


def test_changed_settings_rerun(run_dir, site):
    run_trace(_plan(run_dir, site))
    assert not is_current(_plan(run_dir, site, end_step=3200))
    assert not is_current(_plan(run_dir, site, n_mpi=4))  # re_gc samples per rank
    assert is_current(_plan(run_dir, site, omp_threads=8, note="same answer"))


def test_rebuilt_program_reruns(run_dir, site):
    plan = _plan(run_dir, site)
    run_trace(plan)
    exe = plan.exe
    os.utime(exe, ns=(exe.stat().st_atime_ns, exe.stat().st_mtime_ns + 10**9))
    assert not is_current(_plan(run_dir, site))


def test_restaging_keeps_files_ashen_does_not_manage(run_dir, site):
    plan = _plan(run_dir, site)
    plan.work_dir.mkdir(parents=True)
    (plan.work_dir / "my_notes.txt").write_text("keep", encoding="utf-8")
    run_trace(plan)
    run_trace(_plan(run_dir, site, end_step=3200))
    assert (plan.work_dir / "my_notes.txt").read_text(encoding="utf-8") == "keep"
    assert not (plan.work_dir / "jorek000002.h5").exists()


def test_ex7_stopping_at_a_lost_particle_is_a_result(run_dir, site, monkeypatch):
    monkeypatch.setenv("STUB_LOST", "1")
    plan = _plan(run_dir, site, program="ex7_jorek")
    result = run_trace(plan)
    assert result.ran and result.lost
    assert is_current(plan)
    assert run_trace(plan).lost  # remembered when cached


def test_failing_program_quotes_its_log(run_dir, site, monkeypatch):
    monkeypatch.setenv("STUB_EXIT", "3")
    plan = _plan(run_dir, site)
    with pytest.raises(TraceError, match="exited 3(.|\n)*program failed"):
        run_trace(plan)
    assert not is_current(plan)
    assert (plan.work_dir / META_FILE).is_file()


def test_program_without_outputs_is_an_error(run_dir, site, monkeypatch):
    monkeypatch.setenv("STUB_NO_OUTPUT", "1")
    with pytest.raises(TraceError, match=r"did not write \['part_diag.h5', 'part_restart.h5'\]"):
        run_trace(_plan(run_dir, site))


def test_missing_program_says_how_to_build_it(run_dir, site):
    (run_dir / "exe" / RE_GC).unlink()
    with pytest.raises(TraceError) as info:
        run_trace(_plan(run_dir, site))
    message = str(info.value)
    assert f"trace_exe './exe/{RE_GC}' not found" in message
    assert "relative to the run folder" in message
    assert f"make {RE_GC}" in message


def test_missing_unrecognised_exe_has_no_build_hint(run_dir, site):
    with pytest.raises(TraceError) as info:
        run_trace(_plan(run_dir, site, exe="./exe/my_tracer"))
    assert "not found" in str(info.value)
    assert "make" not in str(info.value)


# --- an executable ashen does not recognise ------------------------------------


def test_unrecognised_exe_runs_as_is(run_dir, site, monkeypatch):
    """No expected outputs: a zero exit is success, even writing nothing."""
    install_program(run_dir, "my_tracer")
    monkeypatch.setenv("STUB_NO_OUTPUT", "1")
    plan = _plan(run_dir, site, exe="./exe/my_tracer")
    assert run_trace(plan).ran
    assert is_current(plan)


def test_unrecognised_exe_gets_no_start_time_check_and_one_restart_is_enough(run_dir, site):
    plan = _plan(run_dir, site, exe="./exe/my_tracer", start_step=3400)
    assert plan.steps == [3400]
    assert plan.warnings == []


def test_stale_known_outputs_are_cleared_for_any_exe(run_dir, site, monkeypatch):
    """A diag file left by an earlier trace in the same folder must not be
    mistaken for this one's."""
    install_program(run_dir, "my_tracer")
    monkeypatch.setenv("STUB_NO_OUTPUT", "1")
    plan = _plan(run_dir, site, exe="./exe/my_tracer")
    plan.work_dir.mkdir(parents=True)
    (plan.work_dir / "diag.h5").write_bytes(b"stale")
    run_trace(plan)
    assert not (plan.work_dir / "diag.h5").exists()


# --- trace_inputs and trace_gc -----------------------------------------------------


@pytest.fixture
def params(tmp_path):
    path = tmp_path / "inputs" / "trace_params.nml"
    path.parent.mkdir()
    path.write_text("&trace\n n_markers = 1\n/\n", encoding="utf-8")
    return path


def test_trace_gc_needs_its_params_file(run_dir, site):
    with pytest.raises(TraceError, match=r"trace_gc reads \['trace_params.nml'\].*trace_inputs"):
        _plan(run_dir, site, program="trace_gc")


def test_inputs_are_copied_in_under_their_own_names(run_dir, site, params):
    plan = _plan(run_dir, site, program="trace_gc", inputs=[params])
    assert plan.copies == [("trace_params.nml", params)]
    assert run_trace(plan).ran
    staged = plan.work_dir / "trace_params.nml"
    assert not staged.is_symlink()
    assert staged.read_text(encoding="utf-8") == params.read_text(encoding="utf-8")


def test_missing_input_file(run_dir, site, tmp_path):
    with pytest.raises(TraceError, match="trace_inputs file .* not found"):
        _plan(run_dir, site, exe="./exe/my_tracer", inputs=[tmp_path / "nope.nml"])


def test_edited_input_reruns(run_dir, site, params):
    run_trace(_plan(run_dir, site, program="trace_gc", inputs=[params]))
    params.write_text("&trace\n n_markers = 2\n/\n", encoding="utf-8")
    assert not is_current(_plan(run_dir, site, program="trace_gc", inputs=[params]))


def test_an_input_dropped_from_the_list_is_cleared(run_dir, site, params, tmp_path):
    extra = tmp_path / "inputs" / "extra.txt"
    extra.write_text("x", encoding="utf-8")
    plan = _plan(run_dir, site, program="trace_gc", inputs=[params, extra])
    run_trace(plan)
    assert (plan.work_dir / "extra.txt").is_file()
    run_trace(_plan(run_dir, site, program="trace_gc", inputs=[params]))
    assert not (plan.work_dir / "extra.txt").exists()


def test_trace_gc_outputs_are_what_the_particles_plot_reads(run_dir, site, params, monkeypatch):
    """trace_gc writes part_restart.h5 at the end; a run that doesn't is a
    failure, as for any recognised program."""
    monkeypatch.setenv("STUB_NO_OUTPUT", "1")
    with pytest.raises(TraceError, match=r"did not write \['trace_diag.h5', 'part_restart.h5'\]"):
        run_trace(_plan(run_dir, site, program="trace_gc", inputs=[params]))
