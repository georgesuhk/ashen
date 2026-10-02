"""ashen.ptracing -- planning, staging and running a trace, against a stub
JOREK particle program (no JOREK, no MPI)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ashen.cases import Case
from ashen.config import Diagnostics, Launch, Site
from ashen.ptracing import (
    LOG_FILE,
    META_FILE,
    PARTICLES_FILE,
    PtraceError,
    is_current,
    plan_ptrace,
    run_ptrace,
    stage,
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
[ -n "$STUB_NOTE" ] && echo " ptrace_gc: NOTE: stopped early -- t_span runs past the last restart"
[ -n "$STUB_EXIT" ] && { echo "program failed"; exit "$STUB_EXIT"; }
[ -n "$STUB_LOST" ] && { echo " PARTICLE IS LOST, STOPPING"; exit 0; }
[ -n "$STUB_NO_OUTPUT" ] && exit 0
case "$(basename "$0")" in
  re_gc*) touch part_diag.h5 part_restart.h5 part_restart_0001.h5 ;;
  ptrace_gc) touch ptrace_diag.h5 part_restart.h5 part_restart_s003000_t2.500000E-03.h5 ;;
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
    for program in (RE_GC, "ex6_jorek", "ex7_jorek", "ptrace_gc"):
        install_program(run, program)
    return run


@pytest.fixture
def run_dir(tmp_path):
    return make_run(tmp_path / "campaign")


def _case(**overrides) -> Case:
    """A case named "run" that traces ./exe/RE_GC from step 3000. Overrides
    are ptrace_* fields without their prefix (start_step=, exe=, ...), plus
    program=NAME as shorthand for exe="./exe/NAME"."""
    settings = dict(name="run", steps=[3000], ptrace_exe=f"./exe/{RE_GC}", ptrace_start_step=3000)
    if "program" in overrides:
        overrides["exe"] = f"./exe/{overrides.pop('program')}"
    for key, value in overrides.items():
        settings[key if key == "note" else f"ptrace_{key}"] = value
    return Case(**settings)


def _plan(run_dir, site, **overrides):
    return plan_ptrace(_case(**overrides), run_dir, site, omp_threads=4)


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
        # and as current_pdf_simple's profile (ptrace_pdf_step's otherwise);
        # not jorek[0-9]*, so last_file_before_time never sees it
        ("jorek_pdf.h5", "jorek03000.h5"),
    ]


def test_last_file_before_time_sees_ascending_step_numbers(tmp_path, site):
    """Its `ls | grep -o '[0-9]\\{5\\}'` over both widths must stay in
    ascending order for its bisection to work -- checked on a real staged
    folder with more restarts than one decade of indices."""
    run = make_run(tmp_path, steps=range(0, 2600, 100))
    plan = plan_ptrace(_case(program="ex7_jorek", start_step=0), run, site, omp_threads=1)
    run_ptrace(plan)
    seen = [int(n) for n in (plan.work_dir / "filenums_seen.txt").read_text().split()]
    assert seen == sorted(seen)
    assert set(seen) == set(range(len(plan.steps)))


def test_a_case_without_trace_exe_cannot_be_planned(run_dir, site):
    with pytest.raises(PtraceError, match="has no ptrace_exe"):
        plan_ptrace(Case(name="run", steps=[3000]), run_dir, site, omp_threads=1)


def test_end_step_limits_the_restarts(run_dir, site):
    assert _plan(run_dir, site, end_step=3200).steps == [3000, 3200]


def test_missing_start_step_names_nearby_steps(run_dir, site):
    with pytest.raises(PtraceError, match=r"no restart for ptrace_start_step 3100; nearby: \[3000, 3200, 3400\]"):
        _plan(run_dir, site, start_step=3100)


def test_a_single_restart_is_allowed(run_dir, site):
    """Whether one restart is enough is the program's business (ptrace_gc
    with t_span, re_gc with its frozen field) -- ashen doesn't guess by name."""
    for program in (RE_GC, "ex7_jorek", "ptrace_gc"):
        assert _plan(run_dir, site, program=program, start_step=3400).steps == [3400]


def test_namelist_and_present_profiles_are_linked(run_dir, site):
    names = [name for name, _ in _plan(run_dir, site).links]
    assert "in_main" in names
    assert "T_prof.dat" in names
    assert "rho_prof.dat" not in names  # absent from the run: skipped, not dangling


# --- executable and command ---------------------------------------------------


def test_exe_is_relative_to_the_run_folder(run_dir, site):
    plan = _plan(run_dir, site)
    assert plan.exe == run_dir / "exe" / RE_GC
    assert plan.work_dir == run_dir / "ptrace" / RE_GC


def test_exe_outside_exe_folder(run_dir, site):
    """Any path, not only under exe/ -- normalised, so ../ works too."""
    plan = _plan(run_dir, site, exe="../tools/re_gc_current_density_initialisation")
    assert plan.exe == run_dir.parent / "tools" / RE_GC


def test_absolute_exe(run_dir, site, tmp_path):
    exe = tmp_path / "bin" / "ex7_jorek"
    assert _plan(run_dir, site, exe=str(exe)).exe == exe


def test_each_exe_name_gets_its_own_folder(run_dir, site):
    plan = _plan(run_dir, site, exe="./exe/re_gc_model600")
    assert plan.work_dir == run_dir / "ptrace" / "re_gc_model600"


def test_omp_threads_are_exported_after_the_prelude(run_dir, site):
    """site.example.toml's prelude exports its own OMP_NUM_THREADS."""
    lines = _plan(run_dir, site).command.splitlines()
    assert lines[0] == "export OMP_NUM_THREADS=10"
    assert lines[1] == "export OMP_NUM_THREADS=4"
    assert lines[2].endswith(f"{RE_GC} < in_main")


def test_trace_own_omp_threads_override_the_default(run_dir, site):
    assert _plan(run_dir, site, omp_threads=2).omp_threads == 2


# --- particles --------------------------------------------------------------------


def test_particles_are_copied_not_linked(run_dir, site, tmp_path):
    """The program overwrites part_restart.h5 at the end; through a link
    that would clobber the original."""
    source = tmp_path / "seed.h5"
    source.write_bytes(b"seed particles")
    plan = _plan(run_dir, site, particles=source)
    run_ptrace(plan)
    staged = plan.work_dir / "particles_seen.h5"
    assert staged.read_bytes() == b"seed particles"
    assert source.read_bytes() == b"seed particles"
    assert not (plan.work_dir / PARTICLES_FILE).is_symlink()


def test_missing_particles_file(run_dir, site, tmp_path):
    with pytest.raises(PtraceError, match="ptrace_particles .* not found"):
        _plan(run_dir, site, particles=tmp_path / "nope.h5")


def test_leftover_particles_are_cleared_before_a_fresh_run(run_dir, site):
    """re_gc starts from part_restart.h5 if it exists -- the previous
    trace's final state must not become this one's start."""
    plan = _plan(run_dir, site)
    run_ptrace(plan)
    run_ptrace(plan, force=True)
    assert not (plan.work_dir / "particles_seen.h5").exists()


# --- running and caching ------------------------------------------------------------


def test_stage_links_relatively(run_dir, site):
    plan = _plan(run_dir, site)
    stage(plan)
    link = plan.work_dir / "jorek000001.h5"
    assert link.is_symlink()
    assert not Path(os.readlink(link)).is_absolute()
    assert link.resolve() == (run_dir / "jorek03200.h5").resolve()


def test_run_stages_and_runs_the_program(run_dir, site):
    plan = _plan(run_dir, site)
    result = run_ptrace(plan)
    assert result.ran and not result.lost

    work = plan.work_dir
    assert work == run_dir / "ptrace" / RE_GC
    assert (work / "stdin_seen.txt").read_text(encoding="utf-8") == "&in1\n&end\n"
    assert (work / "omp_seen.txt").read_text(encoding="utf-8").strip() == "4"
    assert "program running" in (work / LOG_FILE).read_text(encoding="utf-8")
    assert is_current(plan)


def test_a_completed_trace_is_cached(run_dir, site):
    plan = _plan(run_dir, site)
    run_ptrace(plan)
    (plan.work_dir / "listing.txt").unlink()
    assert run_ptrace(plan).ran is False
    assert not (plan.work_dir / "listing.txt").exists()


def test_force_reruns(run_dir, site):
    plan = _plan(run_dir, site)
    run_ptrace(plan)
    assert run_ptrace(plan, force=True).ran is True


def test_changed_settings_rerun(run_dir, site):
    run_ptrace(_plan(run_dir, site))
    assert not is_current(_plan(run_dir, site, end_step=3200))
    assert not is_current(_plan(run_dir, site, n_mpi=4))  # re_gc samples per rank
    assert is_current(_plan(run_dir, site, omp_threads=8, note="same answer"))


def test_rebuilt_program_reruns(run_dir, site):
    plan = _plan(run_dir, site)
    run_ptrace(plan)
    exe = plan.exe
    os.utime(exe, ns=(exe.stat().st_atime_ns, exe.stat().st_mtime_ns + 10**9))
    assert not is_current(_plan(run_dir, site))


def test_restaging_keeps_files_ashen_does_not_manage(run_dir, site):
    plan = _plan(run_dir, site)
    plan.work_dir.mkdir(parents=True)
    (plan.work_dir / "my_notes.txt").write_text("keep", encoding="utf-8")
    run_ptrace(plan)
    run_ptrace(_plan(run_dir, site, end_step=3200))
    assert (plan.work_dir / "my_notes.txt").read_text(encoding="utf-8") == "keep"
    assert not (plan.work_dir / "jorek000002.h5").exists()


def test_ex7_stopping_at_a_lost_particle_is_a_result(run_dir, site, monkeypatch):
    monkeypatch.setenv("STUB_LOST", "1")
    plan = _plan(run_dir, site, program="ex7_jorek")
    result = run_ptrace(plan)
    assert result.ran and result.lost
    assert is_current(plan)
    assert run_ptrace(plan).lost  # remembered when cached


def test_failing_program_quotes_its_log(run_dir, site, monkeypatch):
    monkeypatch.setenv("STUB_EXIT", "3")
    plan = _plan(run_dir, site)
    with pytest.raises(PtraceError, match="exited 3(.|\n)*program failed"):
        run_ptrace(plan)
    assert not is_current(plan)
    assert (plan.work_dir / META_FILE).is_file()


def test_no_particle_file_written_is_a_note(run_dir, site, monkeypatch):
    """Every JOREK particle program writes part_restart.h5 at the end;
    exiting 0 without it is still a success, but worth saying."""
    monkeypatch.setenv("STUB_NO_OUTPUT", "1")
    result = run_ptrace(_plan(run_dir, site))
    assert result.ran
    (note,) = result.notes
    assert "exited 0 without writing part_restart.h5" in note


def test_missing_program_says_how_to_build_it(run_dir, site):
    (run_dir / "exe" / RE_GC).unlink()
    with pytest.raises(PtraceError) as info:
        run_ptrace(_plan(run_dir, site))
    message = str(info.value)
    assert f"ptrace_exe './exe/{RE_GC}' not found" in message
    assert "relative to the run folder" in message
    assert "make <program>" in message


def test_any_exe_name_runs_the_same(run_dir, site, monkeypatch):
    """Nothing depends on the name: my_tracer's lost-particle stop and notes
    are read from its log like any other's."""
    install_program(run_dir, "my_tracer")
    monkeypatch.setenv("STUB_LOST", "1")
    plan = _plan(run_dir, site, exe="./exe/my_tracer")
    result = run_ptrace(plan)
    assert result.ran and result.lost
    assert is_current(plan)


def test_stale_known_outputs_are_cleared_for_any_exe(run_dir, site, monkeypatch):
    """A diag file left by an earlier trace in the same folder must not be
    mistaken for this one's."""
    install_program(run_dir, "my_tracer")
    monkeypatch.setenv("STUB_NO_OUTPUT", "1")
    plan = _plan(run_dir, site, exe="./exe/my_tracer")
    plan.work_dir.mkdir(parents=True)
    (plan.work_dir / "diag.h5").write_bytes(b"stale")
    run_ptrace(plan)
    assert not (plan.work_dir / "diag.h5").exists()


# --- ptrace_inputs and ptrace_gc -----------------------------------------------------

#: What a current_pdf_simple trace cannot do without.
PDF = {"initialiser": "current_pdf_simple", "n_markers": 2, "E_kin_eV": [1e7],
       "cos_pitch": [0.9]}


@pytest.fixture
def params(tmp_path):
    path = tmp_path / "inputs" / "seeds.dat"
    path.parent.mkdir()
    path.write_text("1 2 3\n", encoding="utf-8")
    return path


def test_settings_are_written_complete_and_kept_as_the_record(run_dir, site):
    """Every setting -- the case's keys over the defaults -- goes into
    ptrace_settings.nml, which stays in the trace folder after the run."""
    plan = _plan(run_dir, site, program="ptrace_gc",
                 settings={**PDF, "dt": 5e-11, "hold_last_field": True})
    assert plan.copies == []
    (name, text), = plan.writes
    assert name == "ptrace_settings.nml"
    lines = text.splitlines()
    assert lines[0] == ("! case run: restart steps 3000..3400 (3), linked from index 0; "
                        "jorek_pdf.h5 is step 3000")
    for line in ("&ptrace", "  restart_index = 0", "  hold_last_field = .true.", "  dt = 5d-11",
                 "  initialiser = 'current_pdf_simple'", "  n_markers = 2",
                 "  E_kin_eV = 10000000.0",
                 # not set by the case: the defaults, spelled out
                 "  diag_step = 1d-08", "  snapshot_step = 0.0", "  n_snapshots = 100",
                 "  charge = -1", "  seed = 1"):
        assert line in lines
    assert plan.settings["diag_step"] == 1e-8
    run_ptrace(plan)
    assert (plan.work_dir / "ptrace_settings.nml").read_text(encoding="utf-8") == text


def test_settings_are_in_the_dry_run_with_defaults_marked(run_dir, site):
    lines = _plan(run_dir, site, program="ptrace_gc", settings={**PDF, "dt": 5e-11}).describe()
    assert "write    ptrace_settings.nml:" in lines
    shown = [line.strip() for line in lines]
    assert any(line.startswith("dt") and line.endswith("= 5e-11") for line in shown)
    assert any(line.startswith("diag_step") and line.endswith("= 1e-08   (default)") for line in shown)


def test_changed_settings_rerun_and_a_case_without_any_writes_no_file(run_dir, site):
    plan = _plan(run_dir, site, program="ptrace_gc", settings={**PDF, "dt": 1e-10})
    run_ptrace(plan)
    assert is_current(_plan(run_dir, site, program="ptrace_gc", settings={**PDF, "dt": 1e-10}))
    assert not is_current(_plan(run_dir, site, program="ptrace_gc", settings={**PDF, "dt": 2e-10}))
    bare = _plan(run_dir, site, program="ptrace_gc")
    assert bare.writes == [] and bare.settings == {}
    assert not is_current(bare)
    run_ptrace(bare)
    assert not (bare.work_dir / "ptrace_settings.nml").exists()


def test_setting_a_default_explicitly_is_the_same_trace(run_dir, site):
    """The fingerprint is over every setting the trace runs with, so naming
    a default changes nothing -- and a default changing in ashen would."""
    run_ptrace(_plan(run_dir, site, program="ptrace_gc", settings=dict(PDF)))
    assert is_current(_plan(run_dir, site, program="ptrace_gc", settings={**PDF, "dt": 1e-10}))


def test_settings_that_do_not_make_a_trace(run_dir, site):
    with pytest.raises(PtraceError, match="case 'run': ptrace_n_markers is not set"):
        _plan(run_dir, site, program="ptrace_gc", settings={"dt": 1e-10})


def test_settings_files_of_the_old_scheme_are_cleared(run_dir, site):
    """A folder traced before ptrace_settings.nml may still hold the two
    files ptrace_gc used to read: gone on restaging, so nobody takes them
    for what the trace ran with."""
    plan = _plan(run_dir, site, program="ptrace_gc", settings=dict(PDF))
    plan.work_dir.mkdir(parents=True)
    for name in ("ptrace_params.nml", "ptrace_overrides.nml"):
        (plan.work_dir / name).write_text("&ptrace\n/\n", encoding="utf-8")
    run_ptrace(plan)
    assert sorted(p.name for p in plan.work_dir.glob("ptrace_*.nml")) == ["ptrace_settings.nml"]


def test_inputs_are_copied_in_under_their_own_names(run_dir, site, params):
    plan = _plan(run_dir, site, program="ptrace_gc", inputs=[params])
    assert plan.copies == [("seeds.dat", params)]
    assert run_ptrace(plan).ran
    staged = plan.work_dir / "seeds.dat"
    assert not staged.is_symlink()
    assert staged.read_text(encoding="utf-8") == params.read_text(encoding="utf-8")


def test_missing_input_file(run_dir, site, tmp_path):
    with pytest.raises(PtraceError, match="ptrace_inputs file .* not found"):
        _plan(run_dir, site, exe="./exe/my_tracer", inputs=[tmp_path / "nope.nml"])


def test_edited_input_reruns(run_dir, site, params):
    run_ptrace(_plan(run_dir, site, program="ptrace_gc", inputs=[params]))
    params.write_text("4 5 6\n", encoding="utf-8")
    assert not is_current(_plan(run_dir, site, program="ptrace_gc", inputs=[params]))


def test_an_input_dropped_from_the_list_is_cleared(run_dir, site, params, tmp_path):
    extra = tmp_path / "inputs" / "extra.txt"
    extra.write_text("x", encoding="utf-8")
    plan = _plan(run_dir, site, program="ptrace_gc", inputs=[params, extra])
    run_ptrace(plan)
    assert (plan.work_dir / "extra.txt").is_file()
    run_ptrace(_plan(run_dir, site, program="ptrace_gc", inputs=[params]))
    assert not (plan.work_dir / "extra.txt").exists()


def test_trace_paths_are_relative_to_the_run_folder(run_dir, site):
    """ptrace_inputs and ptrace_particles resolve like ptrace_exe: a bare name
    is the file in the run folder, and "../" works."""
    (run_dir / "seeds.dat").write_text("1\n", encoding="utf-8")
    (run_dir.parent / "shared.h5").write_bytes(b"seed")
    plan = _plan(run_dir, site, inputs=["seeds.dat"], particles="../shared.h5")
    assert dict(plan.copies) == {
        "part_restart.h5": run_dir.parent / "shared.h5",
        "seeds.dat": run_dir / "seeds.dat",
    }


def test_missing_relative_input_names_where_it_looked(run_dir, site):
    with pytest.raises(PtraceError) as info:
        _plan(run_dir, site, exe="./exe/my_tracer", inputs=["seeds.dat"])
    message = str(info.value)
    assert f"looked for {run_dir / 'seeds.dat'}" in message
    assert "relative to the run folder" in message


def test_restart_links_are_there_for_the_run_and_gone_after(run_dir, site):
    plan = _plan(run_dir, site)
    run_ptrace(plan)
    seen = (plan.work_dir / "listing.txt").read_text(encoding="utf-8").split()
    assert {"jorek000000.h5", "jorek00000.h5", "jorek_restart.h5"} <= set(seen)
    left = sorted(p.name for p in plan.work_dir.iterdir())
    assert not [name for name in left if name.startswith("jorek")]
    assert "in_main" in left  # the namelist link is not a restart


def test_restart_links_removed_even_when_the_program_fails(run_dir, site, monkeypatch):
    monkeypatch.setenv("STUB_EXIT", "2")
    plan = _plan(run_dir, site)
    with pytest.raises(PtraceError):
        run_ptrace(plan)
    assert not list(plan.work_dir.glob("jorek*"))


def test_dry_run_marks_the_restart_links(run_dir, site):
    lines = _plan(run_dir, site).describe()
    assert any(l.startswith("link     jorek000000.h5") and l.endswith("(removed after the run)") for l in lines)
    assert any(l.startswith("link     in_main") and "removed" not in l for l in lines)


def test_program_notes_are_reported_and_remembered(run_dir, site, params, monkeypatch):
    """ptrace_gc stopping at the last restart before t_span is done is a
    success -- outputs written -- but one the user should hear about, on the
    run and again when it is [cached]."""
    monkeypatch.setenv("STUB_NOTE", "1")
    plan = _plan(run_dir, site, program="ptrace_gc", inputs=[params])
    result = run_ptrace(plan)
    assert result.ran
    assert result.notes == ("stopped early -- t_span runs past the last restart",)
    monkeypatch.delenv("STUB_NOTE")
    cached = run_ptrace(_plan(run_dir, site, program="ptrace_gc", inputs=[params]))
    assert not cached.ran and cached.notes == result.notes


def test_notes_from_an_exe_under_any_name(run_dir, site, monkeypatch):
    """ptrace_gc built as, say, ptrace_gc_refluid: ashen doesn't know the
    name, but its notes are still repeated."""
    install_program(run_dir, "ptrace_gc_refluid")
    monkeypatch.setenv("STUB_NOTE", "1")
    result = run_ptrace(_plan(run_dir, site, exe="./exe/ptrace_gc_refluid"))
    assert result.notes == ("stopped early -- t_span runs past the last restart",)



def test_pdf_restart_is_the_start_step_by_default(run_dir, site):
    links = dict(_plan(run_dir, site).links)
    assert links["jorek_pdf.h5"] == run_dir / "jorek03000.h5"


def test_pdf_step_chooses_the_current_profile_restart(run_dir, site):
    """Any step of the run -- here one after the traced range."""
    plan = _plan(run_dir, site, end_step=3200, pdf_step=3400)
    assert dict(plan.links)["jorek_pdf.h5"] == run_dir / "jorek03400.h5"
    assert "jorek_pdf.h5" in plan.restart_links  # removed after the run, like the others
    run_ptrace(plan)
    assert not (plan.work_dir / "jorek_pdf.h5").exists()
    assert not is_current(_plan(run_dir, site, end_step=3200, pdf_step=3000))


def test_missing_pdf_step(run_dir, site):
    with pytest.raises(PtraceError, match="no restart for ptrace_pdf_step 3100"):
        _plan(run_dir, site, pdf_step=3100)
