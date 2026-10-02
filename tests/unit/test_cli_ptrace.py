"""ashen.cli.ptrace -- end to end from cases.toml and site.toml, against the
stub program from test_ptracing."""

from __future__ import annotations

import os
import textwrap

import pytest

from ashen.cli import ptrace as ptrace_cli
from test_ptracing import make_run

pytestmark = pytest.mark.skipif(os.name == "nt", reason="stub program is a POSIX shell script")

CASES = """
[cases.run]
steps            = [3000]
note             = "runaways from the current profile"
ptrace_exe        = "./exe/re_gc_current_density_initialisation"
ptrace_start_step = 3000
ptrace_n_mpi      = 2

[cases.other]
steps            = [3200]
ptrace_exe        = "./exe/ex7_jorek"
ptrace_start_step = 3200

[cases.untraced]
steps = [3000]
"""


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    make_run(root)
    make_run(root, name="other")
    (root / "site.toml").write_text(textwrap.dedent("""
        [paths]
        exe         = "./exe"
        template    = "./template"
        jobscripts  = "./jobscripts"
        jorek       = "../jorek"
        jorek_re    = "../jorek_RE"
        castor_root = "./castor"

        [launch]
        mpirun = ""

        [diagnostics]
        omp_threads = 2
    """), encoding="utf-8")
    (root / "cases.toml").write_text(CASES, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.delenv("ASHEN_SITE", raising=False)
    return root


def test_list_shows_tracing_cases(campaign, capsys):
    assert ptrace_cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert ("run: ./exe/re_gc_current_density_initialisation from step 3000 "
            "to the last restart -- runaways from the current profile") in out
    assert "other: ./exe/ex7_jorek from step 3200 to the last restart\n" in out
    assert "untraced" not in out
    assert "recognis" not in out


def test_dry_run_writes_nothing(campaign, capsys):
    assert ptrace_cli.main(["--case", "run", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "link     jorek000001.h5 ->" in out
    assert "link     jorek00001.h5 ->" in out
    assert "export OMP_NUM_THREADS=2" in out
    assert not (campaign / "run" / "trace").exists()


def test_runs_every_tracing_case_then_reports_them_cached(campaign, capsys):
    assert ptrace_cli.main(["--run_i"]) == 0
    out = capsys.readouterr().out
    assert ("==== run (./exe/re_gc_current_density_initialisation) ====\n"
            "  steps 3000..3400 (3 restart(s))") in out
    assert "==== other (./exe/ex7_jorek) ====" in out
    assert "untraced" not in out
    assert (campaign / "run" / "ptrace" / "re_gc_current_density_initialisation" / "part_diag.h5").is_file()
    assert (campaign / "other" / "ptrace" / "ex7_jorek" / "diag.h5").is_file()

    assert ptrace_cli.main(["--run_i"]) == 0
    assert capsys.readouterr().out.count("[cached]") == 2
    # and without a launch flag, just where they stand
    assert ptrace_cli.main([]) == 0
    assert capsys.readouterr().out.count("[cached] steps") == 2


def test_lost_particle_is_reported_not_failed(campaign, capsys, monkeypatch):
    monkeypatch.setenv("STUB_LOST", "1")
    assert ptrace_cli.main(["--case", "other", "--run_i"]) == 0
    assert "stopped at its first lost particle" in capsys.readouterr().out


def test_unknown_case(campaign, capsys):
    assert ptrace_cli.main(["--case", "nope"]) == 1
    assert "unknown case(s) ['nope']" in capsys.readouterr().err


def test_selecting_a_case_that_does_not_trace(campaign, capsys):
    assert ptrace_cli.main(["--case", "untraced"]) == 1
    assert "case(s) ['untraced'] set no ptrace_exe" in capsys.readouterr().err


def test_one_failing_trace_does_not_stop_the_others(campaign, capsys):
    (campaign / "cases.toml").write_text(
        CASES.replace("ptrace_start_step = 3200", "ptrace_start_step = 3100"), encoding="utf-8"
    )
    assert ptrace_cli.main(["--run_i"]) == 1
    err = capsys.readouterr().err
    assert "other: case 'other': no restart for ptrace_start_step 3100" in err
    assert "1 of 2 trace(s) failed: other" in err
    assert (campaign / "run" / "ptrace" / "re_gc_current_density_initialisation" / "part_diag.h5").is_file()


def test_invalid_trace_settings_are_reported(campaign, capsys):
    (campaign / "cases.toml").write_text(
        CASES.replace("ptrace_start_step = 3200", "ptrace_start_step = -1"), encoding="utf-8"
    )
    assert ptrace_cli.main(["--list"]) == 1
    assert "ptrace_start_step must be >= 0" in capsys.readouterr().err



# --- launch modes ----------------------------------------------------------------


def test_without_a_launch_flag_nothing_runs(campaign, capsys):
    assert ptrace_cli.main(["--case", "run"]) == 0
    out = capsys.readouterr().out
    assert "not traced yet (steps 3000..3400 (3 restart(s))); --run_i to trace here, --run to queue it" in out
    assert not (campaign / "run" / "ptrace").exists()


def test_status_says_when_settings_changed(campaign, capsys):
    assert ptrace_cli.main(["--case", "run", "--run_i"]) == 0
    (campaign / "cases.toml").write_text(
        CASES.replace("ptrace_n_mpi      = 2", "ptrace_n_mpi      = 3"), encoding="utf-8")
    capsys.readouterr()
    assert ptrace_cli.main(["--case", "run"]) == 0
    assert "out of date or unfinished" in capsys.readouterr().out


def test_default_steps_from_the_cli(campaign, capsys):
    (campaign / "cases.toml").write_text(
        CASES.replace("ptrace_start_step = 3200\n", ""), encoding="utf-8")
    assert ptrace_cli.main(["--list"]) == 0
    assert "other: ./exe/ex7_jorek from the first restart to the last restart" in capsys.readouterr().out
    assert ptrace_cli.main(["--case", "other", "--run_i"]) == 0
    assert "steps 3000..3400 (3 restart(s))" in capsys.readouterr().out


def test_job_needs_run(campaign, capsys):
    assert ptrace_cli.main(["-job", "23h"]) == 1
    assert "-job chooses the jobscript for --run" in capsys.readouterr().err


def test_run_and_run_i_exclude_each_other(campaign):
    with pytest.raises(SystemExit):
        ptrace_cli.main(["--run", "--run_i"])


@pytest.fixture
def queued_campaign(campaign, monkeypatch):
    bin_dir = campaign / "fake_bin"
    bin_dir.mkdir()
    for name, text in (
        ("sbatch", 'echo "$@" > sbatch_args.txt\necho 777\n'),
        ("squeue", '[ -n "$FAKE_SQUEUE" ] && echo "$FAKE_SQUEUE"\nexit 0\n'),
        ("sacct", 'echo COMPLETED\n'),
    ):
        (bin_dir / name).write_text("#!/bin/sh\n" + text, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    (campaign / "jobscripts").mkdir()
    for job in ("2h", "23h"):
        (campaign / "jobscripts" / job).write_text(f"#!/bin/bash\n# {job}\n", encoding="utf-8")
    return campaign


def test_run_queues_with_the_2h_jobscript_by_default(queued_campaign, capsys):
    assert ptrace_cli.main(["--case", "run", "--run"]) == 0
    out = capsys.readouterr().out
    assert "queued: job 777 (2h)" in out
    folder = queued_campaign / "run" / "ptrace" / "re_gc_current_density_initialisation"
    args = (folder / "sbatch_args.txt").read_text().split()
    assert args[:2] == ["--parsable", str(queued_campaign / "jobscripts" / "2h")]


def test_job_flag_picks_the_jobscript(queued_campaign, capsys):
    assert ptrace_cli.main(["--case", "run", "--run", "-job", "23h"]) == 0
    assert "queued: job 777 (23h)" in capsys.readouterr().out
    assert ptrace_cli.main(["--case", "run", "--run", "--job", "5h"]) == 1
    assert "no jobscript '5h'" in capsys.readouterr().err


def test_status_follows_a_queued_trace_to_the_end(queued_campaign, capsys, monkeypatch):
    assert ptrace_cli.main(["--case", "run", "--run"]) == 0
    folder = queued_campaign / "run" / "ptrace" / "re_gc_current_density_initialisation"
    monkeypatch.setenv("FAKE_SQUEUE", "RUNNING")
    capsys.readouterr()
    assert ptrace_cli.main(["--case", "run"]) == 0
    assert "job 777: running" in capsys.readouterr().out
    assert ptrace_cli.main(["--case", "run", "--run_i"]) == 1
    assert "job 777 (RUNNING) is still queued or running" in capsys.readouterr().err

    # the job ends, having written what the program writes
    monkeypatch.delenv("FAKE_SQUEUE")
    (folder / "part_restart.h5").write_bytes(b"out")
    assert ptrace_cli.main(["--case", "run"]) == 0
    assert "job 777 done" in capsys.readouterr().out
    assert ptrace_cli.main(["--case", "run"]) == 0
    assert "[cached] steps 3000..3400" in capsys.readouterr().out
    assert ptrace_cli.main(["--case", "run", "--run"]) == 0
    assert "[cached]" in capsys.readouterr().out


# --- --case patterns ---------------------------------------------------------------


def test_a_pattern_picks_the_matching_cases_that_trace(campaign, capsys):
    """"*" matches run, other and untraced; only the first two trace."""
    assert ptrace_cli.main(["--case", "*", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "==== run (" in out and "==== other (" in out and "untraced" not in out
    assert ptrace_cli.main(["--case", "o*", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "==== other (" in out and "==== run (" not in out


def test_a_pattern_matching_only_untraced_cases(campaign, capsys):
    assert ptrace_cli.main(["--case", "untr*"]) == 1
    assert "none of the cases ['untr*'] select sets ptrace_exe" in capsys.readouterr().err


# --- ptrace_gc settings: printed when a trace runs or is queued ---------------------

GC_CASES = CASES + """
[cases.gc]
steps              = [3000]
ptrace_exe         = "./exe/ptrace_gc"
ptrace_initialiser = "current_pdf_simple"
ptrace_n_markers   = 500
ptrace_E_kin_eV    = 1e7
ptrace_cos_pitch   = 0.9
ptrace_dt          = 5e-11
"""


def _gc(campaign):
    make_run(campaign, name="gc")
    (campaign / "cases.toml").write_text(GC_CASES, encoding="utf-8")
    return campaign / "gc" / "ptrace" / "ptrace_gc"


def test_settings_are_printed_when_a_trace_runs_but_not_when_cached(campaign, capsys):
    folder = _gc(campaign)
    assert ptrace_cli.main(["--case", "gc", "--run_i"]) == 0
    out = capsys.readouterr().out
    assert "  settings (ptrace_settings.nml in the trace folder):" in out
    lines = [line.strip() for line in out.splitlines()]
    assert any(l.startswith("n_markers") and l.endswith("= 500") for l in lines)
    assert any(l.startswith("dt") and l.endswith("= 5e-11") for l in lines)
    assert any(l.startswith("diag_step") and l.endswith("= 1e-08   (default)") for l in lines)
    # the record of what it ran with, in its folder
    recorded = (folder / "ptrace_settings.nml").read_text(encoding="utf-8")
    assert "  n_markers = 500\n" in recorded and "  diag_step = 1d-08\n" in recorded

    assert ptrace_cli.main(["--case", "gc", "--run_i"]) == 0
    out = capsys.readouterr().out
    assert "[cached]" in out and "settings (" not in out
    assert ptrace_cli.main(["--case", "gc", "--run_i", "--force"]) == 0
    assert "settings (" in capsys.readouterr().out


def test_settings_are_printed_when_a_trace_is_queued(queued_campaign, capsys):
    _gc(queued_campaign)
    assert ptrace_cli.main(["--case", "gc", "--run"]) == 0
    out = capsys.readouterr().out
    assert out.index("settings (ptrace_settings.nml") < out.index("queued: job 777")
    assert "initialiser" in out and "current_pdf_simple" in out


def test_a_program_without_settings_prints_none(campaign, capsys):
    assert ptrace_cli.main(["--case", "run", "--run_i"]) == 0
    assert "settings (" not in capsys.readouterr().out


def test_a_params_file_in_ptrace_inputs_is_refused(campaign, capsys):
    _gc(campaign)
    (campaign / "cases.toml").write_text(
        GC_CASES + 'ptrace_inputs = ["ptrace_params.nml"]\n', encoding="utf-8")
    assert ptrace_cli.main(["--case", "gc", "--run_i"]) == 1
    assert "ptrace_gc no longer reads a settings file of yours" in capsys.readouterr().err
