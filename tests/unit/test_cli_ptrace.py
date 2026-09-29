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


def test_list_shows_tracing_cases_and_programs(campaign, capsys):
    assert ptrace_cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert ("run: ./exe/re_gc_current_density_initialisation from step 3000 "
            "[recognised as re_gc_current_density_initialisation] "
            "-- runaways from the current profile") in out
    assert "other: ./exe/ex7_jorek from step 3200 [recognised as ex7_jorek]" in out
    assert "untraced" not in out
    assert "  ex6_jorek: one relativistic full-orbit electron" in out


def test_dry_run_writes_nothing(campaign, capsys):
    assert ptrace_cli.main(["--case", "run", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "link     jorek000001.h5 ->" in out
    assert "link     jorek00001.h5 ->" in out
    assert "export OMP_NUM_THREADS=2" in out
    assert not (campaign / "run" / "trace").exists()


def test_runs_every_tracing_case_then_reports_them_cached(campaign, capsys):
    assert ptrace_cli.main([]) == 0
    out = capsys.readouterr().out
    assert ("==== run (./exe/re_gc_current_density_initialisation) ====\n"
            "  recognised as re_gc_current_density_initialisation\n"
            "  steps 3000..3400 (3 restart(s))") in out
    assert "==== other (./exe/ex7_jorek) ====" in out
    assert "untraced" not in out
    # ex7's fixed 2.5 ms start can't be checked without zeroD here
    assert "warning: ex7_jorek always starts at t = 0.0025 s" in out
    assert (campaign / "run" / "ptrace" / "re_gc_current_density_initialisation" / "part_diag.h5").is_file()
    assert (campaign / "other" / "ptrace" / "ex7_jorek" / "diag.h5").is_file()

    assert ptrace_cli.main([]) == 0
    assert capsys.readouterr().out.count("[cached]") == 2


def test_lost_particle_is_reported_not_failed(campaign, capsys, monkeypatch):
    monkeypatch.setenv("STUB_LOST", "1")
    assert ptrace_cli.main(["--case", "other"]) == 0
    assert "stopped at a lost particle (ex7_jorek stops at the first)" in capsys.readouterr().out


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
    assert ptrace_cli.main([]) == 1
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


def test_unrecognised_exe_is_labelled(campaign, capsys):
    (campaign / "cases.toml").write_text(
        CASES.replace('"./exe/ex7_jorek"', '"./exe/my_tracer"'), encoding="utf-8"
    )
    assert ptrace_cli.main(["--list"]) == 0
    assert ("other: ./exe/my_tracer from step 3200 "
            "[not a program ashen knows: run as-is, success = exit code 0]") in capsys.readouterr().out
