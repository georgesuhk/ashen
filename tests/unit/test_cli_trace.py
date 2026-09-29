"""ashen.cli.trace -- end to end from cases.toml and site.toml, against the
stub program from test_tracing."""

from __future__ import annotations

import os
import textwrap

import pytest

from ashen.cli import trace as trace_cli
from test_tracing import make_run

pytestmark = pytest.mark.skipif(os.name == "nt", reason="stub program is a POSIX shell script")

CASES = """
[cases.run]
steps            = [3000]
note             = "runaways from the current profile"
trace_program    = "re_gc_current_density_initialisation"
trace_start_step = 3000
trace_n_mpi      = 2

[cases.other]
steps            = [3200]
trace_program    = "ex7_jorek"
trace_start_step = 3200

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
    assert trace_cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert ("run: re_gc_current_density_initialisation from step 3000 "
            "-- runaways from the current profile") in out
    assert "other: ex7_jorek from step 3200" in out
    assert "untraced" not in out
    assert "  ex6_jorek: one relativistic full-orbit electron" in out


def test_dry_run_writes_nothing(campaign, capsys):
    assert trace_cli.main(["--case", "run", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "link     jorek000001.h5 ->" in out
    assert "link     jorek00001.h5 ->" in out
    assert "export OMP_NUM_THREADS=2" in out
    assert not (campaign / "run" / "trace").exists()


def test_runs_every_tracing_case_then_reports_them_cached(campaign, capsys):
    assert trace_cli.main([]) == 0
    out = capsys.readouterr().out
    assert "==== run (re_gc_current_density_initialisation) ====\n  steps 3000..3400 (3 restart(s))" in out
    assert "==== other (ex7_jorek) ====" in out
    assert "untraced" not in out
    # ex7's fixed 2.5 ms start can't be checked without zeroD here
    assert "warning: ex7_jorek always starts at t = 0.0025 s" in out
    assert (campaign / "run" / "trace" / "re_gc_current_density_initialisation" / "part_diag.h5").is_file()
    assert (campaign / "other" / "trace" / "ex7_jorek" / "diag.h5").is_file()

    assert trace_cli.main([]) == 0
    assert capsys.readouterr().out.count("[cached]") == 2


def test_lost_particle_is_reported_not_failed(campaign, capsys, monkeypatch):
    monkeypatch.setenv("STUB_LOST", "1")
    assert trace_cli.main(["--case", "other"]) == 0
    assert "stopped at a lost particle (ex7_jorek stops at the first)" in capsys.readouterr().out


def test_unknown_case(campaign, capsys):
    assert trace_cli.main(["--case", "nope"]) == 1
    assert "unknown case(s) ['nope']" in capsys.readouterr().err


def test_selecting_a_case_that_does_not_trace(campaign, capsys):
    assert trace_cli.main(["--case", "untraced"]) == 1
    assert "case(s) ['untraced'] set no trace_program" in capsys.readouterr().err


def test_one_failing_trace_does_not_stop_the_others(campaign, capsys):
    (campaign / "cases.toml").write_text(
        CASES.replace("trace_start_step = 3200", "trace_start_step = 3100"), encoding="utf-8"
    )
    assert trace_cli.main([]) == 1
    err = capsys.readouterr().err
    assert "other: case 'other': no restart for trace_start_step 3100" in err
    assert "1 of 2 trace(s) failed: other" in err
    assert (campaign / "run" / "trace" / "re_gc_current_density_initialisation" / "part_diag.h5").is_file()


def test_invalid_trace_settings_are_reported(campaign, capsys):
    (campaign / "cases.toml").write_text(
        CASES.replace('"ex7_jorek"', '"ex9_jorek"'), encoding="utf-8"
    )
    assert trace_cli.main(["--list"]) == 1
    assert "trace_program must be one of" in capsys.readouterr().err
