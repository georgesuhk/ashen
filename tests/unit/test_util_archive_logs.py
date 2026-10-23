"""util --func archive_logs / tstep_harvest: each job's log kept once,
gzipped, SLURM's copies deleted only once kept; the time-step record."""

from __future__ import annotations

import os
import time

import pytest

from ashen.cli import util as util_cli
from support import jorek_log_text


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "jorek00000.h5").touch()
    crashed = jorek_log_text([(1.0, 2, True)] * 3 + [(1.0, 200, True)], ending="gmresfail")
    restart = jorek_log_text([(0.5, 25, True)] * 4, first_index=3, t0=2.0)
    (run / "1.8_1e-05.1.out").write_text(crashed)
    (run / "1.8_1e-05.2.out").write_text(restart)
    (run / "log").write_text(restart)                        # = job 2's .out
    (run / "trace0000.out").write_text("ptrace\n")
    then = time.time() - 3600
    for path in run.iterdir():
        os.utime(path, (then, then))
    (tmp_path / "cases.toml").write_text("[cases.run]\nsteps = [0]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return run


def test_preview_changes_nothing(campaign, capsys):
    before = sorted(p.name for p in campaign.iterdir())
    assert util_cli.main(["--func", "archive_logs"]) == 0
    out = capsys.readouterr().out
    assert sorted(p.name for p in campaign.iterdir()) == before
    assert "logs/log_s00001-s00004_gmresfail.gz" in out and "would delete" in out
    assert "add --apply" in out


def test_apply_keeps_each_job_once_and_deletes_the_out_copies(campaign, capsys):
    assert util_cli.main(["--func", "archive_logs", "--apply"]) == 0
    kept = sorted(p.name for p in (campaign / "logs").glob("*.gz"))
    assert kept == ["log_s00001-s00004_gmresfail.gz", "log_s00003-s00006.gz"]
    assert not list(campaign.glob("1.8_1e-05.*.out"))
    assert (campaign / "log").is_file()                       # read by ashen; the next launch replaces it
    assert (campaign / "trace0000.out").is_file()              # not a JOREK log


def test_a_job_that_may_still_run_keeps_its_out(campaign, capsys):
    os.utime(campaign / "1.8_1e-05.2.out", None)              # written just now
    assert util_cli.main(["--func", "archive_logs", "--apply"]) == 0
    assert (campaign / "1.8_1e-05.2.out").is_file()
    assert "a job may still be running" in capsys.readouterr().out


def test_tstep_harvest_lists_each_job(campaign, capsys):
    assert util_cli.main(["--func", "tstep_harvest"]) == 0
    out = capsys.readouterr().out
    assert "steps 1-4 (t 0-4), dt 1, 2-200 GMRES iterations, ended: gmresfail" in out
    assert "steps 3-6 (t 2-4), dt 0.5, 25-25 GMRES iterations" in out
    assert (campaign / "tstep_history.json.gz").is_file()
