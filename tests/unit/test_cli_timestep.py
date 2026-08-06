"""ashen.cli.timestep -- end-to-end against a synthetic run folder, with
step_time monkeypatched (no real jorek2_postproc needed)."""

from __future__ import annotations

import pytest

from ashen.cli import timestep as timestep_cli
from ashen.diagnostics.timestep import StepTime


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "jorek000100.h5").write_bytes(b"")
    (run_dir / "jorek000200.h5").write_bytes(b"")
    (run_dir / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    monkeypatch.chdir(run_dir)
    return run_dir


def _stub_step_time(monkeypatch, times: dict[int, tuple[float, float]]):
    def fake(run, paths, step):
        time_si, time_jorek = times[step]
        return StepTime(step=step, time_si=time_si, time_jorek=time_jorek)

    monkeypatch.setattr(timestep_cli, "step_time", fake)


def test_one_step_prints_both_units(run_dir, monkeypatch, capsys):
    _stub_step_time(monkeypatch, {100: (1.5e-4, 452.3)})

    assert timestep_cli.main(["100"]) == 0
    out = capsys.readouterr().out
    assert "step 100" in out
    assert "1.500000e-04 s" in out
    assert "4.523000e+02" in out


def test_two_steps_prints_delta_and_rate(run_dir, monkeypatch, capsys):
    _stub_step_time(monkeypatch, {100: (1.0e-4, 100.0), 200: (3.0e-4, 300.0)})

    assert timestep_cli.main(["100", "200"]) == 0
    out = capsys.readouterr().out
    assert "step 100" in out
    assert "step 200" in out
    assert "2.000000e-04 s" in out  # delta SI
    assert "2.000000e+02" in out  # delta jorek
    assert "s/step" in out


def test_more_than_two_steps_is_an_error(run_dir, capsys):
    assert timestep_cli.main(["100", "200", "300"]) == 1
    assert "at most two steps" in capsys.readouterr().out


def test_no_restart_files_is_an_error(tmp_path, monkeypatch, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)

    assert timestep_cli.main(["100"]) == 1
    assert "error" in capsys.readouterr().out


def test_missing_restart_step_is_reported(run_dir, monkeypatch, capsys):
    from ashen.jorek2 import MissingRestartError

    def fake(run, paths, step):
        raise MissingRestartError(f"restart file not found for step {step}")

    monkeypatch.setattr(timestep_cli, "step_time", fake)

    assert timestep_cli.main(["999"]) == 1
    assert "error" in capsys.readouterr().out
