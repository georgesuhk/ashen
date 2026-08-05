"""Tests for ashen.cli.analyse's zeroD gathering: cache-gating, fan-out, and
the missing-restart warn-and-continue behaviour.

Exercises ``_gather_zero_d`` directly (not the full CLI) since it needs only
a ``Jorek2Run``/``RunPaths`` pair and a stand-in ``run_zero_d`` -- no real
JOREK, no cases.toml plumbing.
"""

from __future__ import annotations

import pytest

from ashen.cli import analyse as analyse_cli
from ashen.jorek2 import Jorek2Run, MissingRestartError
from ashen.paths import RunPaths


@pytest.fixture
def jrun_and_paths(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run = Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=run_dir / "in_main", pad_width=6)
    paths = RunPaths(run_dir, pad_width=6)
    return run, paths


def test_missing_restart_is_warned_and_skipped(jrun_and_paths, monkeypatch):
    """A step with no restart file yet must not abort the rest of the range --
    ports the behaviour George asked for: warn and move on to the next step."""
    run, paths = jrun_and_paths

    def fake_run_zero_d(jrun, step, paths):
        if step == 200:
            raise MissingRestartError(f"restart file not found: step {step}")
        paths.zero_d(step).parent.mkdir(parents=True, exist_ok=True)
        paths.zero_d(step).write_text("Time Energy\n1.0 1.0\n", encoding="utf-8")

    monkeypatch.setattr(analyse_cli, "run_zero_d", fake_run_zero_d)

    with pytest.warns(UserWarning, match="skipping zerod step 200"):
        analyse_cli._gather_zero_d(run, paths, [100, 200, 300], force=False, n_workers=1)

    assert paths.zero_d(100).is_file()
    assert not paths.zero_d(200).is_file()
    assert paths.zero_d(300).is_file()


def test_cached_steps_are_not_recomputed(jrun_and_paths, monkeypatch, capsys):
    run, paths = jrun_and_paths
    paths.zero_d(100).parent.mkdir(parents=True, exist_ok=True)
    paths.zero_d(100).write_text("Time Energy\n1.0 1.0\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        analyse_cli, "run_zero_d", lambda jrun, step, paths: calls.append(step)
    )

    analyse_cli._gather_zero_d(run, paths, [100, 200], force=False, n_workers=1)

    assert calls == [200]
    assert "[cached]" in capsys.readouterr().out


def test_force_recomputes_cached_steps(jrun_and_paths, monkeypatch):
    run, paths = jrun_and_paths
    paths.zero_d(100).parent.mkdir(parents=True, exist_ok=True)
    paths.zero_d(100).write_text("Time Energy\n1.0 1.0\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        analyse_cli, "run_zero_d", lambda jrun, step, paths: calls.append(step)
    )

    analyse_cli._gather_zero_d(run, paths, [100], force=True, n_workers=1)

    assert calls == [100]
