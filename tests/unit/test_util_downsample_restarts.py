"""util --func downsample_restarts: keep the restarts at multiples of
--every, and every one anything still needs."""

from __future__ import annotations

import pytest

from ashen.cli import util as util_cli
from ashen.restart_thin import plan_thin


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    for step in range(0, 401, 20):                    # every 20 steps, 0..400
        (run / f"jorek{step:05d}.h5").write_bytes(b"x" * 1000)
    (run / "jorek_restart.h5").write_bytes(b"latest")
    (run / "in_main").write_text("&in1\n&end\n")
    (tmp_path / "cases.toml").write_text("[cases.run]\nsteps = [100, 200]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return run


def _steps(run):
    return sorted(int(p.name[5:-3]) for p in run.glob("jorek[0-9]*.h5"))


def test_nothing_is_deleted_without_apply(campaign, capsys):
    assert util_cli.main(["--func", "downsample_restarts", "--every", "40"]) == 0
    out = capsys.readouterr().out
    assert len(_steps(campaign)) == 21
    assert "21 restart step(s): would delete 9, keeping 12 (9.0 kB)" in out
    assert "nothing deleted yet; add --apply to delete (it cannot be undone)" in out


def test_every_40_keeps_multiples_of_40_and_what_cases_toml_uses(campaign, capsys):
    assert util_cli.main(["--func", "downsample_restarts", "--every", "40", "--apply"]) == 0
    out = capsys.readouterr().out
    # 0, 40, ..., 400 -- and 100, which cases.toml asks for (200 is a multiple)
    assert _steps(campaign) == [0, 40, 80, 100, 120, 160, 200, 240, 280, 320, 360, 400]
    assert "kept, not a multiple of 40 but cases.toml uses it: 100" in out
    assert "deleted 9 restart step(s), freeing 9.0 kB" in out
    assert (campaign / "jorek_restart.h5").is_file() and (campaign / "in_main").is_file()


def test_first_and_last_are_kept(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for step in (20, 40, 60, 80, 100, 110):
        (run / f"jorek{step:06d}.h5").write_bytes(b"x")
    plan = plan_thin(run, 40)
    assert sorted({s for s, _, _ in plan.delete}) == [60, 100]
    assert plan.kept_anyway == {20: "the run's first restart", 110: "the run's last restart"}


def test_both_widths_of_one_step_go_together(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for name in ("jorek00000.h5", "jorek00020.h5", "jorek000020.h5", "jorek00040.h5", "jorek00060.h5"):
        (run / name).write_bytes(b"x")
    plan = plan_thin(run, 40)
    assert sorted(p.name for _, p, _ in plan.delete) == ["jorek000020.h5", "jorek00020.h5"]


def test_every_kind_of_step_a_case_names_is_kept(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    for step in range(0, 201, 10):
        (run / f"jorek{step:05d}.h5").write_bytes(b"x")
    (run / "jorek_pdf.h5").write_bytes(b"x")          # not a numbered restart: untouched
    (tmp_path / "cases.toml").write_text(
        "[cases.run]\nsteps = [30]\nptrace_exe = \"./exe/p\"\nptrace_start_step = 50\n"
        "ptrace_end_step = 170\nptrace_pdf_step = 90\n\n[cases.run.four]\nsteps = [110]\n",
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert util_cli.main(["--func", "downsample_restarts", "--every", "40", "--apply"]) == 0
    kept = sorted(int(p.name[5:-3]) for p in run.glob("jorek[0-9]*.h5"))
    assert kept == [0, 30, 40, 50, 80, 90, 110, 120, 160, 170, 200]
    assert (run / "jorek_pdf.h5").is_file()


def test_every_is_required_and_positive(campaign, capsys):
    assert util_cli.main(["--func", "downsample_restarts"]) == 1
    assert "needs --every STEPS" in capsys.readouterr().err
    assert util_cli.main(["--func", "downsample_restarts", "--every", "0"]) == 1
    assert len(_steps(campaign)) == 21
