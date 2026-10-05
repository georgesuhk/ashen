"""util --func delete_figures: the figures plot drew go, everything it
drew them from stays."""

from __future__ import annotations

import pytest

from ashen.cli import util as util_cli
from ashen.figure_clean import figure_files


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    run = tmp_path / "run"
    files = {
        # figures, next to the caches they were drawn from
        "poinc_dir/3000_poincare.png": 400, "poinc_dir/LCTT_200_5800.png": 300,
        "poinc_dir/theta_hist.png": 100, "four_dir/Psi_modes_lin.png": 200,
        "profiles/psi_T_profile.png": 100, "profiles/psi_T_profile.gif": 900,
        "ptrace/ptrace_gc/E10000000eV_n1000/particles.png": 50,
        "ptrace/ptrace_gc/E10000000eV_n1000/particles.gif": 500,
        "ptrace/ptrace_gc/E10000000eV_n1000/particle_exits_psi0.8-1.png": 50,
        "ptrace/ptrace_gc/E10000000eV_n1000/particle_wetted_counts.png": 50,
        "ptrace/ptrace_gc/E10000000eV_n1000/particle_loss.png": 50,
        "ptrace/ex7_jorek/particle_exits.gif": 70,
        # not figures, or not plot's: stay
        "poinc_dir/poinc_s003000.h5": 9000, "four_dir/four_s3000.h5": 9000,
        "postproc/zeroD_quantities_s03000.dat": 10,
        "ptrace/ptrace_gc/E10000000eV_n1000/ptrace_diag.h5": 5000,
        "ptrace/ptrace_gc/E10000000eV_n1000/particle_wetted.json": 10,
        "ptrace/ptrace_gc/E10000000eV_n1000/part_restart.h5": 100,
        "ptrace/ptrace_gc/E10000000eV_n1000/my_sketch.png": 10,
        "notes.png": 10, "jorek03000.h5": 100,
    }
    for name, size in files.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "eta_scan_theta_hist.png").write_bytes(b"x")
    (tmp_path / "cases.toml").write_text("[cases.run]\nsteps = [3000]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


FIGURES = 12


def test_only_plots_figures_are_found(campaign):
    found = {p.relative_to(campaign / "run").as_posix() for p in figure_files(campaign / "run")}
    assert len(found) == FIGURES
    assert "ptrace/ptrace_gc/E10000000eV_n1000/my_sketch.png" not in found
    assert "notes.png" not in found
    assert not any(p.endswith((".h5", ".json", ".dat")) for p in found)


def test_delete_figures(campaign, capsys):
    assert util_cli.main(["--func", "delete_figures", "--apply"]) == 0
    out = capsys.readouterr().out
    assert f"deleted {FIGURES} figure(s), freeing 2.8 kB -- `plot` draws them again" in out
    run = campaign / "run"
    assert figure_files(run) == []
    for kept in ("poinc_dir/poinc_s003000.h5", "four_dir/four_s3000.h5",
                 "ptrace/ptrace_gc/E10000000eV_n1000/ptrace_diag.h5",
                 "ptrace/ptrace_gc/E10000000eV_n1000/particle_wetted.json",
                 "ptrace/ptrace_gc/E10000000eV_n1000/my_sketch.png", "notes.png", "jorek03000.h5"):
        assert (run / kept).is_file(), kept
    # comparison figures belong to no one run
    assert (campaign / "figures" / "eta_scan_theta_hist.png").is_file()


def test_without_apply_it_only_says_what_it_would_free(campaign, capsys):
    assert util_cli.main(["--func", "delete_figures"]) == 0
    out = capsys.readouterr().out
    assert (f"would delete {FIGURES} figure(s), freeing 2.8 kB -- nothing deleted yet; "
            "add --apply to delete") in out
    assert "would delete 2 figure(s) in profiles/" in out
    assert len(figure_files(campaign / "run")) == FIGURES


def test_a_linked_figure_is_left(campaign):
    target = campaign / "elsewhere.png"
    target.write_bytes(b"x")
    (campaign / "run" / "profiles" / "linked.png").symlink_to(target)
    util_cli.main(["--func", "delete_figures", "--apply"])
    assert (campaign / "run" / "profiles" / "linked.png").is_symlink() and target.is_file()
