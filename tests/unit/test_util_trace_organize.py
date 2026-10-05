"""util --func trace_organize: restart links out of trace folders, and
traces from older layouts into ptrace/<exe>/<label>/."""

from __future__ import annotations

import json
import os

import pytest

from ashen.cli import util as util_cli
from ashen.trace_tidy import plan_tidy, read_label_settings

pytestmark = pytest.mark.skipif(os.name == "nt", reason="symlinks")

CASES = """
[cases.run]
steps = [3000]

[cases.other]
steps = [3000]
"""


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    for name in ("run", "other"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "jorek03000.h5").write_bytes(b"restart" * 100)
    (tmp_path / "cases.toml").write_text(CASES, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _files(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in names.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def _link_restarts(folder, run):
    for name in ("jorek000000.h5", "jorek00000.h5", "jorek_restart.h5", "jorek_pdf.h5"):
        (folder / name).symlink_to(os.path.relpath(run / "jorek03000.h5", folder))


SETTINGS = "&ptrace\n  restart_index = 0\n  n_markers = 1000\n  E_kin_eV = 10000000.0\n/\n"


# --- reading what an old trace ran with ------------------------------------------------


@pytest.mark.parametrize("files, expected", [
    ({"ptrace_settings.nml": SETTINGS}, {"E_kin_eV": [1e7], "n_markers": 1000}),
    # the old pair: overrides read over params
    ({"ptrace_params.nml": "&ptrace\n n_markers = 5\n E_kin_eV = 1.d6 ! low\n/\n",
      "ptrace_overrides.nml": "! from cases.toml\n&ptrace\n  E_kin_eV = 2.5d6\n/\n"},
     {"E_kin_eV": [2.5e6], "n_markers": 5}),
    # oldest: trace_params.nml, per-marker lists and repeat counts
    ({"trace_params.nml": "&ptrace\n n_markers = 3\n E_kin_eV = 1.d6, 2*5.d6\n"
                          " R0 = 1.4, 1.5, 1.6\n/\n"},
     {"E_kin_eV": [1e6, 5e6, 5e6], "n_markers": 3}),
    # a list longer than the markers: only theirs
    ({"ptrace_params.nml": "&ptrace\n n_markers = 2\n E_kin_eV = 1.d6, 1.d6, 9.d9\n/\n"},
     {"E_kin_eV": [1e6, 1e6], "n_markers": 2}),
    # wrapped over lines
    ({"ptrace_settings.nml": "&ptrace\n  n_markers = 7\n  E_kin_eV = 1.0, 2.0,\n"
                             "              3.0, 4.0, 5.0, 6.0, 7.0\n/\n"},
     {"E_kin_eV": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "n_markers": 7}),
    ({}, {}),
])
def test_read_label_settings(tmp_path, files, expected):
    assert read_label_settings(_files(tmp_path / "t", files)) == expected


# --- restart links ----------------------------------------------------------------------


def test_restart_links_and_copies_are_removed_from_every_trace_folder(campaign, capsys):
    run = campaign / "run"
    here = _files(run / "ptrace" / "ptrace_gc" / "E10000000eV_n1000",
                  {"ptrace_settings.nml": SETTINGS, "ptrace_meta.json": "{}"})
    _link_restarts(here, run)
    copy = here / "jorek000001.h5"
    copy.write_bytes(b"x" * 5000)                     # a copied restart: real bytes
    other = _files(run / "ptrace" / "ex7_jorek", {"diag.h5": ""})
    _link_restarts(other, run)

    assert util_cli.main(["--func", "trace_organize"]) == 0
    out = capsys.readouterr().out
    assert not list(here.glob("jorek*")) and not list(other.glob("jorek*"))
    assert (run / "jorek03000.h5").is_file()          # the run's own restart is untouched
    assert (here / "ptrace_settings.nml").is_file()
    assert "remove   ptrace/ptrace_gc/E10000000eV_n1000/jorek000001.h5  (copy, 5.0 kB)" in out
    assert "removed 9 restart link(s) or copies (freeing 5.0 kB); moved 0 trace(s)" in out


def test_dry_run_changes_nothing(campaign, capsys):
    run = campaign / "run"
    loose = _files(run / "ptrace" / "ptrace_gc", {"ptrace_settings.nml": SETTINGS,
                                                  "ptrace_diag.h5": "d"})
    _link_restarts(loose, run)
    assert util_cli.main(["--func", "trace_organize", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would remove 4 restart link(s) or copies; would move 1 trace(s)" in out
    assert len(list(loose.glob("jorek*"))) == 4 and (loose / "ptrace_diag.h5").is_file()


def test_a_folder_whose_job_may_be_running_is_left_alone(campaign, capsys, monkeypatch):
    run = campaign / "run"
    busy = _files(run / "ptrace" / "ptrace_gc" / "E10000000eV_n1000",
                  {"ptrace_settings.nml": SETTINGS})
    (busy / "ptrace_meta.json").write_text(json.dumps({"job_id": "77", "complete": False}))
    _link_restarts(busy, run)
    bin_dir = campaign / "bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text("#!/bin/sh\necho RUNNING\n", encoding="utf-8")
    (bin_dir / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    assert util_cli.main(["--func", "trace_organize"]) == 0
    out = capsys.readouterr().out
    assert len(list(busy.glob("jorek*"))) == 4
    assert "keep     ptrace/ptrace_gc/E10000000eV_n1000/  (job 77 is running;" in out


# --- older layouts ----------------------------------------------------------------------


def test_loose_trace_with_settings_moves_into_its_label(campaign, capsys):
    run = campaign / "run"
    loose = _files(run / "ptrace" / "ptrace_gc", {
        "ptrace_settings.nml": SETTINGS, "ptrace_diag.h5": "d", "ptrace.log": "log",
        "ptrace_meta.json": "{}", "part_restart.h5": "p", "part_restart_s003000_t1E-06.h5": "s",
        "2h.123.out": "o"})
    assert util_cli.main(["--func", "trace_organize"]) == 0
    target = loose / "E10000000eV_n1000"
    assert sorted(p.name for p in target.iterdir()) == sorted([
        "ptrace_settings.nml", "ptrace_diag.h5", "ptrace.log", "ptrace_meta.json",
        "part_restart.h5", "part_restart_s003000_t1E-06.h5", "2h.123.out"])
    assert [p.name for p in loose.iterdir()] == ["E10000000eV_n1000"]
    assert "move     ptrace/ptrace_gc/ -> ptrace/ptrace_gc/E10000000eV_n1000/  " \
           "(E10000000eV_n1000)" in capsys.readouterr().out


def test_old_params_and_overrides_pair(campaign):
    loose = _files(campaign / "run" / "ptrace" / "ptrace_gc_refluid", {
        "ptrace_params.nml": "&ptrace\n n_markers = 500\n E_kin_eV = 1.d6\n/\n",
        "ptrace_overrides.nml": "&ptrace\n E_kin_eV = 2.d7\n/\n", "ptrace_diag.h5": "d"})
    assert util_cli.main(["--func", "trace_organize"]) == 0
    assert (loose / "E20000000eV_n500" / "ptrace_overrides.nml").is_file()


def test_oldest_layout_moves_and_takes_todays_names(campaign, capsys):
    run = campaign / "run"
    old = _files(run / "trace" / "trace_gc", {
        "trace_params.nml": "&ptrace\n n_markers = 5\n E_kin_eV = 1.d7\n/\n",
        "trace_diag.h5": "d", "trace.log": "log", "trace_meta.json": "{}",
        "part_restart.h5": "p"})
    _link_restarts(old, run)
    assert util_cli.main(["--func", "trace_organize"]) == 0
    target = run / "ptrace" / "trace_gc" / "E10000000eV_n5"
    assert sorted(p.name for p in target.iterdir()) == sorted([
        "trace_params.nml", "ptrace_diag.h5", "ptrace.log", "ptrace_meta.json", "part_restart.h5"])
    assert not (run / "trace").exists()               # emptied, so gone
    out = capsys.readouterr().out
    assert "trace.log -> ptrace.log" in out and "trace_diag.h5 -> ptrace_diag.h5" in out


def test_oldest_layout_of_a_program_without_settings(campaign):
    run = campaign / "run"
    _files(run / "trace" / "re_gc_current_density_initialisation",
           {"part_diag.h5": "d", "part_restart.h5": "p", "trace.log": "l"})
    assert util_cli.main(["--func", "trace_organize"]) == 0
    home = run / "ptrace" / "re_gc_current_density_initialisation"
    assert sorted(p.name for p in home.iterdir()) == ["part_diag.h5", "part_restart.h5", "ptrace.log"]


def test_a_program_without_settings_is_already_home(campaign, capsys):
    _files(campaign / "run" / "ptrace" / "ex7_jorek", {"diag.h5": "d", "ptrace.log": "l"})
    plan = plan_tidy(campaign / "run")
    assert plan.actions == []


def test_a_gc_trace_without_its_settings_is_left_with_a_reason(campaign, capsys):
    loose = _files(campaign / "run" / "ptrace" / "ptrace_gc", {"ptrace_diag.h5": "d"})
    assert util_cli.main(["--func", "trace_organize"]) == 0
    assert (loose / "ptrace_diag.h5").is_file()
    assert "settings files do not say its energy and marker count" in capsys.readouterr().out


def test_a_taken_place_is_not_overwritten(campaign, capsys):
    exe = campaign / "run" / "ptrace" / "ptrace_gc"
    _files(exe, {"ptrace_settings.nml": SETTINGS, "ptrace_diag.h5": "old"})
    _files(exe / "E10000000eV_n1000", {"ptrace_settings.nml": SETTINGS, "ptrace_diag.h5": "new"})
    assert util_cli.main(["--func", "trace_organize"]) == 0
    assert (exe / "ptrace_diag.h5").read_text() == "old"
    assert (exe / "E10000000eV_n1000" / "ptrace_diag.h5").read_text() == "new"
    assert "E10000000eV_n1000/ already holds a trace" in capsys.readouterr().out


def test_labelled_folders_are_todays_layout_and_stay(campaign):
    here = _files(campaign / "run" / "ptrace" / "ptrace_gc" / "E5eV_n3",
                  {"ptrace_settings.nml": SETTINGS, "ptrace_diag.h5": "d"})
    assert util_cli.main(["--func", "trace_organize"]) == 0
    assert (here / "ptrace_diag.h5").is_file()        # its settings say otherwise; not moved


def test_case_selection_and_required_func(campaign, capsys):
    loose = _files(campaign / "other" / "ptrace" / "ptrace_gc",
                   {"ptrace_settings.nml": SETTINGS, "ptrace_diag.h5": "d"})
    assert util_cli.main(["--func", "trace_organize", "--case", "run"]) == 0
    assert (loose / "ptrace_diag.h5").is_file()
    with pytest.raises(SystemExit):
        util_cli.main([])
    with pytest.raises(SystemExit):
        util_cli.main(["--func", "nope"])
