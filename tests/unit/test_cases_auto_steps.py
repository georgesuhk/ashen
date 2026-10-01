"""cases.toml steps tables that leave start and/or stop to the run's own
restarts (ashen.cases._steps_from_range_dict)."""

from __future__ import annotations

import pytest

from ashen.cases import CasesError, load_cases


@pytest.fixture
def root(tmp_path):
    """A run with restarts every 200 steps from 0 to 2000."""
    run = tmp_path / "run"
    run.mkdir()
    for step in range(0, 2001, 200):
        (run / f"jorek{step:05d}.h5").write_bytes(b"")
    (run / "jorek_restart.h5").write_bytes(b"")
    return tmp_path


def steps(root, spec, case="run", extra=""):
    path = root / "cases.toml"
    path.write_text(f'[cases."{case}"]\nsteps = {spec}\n{extra}', encoding="utf-8")
    return load_cases(path, run_root=root)[case]


@pytest.mark.parametrize("spec, expected", [
    ("{ step = 400 }", [0, 400, 800, 1200, 1600, 2000]),
    ("{ step = 600 }", [0, 600, 1200, 1800]),
    # every restart, last one included
    ("{}", list(range(0, 2001, 200))),
    ("{ start = 1000, step = 400 }", [1000, 1400, 1800]),
    ("{ start = 600 }", list(range(600, 2001, 200))),
    # an explicit stop stays exclusive, as with both ends given
    ("{ stop = 1200, step = 400 }", [0, 400, 800]),
    # only restarts that exist: 100, 300, ... are not there
    ("{ step = 100 }", list(range(0, 2001, 200))),
    ("{ start = 100, step = 200 }", []),
    # mixed with explicit steps
    ("[50, { start = 1600, step = 200 }]", [50, 1600, 1800, 2000]),
])
def test_open_ended_ranges_come_from_the_restarts(root, spec, expected):
    assert steps(root, spec).steps == expected


def test_both_ends_given_is_unchanged_and_ignores_the_restarts(root):
    assert steps(root, "{ start = 100, stop = 500, step = 100 }").steps == [100, 200, 300, 400]


def test_start_is_the_first_restart_not_zero(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for step in (200, 400, 600, 800, 1000):
        (run / f"jorek{step:06d}.h5").write_bytes(b"")
    assert steps(tmp_path, "{ step = 400 }").steps == [200, 600, 1000]


def test_a_new_restart_is_picked_up_on_the_next_load(root):
    assert steps(root, "{ step = 400 }").steps[-1] == 2000
    (root / "run" / "jorek02400.h5").write_bytes(b"")
    assert steps(root, "{ step = 400 }").steps[-1] == 2400


def test_a_missing_or_empty_run_folder_gives_no_steps(root):
    assert steps(root, "{ step = 400 }", case="not_there").steps == []
    (root / "empty").mkdir()
    assert steps(root, "{ step = 400 }", case="empty").steps == []


def test_defaults_resolve_per_case(root):
    other = root / "other"
    other.mkdir()
    for step in (0, 400, 800):
        (other / f"jorek{step:05d}.h5").write_bytes(b"")
    path = root / "cases.toml"
    path.write_text(
        "[defaults]\nsteps = { step = 400 }\n\n[cases.run]\n\n[cases.other]\n", encoding="utf-8")
    cases = load_cases(path, run_root=root)
    assert cases["run"].steps[-1] == 2000 and cases["other"].steps == [0, 400, 800]


def test_per_diag_override(root):
    case = steps(root, "{ step = 400 }", extra='\n[cases.run.four]\nsteps = { start = 1600 }\n')
    assert case.steps_for("four") == [1600, 1800, 2000]
    assert case.steps_for("poincare")[0] == 0


def test_run_root_defaults_to_the_current_directory(root, monkeypatch):
    path = root / "cases.toml"
    path.write_text("[cases.run]\nsteps = { step = 1000 }\n", encoding="utf-8")
    monkeypatch.chdir(root)
    assert load_cases(path)["run"].steps == [0, 1000, 2000]


@pytest.mark.parametrize("spec, message", [
    ("{ step = 0 }", "step must be >= 1"),
    ("{ stpe = 400 }", r"unknown key\(s\) \['stpe'\]"),
    ('{ step = "x" }', "must be whole numbers"),
])
def test_bad_tables(root, spec, message):
    with pytest.raises(CasesError, match=message):
        steps(root, spec)
