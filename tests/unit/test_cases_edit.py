"""Setting a case's steps in cases.toml without disturbing the rest
(ashen.cases_edit)."""

from __future__ import annotations

import tomllib

import pytest

from ashen.cases import CasesError, load_cases
from ashen.cases_edit import format_steps_table, set_case_steps

SPEC = {"start": 0, "step": 400}
LINE = "steps = { start = 0, step = 400 }\n"

FILE = '''\
# campaign notes
[defaults]
n_turns = 500            # a comment to keep

[cases."qa2.1/eta1e-3"]
note  = "first"
steps = { start = 200, stop = 5800, step = 200 }   # old range
psi_n_in = [0.1, 0.5]

[cases."qa2.1/eta1e-3".four]
steps = [1000, 2000]

[cases."qa2.1/eta1e-4"]
  steps = [
      200, 400,          # early
      { start = 1000, stop = 2000, step = 500 },
  ]
  note = "second"

[cases.plain]
note = "takes its steps from somewhere else"
'''


@pytest.fixture
def path(tmp_path):
    out = tmp_path / "cases.toml"
    out.write_text(FILE, encoding="utf-8")
    return out


def test_format():
    assert format_steps_table(SPEC) == "{ start = 0, step = 400 }"
    assert format_steps_table({}) == "{}"


def test_replaces_a_one_line_value_and_nothing_else(path):
    assert set_case_steps(path, "qa2.1/eta1e-3", SPEC) == "replaced"
    old, new = FILE.splitlines(keepends=True), path.read_text().splitlines(keepends=True)
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    assert len(old) == len(new) and changed == [6] and new[6] == LINE
    data = tomllib.loads(path.read_text())
    assert data["cases"]["qa2.1/eta1e-3"]["steps"] == SPEC
    assert data["cases"]["qa2.1/eta1e-3"]["four"]["steps"] == [1000, 2000]   # override kept


def test_replaces_a_list_over_several_lines_keeping_its_indent(path):
    assert set_case_steps(path, "qa2.1/eta1e-4", SPEC) == "replaced"
    text = path.read_text()
    assert "  " + LINE + '  note = "second"\n' in text and "# early" not in text
    assert tomllib.loads(text)["cases"]["qa2.1/eta1e-4"] == {"steps": SPEC, "note": "second"}
    assert "# a comment to keep" in text and "# old range" in text


def test_adds_the_key_to_a_case_without_steps(path):
    assert set_case_steps(path, "plain", SPEC) == "added to the case"
    assert "[cases.plain]\n" + LINE + "note = " in path.read_text()


def test_adds_a_case_that_is_not_listed(path):
    assert set_case_steps(path, "qa3.3/new run", SPEC) == "added the case"
    text = path.read_text()
    assert text.startswith(FILE) and text.endswith('\n[cases."qa3.3/new run"]\n' + LINE)
    assert tomllib.loads(text)["cases"]["qa3.3/new run"] == {"steps": SPEC}


def test_the_result_loads_and_follows_the_restarts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for step in range(0, 2001, 200):
        (run / f"jorek{step:05d}.h5").write_bytes(b"")
    path = tmp_path / "cases.toml"
    path.write_text('[cases."run"]\nsteps = [0]\n', encoding="utf-8")
    set_case_steps(path, "run", SPEC)
    assert load_cases(path, run_root=tmp_path)["run"].steps == [0, 400, 800, 1200, 1600, 2000]


def test_writes_nothing_when_the_file_or_the_edit_is_not_sound(tmp_path):
    bad = tmp_path / "cases.toml"
    bad.write_text("[cases.a\nsteps = [1]\n", encoding="utf-8")
    with pytest.raises(CasesError, match="malformed TOML"):
        set_case_steps(bad, "a", SPEC)
    assert bad.read_text() == "[cases.a\nsteps = [1]\n"

    # the case written as an inline table: no [cases.a] header to edit under
    inline = '[cases]\na = { steps = [1], note = "x" }\n'
    bad.write_text(inline, encoding="utf-8")
    with pytest.raises(CasesError, match="nothing was written"):
        set_case_steps(bad, "a", SPEC)
    assert bad.read_text() == inline
    assert list(tmp_path.iterdir()) == [bad]                   # no temp file left
