"""--case values: names, patterns and folders (ashen.cli._common), and how
analyse and ptrace take them."""

from __future__ import annotations

import pytest

from ashen.cli import analyse as analyse_cli
from ashen.cli._common import matching_cases, resolve_selection

NAMES = [
    "qa2.1_g2.3/eta1e-3_RE",
    "qa2.1_g2.3/eta1e-4_RE",
    "qa2.1_g2.5/eta1e-3_RE",
    "qa2.3_g2.3/eta1e-3_RE",
    "single",
]
CASES = dict.fromkeys(NAMES)


@pytest.mark.parametrize("selector, expected", [
    ("single", ["single"]),
    ("qa2.1_g2.3/eta1e-3_RE", ["qa2.1_g2.3/eta1e-3_RE"]),
    # a pattern: * crosses the "/" between folder and run
    ("qa2.1*", NAMES[:3]),
    ("*eta1e-3_RE", [NAMES[0], NAMES[2], NAMES[3]]),
    ("qa2.?_g2.3/*", [NAMES[0], NAMES[1], NAMES[3]]),
    ("qa2.[13]_g2.3/eta1e-3*", [NAMES[0], NAMES[3]]),
    # a pattern that matches the folder the cases are in
    ("qa2.1_g2.?", NAMES[:3]),
    # a folder -- what the shell turns an unquoted qa2.1* into
    ("qa2.1_g2.3", NAMES[:2]),
    ("qa2.1_g2.3/", NAMES[:2]),
    # not a prefix match: a folder is a whole path component
    ("qa2.1", []),
    ("sing", []),
    ("nope*", []),
    ("QA2.1*", []),
])
def test_matching_cases(selector, expected):
    assert matching_cases(selector, NAMES) == expected


def test_selection_is_in_cases_order_without_repeats():
    assert resolve_selection(["single", "qa2.1*", "qa2.1_g2.3"], CASES) == [*NAMES[:3], "single"]


def test_nothing_selected_is_every_case():
    assert resolve_selection(None, CASES) == NAMES


def test_a_value_that_selects_nothing_is_reported(capsys):
    assert resolve_selection(["qa2.1*", "qa9*", "nope"], CASES) is None
    err = capsys.readouterr().err
    assert "unknown case(s) ['qa9*', 'nope']" in err
    assert "quote a pattern" in err


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    body = "".join(f'[cases."{name}"]\nsteps = [100]\n\n' for name in NAMES)
    (tmp_path / "cases.toml").write_text(body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASHEN_SITE", raising=False)
    ran = []
    monkeypatch.setattr(analyse_cli, "_run_case", lambda case, **kwargs: ran.append(case.name))
    return ran


def test_analyse_takes_a_pattern(campaign):
    assert analyse_cli.main(["--diag", "poincare", "--case", "qa2.1*"]) == 0
    assert campaign == NAMES[:3]


def test_analyse_takes_what_the_shell_expands_an_unquoted_pattern_to(campaign):
    """`--case qa2.1*` unquoted arrives as the run folders, several after one --case."""
    assert analyse_cli.main(
        ["--diag", "poincare", "--case", "qa2.1_g2.3", "qa2.1_g2.5", "--force"]
    ) == 0
    assert campaign == NAMES[:3]


def test_analyse_repeated_case_flags_still_add_up(campaign):
    assert analyse_cli.main(["--case", "single", "--case", "qa2.3*"]) == 0
    assert campaign == [NAMES[3], "single"]


def test_analyse_pattern_matching_nothing(campaign, capsys):
    assert analyse_cli.main(["--case", "qa7*"]) == 1
    assert campaign == []
    assert "unknown case(s) ['qa7*']" in capsys.readouterr().err


def test_analyse_passes_the_matched_cases_on_to_plot(campaign, monkeypatch):
    seen = []
    monkeypatch.setattr(analyse_cli.plot_cli, "main", lambda argv: seen.append(argv) or 0)
    assert analyse_cli.main(["--diag", "four", "--case", "qa2.1_g2.3", "-plot", "four"]) == 0
    assert seen[0][:6] == ["--cases", "cases.toml", "--case", NAMES[0], "--case", NAMES[1]]
