"""ashen.comparisons -- [comparisons.*] tables in cases.toml, grouping
already-defined cases for cross-case figures (e.g. a resistivity scan)."""

from __future__ import annotations

import pytest

from ashen.cases import CasesError, load_cases
from ashen.comparisons import load_comparisons


def _write(tmp_path, text):
    path = tmp_path / "cases.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _cases_block():
    return (
        '[cases.a]\nsteps = [1]\n'
        '[cases.b]\nsteps = [1]\n'
        '[cases.c]\nsteps = [1]\n'
    )


def test_no_comparisons_section_returns_empty_dict(tmp_path):
    path = _write(tmp_path, _cases_block())
    cases = load_cases(path)
    assert load_comparisons(path, cases) == {}


def test_comparison_with_explicit_labels(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + """
        [comparisons.scan]
        note = "eta scan"
        cases = ["a", "b", "c"]
        labels = ["1e-3", "1e-4", "1e-5"]
        n_cols = 5
        """,
    )
    cases = load_cases(path)
    comparisons = load_comparisons(path, cases)
    comparison = comparisons["scan"]
    assert comparison.name == "scan"
    assert comparison.note == "eta scan"
    assert comparison.cases == ["a", "b", "c"]
    assert comparison.labels == ["1e-3", "1e-4", "1e-5"]
    assert comparison.n_cols == 5
    assert comparison.labelled_cases() == [
        ("1e-3", "a"), ("1e-4", "b"), ("1e-5", "c"),
    ]


def test_comparison_without_labels_defaults_to_case_names(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a", "b"]\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["scan"]
    assert comparison.labels == []
    assert comparison.labelled_cases() == [("a", "a"), ("b", "b")]


def test_comparison_default_n_cols(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a"]\n',
    )
    cases = load_cases(path)
    assert load_comparisons(path, cases)["scan"].n_cols == 4


def test_comparison_naming_undefined_case_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a", "ghost"]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="undefined case"):
        load_comparisons(path, cases)


def test_comparison_with_empty_cases_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = []\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="no 'cases'"):
        load_comparisons(path, cases)


def test_comparison_missing_cases_key_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\nnote = "oops"\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="no 'cases'"):
        load_comparisons(path, cases)


def test_comparison_labels_length_mismatch_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a", "b"]\nlabels = ["only-one"]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="labels"):
        load_comparisons(path, cases)


def test_comparison_unknown_key_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a"]\nbogus = true\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="unknown key"):
        load_comparisons(path, cases)


def test_multiple_comparisons(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.one]\ncases = ["a"]\n'
        + '[comparisons.two]\ncases = ["b", "c"]\n',
    )
    cases = load_cases(path)
    comparisons = load_comparisons(path, cases)
    assert sorted(comparisons) == ["one", "two"]


def test_missing_file_raises(tmp_path):
    cases = {}
    with pytest.raises(CasesError, match="not found"):
        load_comparisons(tmp_path / "does_not_exist.toml", cases)
