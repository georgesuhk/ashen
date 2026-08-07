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


def test_comparison_with_explicit_x_tick_labels(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + """
        [comparisons.scan]
        note = "eta scan"
        cases = ["a", "b", "c"]
        x_tick_labels = ["1e-3", "1e-4", "1e-5"]
        n_cols = 5
        """,
    )
    cases = load_cases(path)
    comparisons = load_comparisons(path, cases)
    comparison = comparisons["scan"]
    assert comparison.name == "scan"
    assert comparison.note == "eta scan"
    assert comparison.cases == ["a", "b", "c"]
    assert comparison.x_tick_labels == ["1e-3", "1e-4", "1e-5"]
    assert comparison.n_cols == 5
    assert comparison.labelled_cases() == [
        ("1e-3", "a"), ("1e-4", "b"), ("1e-5", "c"),
    ]


def test_comparison_without_x_tick_labels_defaults_to_case_names(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a", "b"]\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["scan"]
    assert comparison.x_tick_labels == []
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


def test_comparison_x_tick_labels_length_mismatch_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a", "b"]\nx_tick_labels = ["only-one"]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="x_tick_labels"):
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


# --- x_values / x_label: scan-parameter plots (wetted fraction vs. eta) ----------


def test_x_values_default_to_none(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a", "b"]\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["scan"]
    assert comparison.x_values is None
    assert comparison.x_label == ""


def test_x_values_and_x_label_are_parsed(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + 'cases    = ["a", "b", "c"]\n'
        + 'x_values = [1e-3, 1e-4, 1e-5]\n'
        + 'x_label  = "$\\\\eta$"\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["scan"]
    assert comparison.x_values == pytest.approx([1e-3, 1e-4, 1e-5])
    assert comparison.x_label == r"$\eta$"


def test_x_values_length_mismatch_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a", "b"]\nx_values = [1.0]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="x_values"):
        load_comparisons(path, cases)


# --- comparison-level analysis-parameter overrides -------------------------------


def test_theta_overrides_default_to_none(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a"]\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["scan"]
    assert comparison.theta_target_psi is None
    assert comparison.theta_bins is None
    assert comparison.theta_psi_n_range is None
    assert comparison.theta_wetted_threshold is None


def test_theta_overrides_are_parsed(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + 'cases                  = ["a", "b"]\n'
        + 'theta_target_psi       = 1.1\n'
        + 'theta_bins             = 200\n'
        + 'theta_psi_n_range      = [0.3, 0.7]\n'
        + 'theta_wetted_threshold = 0.005\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["scan"]
    assert comparison.theta_target_psi == pytest.approx(1.1)
    assert comparison.theta_bins == 200
    assert comparison.theta_psi_n_range == [0.3, 0.7]
    assert comparison.theta_wetted_threshold == pytest.approx(0.005)


def test_theta_psi_n_range_override_rejects_non_pair(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a"]\ntheta_psi_n_range = [0.5]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="\\[min, max\\]"):
        load_comparisons(path, cases)


def test_theta_psi_n_range_override_rejects_min_not_less_than_max(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\ncases = ["a"]\ntheta_psi_n_range = [0.8, 0.2]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="min < max"):
        load_comparisons(path, cases)


def test_theta_wetted_threshold_override_rejects_non_positive(tmp_path):
    path = _write(
        tmp_path,
        _cases_block() + '[comparisons.scan]\ncases = ["a"]\ntheta_wetted_threshold = 0\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="must be positive"):
        load_comparisons(path, cases)


# --- datasets: grouped sub-scans sharing one comparison's x-axis -----------------


def test_datasets_are_parsed(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.wetted_vs_eta]\n'
        + 'x_label = "$\\\\eta$"\n'
        + '[comparisons.wetted_vs_eta.datasets.normal]\n'
        + 'cases = ["a", "b"]\n'
        + '[comparisons.wetted_vs_eta.datasets.rho19]\n'
        + 'cases = ["c"]\n'
        + 'color = "tab:red"\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["wetted_vs_eta"]
    assert comparison.cases == []
    assert set(comparison.datasets) == {"normal", "rho19"}
    normal = comparison.datasets["normal"]
    assert normal.name == "normal"
    assert normal.cases == ["a", "b"]
    assert normal.color is None
    rho19 = comparison.datasets["rho19"]
    assert rho19.cases == ["c"]
    assert rho19.color == "tab:red"


def test_dataset_labelled_cases_defaults_to_case_names():
    from ashen.comparisons import Dataset

    ds = Dataset(name="normal", cases=["a", "b"])
    assert ds.labelled_cases() == [("a", "a"), ("b", "b")]


def test_dataset_labelled_cases_uses_its_own_x_tick_labels():
    from ashen.comparisons import Dataset

    ds = Dataset(name="normal", cases=["a", "b"], x_tick_labels=["1e-3", "1e-4"])
    assert ds.labelled_cases() == [("1e-3", "a"), ("1e-4", "b")]


def test_dataset_series_label_defaults_to_name():
    from ashen.comparisons import Dataset

    ds = Dataset(name="rho19", cases=["a"])
    assert ds.series_label == "rho19"


def test_dataset_series_label_uses_dataset_label_when_set():
    from ashen.comparisons import Dataset

    ds = Dataset(name="rho19", cases=["a"], dataset_label=r"$\rho^* = 19$")
    assert ds.series_label == r"$\rho^* = 19$"


def test_dataset_inherits_comparisons_x_values_by_default(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.wetted_vs_eta]\n'
        + 'x_values = [1e-3, 1e-4]\n'
        + '[comparisons.wetted_vs_eta.datasets.normal]\n'
        + 'cases = ["a", "b"]\n'
        + '[comparisons.wetted_vs_eta.datasets.rho19]\n'
        + 'cases = ["c", "a"]\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["wetted_vs_eta"]
    assert comparison.datasets["normal"].x_values == pytest.approx([1e-3, 1e-4])
    assert comparison.datasets["rho19"].x_values == pytest.approx([1e-3, 1e-4])


def test_dataset_own_x_values_overrides_the_comparisons(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.wetted_vs_eta]\n'
        + 'x_values = [1e-3, 1e-4]\n'
        + '[comparisons.wetted_vs_eta.datasets.normal]\n'
        + 'cases = ["a", "b"]\n'
        + '[comparisons.wetted_vs_eta.datasets.rho19]\n'
        + 'cases = ["c", "a"]\n'
        + 'x_values = [2e-3, 2e-4]\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["wetted_vs_eta"]
    assert comparison.datasets["normal"].x_values == pytest.approx([1e-3, 1e-4])
    assert comparison.datasets["rho19"].x_values == pytest.approx([2e-3, 2e-4])


def test_dataset_with_neither_own_nor_inherited_x_values_is_none(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.wetted_vs_eta]\n'
        + '[comparisons.wetted_vs_eta.datasets.normal]\n'
        + 'cases = ["a"]\n',
    )
    cases = load_cases(path)
    comparison = load_comparisons(path, cases)["wetted_vs_eta"]
    assert comparison.datasets["normal"].x_values is None


def test_setting_both_cases_and_datasets_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + 'cases = ["a"]\n'
        + '[comparisons.scan.datasets.normal]\n'
        + 'cases = ["b"]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="both 'cases' and 'datasets'"):
        load_comparisons(path, cases)


def test_setting_neither_cases_nor_datasets_raises(tmp_path):
    path = _write(tmp_path, _cases_block() + '[comparisons.scan]\nnote = "oops"\n')
    cases = load_cases(path)
    with pytest.raises(CasesError, match="no 'cases'.*no 'datasets'"):
        load_comparisons(path, cases)


def test_top_level_x_tick_labels_with_datasets_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + 'x_tick_labels = ["x"]\n'
        + '[comparisons.scan.datasets.normal]\n'
        + 'cases = ["a"]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="x_tick_labels.*datasets"):
        load_comparisons(path, cases)


def test_dataset_naming_undefined_case_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + '[comparisons.scan.datasets.normal]\n'
        + 'cases = ["ghost"]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="undefined case"):
        load_comparisons(path, cases)


def test_dataset_with_empty_cases_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + '[comparisons.scan.datasets.normal]\n'
        + 'cases = []\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="no 'cases'"):
        load_comparisons(path, cases)


def test_dataset_x_tick_labels_length_mismatch_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + '[comparisons.scan.datasets.normal]\n'
        + 'cases = ["a", "b"]\n'
        + 'x_tick_labels = ["only-one"]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="x_tick_labels"):
        load_comparisons(path, cases)


def test_dataset_label_field_is_parsed(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + '[comparisons.scan.datasets.normal]\n'
        + 'cases = ["a"]\n'
        + "dataset_label = '$\\rho^* = 19$'\n",  # TOML literal string: no escaping
    )
    cases = load_cases(path)
    ds = load_comparisons(path, cases)["scan"].datasets["normal"]
    assert ds.dataset_label == r"$\rho^* = 19$"
    assert ds.series_label == r"$\rho^* = 19$"


def test_dataset_x_values_length_mismatch_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + '[comparisons.scan.datasets.normal]\n'
        + 'cases = ["a", "b"]\n'
        + 'x_values = [1.0]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="x_values"):
        load_comparisons(path, cases)


def test_dataset_inherited_x_values_length_mismatch_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + 'x_values = [1.0, 2.0]\n'
        + '[comparisons.scan.datasets.normal]\n'
        + 'cases = ["a"]\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="x_values"):
        load_comparisons(path, cases)


def test_dataset_unknown_key_raises(tmp_path):
    path = _write(
        tmp_path,
        _cases_block()
        + '[comparisons.scan]\n'
        + '[comparisons.scan.datasets.normal]\n'
        + 'cases = ["a"]\n'
        + 'bogus = true\n',
    )
    cases = load_cases(path)
    with pytest.raises(CasesError, match="unknown key"):
        load_comparisons(path, cases)
