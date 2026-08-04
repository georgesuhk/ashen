"""Tests for namelist reading, editing and the semantic comparator.

The comparator is what the golden tests rely on, so it is tested directly here
rather than only through them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ashen.namelist import (
    NamelistError,
    effective_fields,
    format_boundary_block,
    fortran_literal,
    parse_value,
    read_field,
    set_boundary_block,
    set_fields,
    write_boundary_file,
)

SAMPLE = """\
&in1
 restart = .f.
 tstep_n = 1
 nstep_n = 1200
 eta = 1.d-6                    ! resistivity
 eta_num = 1d-8
 central_density = 1.d-2
 R_geo = 1.368
 ffprime_file = 'ffprime_prof.dat'
 n_boundary = 3
R_boundary(  1) = 1.00, Z_boundary(  1) = 0.10, psi_boundary(  1) = 0.98
R_boundary(  2) = 1.10, Z_boundary(  2) = 0.20, psi_boundary(  2) = 0.98
R_boundary(  3) = 1.20, Z_boundary(  3) = 0.30, psi_boundary(  3) = 0.98
&end
"""


@pytest.fixture
def namelist(tmp_path) -> Path:
    path = tmp_path / "in_main"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


# --- fortran_literal ---------------------------------------------------------


def test_float_becomes_double_precision_literal():
    assert fortran_literal(1e-6) == "1.0d-6"
    assert fortran_literal(0.03) == "0.03d0"


def test_float_formatting_does_not_leak_representation_noise():
    """Fixed-precision formatting would give '9.9999999999999995d-7'."""
    assert fortran_literal(1e-6) == "1.0d-6"
    assert fortran_literal(0.1) == "0.1d0"


def test_non_integer_mantissa_is_not_mangled():
    """Regression: the old .replace("e", ".d") produced '1.5.d+20'."""
    rendered = fortran_literal(1.5e20)
    assert rendered == "1.5d20"
    assert ".d" not in rendered
    assert rendered.count(".") == 1


def test_whole_floats_stay_double_not_integer():
    """The old code emitted '100' -- an integer literal in a double field."""
    assert fortran_literal(100.0) == "100.0d0"


def test_bools_render_as_fortran_logicals():
    assert fortran_literal(True) == ".t."
    assert fortran_literal(False) == ".f."


def test_ints_stay_ints():
    assert fortran_literal(1200) == "1200"


def test_strings_pass_through_untouched():
    assert fortran_literal(".t.") == ".t."
    assert fortran_literal("'in_bnd.dat'") == "'in_bnd.dat'"


def test_sequences_render_as_fortran_lists():
    assert fortran_literal([1, 2]) == "1, 2"


def test_non_finite_is_refused():
    with pytest.raises(NamelistError):
        fortran_literal(float("nan"))


@pytest.mark.parametrize("value", [1e-6, 1.5e20, 0.03, 100.0, 3.7, 1e-12])
def test_literals_round_trip_through_the_parser(value):
    assert parse_value(fortran_literal(value)) == pytest.approx(value)


# --- parse_value -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1.d-6", 1e-6),
        ("1d-8", 1e-8),
        ("1.0e-6", 1e-6),
        ("1.368", 1.368),
        (".t.", True),
        (".F.", False),
        ("'prof.dat'", "prof.dat"),
        ("1, 2", (1.0, 2.0)),
    ],
)
def test_parse_value(text, expected):
    assert parse_value(text) == expected


def test_unparseable_value_is_kept_as_text():
    assert parse_value("Gears") == "Gears"


# --- effective_fields --------------------------------------------------------


def test_reads_scalars_and_ignores_comments(namelist):
    fields = effective_fields(namelist)

    assert fields["eta"] == pytest.approx(1e-6)
    assert fields["restart"] is False
    assert fields["ffprime_file"] == "ffprime_prof.dat"


def test_similarly_named_fields_stay_distinct(namelist):
    """`eta` must not swallow `eta_num`."""
    fields = effective_fields(namelist)

    assert fields["eta"] == pytest.approx(1e-6)
    assert fields["eta_num"] == pytest.approx(1e-8)


def test_multiple_assignments_on_one_line_are_all_captured(namelist):
    fields = effective_fields(namelist)

    assert fields["r_boundary(1)"] == pytest.approx(1.00)
    assert fields["z_boundary(2)"] == pytest.approx(0.20)
    assert fields["psi_boundary(3)"] == pytest.approx(0.98)


def test_keys_are_case_and_space_insensitive(tmp_path):
    path = tmp_path / "in_x"
    path.write_text("&in1\n ETA = 1.d-6\nR_Boundary( 1 ) = 2.0\n&end\n", encoding="utf-8")

    fields = effective_fields(path)

    assert "eta" in fields
    assert "r_boundary(1)" in fields


def test_duplicate_keys_resolve_last_wins(tmp_path):
    """Fortran namelist semantics, and the basis of the golden comparison."""
    path = tmp_path / "in_dup"
    path.write_text("&in1\n eta = 1.d-6\n eta = 5.d-3\n&end\n", encoding="utf-8")

    assert effective_fields(path)["eta"] == pytest.approx(5e-3)


def test_stacked_boundary_blocks_compare_equal_to_the_last_one(tmp_path):
    """The regression that makes the in_eq cleanup provably safe.

    The shipped template carries seven stacked n_boundary blocks. Retaining
    only the last must produce an identical effective mapping.
    """
    stacked = tmp_path / "in_stacked"
    single = tmp_path / "in_single"

    old_block = "\n".join(format_boundary_block([9.0], [9.0], [0.1], ".2f"))
    new_block = "\n".join(format_boundary_block([1.0], [2.0], [0.98], ".2f"))
    header = "&in1\n eta = 1.d-6\n"

    stacked.write_text(
        header + old_block + "\n" + old_block + "\n" + new_block + "\n&end\n",
        encoding="utf-8",
    )
    single.write_text(header + new_block + "\n&end\n", encoding="utf-8")

    assert effective_fields(stacked) == effective_fields(single)


def test_read_field_casts_and_reports_absence(namelist):
    assert read_field(namelist, "nstep_n") == pytest.approx(1200)
    assert read_field(namelist, "nstep_n", cast=int) == 1200

    with pytest.raises(NamelistError, match="nope"):
        read_field(namelist, "nope")


# --- set_fields --------------------------------------------------------------


def test_edit_replaces_value_and_preserves_comment(namelist):
    set_fields(namelist, {"eta": 5e-3})

    text = namelist.read_text(encoding="utf-8")
    assert "! resistivity" in text
    assert effective_fields(namelist)["eta"] == pytest.approx(5e-3)


def test_edit_does_not_disturb_neighbouring_fields(namelist):
    before = effective_fields(namelist)

    set_fields(namelist, {"eta": 5e-3})

    after = effective_fields(namelist)
    assert after["eta_num"] == before["eta_num"]
    assert after["r_boundary(2)"] == before["r_boundary(2)"]


def test_several_fields_and_files_at_once(tmp_path):
    first = tmp_path / "in_main"
    second = tmp_path / "in_main_r"
    first.write_text(SAMPLE, encoding="utf-8")
    second.write_text(SAMPLE, encoding="utf-8")

    set_fields([first, second], {"eta": 5e-3, "nstep_n": 3000})

    for path in (first, second):
        fields = effective_fields(path)
        assert fields["eta"] == pytest.approx(5e-3)
        assert fields["nstep_n"] == pytest.approx(3000)


def test_missing_field_is_an_error_by_default(namelist):
    with pytest.raises(NamelistError, match="Dre_par"):
        set_fields(namelist, {"Dre_par": 1e-6})


def test_create_missing_inserts_before_the_boundary_block(namelist):
    set_fields(namelist, {"Dre_par": 1e-6}, create_missing=True)

    lines = namelist.read_text(encoding="utf-8").splitlines()
    inserted = next(i for i, line in enumerate(lines) if "Dre_par" in line)
    boundary = next(i for i, line in enumerate(lines) if "n_boundary" in line)

    assert inserted < boundary
    assert effective_fields(namelist)["dre_par"] == pytest.approx(1e-6)


def test_create_missing_falls_back_to_before_end(tmp_path):
    path = tmp_path / "in_nobnd"
    path.write_text("&in1\n eta = 1.d-6\n&end\n", encoding="utf-8")

    set_fields(path, {"Dre_par": 1e-6}, create_missing=True)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert "Dre_par" in lines[-2]
    assert lines[-1].strip().lower() == "&end"


def test_duplicate_field_refuses_the_edit(tmp_path):
    """The setinput.sh guard: never guess which occurrence was meant."""
    path = tmp_path / "in_dup"
    path.write_text("&in1\n eta = 1.d-6\n eta = 2.d-6\n&end\n", encoding="utf-8")

    with pytest.raises(NamelistError, match="appears 2 times"):
        set_fields(path, {"eta": 5e-3})

    # The file is left untouched.
    assert path.read_text(encoding="utf-8").count("eta") == 2


def test_edit_leaves_file_otherwise_byte_identical(namelist):
    before = namelist.read_text(encoding="utf-8").splitlines()

    set_fields(namelist, {"eta": 1e-6})

    after = namelist.read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1


# --- boundary blocks ---------------------------------------------------------


def test_format_boundary_block_shape():
    block = format_boundary_block([1.0, 2.0], [3.0, 4.0], [0.9, 0.9], ".2f")

    assert block[0].strip() == "n_boundary = 2"
    assert len(block) == 3
    assert "R_boundary(  1) = 1.00" in block[1]


def test_mismatched_lengths_are_refused():
    with pytest.raises(NamelistError, match="same length"):
        format_boundary_block([1.0, 2.0], [3.0], [0.9], ".2f")


def test_set_boundary_block_replaces_rather_than_appends(namelist):
    """Regression for the seven stacked blocks in template/copy/in_eq."""
    set_boundary_block(namelist, [5.0, 6.0], [7.0, 8.0], [0.5, 0.5], ".2f")

    text = namelist.read_text(encoding="utf-8")
    assert text.count("n_boundary") == 1

    fields = effective_fields(namelist)
    assert fields["n_boundary"] == pytest.approx(2)
    assert fields["r_boundary(1)"] == pytest.approx(5.0)
    assert "r_boundary(3)" not in fields  # the old third point is gone


def test_repeated_writes_do_not_accumulate(namelist):
    for _ in range(5):
        set_boundary_block(namelist, [1.0], [2.0], [0.9], ".2f")

    assert namelist.read_text(encoding="utf-8").count("n_boundary") == 1


def test_set_boundary_block_preserves_other_fields(namelist):
    set_boundary_block(namelist, [5.0], [7.0], [0.5], ".2f")

    fields = effective_fields(namelist)
    assert fields["eta"] == pytest.approx(1e-6)
    assert fields["ffprime_file"] == "ffprime_prof.dat"


def test_set_boundary_block_needs_an_end_marker(tmp_path):
    path = tmp_path / "in_broken"
    path.write_text("&in1\n eta = 1.d-6\n", encoding="utf-8")

    with pytest.raises(NamelistError, match="&end"):
        set_boundary_block(path, [1.0], [2.0], [0.9])


def test_write_boundary_file_has_no_leading_space(tmp_path):
    path = tmp_path / "in_bnd"

    write_boundary_file(path, [1.0], [2.0], [0.9])

    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("n_boundary")


def test_standalone_and_namelist_blocks_agree(tmp_path, namelist):
    """One implementation, so the two destinations cannot drift apart."""
    standalone = tmp_path / "in_bnd"
    R, Z, psi = [1.0, 2.0], [3.0, 4.0], [0.9, 0.9]

    write_boundary_file(standalone, R, Z, psi, ".2f")
    set_boundary_block(namelist, R, Z, psi, ".2f")

    from_file = [
        line for line in standalone.read_text(encoding="utf-8").splitlines()
        if line.startswith("R_boundary")
    ]
    from_namelist = [
        line for line in namelist.read_text(encoding="utf-8").splitlines()
        if line.startswith("R_boundary")
    ]
    assert from_file == from_namelist
