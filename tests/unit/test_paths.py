"""Tests for run-folder conventions and step padding.

The central property: whatever width a run uses, readers and writers must
agree. The old code mixed a sniffed width with a hardcoded default at
different call sites, so non-6-padded runs silently produced caches nobody
looked for.
"""

from __future__ import annotations

import pytest

from ashen.paths import (
    DEFAULT_PAD_WIDTH,
    JOREK_PAD_WIDTHS,
    PaddingError,
    RunPaths,
    detect_pad_width,
    step_name_variants,
    step_str,
)


def make_run(tmp_path, width: int, steps=(0, 100, 2000)):
    for step in steps:
        (tmp_path / f"jorek{step:0{width}d}.h5").write_bytes(b"")
    return tmp_path


# --- detection ---------------------------------------------------------------


@pytest.mark.parametrize("width", [4, 5, 6, 7])
def test_detects_each_width(tmp_path, width):
    make_run(tmp_path, width)

    assert detect_pad_width(tmp_path) == width


def test_majority_wins_over_a_stray_file(tmp_path):
    make_run(tmp_path, 5, steps=(0, 100, 2000, 3000))
    (tmp_path / "jorek0000000042.h5").write_bytes(b"")

    assert detect_pad_width(tmp_path) == 5


def test_underscored_restart_names_are_recognised(tmp_path):
    (tmp_path / "jorek_restart_00100.h5").write_bytes(b"")

    assert detect_pad_width(tmp_path) == 5


def test_empty_directory_is_an_explicit_error(tmp_path):
    with pytest.raises(PaddingError, match="jorek"):
        detect_pad_width(tmp_path)


def test_unrelated_h5_files_are_ignored(tmp_path):
    (tmp_path / "something00100.h5").write_bytes(b"")

    with pytest.raises(PaddingError):
        detect_pad_width(tmp_path)


# --- the headline property ---------------------------------------------------


@pytest.mark.parametrize("width", [4, 5, 6, 7])
def test_reader_and_writer_agree_on_every_width(tmp_path, width):
    """Regression for the poinc_diag read/write mismatch.

    Previously the .npz was written with width=6 and read with the sniffed
    width, so on a width-5 run the reader looked for a file that never existed.
    """
    run = make_run(tmp_path, width)
    paths = RunPaths.detect(run)

    writer_wrote = paths.poincare_cache(100)
    reader_expects = RunPaths.detect(run).poincare_cache(100)

    assert writer_wrote == reader_expects
    assert paths.pad_width == width


@pytest.mark.parametrize("width", [4, 5, 7])
def test_non_default_width_is_actually_used(tmp_path, width):
    """Guards against silently falling back to 6."""
    run = make_run(tmp_path, width)
    paths = RunPaths.detect(run)

    assert paths.step_str(100) == f"{100:0{width}d}"
    assert paths.step_str(100) != step_str(100, DEFAULT_PAD_WIDTH)


def test_all_artefacts_share_one_width(tmp_path):
    run = make_run(tmp_path, 5)
    paths = RunPaths.detect(run)

    stamped = [
        paths.restart(100).name,
        paths.zero_d(100).name,
        paths.poincare_cache(100).name,
        paths.flux_surface(0.5, 100).name,
        paths.profile_cache("Psi_N", "currdens", 100).name,
        paths.profile_cache("Psi_N", "currdens", 100, "average").name,
    ]
    assert all("00100" in name for name in stamped)
    assert not any("000100" in name for name in stamped)


# --- filenames ---------------------------------------------------------------


def test_step_str_accepts_floats_from_numpy_ranges():
    assert step_str(100.0, 6) == "000100"


def test_restart_filename(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)

    assert paths.restart(100).name == "jorek000100.h5"
    assert paths.restart(100, prefix="jorek2", ext=".rst").name == "jorek2000100.rst"


def test_live_restart_is_not_padded(tmp_path):
    """jorek2_* tools always read a fixed name, not a stepped one."""
    paths = RunPaths(tmp_path, pad_width=5)

    assert paths.live_restart.name == "jorek_restart.h5"


def test_postproc_artefact_names(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)

    assert paths.zero_d(10).name == "zeroD_quantities_s000010.dat"
    assert paths.flux_surface(0.5, 10).name == "fluxsurface_at_psi_0.500_s000010.dat"
    assert paths.poincare_cache(10).name == "poinc_s000010.h5"
    assert paths.poincare_cache_legacy(10, "psi_n").name == "poinc_t000010_psi_n.npz"


def test_zero_d_jorek_units_gets_a_distinct_name(tmp_path):
    """The two unit systems must not collide on disk -- JOREK itself always
    writes zeroD_quantities_s<step>.dat regardless of units mode, so ashen's
    own path convention is what tells them apart after the fact."""
    paths = RunPaths(tmp_path, pad_width=6)

    assert paths.zero_d(10, si_units=False).name == "zeroD_quantities_jorek_s000010.dat"
    assert paths.zero_d(10, si_units=True) != paths.zero_d(10, si_units=False)


def test_profile_cache_midplane_keeps_the_legacy_name(tmp_path):
    """midplane must keep exactly the legacy .npz name/path -- the one cache
    KNOWN_ISSUES.md #5 promises is unaffected by later changes, since legacy
    plot_postproc_profs still reads it directly."""
    paths = RunPaths(tmp_path, pad_width=6)
    cache = paths.profile_cache("Psi_N", "currdens", 100)
    assert cache.name == "Psi_N_currdens_000100.npz"
    assert cache.parent == tmp_path / "postproc"


def test_profile_cache_non_midplane_modes_get_a_suffix(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)
    average = paths.profile_cache("Psi_N", "currdens", 100, "average")
    outer = paths.profile_cache("Psi_N", "currdens", 100, "midplane outer")
    assert average.name == "Psi_N_currdens_average_000100.npz"
    assert outer.name == "Psi_N_currdens_midplane-outer_000100.npz"


def test_profile_cache_modes_do_not_collide(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)
    names = {
        paths.profile_cache("Psi_N", "currdens", 100, mode).name
        for mode in ("midplane", "midplane outer", "midplane inner", "average")
    }
    assert len(names) == 4


def test_artefacts_live_under_the_run_directory(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)

    assert paths.zero_d(10).parent == tmp_path / "postproc"
    assert paths.poincare_cache(10).parent == tmp_path / "poinc_dir"
    assert paths.in_eq.parent == tmp_path


def test_namelists_are_the_three_the_runner_edits(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)

    assert [p.name for p in paths.namelists] == ["in_eq", "in_main", "in_main_r"]


def test_flux_surface_psi_uses_three_decimals(tmp_path):
    """Matches the convention poinc_diag.py:100 writes and reads."""
    paths = RunPaths(tmp_path, pad_width=6)

    assert "psi_0.010_" in paths.flux_surface(0.01, 0).name
    assert "psi_0.950_" in paths.flux_surface(0.95, 0).name


# --- mixed step padding between JOREK builds -------------------------------
#
# A postproc binary names its outputs with rst_file_ind_fmt(1) alone, while
# its importer accepts either width -- so which spelling lands in a run
# folder is a property of the binary, not of the run, and one folder can end
# up holding both. See paths.JOREK_PAD_WIDTHS.


def test_step_name_variants_repads_only_the_step():
    """The psi value in a flux-surface name must survive untouched."""
    assert step_name_variants("fluxsurface_at_psi_0.200_s08002.dat", 8002) == [
        "fluxsurface_at_psi_0.200_s008002.dat"
    ]


def test_step_name_variants_excludes_the_spelling_given():
    """These are the alternatives to what the caller already tried."""
    variants = step_name_variants("zeroD_quantities_s008002.dat", 8002)
    assert "zeroD_quantities_s008002.dat" not in variants
    assert variants == ["zeroD_quantities_s08002.dat"]


def test_zero_d_finds_the_other_width(tmp_path):
    """pad_width 5 from the restarts, but the tool wrote a 6-wide name."""
    paths = RunPaths(tmp_path, pad_width=5)
    paths.postproc_dir.mkdir()
    written = paths.postproc_dir / "zeroD_quantities_s008002.dat"
    written.write_text("x", encoding="utf-8")
    assert paths.zero_d(8002) == written


def test_flux_surface_finds_the_other_width(tmp_path):
    paths = RunPaths(tmp_path, pad_width=5)
    paths.postproc_dir.mkdir()
    written = paths.postproc_dir / "fluxsurface_at_psi_0.200_s008002.dat"
    written.write_text("x", encoding="utf-8")
    assert paths.flux_surface(0.2, 8002) == written


def test_qprofile_finds_the_other_width(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)
    paths.postproc_dir.mkdir()
    written = paths.postproc_dir / "qprofile_s08002.dat"
    written.write_text("x", encoding="utf-8")
    assert paths.qprofile(8002) == written


def test_restart_finds_the_other_width(tmp_path):
    """A run continued under a different build holds both spellings; neither
    half of its steps should become unreachable."""
    paths = RunPaths(tmp_path, pad_width=6)
    written = tmp_path / "jorek08002.h5"
    written.write_bytes(b"x")
    assert paths.restart(8002) == written


def test_the_run_s_own_width_wins_when_both_exist(tmp_path):
    """Resolution is a fallback, not a search: an exact match is never
    displaced by a variant that also happens to be there."""
    paths = RunPaths(tmp_path, pad_width=5)
    paths.postproc_dir.mkdir()
    canonical = paths.postproc_dir / "zeroD_quantities_s08002.dat"
    canonical.write_text("mine", encoding="utf-8")
    (paths.postproc_dir / "zeroD_quantities_s008002.dat").write_text(
        "other", encoding="utf-8"
    )
    assert paths.zero_d(8002) == canonical


def test_missing_resolves_to_the_canonical_name(tmp_path):
    """With nothing on disk the canonical spelling comes back, so the path
    stays usable as a write target and "expected <path>" stays predictable."""
    paths = RunPaths(tmp_path, pad_width=5)
    assert paths.zero_d(8002).name == "zeroD_quantities_s08002.dat"


def test_caches_ashen_writes_are_not_width_tolerant(tmp_path):
    """The Poincare/profile/four caches are written and read only by ashen.
    A second accepted spelling there would be a way to end up with two
    caches for one step, not a way to find the one that exists."""
    paths = RunPaths(tmp_path, pad_width=5)
    paths.poinc_dir.mkdir()
    (paths.poinc_dir / "poinc_s008002.h5").write_bytes(b"x")
    assert paths.poincare_cache(8002).name == "poinc_s08002.h5"
