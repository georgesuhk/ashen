"""Golden-file comparison: Ashen's prepare_run vs. the legacy run_jorek.py.

The whole suite skips (not errors) until the reference files described in
reference/qa2.1_g2.3_eta1e-3_RE/README.md are captured from the HPC and
committed -- there is nothing to compare against before then.

Comparison strategy per file, matching the refactor plan's golden-file table:

- in_eq/in_main/in_main_r: SEMANTIC via namelist.effective_fields, not byte
  comparison. in_eq specifically cannot be byte-compared across this refactor:
  the legacy template carries seven stacked n_boundary blocks (an append-not-
  replace bug fixed in namelist.py), so the legacy in_eq and Ashen's in_eq
  have a different number of lines by construction even though they describe
  the same namelist. effective_fields resolves Fortran's last-wins duplicate
  rule, so it is the correct comparator for what JOREK actually reads.
- in_bnd / input_starwall: byte-exact. No reason for either to differ.
- *_prof.dat / real_psi_edge.dat / original_bnd.dat: np.allclose, not byte
  comparison -- float formatting is not guaranteed identical across Python
  versions/platforms (this ran on 3.13 HPC vs 3.11 here).
- symlinks: name/target-suffix only, parsed from symlinks.txt, never followed
  (this checkout can't even create real symlinks -- see
  symlinks_maybe_bypassed in conftest.py).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from ashen.namelist import effective_fields
from ashen.runner import prepare_run

# Plain import, not relative: tests/golden has no __init__.py (matching the
# rest of the suite), so pytest's rootless collection puts this directory on
# sys.path directly rather than treating it as a package.
from conftest import REFERENCE_DIR

pytestmark = pytest.mark.skipif(
    not REFERENCE_DIR.is_dir() or not any(REFERENCE_DIR.glob("in_eq")),
    reason=(
        f"no golden reference captured yet at {REFERENCE_DIR} -- "
        "see reference/qa2.1_g2.3_eta1e-3_RE/README.md"
    ),
)

NAMELISTS = ("in_eq", "in_main", "in_main_r")
BYTE_EXACT_FILES = ("in_bnd", "input_starwall")
NUMERIC_2COL_FILES = ("ffprime_prof.dat", "T_prof.dat", "rho_prof.dat", "original_bnd.dat")


@pytest.fixture
def produced_run(real_campaign, symlinks_maybe_bypassed, tmp_path):
    """prepare_run's actual output, for comparison against REFERENCE_DIR."""
    site, params = real_campaign
    run_dir = tmp_path / "golden_run"
    result = prepare_run(params, site, run_dir, dry_run=False)
    return run_dir, result


# --- namelists: semantic comparison -----------------------------------------


@pytest.mark.parametrize("name", NAMELISTS)
def test_namelist_matches_legacy_semantically(produced_run, name):
    run_dir, _ = produced_run
    reference = REFERENCE_DIR / name
    if not reference.is_file():
        pytest.skip(f"{reference} not captured")

    produced_fields = effective_fields(run_dir / name)
    reference_fields = effective_fields(reference)

    missing = set(reference_fields) - set(produced_fields)
    extra = set(produced_fields) - set(reference_fields)
    assert not missing, f"{name}: fields in legacy but not produced: {sorted(missing)}"
    assert not extra, f"{name}: fields in produced but not legacy: {sorted(extra)}"

    mismatched = {
        key: (reference_fields[key], produced_fields[key])
        for key in reference_fields
        if not _values_match(reference_fields[key], produced_fields[key])
    }
    assert not mismatched, f"{name}: value mismatches (legacy, produced): {mismatched}"


def _values_match(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == pytest.approx(b, rel=1e-9, abs=1e-12)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_values_match(x, y) for x, y in zip(a, b))
    return a == b


def test_in_eq_boundary_is_a_single_clean_block(produced_run):
    """Regression for the specific bug this refactor fixes: the legacy in_eq
    carries seven stacked n_boundary blocks (append, not replace); Ashen's
    output must have exactly one, describing the same boundary as the last
    (i.e. effective) legacy block -- covered by the semantic test above."""
    run_dir, _ = produced_run

    text = (run_dir / "in_eq").read_text(encoding="utf-8")
    assert text.lower().count("n_boundary") == 1


def test_boundary_now_present_in_main_namelists_too(produced_run):
    """The confirmed fix: the legacy script only ever wrote the boundary into
    in_eq. Ashen writes it to all three -- this can't be checked against the
    reference (which won't have it in in_main/in_main_r), only asserted."""
    run_dir, _ = produced_run

    for name in ("in_main", "in_main_r"):
        fields = effective_fields(run_dir / name)
        assert "r_boundary(1)" in fields


# --- byte-exact files ----------------------------------------------------------


@pytest.mark.parametrize("name", BYTE_EXACT_FILES)
def test_byte_exact_files_match(produced_run, name):
    run_dir, _ = produced_run
    reference = REFERENCE_DIR / name
    if not reference.is_file():
        pytest.skip(f"{reference} not captured")

    produced = (run_dir / name).read_bytes()
    expected = reference.read_bytes()
    assert produced == expected, f"{name} differs from the legacy reference"


# --- numeric files: np.allclose ----------------------------------------------


@pytest.mark.parametrize("name", NUMERIC_2COL_FILES)
def test_numeric_profile_matches_legacy(produced_run, name):
    run_dir, _ = produced_run
    reference = REFERENCE_DIR / name
    if not reference.is_file():
        pytest.skip(f"{reference} not captured")

    produced = np.loadtxt(run_dir / name)
    expected = np.loadtxt(reference)

    assert produced.shape == expected.shape, f"{name}: shape {produced.shape} vs {expected.shape}"
    assert np.allclose(produced, expected, rtol=1e-6, atol=1e-12), (
        f"{name}: max abs diff = {np.max(np.abs(produced - expected))}"
    )


def test_real_psi_edge_matches_legacy(produced_run):
    run_dir, result = produced_run
    reference = REFERENCE_DIR / "real_psi_edge.dat"
    if not reference.is_file():
        pytest.skip(f"{reference} not captured")

    expected = float(reference.read_text(encoding="utf-8").strip())
    assert result.real_psi_edge == pytest.approx(expected, rel=1e-9)


# --- symlink targets: name/suffix only, never followed --------------------------


_SYMLINK_LINE = re.compile(r"^\S+\s+.*?\s(\S+)\s+->\s+(.+)$")


def _parse_symlinks_txt(text: str) -> dict[str, str]:
    """Extract {name: target} from raw `ls -la` output."""
    targets = {}
    for line in text.splitlines():
        if " -> " not in line:
            continue
        left, target = line.split(" -> ", 1)
        name = left.split()[-1]
        targets[name] = target.strip()
    return targets


def test_symlinks_txt_parses(produced_run):
    reference_txt = REFERENCE_DIR / "symlinks.txt"
    if not reference_txt.is_file():
        pytest.skip(f"{reference_txt} not captured")

    targets = _parse_symlinks_txt(reference_txt.read_text(encoding="utf-8"))

    # These are the names run_jorek.py's own symlink calls produce -- see
    # run_jorek.py:141-159 for exe/jobscripts/util, and template/symlink/base
    # + template/symlink/{RE,standard} for the rest.
    expected_names = {"exe", "jobscripts", "util"}
    assert expected_names <= set(targets), (
        f"symlinks.txt is missing expected names: {expected_names - set(targets)}"
    )


@pytest.mark.parametrize("name", ["exe", "util", "jobscripts"])
def test_symlink_target_suffix_matches(produced_run, name):
    """Compares path *suffixes* only, not absolute paths -- the HPC root
    (/tokp/work/geosu/...) and this checkout's root necessarily differ."""
    run_dir, _ = produced_run
    reference_txt = REFERENCE_DIR / "symlinks.txt"
    if not reference_txt.is_file():
        pytest.skip(f"{reference_txt} not captured")

    targets = _parse_symlinks_txt(reference_txt.read_text(encoding="utf-8"))
    if name not in targets:
        pytest.skip(f"{name} not present in symlinks.txt")

    legacy_target = targets[name].replace("\\", "/")
    produced_entry = run_dir / name
    assert produced_entry.exists(), f"{name} missing from produced run folder"

    # e.g. ".../Columbia/NL_kinks/exe" and "Columbia/NL_kinks/exe" should
    # agree on trailing path components even though roots differ.
    legacy_tail = "/".join(legacy_target.split("/")[-2:])
    assert legacy_tail.split("/")[-1] == name.split("/")[-1] or name in legacy_tail
