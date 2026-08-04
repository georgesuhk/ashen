"""Fixtures for the golden-file comparison against the legacy run_jorek.py.

Unlike tests/unit, this suite deliberately uses the REAL shotfile, REAL
template tree, and REAL CASTOR3D fixture data -- already present in this
checkout, under Columbia/ and castor3d/ (outside ashen/ itself) -- rather than
synthetic ones. The whole point is comparing against a real reference output
captured by actually running the legacy script on the HPC (see
tests/golden/reference/README.md).
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

from ashen.config import Launch, Site
from ashen.shotfile import load_shotfile
from support import install_symlink_bypass, symlinks_supported

#: ashen/tests/golden/conftest.py -> parents[3] is the repo root
#  (geosu_TOKClone), one level above ashen/ itself.
REPO_ROOT = Path(__file__).resolve().parents[3]
SHOTFILE_PATH = (
    REPO_ROOT / "Columbia" / "NL_kinks" / "qa2.1_g2.3" / "eta1e-3_RE" / "shotfile.py"
)
TEMPLATE_DIR = REPO_ROOT / "Columbia" / "NL_kinks" / "template"
CASTOR_ROOT = REPO_ROOT / "castor3d"
REFERENCE_DIR = Path(__file__).resolve().parent / "reference" / "qa2.1_g2.3_eta1e-3_RE"


def _real_tree_available() -> bool:
    return SHOTFILE_PATH.is_file() and TEMPLATE_DIR.is_dir() and CASTOR_ROOT.is_dir()


#: A copy of the shotfile.py actually used for the HPC capture, if provided.
#: Preferred over the live shotfile: the live one is George's working copy
#: and will keep changing for reasons that have nothing to do with Ashen's
#: correctness (confirmed in practice -- tstep_n drifted between the first
#: capture and when this comparator was built). A golden test that re-reads
#: a moving target isn't golden.
FROZEN_SHOTFILE = REFERENCE_DIR / "shotfile.py"


@pytest.fixture
def real_campaign():
    """The shotfile actually used for the HPC capture (frozen copy preferred,
    live one as fallback) against the real template.

    castor_params.machine_folder is overridden from the shotfile's hardcoded
    HPC path (/tokp/work/geosu/castor3d/DIIID_low_pres) to this checkout's
    local copy of the same data -- the fixtures under castor3d/DIIID_low_pres/
    were copied from that exact HPC path, so this does not change what's
    being computed, only where it's read from.
    """
    if not _real_tree_available():
        pytest.skip(
            "real Columbia/castor3d tree not present in this checkout "
            f"(looked for {SHOTFILE_PATH}, {TEMPLATE_DIR}, {CASTOR_ROOT})"
        )

    shotfile_path = FROZEN_SHOTFILE if FROZEN_SHOTFILE.is_file() else SHOTFILE_PATH
    if shotfile_path is SHOTFILE_PATH:
        warnings.warn(
            f"No frozen shotfile at {FROZEN_SHOTFILE}; comparing against the "
            "live shotfile, which may have changed since the reference was "
            "captured. Copy the shotfile.py used for capture there to pin it.",
            stacklevel=2,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected: shot_folder/template_folder/etc
        params = load_shotfile(shotfile_path)

    local_castor_params = dict(params.castor_params)
    local_castor_params["machine_folder"] = str(CASTOR_ROOT / "DIIID_low_pres")
    params = dataclasses.replace(params, castor_params=local_castor_params)

    site = Site(
        source=REPO_ROOT / "site.toml",
        root=REPO_ROOT,
        paths={
            "exe": REPO_ROOT / "Columbia" / "NL_kinks" / "exe",
            "template": TEMPLATE_DIR,
            # jobscripts/jorek don't exist in this checkout (HPC-only paths);
            # install_symlink_bypass tolerates a missing source directory.
            "jobscripts": REPO_ROOT / "Columbia" / "jobscripts",
            "jorek": REPO_ROOT / "Columbia" / "jorek",
            "jorek_re": REPO_ROOT / "Columbia" / "jorek_RE",
            "castor_root": CASTOR_ROOT,
        },
        launch=Launch(),
    )

    return site, params


@pytest.fixture
def symlinks_maybe_bypassed(monkeypatch, tmp_path):
    """See tests/unit/conftest.py's fixture of the same name -- duplicated
    here (not imported) because tests/unit and tests/golden don't share a
    conftest.py, and this is three lines of glue around tests/support.py."""
    if symlinks_supported(tmp_path):
        return
    install_symlink_bypass(monkeypatch)
