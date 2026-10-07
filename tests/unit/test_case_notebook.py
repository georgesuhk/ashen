"""ashen.case_notebook -- the viewer notebook each run folder gets."""

from __future__ import annotations

import json
from pathlib import Path

from ashen.case_notebook import NOTEBOOK_NAME, case_notebook, default_ashen_src, write_case_notebook
from ashen.runner import prepare_run

REPO = Path(__file__).resolve().parents[2]


def _code(nb) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_is_valid_and_location_independent():
    nb = case_notebook()
    assert nb["nbformat"] == 4 and all("source" in c for c in nb["cells"])
    code = _code(nb)
    assert 'RUN = Path(".").resolve()' in code
    assert repr(default_ashen_src()) in code            # fallback when ashen is not importable
    for view in ("profile_tuner", "boundary_view", "equilibrium_view", "four_view", "profiles_view"):
        assert f"viewer.{view}(RUN" in code
    compile(code, "case_viewer", "exec")                # every code cell parses


def test_default_ashen_src_is_this_checkout():
    assert Path(default_ashen_src()) == REPO / "src"
    assert (Path(default_ashen_src()) / "ashen" / "viewer.py").is_file()


def test_tracked_copy_matches_what_is_generated():
    """notebooks/case_viewer.ipynb is the same notebook, pointing at ../src."""
    tracked = json.loads((REPO / "notebooks" / NOTEBOOK_NAME).read_text())
    assert tracked == case_notebook(ashen_src="../src")


def test_write_never_overwrites_unless_told(tmp_path):
    path = write_case_notebook(tmp_path)
    assert path == tmp_path / NOTEBOOK_NAME and json.loads(path.read_text())["nbformat"] == 4

    path.write_text("mine")
    assert write_case_notebook(tmp_path) is None
    assert path.read_text() == "mine"
    assert write_case_notebook(tmp_path, overwrite=True) == path
    assert path.read_text() != "mine"


def test_prepare_run_adds_the_notebook_once(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    first = prepare_run(params, site, run_dir)
    notebook = run_dir / NOTEBOOK_NAME
    assert notebook.is_file()
    assert any(a.startswith("write ") and NOTEBOOK_NAME in a for a in first.actions)

    notebook.write_text("edited by the user")
    second = prepare_run(params, site, run_dir)
    assert notebook.read_text() == "edited by the user"
    assert any(a.startswith("keep ") and NOTEBOOK_NAME in a for a in second.actions)


def test_dry_run_writes_no_notebook(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)
    assert not (tmp_path / "rundir").exists()
    assert any(NOTEBOOK_NAME in a for a in result.actions)
