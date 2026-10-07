"""The viewer notebook that lives in each run folder.

``run_jorek`` puts ``case_viewer.ipynb`` into a run folder when it prepares
one, if none is there: an existing notebook is never overwritten, so its
saved figures and any edits survive ``run_jorek`` and the viewer's own
Regenerate button. The notebook is a few lines per view; the logic is in
ashen.viewer, so a copy does not go stale as the viewer changes.

Written as plain JSON here rather than copied from a file, so it does not
depend on where ashen is installed or on nbformat.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["NOTEBOOK_NAME", "case_notebook", "default_ashen_src", "write_case_notebook"]

NOTEBOOK_NAME = "case_viewer.ipynb"


def default_ashen_src() -> str:
    """The ``src/`` this ashen was imported from."""
    return str(Path(__file__).resolve().parents[1])


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def case_notebook(ashen_src: str | None = None) -> dict:
    """The notebook as a dict. ``ashen_src`` is where the first cell looks for
    ashen when it is not already importable (a VS Code kernel does not
    always see the shell's PYTHONPATH); default: this checkout's ``src/``."""
    src = default_ashen_src() if ashen_src is None else ashen_src
    cells = [
        _markdown(
            "# Case viewer\n\n"
            "Profiles, boundaries, the equilibrium and diagnostics of the run folder this\n"
            "notebook sits in. Run the cells. Everything is drawn from files already in the\n"
            "folder. JOREK's tools run only when you press a gather button (\"Run analyse ...\",\n"
            "\"Gather ...\"); those run in this notebook until they finish.\n\n"
            "`run_jorek` put this notebook here and will not overwrite it. Needs `ipywidgets`,\n"
            "`matplotlib` and `h5py` (`pip install --user ipywidgets ipykernel h5py`). ashen's\n"
            "README, \"Looking at a case from a notebook\", describes each view."
        ),
        _code(
            "import sys\n"
            "from pathlib import Path\n\n"
            "try:\n"
            "    import ashen\n"
            "except ImportError:   # not on this kernel's path: use the checkout that wrote this notebook\n"
            f"    sys.path.insert(0, {src!r})\n\n"
            "from ashen import viewer\n\n"
            "RUN = Path(\".\").resolve()   # this notebook's folder\n"
            "print(RUN)"
        ),
        _markdown(
            "## Current profile from q0, l_i, qa\n\n"
            "**Save to shotfile** writes the three values and `ffprime_method = \"q_li\"` into\n"
            "`shotfile.py`; **Regenerate inputs** then rewrites this folder's input files from the\n"
            "shotfile. A yellow band says when the two are out of step. `step` is the restart whose\n"
            "q-profile is shown as JOREK's result."
        ),
        _code("viewer.profile_tuner(RUN, step=0)"),
        _markdown("## Boundaries"),
        _code("viewer.boundary_view(RUN)"),
        _markdown(
            "## Equilibrium\n\n"
            "Contours are drawn on the grid nodes: good for looking, not for measuring."
        ),
        _code("viewer.equilibrium_view(RUN)"),
        _markdown("## Fourier modes (`analyse --diag four`)"),
        _code("viewer.four_view(RUN)"),
        _markdown("## Radial profiles (`analyse --diag profiles`)"),
        _code("viewer.profiles_view(RUN)"),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_case_notebook(
    folder: Path | str, *, ashen_src: str | None = None, overwrite: bool = False
) -> Path | None:
    """Write ``case_viewer.ipynb`` into ``folder``. Returns its path, or None
    if one is already there and ``overwrite`` is off."""
    path = Path(folder) / NOTEBOOK_NAME
    if path.exists() and not overwrite:
        return None
    path.write_text(json.dumps(case_notebook(ashen_src), indent=1) + "\n", encoding="utf-8")
    return path
