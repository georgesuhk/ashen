"""The viewer notebook that lives in each run folder.

``run_jorek`` puts ``case_viewer.ipynb`` into a run folder when it prepares
one, if none is there: an existing notebook is never overwritten, so its
saved figures and any edits survive ``run_jorek`` and the viewer's own
Regenerate button. The notebook is one cell that calls
ashen.viewer.case_viewer, after dropping any ashen already imported: a copy
does not go stale as the viewer changes, and running the cell again shows an
updated ashen without restarting the kernel. ``bin/case_viewer`` replaces
notebooks written in an older form.

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
            "notebook sits in. Run the cell. Everything is drawn from files already in the\n"
            "folder. JOREK's tools run only when you press a gather button (\"Run analyse ...\",\n"
            "\"Gather ...\"); those run in this notebook until they finish.\n\n"
            "The views are ashen's `viewer.case_viewer`, loaded afresh each time the cell runs:\n"
            "after updating ashen (`git pull`), run the cell again. This file does not change.\n"
            "Needs `ipywidgets`, `matplotlib` and `h5py`\n"
            "(`pip install --user ipywidgets ipykernel h5py`). ashen's README, \"Looking at a case\n"
            "from a notebook\", describes each view."
        ),
        _code(
            "import sys\n"
            "from pathlib import Path\n\n"
            "for name in [m for m in sys.modules if m == \"ashen\" or m.startswith(\"ashen.\")]:\n"
            "    del sys.modules[name]   # so this cell picks up an updated ashen without a restart\n"
            "try:\n"
            "    import ashen\n"
            "except ImportError:   # not on this kernel's path: use the checkout that wrote this notebook\n"
            f"    sys.path.insert(0, {src!r})\n\n"
            "from ashen import viewer\n\n"
            "viewer.case_viewer(Path(\".\").resolve(), step=0)   # this notebook's folder"
        ),
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
