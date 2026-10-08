"""Write or replace the viewer notebook of run folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ashen.case_notebook import NOTEBOOK_NAME, case_notebook, write_case_notebook

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="case_viewer",
        description=(
            f"Write {NOTEBOOK_NAME} into run folders, replacing one whose cells are not "
            "the current ones (its saved figures and any edits are lost)."
        ),
    )
    parser.add_argument("folders", nargs="*", type=Path, default=[Path(".")],
                        help="run folders (default: the current one)")
    parser.add_argument("--existing", action="store_true",
                        help="only folders that already have the notebook, e.g. with */*/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    current = case_notebook()
    failed = False
    for folder in args.folders:
        path = folder / NOTEBOOK_NAME
        if not folder.is_dir():
            print(f"not a folder: {folder}")
            failed = True
        elif not path.exists():
            if not args.existing:
                write_case_notebook(folder)
                print(f"written:  {path}")
        elif _same(path, current):
            print(f"current:  {path}")
        else:
            write_case_notebook(folder, overwrite=True)
            print(f"replaced: {path}")
    return 1 if failed else 0


def _same(path: Path, notebook: dict) -> bool:
    """Whether the file has this notebook's cells (saved outputs aside)."""
    def cells(nb):
        return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]

    try:
        return cells(json.loads(path.read_text(encoding="utf-8"))) == cells(notebook)
    except (OSError, ValueError, KeyError, TypeError):
        return False
