"""Deleting a run's generated figures (`util --func delete_figures`), to
give back disk space: `plot` draws them again from the gathered data.

Only figures go -- ``.png`` and ``.gif`` files, and only where `plot`
writes them:

- ``poinc_dir/`` (Poincare, connection length, theta_hist, wetted_fraction)
  and ``four_dir/`` (eigenfunctions, mode amplitudes), next to the caches
  there -- the caches (``.h5``, ``.txt``, ...) stay;
- ``profiles/`` (radial profiles and their animations);
- each trace folder: ``particles``, ``particle_exits*``, ``particle_wetted*``
  and ``particle_loss*`` figures -- the trace's outputs and
  ``particle_wetted*.json`` stay.

Comparison figures (the campaign's own ``figures/``) belong to no one run
and are not touched.
"""

from __future__ import annotations

from pathlib import Path

from ashen.paths import RunPaths
from ashen.ptracing import TRACE_DIR

__all__ = ["FIGURE_SUFFIXES", "figure_files"]

#: What counts as a figure.
FIGURE_SUFFIXES = (".png", ".gif")

#: The names `plot` gives the figures it writes into a trace folder.
_TRACE_FIGURES = ("particles", "particle_exits*", "particle_wetted*", "particle_loss*")


def figure_files(run_dir: Path | str) -> list[Path]:
    """Every figure `plot` may have written for the run, as it stands on
    disk: files only, never a link's target."""
    run_dir = Path(run_dir)
    # Only the folder properties are used, which do not depend on the width.
    paths = RunPaths(run_dir=run_dir, pad_width=5)
    found: list[Path] = []
    for folder in (paths.figures_dir, paths.four_dir, paths.profile_figures_dir):
        if folder.is_dir():
            found += [p for p in folder.iterdir()
                      if p.suffix in FIGURE_SUFFIXES and p.is_file() and not p.is_symlink()]
    traces = run_dir / TRACE_DIR
    if traces.is_dir():
        folders = [p for p in traces.glob("*") if p.is_dir()] + \
                  [p for p in traces.glob("*/*") if p.is_dir()]
        for folder in folders:
            for pattern in _TRACE_FIGURES:
                for suffix in FIGURE_SUFFIXES:
                    found += [p for p in folder.glob(pattern + suffix)
                              if p.is_file() and not p.is_symlink()]
    return sorted(set(found))
