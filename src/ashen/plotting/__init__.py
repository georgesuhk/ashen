"""Figure styling and small drawing helpers shared by the plot modules.

Legacy code styled figures by mutating matplotlib.rcParams at IMPORT TIME,
in two modules with two different, only partially overlapping
dictionaries (data.py:5-13, data_jorek.py:13-23). Whichever module
imported last won on any shared key, so effective style depended on
import order, not reproducibility; importing either module for a pure
function (e.g. a connection-length calc) silently restyled every plot
made afterwards, including ones that never touch this package.

Here nothing is mutated at import time. style() is an explicit
rc_context -- apply it only around the plotting calls that want it:

    with style():
        fig, ax = plt.subplots()
        draw_poincare(ax, records, ...)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

__all__ = ["STYLE", "style"]

#: Union of both legacy blocks, with the one real conflict (data.py's
#: concrete font stack vs data_jorek.py's bare "serif") resolved in favour
#: of the more specific one. Import order can no longer matter -- only
#: ever applied through style().
STYLE: dict[str, object] = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "CMU Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
}


@contextmanager
def style(overrides: dict[str, object] | None = None) -> Iterator[None]:
    """Apply STYLE for the duration of the with block only.

    overrides layers on top for a single call site without mutating the
    shared dict -- e.g. a larger font.size for one production figure.

    rc_context only affects artists created (labels, lines, the axes
    frame) while the block is open, not ones added afterwards -- so every
    draw function in this package creates its figure AND every artist on
    it inside one `with style():` block, not opening/closing the context
    around only part of the drawing.
    """
    import matplotlib.pyplot as plt

    rc = {**STYLE, **(overrides or {})}
    with plt.rc_context(rc):
        yield
