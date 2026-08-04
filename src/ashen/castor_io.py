"""Reading CASTOR3D's two-column xmgrace-style profile output.

Shared by :mod:`ashen.boundary` and :mod:`ashen.profiles`, both of which read
files from a CASTOR3D cotrans/scan directory. Ports
``castor3d/util/basics.py:331 load_two_col_data``.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np

__all__ = ["load_two_col_data"]


def load_two_col_data(path: Path | str, data_set: int = 1) -> np.ndarray:
    """Parse an xmgrace-style two-column file into an (N, 2) array.

    Lines that do not split into exactly two whitespace-separated fields (blank
    lines, headers, stray comments) are dropped silently -- this is what lets
    it read both the plain `xn_*` cotrans files and `xm_plasma_0_*_n1`, which
    carries a leading ``#`` comment line.

    ``&&`` markers separate multiple datasets in one file, xmgrace-style;
    ``data_set`` selects which block (1-indexed).
    """
    filtered_lines: list[str] = []
    counter = 1
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "&&" in line:
                if counter == data_set:
                    break
                filtered_lines = []
                counter += 1
                continue
            if not line.strip():
                continue
            if len(line.split()) != 2:
                continue
            filtered_lines.append(line)

    return np.loadtxt(StringIO("".join(filtered_lines)))
