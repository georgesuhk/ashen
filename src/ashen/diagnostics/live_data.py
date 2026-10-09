"""JOREK's live data: the quantities it writes to ``macroscopic_vars.dat``
every time step, as ``util/plot_live_data.sh`` plots them.

The file holds one line per quantity per step::

    @magnetic_energies: %"time"   "E_{mag,00}" "E_{mag,01}" "E_{mag,02}"
    @magnetic_energies_xlabel_si: time [ms]
    @magnetic_energies_x2si:   6.483638667E-05
    @magnetic_energies_logy: 1
    @magnetic_energies:  3.000000000E-02  5.456489831E-02  2.582679735E-45 ...

and grows to tens of MB over a run, so read_live_data picks out one
quantity's lines without parsing the rest (postproc.parse_macroscopic_vars
reads every quantity). The columns are per toroidal harmonic in the order
JOREK numbers them (00 is n = 0); the labels are the file's own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["LIVE_DATA_FILE", "LiveData", "read_live_data"]

LIVE_DATA_FILE = "macroscopic_vars.dat"


@dataclass(frozen=True)
class LiveData:
    quantity: str
    labels: list[str]     #: one per column of ``values``, as the file names them
    x: np.ndarray         #: in SI (xlabel says what) when the file gives a factor
    values: np.ndarray    #: shape (n_rows, n_columns), likewise
    xlabel: str
    ylabel: str
    logy: bool            #: whether plot_live_data draws it on a log axis


def _meta(data: bytes, quantity: str, key: str) -> str | None:
    match = re.search(rb"^@" + quantity.encode() + b"_" + key.encode() + rb":[ \t]*(.*)$",
                      data, re.M)
    return match.group(1).decode(errors="replace").strip() if match else None


def read_live_data(path: Path | str, quantity: str = "magnetic_energies") -> LiveData:
    """One quantity of ``macroscopic_vars.dat`` against time.

    Converted with the file's own ``_x2si`` / ``_y2si`` factors and labelled
    with its ``_xlabel_si`` / ``_ylabel_si`` (for the energies the y factor
    is 1: they stay normalised, as plot_live_data shows them).

    A run restarted from an earlier step writes those times again: where
    the time goes back, the rows it overwrites are dropped, so each time
    appears once and the later run's values stand.

    Raises FileNotFoundError if there is no file, ValueError if the file
    has no such quantity (naming those it has).
    """
    path = Path(path)
    data = path.read_bytes()
    name = re.escape(quantity.encode())
    rows = re.findall(rb"^@" + name + rb":[ \t]*(.*)$", data, re.M)
    if not rows:
        have = sorted({m.decode() for m in re.findall(rb"^@(\w+)_xlabel:", data, re.M)})
        raise ValueError(f"{path}: no {quantity!r}; it has {', '.join(have) or 'nothing'}")

    labels: list[str] = []
    numeric = []
    for row in rows:
        if row.lstrip().startswith(b"%"):
            labels = re.findall(r'"([^"]*)"', row.decode(errors="replace"))[1:]
            continue
        numeric.append(row.replace(b"D", b"E").replace(b"d", b"e"))
    if not numeric:
        raise ValueError(f"{path}: {quantity!r} has a header but no rows yet")
    table = np.array([np.array(row.split(), dtype=float) for row in numeric])

    # keep, for each time, the last run that wrote it
    time = table[:, 0]
    keep = np.ones(len(time), dtype=bool)
    lowest_later = np.inf
    for i in range(len(time) - 1, -1, -1):
        if time[i] >= lowest_later:
            keep[i] = False
        else:
            lowest_later = time[i]
    table = table[keep]

    def factor(key: str) -> float:
        try:
            return float((_meta(data, quantity, key) or "1").replace("D", "E"))
        except ValueError:
            return 1.0

    n_columns = table.shape[1] - 1
    if len(labels) != n_columns:
        labels = [f"column {i + 1}" for i in range(n_columns)]
    return LiveData(
        quantity=quantity,
        labels=labels,
        x=table[:, 0] * factor("x2si"),
        values=table[:, 1:] * factor("y2si"),
        xlabel=_meta(data, quantity, "xlabel_si") or "time",
        ylabel=_meta(data, quantity, "ylabel_si") or quantity,
        logy=(_meta(data, quantity, "logy") or "0").strip() == "1",
    )
