"""A cache of the wall hits behind `--diag particle_wetted`, so replotting
an unchanged trace -- other bins, colour ranges, counts, a psi_n range --
skips reading the diagnostics file and finding where each particle hit.

What is kept is everything particle_wetted takes from the diagnostics
file: each particle's first hit (ashen.diagnostics.particle_wetted.
ParticleHits, for every particle, so any selection of them can be made
from it), the diagnostics times and each particle's psi_n at the first of
them, and how many cut-off rows were left out. It is computed within the
start..end window, so its key is:

- the diagnostics file's name, size and modification time -- a trace that
  is still running, has been repacked or retraced, makes a new one;
- the wall's points, exactly;
- the window's times;
- SCHEMA_VERSION, raised whenever what is computed changes.

Anything else differing makes the cache stale, and it is recomputed and
overwritten. One .npz in the ptrace folder; written to a temporary name
and moved into place, so an interrupted write leaves no half a file.

Pure data: no matplotlib here.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ashen.diagnostics.particle_wetted import ParticleHits

__all__ = ["CACHE_FILE", "SCHEMA_VERSION", "WettedInputs", "cache_key", "read_cache",
           "write_cache"]

#: Raise when particle_hits (or what is stored) changes, so old caches are
#: recomputed rather than trusted.
SCHEMA_VERSION = 1

#: In the ptrace folder.
CACHE_FILE = "particle_wetted_cache.npz"

_ARRAYS = ("time", "start_psi_n", "considered", "hit", "left_grid", "l", "phi")


@dataclass(frozen=True)
class WettedInputs:
    """What particle_wetted needs from the diagnostics file, within the window."""

    #: Diagnostics times [s] within the window.
    time: np.ndarray
    #: Each particle's psi_n at the first of them (empty with no times).
    start_psi_n: np.ndarray
    #: Rows the trace was cut off while writing, left out (read_particle_diag).
    dropped_rows: int
    hits: ParticleHits

    @property
    def n(self) -> int:
        return self.hits.n


def cache_key(diag: Path, boundary: np.ndarray, window) -> dict:
    """The key a cache must have to be used for this diagnostics file, wall
    (its points) and window ((t_min, t_max or None), or None)."""
    stat = Path(diag).stat()
    points = np.ascontiguousarray(boundary, dtype=float)
    return {
        "schema": SCHEMA_VERSION,
        "diag_name": Path(diag).name, "diag_size": stat.st_size,
        "diag_mtime_ns": stat.st_mtime_ns,
        "wall_shape": list(points.shape),
        "wall_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "window": None if window is None else [float(window[0]),
                                               None if window[1] is None else float(window[1])],
    }


def read_cache(path: Path, key: dict) -> WettedInputs | None:
    """The cached inputs at path if its key is `key`; None if there is no
    cache, it is stale, or it can't be read."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["key"])) != key:
                return None
            arrays = {name: data[name] for name in _ARRAYS}
            dropped = int(data["dropped_rows"])
    except (OSError, KeyError, ValueError):
        return None
    return WettedInputs(
        time=arrays["time"], start_psi_n=arrays["start_psi_n"], dropped_rows=dropped,
        hits=ParticleHits(
            considered=arrays["considered"], hit=arrays["hit"], left_grid=arrays["left_grid"],
            l=arrays["l"], phi=arrays["phi"],
        ),
    )


def write_cache(path: Path, key: dict, inputs: WettedInputs) -> None:
    """Write inputs under key to path, replacing any cache there."""
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    hits = inputs.hits
    np.savez(
        tmp, key=np.array(json.dumps(key, sort_keys=True)), time=inputs.time,
        start_psi_n=inputs.start_psi_n, dropped_rows=np.array(inputs.dropped_rows),
        considered=hits.considered, hit=hits.hit, left_grid=hits.left_grid,
        l=hits.l, phi=hits.phi,
    )
    os.replace(tmp, path)
