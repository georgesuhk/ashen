"""HDF5 cache for ``jorek2_four`` output -- one file per step.

Unlike :mod:`ashen.diagnostics.poincare_cache`, this is written whole, not
incrementally: a Fourier decomposition of a single restart file isn't
resumable the way field-line tracing is, so there is nothing to "extend".
``--force`` is the only way to redo a step that already has a cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "FourCacheError", "FourRecord", "SCHEMA_VERSION",
    "write_cache", "read_cache", "count_records",
]

SCHEMA_VERSION = 1


class FourCacheError(RuntimeError):
    """A cache file is malformed, or the wrong schema version."""


def _h5py():
    """Import h5py lazily, with an actionable message.

    Kept out of module import so the rest of ashen stays importable on a
    machine without h5py -- CLAUDE.md's rule that importability must not
    depend on ``pip install``.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise FourCacheError(
            "the jorek2_four cache needs h5py, which is not importable here. "
            "It is present in the HPC environment; on a dev clone install it "
            "with `python -m pip install h5py`."
        ) from exc
    return h5py


@dataclass(frozen=True)
class FourRecord:
    """One ``(variable, toroidal mode n, poloidal mode m)`` Fourier component,
    over the run's radial (``psi_n``) grid."""

    variable: str
    n: int
    m: int
    psi_n: np.ndarray
    real: np.ndarray
    imag: np.ndarray

    @property
    def abs(self) -> np.ndarray:
        return np.hypot(self.real, self.imag)

    @property
    def phase(self) -> np.ndarray:
        return np.arctan2(self.imag, self.real)

    @property
    def group_name(self) -> str:
        """Readable and stable -- not a hash, so ``h5ls -r`` on a cache tells
        you what is in it."""
        return f"{self.variable}/n{self.n:03d}/m{self.m:03d}"


def write_cache(
    path: Path | str, *, step: int, pad_width: int, records: list[FourRecord]
) -> None:
    """Write a step's whole cache in one go.

    Written to a temporary file and renamed into place, so a process killed
    mid-write never leaves a truncated file that the next run's
    cache-existence check would mistake for a finished one.
    """
    h5py = _h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["schema"] = SCHEMA_VERSION
        f.attrs["step"] = int(step)
        f.attrs["pad_width"] = int(pad_width)
        f.attrs["n_records"] = len(records)
        lines = f.create_group("records")
        for record in records:
            group = lines.create_group(record.group_name)
            group.create_dataset("psi_n", data=np.asarray(record.psi_n, dtype=np.float32))
            group.create_dataset("real", data=np.asarray(record.real, dtype=np.float32))
            group.create_dataset("imag", data=np.asarray(record.imag, dtype=np.float32))
    tmp.replace(path)


def read_cache(path: Path | str) -> dict[tuple[str, int, int], FourRecord]:
    """Load every record from a cache file. Returns ``{}`` if it doesn't exist."""
    path = Path(path)
    if not path.is_file():
        return {}

    h5py = _h5py()
    records: dict[tuple[str, int, int], FourRecord] = {}
    with h5py.File(path, "r") as f:
        schema = int(f.attrs.get("schema", -1))
        if schema != SCHEMA_VERSION:
            raise FourCacheError(f"{path}: schema {schema}, expected {SCHEMA_VERSION}")
        for variable, var_group in f["records"].items():
            for n_name, n_group in var_group.items():
                n = int(n_name[1:])
                for m_name, m_group in n_group.items():
                    m = int(m_name[1:])
                    records[(variable, n, m)] = FourRecord(
                        variable=variable,
                        n=n,
                        m=m,
                        psi_n=np.asarray(m_group["psi_n"][:]),
                        real=np.asarray(m_group["real"][:]),
                        imag=np.asarray(m_group["imag"][:]),
                    )
    return records


def count_records(path: Path | str) -> int:
    """The number of cached records, without opening every group -- what
    ``write_cache`` stores its record count for."""
    path = Path(path)
    if not path.is_file():
        return 0
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        schema = int(f.attrs.get("schema", -1))
        if schema != SCHEMA_VERSION:
            raise FourCacheError(f"{path}: schema {schema}, expected {SCHEMA_VERSION}")
        return int(f.attrs.get("n_records", 0))
