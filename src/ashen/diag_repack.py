"""Repacking a particle diagnostics file to the size of its data.

JOREK's write_particle_diagnostics (particles/diagnostics/
mod_particle_diagnostics.f90) creates every dataset chunked 50000
particles x 1 time, uncompressed. HDF5 stores an uncompressed chunk at
full size however much of it the dataset uses, so each diagnostics row
takes the room of 50000 particles: with 1000 markers the file is ~50x
its data -- 2.8 MB a row for ptrace_gc's eleven variables, ~34 GB for
12000 rows.

:func:`repack` rewrites such a file with chunks that fit its data, gzip
(with byte shuffle) on top: lossless -- every value is compared with the
original before the original is replaced -- and read exactly as before by
h5py, so by every plot. The datasets stay extensible, as JOREK made them.
A file that is already compressed, or would not shrink, is left as it is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["RepackResult", "data_size", "diag_files", "is_compressed", "repack"]

#: About how big a chunk of the repacked file is [bytes].
_CHUNK_BYTES = 1 << 20
#: How many bytes of a dataset are copied (and compared) at a time.
_BLOCK_BYTES = 64 << 20
_GZIP_LEVEL = 4


@dataclass(frozen=True)
class RepackResult:
    path: Path
    before: int
    after: int
    #: Why it was left as it is, if it was.
    skipped: str | None = None


def _h5py():
    import h5py

    return h5py


def _datasets(group, prefix=""):
    """(name, dataset) of every dataset under group, depth first."""
    h5py = _h5py()
    for name, item in group.items():
        path = f"{prefix}/{name}"
        if isinstance(item, h5py.Dataset):
            yield path, item
        else:
            yield from _datasets(item, path)


def data_size(path: Path | str) -> int:
    """The bytes of data in the file: what it would take with chunks that
    fit, before compression -- an upper bound on its repacked size, give or
    take HDF5's own bookkeeping."""
    with _h5py().File(path, "r") as f:
        return sum(d.size * d.dtype.itemsize for _, d in _datasets(f))


def is_compressed(path: Path | str) -> bool:
    """Whether every dataset of the file is gzip-compressed already."""
    with _h5py().File(path, "r") as f:
        found = [d.compression for _, d in _datasets(f)]
    return bool(found) and all(c == "gzip" for c in found)


def _chunks(shape, itemsize):
    """Chunks of about _CHUNK_BYTES, whole rows of the trailing axes."""
    if not shape or 0 in shape:
        return None
    row = int(np.prod(shape[1:], dtype=np.int64)) * itemsize
    return (max(1, min(shape[0], _CHUNK_BYTES // max(row, 1))), *shape[1:])


def _blocks(dataset):
    """Slices along the first axis, each about _BLOCK_BYTES."""
    if dataset.shape == ():
        yield ()
        return
    n = dataset.shape[0]
    row = int(np.prod(dataset.shape[1:], dtype=np.int64)) * dataset.dtype.itemsize
    step = max(1, _BLOCK_BYTES // max(row, 1))
    for start in range(0, n, step):
        yield slice(start, min(start + step, n))


def _copy(source, target):
    h5py = _h5py()
    target.attrs.update(dict(source.attrs))
    for name, item in source.items():
        if isinstance(item, h5py.Group):
            _copy(item, target.create_group(name))
            continue
        chunks = _chunks(item.shape, item.dtype.itemsize)
        kwargs = {}
        if chunks is not None:
            kwargs = dict(chunks=chunks, compression="gzip", compression_opts=_GZIP_LEVEL,
                          shuffle=True, maxshape=(None,) * len(item.shape))
        copy = target.create_dataset(name, shape=item.shape, dtype=item.dtype, **kwargs)
        copy.attrs.update(dict(item.attrs))
        if item.size:
            for block in _blocks(item):
                copy[block] = item[block]


def _same(a_path: Path, b_path: Path) -> str | None:
    """None if every dataset and attribute of the two files is identical;
    otherwise what differs."""
    h5py = _h5py()
    with h5py.File(a_path, "r") as a, h5py.File(b_path, "r") as b:
        a_sets, b_sets = dict(_datasets(a)), dict(_datasets(b))
        if a_sets.keys() != b_sets.keys():
            return f"datasets differ: {sorted(a_sets.keys() ^ b_sets.keys())}"
        for name, da in a_sets.items():
            db = b_sets[name]
            if da.shape != db.shape or da.dtype != db.dtype:
                return f"{name}: {da.shape} {da.dtype} became {db.shape} {db.dtype}"
            for block in _blocks(da):
                if not np.array_equal(da[block], db[block], equal_nan=da.dtype.kind == "f"):
                    return f"{name}: values differ"
            if dict(da.attrs).keys() != dict(db.attrs).keys():
                return f"{name}: attributes differ"
    return None


def repack(path: Path | str) -> RepackResult:
    """Rewrite the file compact, in place, if that shrinks it. The new file
    is written beside it, compared value by value, and only then put in
    its place; on any failure the original stays and the copy goes."""
    path = Path(path)
    before = path.stat().st_size
    if not _h5py().is_hdf5(path):
        return RepackResult(path, before, before, skipped="not an HDF5 file")
    if is_compressed(path):
        return RepackResult(path, before, before, skipped="already compressed")
    temp = path.with_name(f".{path.name}.repack")
    try:
        h5py = _h5py()
        with h5py.File(path, "r") as source, h5py.File(temp, "w") as target:
            _copy(source, target)
        difference = _same(path, temp)
        if difference is not None:
            raise ValueError(f"the repacked copy is not identical ({difference})")
        after = temp.stat().st_size
        if after >= before:
            return RepackResult(path, before, before, skipped="would not shrink")
        os.replace(temp, path)
        return RepackResult(path, before, after)
    finally:
        temp.unlink(missing_ok=True)


#: The particle diagnostics files JOREK's writer makes, under every name a
#: trace folder has had (trace_diag.h5: before "trace" became "ptrace").
_DIAG_NAMES = ("ptrace_diag.h5", "part_diag.h5", "diag.h5", "trace_diag.h5")


def diag_files(run_dir: Path | str) -> list[Path]:
    """Every particle diagnostics file in the run's trace folders, in any
    layout: <run>/ptrace/<exe>/, its labelled folders, and <run>/trace/<exe>/."""
    run_dir = Path(run_dir)
    found = []
    for top in ("ptrace", "trace"):
        base = run_dir / top
        if not base.is_dir():
            continue
        folders = [p for p in base.glob("*") if p.is_dir()] + \
                  [p for p in base.glob("*/*") if p.is_dir()]
        for folder in folders:
            found += [folder / n for n in _DIAG_NAMES
                      if (folder / n).is_file() and not (folder / n).is_symlink()]
    return sorted(found)
