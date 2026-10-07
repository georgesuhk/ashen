"""Repacking a particle diagnostics file to the size of its data.

JOREK's write_particle_diagnostics (particles/diagnostics/
mod_particle_diagnostics.f90) creates every dataset chunked 50000
particles x 1 time, uncompressed. HDF5 stores an uncompressed chunk at
full size however much of it the dataset uses, so each diagnostics row
takes the room of 50000 particles: with 1000 markers the file is ~50x
its data -- 2.8 MB a row for ptrace_gc's eleven variables, ~34 GB for
12000 rows.

:func:`repack` rewrites such a file with chunks that fit its data, gzip
(with byte shuffle) on top: lossless, and read exactly as before by h5py,
so by every plot. The datasets stay extensible, as JOREK made them. A file
that is already compressed, or would not shrink, is left as it is.

**Checked before the original is replaced.** Every block of values is
hashed as it is read from the original, and the finished copy is read back
through HDF5 and hashed again: a copy whose hashes differ is thrown away.
So the original -- mostly padding, and HDF5 reads the padding too -- is
read once, not twice.

**On several cores.** HDF5 compresses on one. Here each chunk is shuffled
and deflated on a thread (zlib and the hashing release the GIL) and given
to HDF5 ready-made (write_direct_chunk): the same bytes its own shuffle
and gzip filters would store, in a standard file.
"""

from __future__ import annotations

import hashlib
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["RepackResult", "data_size", "diag_files", "is_compressed", "repack"]

#: About how big a chunk of the repacked file is [bytes].
_CHUNK_BYTES = 1 << 20
#: How many bytes of a dataset are copied (and compared) at a time.
_BLOCK_BYTES = 64 << 20
_GZIP_LEVEL = 4
#: At most this many threads compress and hash.
_MAX_WORKERS = 8
#: dtype kinds whose bytes are the values: bool, integers, floats, complex.
_PLAIN_KINDS = "biufc"


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


def _blocks(dataset, chunk_rows: int | None = None):
    """Slices along the first axis, each about _BLOCK_BYTES -- and a whole
    number of chunk_rows, so no chunk of the copy straddles two blocks."""
    if dataset.shape == ():
        yield ()
        return
    n = dataset.shape[0]
    row = int(np.prod(dataset.shape[1:], dtype=np.int64)) * dataset.dtype.itemsize
    step = max(1, _BLOCK_BYTES // max(row, 1))
    if chunk_rows:
        step = max(chunk_rows, step // chunk_rows * chunk_rows)
    for start in range(0, n, step):
        yield slice(start, min(start + step, n))


def _workers() -> int:
    """Threads to compress and hash with: the cores this process may use,
    at most _MAX_WORKERS (it runs on a login node)."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:          # not on Linux
        cores = os.cpu_count() or 1
    return max(1, min(_MAX_WORKERS, cores))


def _digest(data) -> bytes:
    """A hash of the block's bytes: equal only for identical values (and
    stricter than ==: it tells -0.0 from 0.0, and one NaN from another)."""
    array = np.ascontiguousarray(data)
    if array.dtype.kind in _PLAIN_KINDS:
        raw = array.reshape(-1).view(np.uint8)
    else:                           # strings, objects: no plain buffer
        raw = repr(array.tolist()).encode()
    return hashlib.blake2b(raw, digest_size=20).digest()


def _packed(rows: np.ndarray, chunk: tuple[int, ...]) -> bytes:
    """One chunk of the copy as HDF5 stores it under shuffle + gzip: the
    rows, padded with zeros to a whole chunk; their bytes shuffled (every
    element's first byte, then every second byte, ...); deflated."""
    if rows.shape[0] < chunk[0]:
        full = np.zeros(chunk, dtype=rows.dtype)
        full[:rows.shape[0]] = rows
        rows = full
    rows = np.ascontiguousarray(rows)
    size = rows.dtype.itemsize
    raw = rows.reshape(-1).view(np.uint8)
    if size > 1:
        raw = raw.reshape(-1, size).T
    return zlib.compress(raw.tobytes(), _GZIP_LEVEL)


@dataclass
class _Copied:
    """What was read from one dataset of the original as it was copied."""

    shape: tuple
    dtype: np.dtype
    attrs: frozenset
    chunk_rows: int | None
    #: _digest of each of its _blocks, in order.
    digests: list


def _copy(source, target, pool, copied: dict, prefix: str = "") -> None:
    """Copy source's groups and datasets into target, compact, noting in
    copied what each dataset held. The original is read once, block by
    block; each block is hashed and its chunks shuffled and deflated on the
    pool's threads, and the finished chunks are handed to HDF5 as they are
    (only this thread touches HDF5)."""
    h5py = _h5py()
    target.attrs.update(dict(source.attrs))
    for name, item in source.items():
        path = f"{prefix}/{name}"
        if isinstance(item, h5py.Group):
            _copy(item, target.create_group(name), pool, copied, path)
            continue
        chunks = _chunks(item.shape, item.dtype.itemsize)
        kwargs = {}
        if chunks is not None:
            kwargs = dict(chunks=chunks, compression="gzip", compression_opts=_GZIP_LEVEL,
                          shuffle=True, maxshape=(None,) * len(item.shape))
        copy = target.create_dataset(name, shape=item.shape, dtype=item.dtype, **kwargs)
        copy.attrs.update(dict(item.attrs))
        chunk_rows = chunks[0] if chunks is not None else None
        record = copied[path] = _Copied(
            shape=item.shape, dtype=item.dtype, attrs=frozenset(item.attrs),
            chunk_rows=chunk_rows, digests=[],
        )
        if not item.size:
            continue
        # Chunks made here need a plain numeric type; anything else goes
        # through HDF5's own filters.
        direct = chunks is not None and item.dtype.kind in _PLAIN_KINDS
        for block in _blocks(item, chunk_rows):
            data = item[block]
            digest = pool.submit(_digest, data)
            if direct:
                starts = range(0, data.shape[0], chunk_rows)
                packed = [pool.submit(_packed, data[i:i + chunk_rows], chunks) for i in starts]
                for i, chunk in zip(starts, packed):
                    offset = (block.start + i, *(0,) * (len(chunks) - 1))
                    copy.id.write_direct_chunk(offset, chunk.result())
            else:
                copy[block] = data
            record.digests.append(digest.result())


def _same(copied: dict, path: Path, pool) -> str | None:
    """None if the file at path holds exactly what was copied -- the same
    datasets, shapes, types and attribute names, and every block of values
    read back through HDF5 hashing as the original's did; otherwise what
    differs."""
    with _h5py().File(path, "r") as f:
        found = dict(_datasets(f))
        if found.keys() != copied.keys():
            return f"datasets differ: {sorted(found.keys() ^ copied.keys())}"
        for name, was in copied.items():
            now = found[name]
            if was.shape != now.shape or was.dtype != now.dtype:
                return f"{name}: {was.shape} {was.dtype} became {now.shape} {now.dtype}"
            if was.attrs != frozenset(now.attrs):
                return f"{name}: attributes differ"
            if not now.size:
                continue
            blocks = list(_blocks(now, was.chunk_rows))
            if len(blocks) != len(was.digests):
                return f"{name}: values differ"
            # Hash a few blocks at a time while the next ones are read.
            pending: list = []
            expected = iter(was.digests)
            for block in blocks:
                pending.append(pool.submit(_digest, now[block]))
                if len(pending) >= _workers() and pending.pop(0).result() != next(expected):
                    return f"{name}: values differ"
            if any(job.result() != want for job, want in zip(pending, expected)):
                return f"{name}: values differ"
    return None


def repack(path: Path | str) -> RepackResult:
    """Rewrite the file compact, in place, if that shrinks it. The new file
    is written beside it, read back and checked value for value against
    what was read from the original (by hash, so the original -- mostly
    padding -- is read only once), and only then put in its place; on any
    failure the original stays and the copy goes."""
    path = Path(path)
    before = path.stat().st_size
    if not _h5py().is_hdf5(path):
        return RepackResult(path, before, before, skipped="not an HDF5 file")
    if is_compressed(path):
        return RepackResult(path, before, before, skipped="already compressed")
    temp = path.with_name(f".{path.name}.repack")
    try:
        h5py = _h5py()
        copied: dict[str, _Copied] = {}
        with ThreadPoolExecutor(_workers()) as pool:
            with h5py.File(path, "r") as source, h5py.File(temp, "w") as target:
                _copy(source, target, pool, copied)
            difference = _same(copied, temp, pool)
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
