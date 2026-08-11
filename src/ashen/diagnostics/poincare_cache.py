"""Incremental, per-field-line storage for Poincare traces.

Legacy cache (poinc_diag.py:208-218) wrote 4 np.savez files/step as dense
(n_psi, ang_sample_freq) arrays -- any shape change (new psi_n, higher
n_turns) forced a full retrace.

Unit here is one field line, keyed by its actual start point, not scan
index: `(psi_n_start, R_start, Z_start, phi_start)`. This makes
ang_sample_freq a position generator, not schema: `_sampled_rz`'s
`np.linspace(0, len(fs)-1, n_sample_freq, dtype=int)` keeps endpoints/
coincident samples when raised (e.g. 8->16), so keying on position reuses
those and traces only what's new.

Layout, one HDF5 file per step:
    /                  attrs: schema, step, pad_width
    /lines/<key>/      attrs: psi_n_start, R_start, Z_start, phi_start,
                              n_turns, terminated, n_segments
                       datasets: R, Z, rho, theta (float32, chunked, gzip,
                                 maxshape=(None,))

HDF5 over a rewritten .npz: extend_line resizes in place (tail-only write),
append_line adds a group untouched by existing ones.

Concurrency: one file per step, one process per step (poincare.
run_poincare_scan) -- never two processes on one file. No locking/SWMR;
that invariant must hold.

`rho` stored as jorek2_poincare writes it (poinc_rho-theta.dat); `psi_n =
rho**2` derived on read (legacy cache discarded the raw value).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "PoincareCacheError",
    "LineKey",
    "LineRecord",
    "LineSummary",
    "LineWork",
    "read_cache",
    "read_cache_summary",
    "read_line_tail",
    "read_legacy_cache",
    "read_step",
    "open_cache",
    "append_line",
    "extend_line",
    "plan_work",
]

SCHEMA_VERSION = 1

#: Arrays stored per line, in the order jorek2_poincare's two output files
#: supply them (``poinc_R-Z.dat`` -> R, Z; ``poinc_rho-theta.dat`` -> rho, theta).
_ARRAYS = ("R", "Z", "rho", "theta")

#: Start points are matched by value, so they need a tolerance. 1e-9 is far
#: below the precision of any flux-surface sample and far above float64 noise
#: in the round trip through the ``stpts`` text file (which is read
#: list-directed by jorek2_poincare.f90:113, hence written at full precision).
_KEY_QUANT = 1e-9


class PoincareCacheError(RuntimeError):
    """A cache file is malformed, or an operation is invalid for its state."""


def _h5py():
    """Import h5py lazily, with an actionable message.

    Kept out of module import so the rest of ashen stays importable on a
    machine without h5py -- CLAUDE.md's rule that importability must not
    depend on ``pip install``.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise PoincareCacheError(
            "the Poincare cache needs h5py, which is not importable here. "
            "It is present in the HPC environment; on a dev clone install it "
            "with `python -m pip install h5py`."
        ) from exc
    return h5py


@dataclass(frozen=True)
class LineKey:
    """A field line's identity: where it starts, not where it sits in a scan."""

    psi_n: float
    R: float
    Z: float
    phi: float

    def quantised(self) -> "LineKey":
        """Snapped to :data:`_KEY_QUANT` so equality and hashing are stable
        across a write/read round trip."""
        q = _KEY_QUANT

        def snap(v: float) -> float:
            # NaN is a legitimate value here: it is how a record recovered
            # from a legacy .npz marks "start point unknown".
            return v if math.isnan(v) else round(v / q) * q

        return LineKey(*(snap(v) for v in (self.psi_n, self.R, self.Z, self.phi)))

    @property
    def group_name(self) -> str:
        """Readable and stable -- deliberately not a hash, so ``h5ls -r`` on a
        cache tells you what is in it.

        10 decimal places -- one full order of magnitude finer than
        :data:`_KEY_QUANT`'s 1e-9 grid. This is not cosmetic: it used to be
        ``.6f``, and two field lines whose *quantised* keys are genuinely
        distinct (correctly kept as two separate ``LineWork`` items by
        ``plan_work``/``dict.fromkeys``, since ``LineKey`` equality compares
        the full quantised float) could still format down to an *identical*
        6-decimal-place string -- e.g. quantised values 1e-9 apart both
        round to the same ``"1.262517"``. :func:`append_line` keys its
        already-cached check off this string, not off ``LineKey`` equality,
        so that collision let the second, legitimately-new line's write
        raise ``"already cached -- use extend_line"`` even though it had
        never actually been written. 10 decimal places puts a full decade of
        margin between the quantisation grid and where formatting could
        round two distinct values together.
        """
        return (
            f"psi{self.psi_n:.10f}_R{self.R:.10f}_Z{self.Z:.10f}_phi{self.phi:.10f}"
        )


@dataclass(frozen=True)
class LineRecord:
    """One cached field line."""

    key: LineKey
    n_turns: int
    #: The line left the mesh (``exit L_IT``, jorek2_poincare.f90:315 and
    #: friends), so it produced fewer punctures than turns requested and
    #: tracing it further is meaningless.
    terminated: bool
    #: 1 for a single trace; >1 if it was stitched together by resuming. A
    #: stitched trace is statistically equivalent to an uninterrupted one but
    #: not bit-identical -- see :mod:`ashen.diagnostics.poincare`.
    n_segments: int
    R: np.ndarray
    Z: np.ndarray
    rho: np.ndarray
    theta: np.ndarray

    @property
    def psi_n(self) -> np.ndarray:
        """Normalised flux at each puncture. The legacy cache stored this
        directly (``poincare.py:124``); here it is derived from ``rho``."""
        return self.rho.astype(np.float64) ** 2

    @property
    def n_points(self) -> int:
        return int(self.R.size)

    @property
    def extendable(self) -> bool:
        """False for a terminated line, and for one recovered from a legacy
        ``.npz`` (whose start point was never recorded, so there is no
        trajectory to resume and no key to match against)."""
        return not self.terminated and not any(
            math.isnan(v) for v in (self.key.R, self.key.Z, self.key.phi)
        )


@dataclass(frozen=True)
class LineSummary:
    """A cached line's metadata, without its ``R``/``Z``/``rho``/``theta``
    arrays -- everything :func:`plan_work` needs to decide new/extend/skip,
    from attrs alone. See :func:`read_cache_summary`."""

    key: LineKey
    n_turns: int
    terminated: bool
    n_points: int

    @property
    def extendable(self) -> bool:
        """Same rule as :attr:`LineRecord.extendable`."""
        return not self.terminated and not any(
            math.isnan(v) for v in (self.key.R, self.key.Z, self.key.phi)
        )


@dataclass(frozen=True)
class LineWork:
    """One line's pending work, as handed to the tracer.

    ``resume_from`` is ``None`` for a line that has never been traced;
    otherwise it is the last cached puncture, which is a valid continuation
    point because every puncture lies on the same toroidal plane (see
    :mod:`ashen.diagnostics.poincare`).
    """

    key: LineKey
    n_turns: int
    resume_from: tuple[float, float] | None = None

    @property
    def is_extension(self) -> bool:
        return self.resume_from is not None

    @property
    def start(self) -> tuple[float, float]:
        """``(R, Z)`` to write into ``stpts``."""
        return self.resume_from if self.resume_from is not None else (self.key.R, self.key.Z)


# --- reading ------------------------------------------------------------------


def read_cache(path: Path | str) -> dict[LineKey, LineRecord]:
    """Load every line from a cache file. Returns ``{}`` if it doesn't exist."""
    path = Path(path)
    if not path.is_file():
        return {}

    h5py = _h5py()
    records: dict[LineKey, LineRecord] = {}
    with h5py.File(path, "r") as f:
        schema = int(f.attrs.get("schema", -1))
        if schema != SCHEMA_VERSION:
            raise PoincareCacheError(
                f"{path}: schema {schema}, expected {SCHEMA_VERSION}"
            )
        for name, group in f.get("lines", {}).items():
            key = LineKey(
                psi_n=float(group.attrs["psi_n_start"]),
                R=float(group.attrs["R_start"]),
                Z=float(group.attrs["Z_start"]),
                phi=float(group.attrs["phi_start"]),
            ).quantised()
            missing = [a for a in _ARRAYS if a not in group]
            if missing:
                raise PoincareCacheError(
                    f"{path}: line {name!r} is missing {', '.join(missing)}"
                )
            records[key] = LineRecord(
                key=key,
                n_turns=int(group.attrs["n_turns"]),
                terminated=bool(group.attrs["terminated"]),
                n_segments=int(group.attrs["n_segments"]),
                **{a: np.asarray(group[a][:]) for a in _ARRAYS},
            )
    return records


def read_cache_summary(path: Path | str) -> dict[LineKey, LineSummary]:
    """Like :func:`read_cache`, but reads only each line's attrs, never its
    ``R``/``Z``/``rho``/``theta`` arrays -- attrs are plain HDF5 attributes,
    not gzip-chunked data, so this skips decompressing every already-cached
    line's full trajectory just to check whether it needs more work. That
    check (:func:`plan_work`) is the common case on every rerun of a scan
    that only adds a few new steps: every untouched step still opened its
    cache before this existed, and paid to decompress lines it was about to
    decide needed nothing.
    """
    path = Path(path)
    if not path.is_file():
        return {}

    h5py = _h5py()
    summaries: dict[LineKey, LineSummary] = {}
    with h5py.File(path, "r") as f:
        schema = int(f.attrs.get("schema", -1))
        if schema != SCHEMA_VERSION:
            raise PoincareCacheError(
                f"{path}: schema {schema}, expected {SCHEMA_VERSION}"
            )
        for name, group in f.get("lines", {}).items():
            key = LineKey(
                psi_n=float(group.attrs["psi_n_start"]),
                R=float(group.attrs["R_start"]),
                Z=float(group.attrs["Z_start"]),
                phi=float(group.attrs["phi_start"]),
            ).quantised()
            missing = [a for a in _ARRAYS if a not in group]
            if missing:
                raise PoincareCacheError(
                    f"{path}: line {name!r} is missing {', '.join(missing)}"
                )
            summaries[key] = LineSummary(
                key=key,
                n_turns=int(group.attrs["n_turns"]),
                terminated=bool(group.attrs["terminated"]),
                n_points=int(group["R"].shape[0]),
            )
    return summaries


def read_line_tail(path: Path | str, key: LineKey) -> tuple[float, float]:
    """One cached line's last ``(R, Z)`` puncture -- the one array read
    :func:`plan_work` can't avoid for a line it decides to extend, deferred
    here so it's paid only for that line, not every cached line."""
    key = key.quantised()
    h5py = _h5py()
    with h5py.File(Path(path), "r") as f:
        group = f["lines"][key.group_name]
        return float(group["R"][-1]), float(group["Z"][-1])


def read_step(paths, step: int | float) -> dict[LineKey, LineRecord]:
    """A step's cache, new format if present, else the legacy ``.npz`` set.

    Convenience for readers (plotting, notebooks) that don't want to know
    which format a given step happens to be in -- gathering always writes the
    new format, but a step traced before Phase 4b only has the old one.
    """
    records = read_cache(paths.poincare_cache(step))
    if records:
        return records
    return read_legacy_cache(paths.poinc_dir, paths.step_str(step))


def read_legacy_cache(poinc_dir: Path | str, step_str: str) -> dict[LineKey, LineRecord]:
    """Load the pre-Phase-4b four-file ``.npz`` set for one step.

    Kept so nothing already computed is lost. The legacy files record only the
    starting ``psi_n``, never the ``(R, Z)`` the line actually started from
    (``poinc_diag.py:208-218``), so the recovered records get ``NaN`` start
    coordinates and are :attr:`~LineRecord.extendable`-false: they can be read
    and plotted, but the first time such a step is extended it is retraced from
    scratch, once.
    """
    poinc_dir = Path(poinc_dir)
    parts = {}
    for kind in ("psi_n", "theta", "R", "Z"):
        f = poinc_dir / f"poinc_t{step_str}_{kind}.npz"
        if not f.is_file():
            return {}
        parts[kind] = np.load(f, allow_pickle=True)

    psi_in = np.asarray(parts["psi_n"]["in_val"], dtype=float)
    records: dict[LineKey, LineRecord] = {}
    for i, psi_n in enumerate(psi_in):
        row = parts["psi_n"]["out_val"][i]
        for j in range(len(row)):
            # Angular samples share a psi_n and have no recorded start point,
            # so they cannot be told apart by position. Disambiguate by index
            # in the phi slot -- these keys are only ever used for reading,
            # never for matching against new work (extendable is false).
            key = LineKey(psi_n=float(psi_n), R=math.nan, Z=math.nan, phi=float(j))
            rho = np.sqrt(np.maximum(np.asarray(row[j], dtype=float), 0.0))
            records[key] = LineRecord(
                key=key,
                n_turns=int(rho.size),
                terminated=False,
                n_segments=1,
                R=np.asarray(parts["R"]["out_val"][i][j], dtype=np.float32),
                Z=np.asarray(parts["Z"]["out_val"][i][j], dtype=np.float32),
                rho=rho.astype(np.float32),
                theta=np.asarray(parts["theta"]["out_val"][i][j], dtype=np.float32),
            )
    return records


# --- writing ------------------------------------------------------------------


def open_cache(path: Path | str, *, step: int, pad_width: int):
    """Open (creating if needed) a cache file for writing.

    Use as a context manager; the returned handle is passed to
    :func:`append_line` / :func:`extend_line`.
    """
    h5py = _h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "a")
    if "schema" not in f.attrs:
        f.attrs["schema"] = SCHEMA_VERSION
        f.attrs["step"] = int(step)
        f.attrs["pad_width"] = int(pad_width)
        f.create_group("lines")
    elif int(f.attrs["schema"]) != SCHEMA_VERSION:
        schema = int(f.attrs["schema"])
        f.close()
        raise PoincareCacheError(f"{path}: schema {schema}, expected {SCHEMA_VERSION}")
    return f


def append_line(
    handle,
    key: LineKey,
    arrays: Mapping[str, np.ndarray],
    *,
    n_turns: int,
    terminated: bool,
    replace: bool = False,
) -> None:
    """Add a never-before-traced line. Existing groups are untouched.

    ``replace=True`` overwrites an existing group instead of raising -- used
    for the one case :func:`plan_work` schedules a retrace of a cached line:
    a record with zero punctures, which has no puncture to resume from.
    """
    key = key.quantised()
    lines = handle["lines"]
    if key.group_name in lines:
        if not replace:
            raise PoincareCacheError(
                f"line {key.group_name!r} already cached -- use extend_line"
            )
        del lines[key.group_name]
    _check_arrays(arrays)

    group = lines.create_group(key.group_name)
    group.attrs["psi_n_start"] = float(key.psi_n)
    group.attrs["R_start"] = float(key.R)
    group.attrs["Z_start"] = float(key.Z)
    group.attrs["phi_start"] = float(key.phi)
    group.attrs["n_turns"] = int(n_turns)
    group.attrs["terminated"] = bool(terminated)
    group.attrs["n_segments"] = 1
    for name in _ARRAYS:
        data = np.asarray(arrays[name], dtype=np.float32)
        group.create_dataset(
            name,
            data=data,
            maxshape=(None,),
            chunks=(max(int(data.size), 1),),
            compression="gzip",
        )


def extend_line(
    handle,
    key: LineKey,
    arrays: Mapping[str, np.ndarray],
    *,
    added_turns: int,
    terminated: bool,
) -> None:
    """Append a resumed segment to an already-cached line, in place.

    Each dataset is resized and only the tail is written -- the existing
    samples are never rewritten, which is the whole reason this cache is HDF5.
    """
    key = key.quantised()
    lines = handle["lines"]
    if key.group_name not in lines:
        raise PoincareCacheError(f"line {key.group_name!r} not cached -- use append_line")
    group = lines[key.group_name]
    if bool(group.attrs["terminated"]):
        raise PoincareCacheError(
            f"line {key.group_name!r} left the mesh; there is nothing to resume"
        )
    _check_arrays(arrays)

    for name in _ARRAYS:
        data = np.asarray(arrays[name], dtype=np.float32)
        ds = group[name]
        old = ds.shape[0]
        ds.resize((old + data.size,))
        ds[old:] = data
    group.attrs["n_turns"] = int(group.attrs["n_turns"]) + int(added_turns)
    group.attrs["terminated"] = bool(terminated)
    group.attrs["n_segments"] = int(group.attrs["n_segments"]) + 1


def _check_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    missing = [a for a in _ARRAYS if a not in arrays]
    if missing:
        raise PoincareCacheError(f"missing array(s): {', '.join(missing)}")
    sizes = {a: np.asarray(arrays[a]).size for a in _ARRAYS}
    if len(set(sizes.values())) != 1:
        raise PoincareCacheError(f"array length mismatch: {sizes}")


# --- planning -----------------------------------------------------------------


def plan_work(
    cached: Mapping[LineKey, LineRecord] | Mapping[LineKey, LineSummary],
    requested: Iterable[LineKey],
    n_turns: int,
    *,
    last_point: Callable[[LineKey, LineRecord], tuple[float, float]] | None = None,
) -> tuple[list[LineWork], list[LineWork]]:
    """Split a request into ``(new, extensions)``.

    A requested line is *new* if its start point isn't cached, and an
    *extension* if it is cached with fewer than ``n_turns`` turns and is
    :attr:`~LineRecord.extendable`. A line already at or beyond ``n_turns``,
    or one that left the mesh, produces no work at all -- which is what makes
    widening ``psi_n_in`` or raising ``n_turns`` cost only the increment.

    ``cached`` normally maps to :class:`LineRecord` (full arrays already in
    memory); ``last_point`` then defaults to reading ``record.R[-1]``/
    ``record.Z[-1]`` directly. Pass :class:`LineSummary` values instead (from
    :func:`read_cache_summary`) with a ``last_point`` that fetches the tail on
    demand (e.g. :func:`read_line_tail`) to plan a step's work without ever
    loading a line's full trajectory unless it's actually being extended.
    """
    if last_point is None:
        last_point = lambda key, record: (float(record.R[-1]), float(record.Z[-1]))  # noqa: E731

    new: list[LineWork] = []
    extensions: list[LineWork] = []
    for key in requested:
        key = key.quantised()
        record = cached.get(key)
        if record is None:
            new.append(LineWork(key=key, n_turns=int(n_turns)))
            continue
        if record.n_turns >= n_turns:
            continue
        if record.terminated:
            # The line left the mesh; there is no more trajectory to have.
            continue
        if record.n_points == 0 or not record.extendable:
            # Nothing to resume from -- either no puncture was recorded, or
            # the record came from a legacy .npz that never stored a start
            # point. Retrace from the original position instead.
            new.append(LineWork(key=key, n_turns=int(n_turns)))
            continue
        extensions.append(
            LineWork(
                key=key,
                n_turns=int(n_turns) - record.n_turns,
                resume_from=last_point(key, record),
            )
        )
    return new, extensions
