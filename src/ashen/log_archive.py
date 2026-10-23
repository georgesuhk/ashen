"""Keeping every JOREK job's log, compressed, in the run folder.

A main run or restart tees its output into ``log`` (``runner.submit_main``,
``submit_restart``), so each launch overwrites the previous job's log --
and with it the step-by-step record of how far the run got at which time
step and how hard each step was for the solver (the GMRES iteration count),
which ``ashen.tstep_history`` reads. Batch jobs also leave SLURM's copy of
the same stream, ``<jobname>.<jobid>.out`` (the jobscripts'
``#SBATCH -o ./%x.%j.out``), so most of that record is on disk, but twice
over and uncompressed (a few hundred MB per job).

Here each job's log is kept once, gzipped (about 16x smaller), as::

    logs/log_s000961-s004052.gz
    logs/log_s000001-s000986_gmresfail.gz

named by the first and last time step it ran (``* time step :`` banners,
padded like the run's restarts) and how it ended, where the log says:
``gmresfail`` (``>>>>> NO CONVERGENCE AFTER ... ITERATIONS. ABORTING``,
jorek2_main.f90), ``nan`` (``NaNs DETECTED``), ``sigterm`` (``SIGTERM
RECEIVED``), ``walltime`` (SLURM's ``.err``: ``DUE TO TIME LIMIT``), or
nothing (ran to its end, or stopped some other way); ``_mixed`` when
another job's output follows in the same file (two jobs launched in one
folder at once: see banners_follow). ``logs/index.json``
records each archive's sha256, so the same log found again (``log`` and its
``.out``) is kept once.

The source is copied, never moved: a job may still be writing it, and the
next launch's tee truncates ``log`` anyway.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from ashen.padding import DEFAULT_PAD_WIDTH, PaddingError, detect_pad_width

__all__ = [
    "ARCHIVE_DIR", "INDEX_FILE", "LogSpan", "banners_follow", "ArchiveAction", "open_log", "scan_span",
    "job_logs", "archived_logs", "read_index", "plan_archive", "apply_archive", "archive_current_log",
]

ARCHIVE_DIR = "logs"
INDEX_FILE = "index.json"
LIVE_LOG = "log"

#: A file written to more recently than this [s] may belong to a job still
#: running: archived as it stands, never deleted.
RECENT = 600

#: ``<jobname>.<jobid>.out``, SLURM's stdout for the jobscripts' -o pattern.
#: ptrace's ``trace0000.out`` has no job id and does not match.
_JOB_OUT = re.compile(r"^.+\.\d+\.out$")

BANNER = re.compile(r"^\s*\*\s+time step :\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)")

#: Line markers of how a job ended, first match wins (jorek2_main.f90).
_ENDINGS = (
    ("gmresfail", "NO CONVERGENCE AFTER"),
    ("nan", "NaNs DETECTED"),
    ("sigterm", "SIGTERM RECEIVED"),
)
_WALLTIME = "DUE TO TIME LIMIT"


def open_log(path: Path | str) -> TextIO:
    """A log for reading as text, gzipped (``.gz``) or not."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class LogSpan:
    """Which time steps a job log ran and how it ended."""

    first: int | None     #: index_now of the first ``* time step :`` banner
    last: int | None      #: and of the last (the last step the job began)
    n_steps: int
    outcome: str | None   #: gmresfail, nan, sigterm, walltime or None
    mixed: bool = False   #: another job's output follows (see banners_follow)


def banners_follow(before: tuple, after: tuple) -> bool:
    """Whether banner ``after`` (jstep, istep, index) is the step after ``before``.

    Two jobs launched in the same folder at once both write ``log``: the
    later one's tee truncates it while the earlier keeps writing at its own
    offset, so the file holds one job's steps and then the other's. The
    first banner that does not follow marks where."""
    if after[2] != before[2] + 1:
        return False
    return (after[0] == before[0] and after[1] == before[1] + 1) or \
        (after[0] > before[0] and after[1] == 1)


def _err_for(path: Path) -> Path | None:
    if _JOB_OUT.match(path.name):
        err = path.with_suffix(".err")
        return err if err.is_file() else None
    return None


def scan_span(path: Path | str) -> LogSpan:
    """Read a log through once for its step range and ending; where another
    job's output follows (``mixed``), up to there."""
    path = Path(path)
    first = None
    previous = None
    n = 0
    outcome = None
    mixed = False
    with open_log(path) as f:
        for line in f:
            if "time step :" in line:
                m = BANNER.match(line)
                if m:
                    banner = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    if previous is not None and not banners_follow(previous, banner):
                        mixed = True
                        break
                    first = banner[2] if first is None else first
                    previous = banner
                    n += 1
                    continue
            if outcome is None and ">>>>>" in line:
                for kind, marker in _ENDINGS:
                    if marker in line:
                        outcome = kind
                        break
    last = previous[2] if previous else None
    if mixed:
        return LogSpan(first, last, n, outcome, mixed=True)
    if outcome is None:
        err = _err_for(path)
        if err is not None and _WALLTIME in err.read_text(encoding="utf-8", errors="replace"):
            outcome = "walltime"
    return LogSpan(first, last, n, outcome)


def archive_name(span: LogSpan, width: int, mtime: float) -> str:
    """``log_s<first>-s<last>[_<outcome>].gz``; a log with no time step (a
    job that stopped while starting) is named by when it was written."""
    if span.first is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(mtime))
        stem = f"log_nosteps_{stamp}"
    else:
        stem = f"log_s{span.first:0{width}d}-s{span.last:0{width}d}"
    return stem + (f"_{span.outcome}" if span.outcome else "") + \
        ("_mixed" if span.mixed else "") + ".gz"


def job_logs(run_dir: Path | str) -> list[Path]:
    """The job logs in a run folder, oldest first: ``log`` and SLURM's
    ``<jobname>.<jobid>.out`` files that hold JOREK time steps or its
    start-up (a ``.out`` of anything else is not a JOREK log)."""
    run_dir = Path(run_dir)
    found = [p for p in run_dir.iterdir() if p.is_file() and _JOB_OUT.match(p.name)]
    found = [p for p in found if _looks_like_jorek(p)]
    live = run_dir / LIVE_LOG
    if live.is_file():
        found.append(live)
    return sorted(found, key=lambda p: p.stat().st_mtime)


def _looks_like_jorek(path: Path) -> bool:
    """A JOREK main-run log starts with its build information."""
    with open_log(path) as f:
        head = f.read(4000)
    return "GIT revision" in head or "time step :" in head


def archived_logs(run_dir: Path | str) -> list[Path]:
    """The archived logs, by name (so by first step)."""
    folder = Path(run_dir) / ARCHIVE_DIR
    return sorted(folder.glob("log_*.gz")) if folder.is_dir() else []


def read_index(folder: Path) -> dict:
    try:
        return json.loads((folder / INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_index(folder: Path, index: dict) -> None:
    tmp = folder / (INDEX_FILE + ".tmp")
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, folder / INDEX_FILE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArchiveAction:
    """What archiving one log does (or would do)."""

    source: Path
    kind: str                 #: "archive", "duplicate" (kept already) or "empty"
    target: Path | None       #: the archive it becomes, or the one that holds it
    span: LogSpan
    size: int                 #: the source's size [B]
    sha256: str
    busy: bool                #: written to within RECENT s: a job may be running

    def describe(self, run_dir: Path) -> str:
        where = self.source.relative_to(run_dir)
        steps = "no time steps" if self.span.first is None else \
            f"steps {self.span.first}-{self.span.last}"
        ending = f", ended: {self.span.outcome}" if self.span.outcome else ""
        if self.span.mixed:
            ending += ", then another job's output: two jobs ran here at once"
        note = " (written to recently: a job may still be running)" if self.busy else ""
        if self.kind == "duplicate":
            return f"{where}: already kept as {self.target.relative_to(run_dir)}{note}"
        if self.kind == "empty":
            return f"{where}: empty, nothing to keep"
        return f"{where} -> {self.target.relative_to(run_dir)}  ({steps}{ending}){note}"


def _unique_target(folder: Path, name: str, taken: set[str]) -> Path:
    stem = name[:-3]
    candidate, k = name, 2
    while candidate in taken or (folder / candidate).exists():
        candidate = f"{stem}_{k}.gz"
        k += 1
    return folder / candidate


def plan_archive(run_dir: Path | str, sources: list[Path] | None = None) -> list[ArchiveAction]:
    """What archiving ``sources`` (default: every job log in the folder) would do."""
    run_dir = Path(run_dir)
    folder = run_dir / ARCHIVE_DIR
    index = read_index(folder)
    by_sha = {entry["sha256"]: name for name, entry in index.items()
              if (folder / name).is_file()}
    try:
        width = detect_pad_width(run_dir)
    except PaddingError:    # no restart written yet
        width = DEFAULT_PAD_WIDTH
    taken: set[str] = set()
    actions = []
    for source in (job_logs(run_dir) if sources is None else sources):
        stat = source.stat()
        busy = time.time() - stat.st_mtime < RECENT
        if stat.st_size == 0:
            actions.append(ArchiveAction(source, "empty", None, LogSpan(None, None, 0, None),
                                         0, "", busy))
            continue
        sha = _sha256(source)
        span = scan_span(source)
        if sha in by_sha:
            actions.append(ArchiveAction(source, "duplicate", folder / by_sha[sha], span,
                                         stat.st_size, sha, busy))
            continue
        target = _unique_target(folder, archive_name(span, width, stat.st_mtime), taken)
        taken.add(target.name)
        by_sha[sha] = target.name
        actions.append(ArchiveAction(source, "archive", target, span, stat.st_size, sha, busy))
    return actions


def apply_archive(action: ArchiveAction) -> int:
    """Carry out one planned action; the archive's size [B] (0 if none written).

    Written to a temporary file and renamed into place, then recorded in the
    index, so an interrupted archive leaves no half-written ``.gz`` behind."""
    if action.kind != "archive":
        return 0
    folder = action.target.parent
    folder.mkdir(exist_ok=True)
    tmp = folder / (action.target.name + ".tmp")
    with open(action.source, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    os.replace(tmp, action.target)
    index = read_index(folder)
    index[action.target.name] = {
        "sha256": action.sha256,
        "bytes": action.size,
        "first_step": action.span.first,
        "last_step": action.span.last,
        "outcome": action.span.outcome,
        "source": action.source.name,
        "archived": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_index(folder, index)
    return action.target.stat().st_size


def archive_current_log(run_dir: Path | str, *, dry_run: bool = False) -> str | None:
    """Keep the run folder's ``log`` before a launch overwrites it.

    Returns a line saying what was (or would be) done, None if there is no
    log. Raises OSError if it cannot be kept: the caller should not launch,
    or that job's record is lost."""
    run_dir = Path(run_dir)
    live = run_dir / LIVE_LOG
    if not live.is_file():
        return None
    (action,) = plan_archive(run_dir, [live])
    if not dry_run:
        apply_archive(action)
    if action.kind == "archive":
        verb = "would keep" if dry_run else "kept"
        return f"{verb} the previous log as {action.target.relative_to(run_dir)}"
    if action.kind == "duplicate":
        return f"the previous log is kept already, as {action.target.relative_to(run_dir)}"
    return None
