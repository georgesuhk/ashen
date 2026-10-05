"""Tidying a run's trace folders (`util --func trace_organize`).

Two jobs:

**Restart links.** ashen links (or, on a filesystem without symlinks,
copies) a run's restarts into a trace folder as ``jorek*.h5`` while the
program runs, and removes them when it is done -- but a trace that was
interrupted, failed to stage, or whose queued job was never concluded
leaves them behind. They are only ever the program's input, never a
result, so they go; a copy frees its size. A folder whose queued job may
still be running is left alone: the job reads them.

**Older layouts.** Traces from earlier versions of ashen sit where the
plots no longer look:

- ``<run>/trace/<exe>/`` -- before "trace" became "ptrace": ``trace.log``,
  ``trace_meta.json``, ``trace_diag.h5``, settings in ``trace_params.nml``;
- loose in ``<run>/ptrace/<exe>/`` -- before a ptrace_gc trace got a folder
  per energy and marker count: settings in ``ptrace_params.nml`` and
  ``ptrace_overrides.nml``, or later ``ptrace_settings.nml``.

Each is moved to ``<run>/ptrace/<exe>/<label>/`` -- the label
(ptracing.settings_label) read from the trace's *own* settings files, so it
names what that trace ran with, whatever cases.toml says now -- and its
files renamed to today's names (``trace.log`` -> ``ptrace.log``, ...).
A trace with no settings file and no ptrace_gc output is a program that
takes none (re_gc, ex6/ex7): it belongs in ``<run>/ptrace/<exe>/`` itself.
A trace whose energy cannot be told, or whose place is taken, is left
where it is, and said so.

Nothing here runs a program or reads a restart; :func:`plan_tidy` decides
everything first, :func:`apply_tidy` then does it.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ashen.ptracing import META_FILE, TRACE_DIR, PtraceError, poll_job, settings_label

__all__ = ["TidyAction", "TidyPlan", "apply_tidy", "plan_tidy", "read_label_settings"]

#: The folder traces lived in before "trace" became "ptrace".
OLD_TRACE_DIR = "trace"

#: Old names of files a trace folder holds, and today's.
RENAMES = {
    "trace.log": "ptrace.log",
    "trace_meta.json": "ptrace_meta.json",
    "trace_diag.h5": "ptrace_diag.h5",
}

#: Files whose being there makes a folder hold a trace (any layout).
_TRACE_FILES = (
    "ptrace_meta.json", "trace_meta.json", "ptrace.log", "trace.log",
    "ptrace_diag.h5", "trace_diag.h5", "part_diag.h5", "diag.h5", "part_restart.h5",
)

#: Settings files, the later read over the earlier: what a trace ran with.
_SETTINGS_FILES = ("trace_params.nml", "ptrace_params.nml", "ptrace_overrides.nml",
                   "ptrace_settings.nml")

#: ptrace_gc's diagnostics file, under either name: a trace that has it ran
#: ptrace_gc, so it needs a label.
_GC_DIAG = ("ptrace_diag.h5", "trace_diag.h5")


@dataclass(frozen=True)
class TidyAction:
    """One thing trace_organize does, or would do."""

    #: "remove" (a restart link or copy), "move" (a whole trace), or "keep"
    #: (left as it is, with why).
    kind: str
    path: Path
    target: Path | None = None
    #: Bytes a removal frees (0 for a link).
    size: int = 0
    note: str = ""
    #: For "move": (old name, new name) of the files renamed on the way.
    renames: tuple[tuple[str, str], ...] = ()

    def describe(self, root: Path) -> str:
        def rel(path: Path) -> str:
            try:
                return str(path.relative_to(root))
            except ValueError:
                return str(path)

        if self.kind == "remove":
            what = f"copy, {_size(self.size)}" if self.size else "link"
            return f"remove   {rel(self.path)}  ({what})"
        if self.kind == "move":
            renamed = "".join(f"; {old} -> {new}" for old, new in self.renames)
            return f"move     {rel(self.path)}/ -> {rel(self.target)}/  ({self.note}{renamed})"
        return f"keep     {rel(self.path)}/  ({self.note})"


@dataclass
class TidyPlan:
    """Everything trace_organize will do for one run folder."""

    run_dir: Path
    actions: list[TidyAction] = field(default_factory=list)

    @property
    def freed(self) -> int:
        return sum(a.size for a in self.actions if a.kind == "remove")


def _size(n: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def _holds_trace(folder: Path) -> bool:
    return any((folder / name).exists() or (folder / name).is_symlink() for name in _TRACE_FILES)


# --- the settings a trace ran with ---------------------------------------------------

_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)\s*(?:\(\s*\d+\s*\))?\s*=")


def _namelist_values(text: str) -> dict[str, list[str]]:
    """{name: raw values} of the &ptrace group in a namelist's text, a
    later assignment winning; values split on commas and blanks, with
    repeat counts (3*1.0) expanded."""
    text = "\n".join(line.split("!", 1)[0] for line in text.splitlines())
    start = re.search(r"&ptrace\b", text, re.IGNORECASE)
    if start is None:
        return {}
    body = text[start.end():]
    end = re.search(r"^\s*/|&end", body, re.MULTILINE | re.IGNORECASE)
    body = body[:end.start()] if end else body
    found: dict[str, list[str]] = {}
    marks = list(_ASSIGNMENT.finditer(body))
    for here, after in zip(marks, [*marks[1:], None]):
        raw = body[here.end():after.start() if after else len(body)]
        values = []
        for token in re.split(r"[,\s]+", raw.strip()):
            if not token:
                continue
            count, star, value = token.rpartition("*")
            values += [value] * int(count) if star and count.isdigit() else [token]
        found[here.group(1).lower()] = values
    return found


def _number(text: str) -> float:
    return float(text.lower().replace("d", "e"))


def read_label_settings(folder: Path) -> dict:
    """{"E_kin_eV": [...], "n_markers": n} as far as the folder's settings
    files say -- each read over the one before, as the tracer read them.
    Empty if it has none, or they say neither."""
    values: dict[str, list[str]] = {}
    for name in _SETTINGS_FILES:
        path = folder / name
        if path.is_file():
            values.update(_namelist_values(path.read_text(encoding="utf-8", errors="replace")))
    settings: dict = {}
    try:
        if values.get("e_kin_ev"):
            settings["E_kin_eV"] = [_number(v) for v in values["e_kin_ev"]]
        if values.get("n_markers"):
            settings["n_markers"] = int(_number(values["n_markers"][0]))
    except ValueError:
        return {}
    if "E_kin_eV" in settings and "n_markers" in settings:
        # One energy per marker under 'markers'; a list longer than the
        # markers (an old file's spare entries) says nothing of the rest.
        settings["E_kin_eV"] = settings["E_kin_eV"][:settings["n_markers"]]
    return settings


# --- planning ---------------------------------------------------------------------------


def _restart_actions(folder: Path) -> list[TidyAction]:
    actions = []
    for entry in sorted(folder.glob("jorek*.h5")):
        if entry.is_symlink():
            actions.append(TidyAction("remove", entry))
        elif entry.is_file():
            actions.append(TidyAction("remove", entry, size=entry.stat().st_size))
    return actions


def _job_may_be_running(folder: Path) -> str | None:
    """Why the folder must be left alone, if a job queued for it has not
    been seen to end -- asking SLURM, as `ptrace` does, which also
    concludes a job that has ended. None if there is no such job."""
    try:
        queued = poll_job(folder)
    except PtraceError as exc:
        return f"a queued job's state is unknown -- {exc}"
    if queued is not None and queued.running:
        return f"job {queued.job_id} is {queued.state.lower()}"
    return None


def _move_action(source: Path, target: Path, note: str) -> TidyAction:
    renames = tuple(
        (old, new) for old, new in RENAMES.items()
        if (source / old).exists() and not (source / new).exists()
    )
    return TidyAction("move", source, target=target, note=note, renames=renames)


def _place_of(folder: Path, exe_dir: Path) -> TidyAction | None:
    """Where an older-layout trace in `folder` belongs, under `exe_dir`
    (``<run>/ptrace/<exe>``): a move, a keep with why, or None when it is
    where it belongs already."""
    settings = read_label_settings(folder)
    label = settings_label(settings)
    if label is None:
        if any((folder / name).exists() for name in _GC_DIAG):
            return TidyAction("keep", folder, note=(
                "a ptrace_gc trace whose settings files do not say its energy and marker "
                "count; move it into ptrace/<exe>/E<eV>eV_n<markers>/ by hand"))
        if folder == exe_dir:
            return None
        target = exe_dir
    else:
        target = exe_dir / label
    if target != folder and target.is_dir() and _holds_trace(target):
        return TidyAction("keep", folder, note=f"{target.name}/ already holds a trace")
    what = label or "a program without settings"
    return _move_action(folder, target, note=what)


def plan_tidy(run_dir: Path | str) -> TidyPlan:
    """Decide what trace_organize does for one run folder; touch nothing
    (except concluding a queued job that has ended, as `ptrace` would)."""
    run_dir = Path(run_dir)
    plan = TidyPlan(run_dir=run_dir)
    for top in (TRACE_DIR, OLD_TRACE_DIR):
        base = run_dir / top
        if not base.is_dir():
            continue
        for exe_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.is_symlink()):
            folders = [exe_dir, *sorted(p for p in exe_dir.iterdir()
                                        if p.is_dir() and not p.is_symlink())]
            home = run_dir / TRACE_DIR / exe_dir.name
            for folder in folders:
                busy = _job_may_be_running(folder) if (folder / META_FILE).is_file() else None
                if busy:
                    plan.actions.append(TidyAction("keep", folder, note=(
                        f"{busy}; its restart links stay while it runs -- `ptrace` once it "
                        "has ended")))
                    continue
                plan.actions += _restart_actions(folder)
                if folder != exe_dir and top == TRACE_DIR:
                    continue  # a labelled folder: today's layout
                if not _holds_trace(folder):
                    continue
                placed = _place_of(folder, home)
                if placed is not None:
                    plan.actions.append(placed)
    return plan


# --- doing it -----------------------------------------------------------------------------


def _move_trace(action: TidyAction) -> None:
    """Move a trace's files (not the folders under it -- those are other
    traces) into the target, renaming old names to today's; remove the
    source folder if that empties it."""
    source, target = action.path, action.target
    renamed = dict(action.renames)
    target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            continue
        destination = target / renamed.get(entry.name, entry.name)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"{destination} already exists; {entry} left where it is")
        shutil.move(str(entry), str(destination))
    for folder in (source, source.parent):
        try:
            folder.rmdir()  # only if empty
        except OSError:
            break


def apply_tidy(plan: TidyPlan) -> None:
    """Carry the plan out: removals first, so no restart link travels."""
    for action in plan.actions:
        if action.kind == "remove":
            os.unlink(action.path)
    for action in plan.actions:
        if action.kind == "move":
            _move_trace(action)
