"""A scan file: several run folders from one base shotfile.

A scan is a small TOML file in a campaign that names a base shotfile and the
shotfile fields that vary::

    base   = "base_shotfile.py"        # an ordinary shotfile: what the runs share
    folder = "qa2.1_li1.0_q01.0"       # parent folder of the runs
    name   = "eta{eta}"                # run folder name, from the varied values
    copy   = ["plasma_bnd.dat"]        # files each run folder needs (optional)

    [vary]                             # every combination of these ...
    eta = [1e-3, 1e-4, 1e-5]

    # [[runs]]                         # ... or, instead, hand-picked ones
    # eta = 1e-3

Each run folder gets the base shotfile with only the varied lines rewritten
(ashen.shotfile.shotfile_text_with), so every folder still holds its own
complete shotfile.py and run_jorek, the viewer and analyse work in it as in
any other. All paths are relative to the campaign root, the folder holding
site.toml. Folder names are built from the values and never parsed back.

Applying a scan only ever acts on run folders it creates: one already there
with the same shotfile is not touched, prepared or launched again, so a scan
can be extended and re-applied while its earlier runs are going. One with a
different shotfile.py is left alone unless forced. Unless ``cases = false``, the runs are added to the campaign's
cases.toml (append only: nothing already there is edited), with a comparison
named after the scan file.
"""

from __future__ import annotations

import itertools
import shutil
import string
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ashen.shotfile import ShotfileError, ShotParams, shotfile_text_with

__all__ = [
    "PlannedRun",
    "Scan",
    "ScanError",
    "apply_plan",
    "cases_toml_additions",
    "format_value",
    "load_scan",
    "plan_scan",
    "update_cases_toml",
]

_KEYS = {"base", "folder", "name", "copy", "vary", "runs", "cases", "comparison"}


class ScanError(RuntimeError):
    """A scan file that cannot be read, or asks for something inconsistent."""


def format_value(value: Any) -> str:
    """A value as it appears in a folder name: 1e-3 not 0.001, 2.1 not 2.10.

    Floats below 0.01 or from 10000 up are written as mantissa and exponent
    with no padding (``1e-3``, ``2.5e-4``, ``1e5``); others plainly. Whole
    floats lose their ``.0`` only in exponent form: 2.0 stays ``2.0``.
    """
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    if value == 0 or 1e-2 <= abs(value) < 1e4:
        return repr(value)
    mantissa, exponent = f"{value:.12e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


@dataclass(frozen=True)
class Scan:
    source: Path                 #: the scan file
    base: str                    #: base shotfile, relative to the campaign root
    folder: str                  #: parent folder of the runs, relative to the root
    name: str                    #: run folder name pattern
    runs: list[dict[str, Any]]   #: the varied values of each run, in order
    copy: list[str] = field(default_factory=list)
    cases: bool = True           #: add the runs to cases.toml
    comparison: str = ""         #: name of the comparison; default: the file's stem

    @property
    def varied(self) -> list[str]:
        """The varied keys, in the order first seen."""
        seen: dict[str, None] = {}
        for run in self.runs:
            seen.update(dict.fromkeys(run))
        return list(seen)

    def run_name(self, values: dict[str, Any]) -> str:
        """The run folder name for one run's values."""
        parts = []
        for literal, key, spec, _ in string.Formatter().parse(self.name):
            parts.append(literal)
            if key is None:
                continue
            if key not in values:
                raise ScanError(
                    f"{self.source}: name = {self.name!r} uses {{{key}}}, which this run "
                    f"does not vary (it varies {', '.join(values) or 'nothing'})"
                )
            parts.append(format(values[key], spec) if spec else format_value(values[key]))
        return "".join(parts)


def load_scan(path: Path | str) -> Scan:
    """Read and check a scan file."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ScanError(f"{path}: {exc}") from exc

    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise ScanError(f"{path}: unknown key(s) {unknown}; a scan takes {sorted(_KEYS)}")
    for key in ("base", "folder", "name"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ScanError(f"{path}: {key} is required and must be a string")
    if ("vary" in data) == ("runs" in data):
        raise ScanError(f"{path}: give either [vary] or [[runs]], one of them")

    if "vary" in data:
        vary = data["vary"]
        if not isinstance(vary, dict) or not vary:
            raise ScanError(f"{path}: [vary] needs at least one key")
        for key, values in vary.items():
            if not isinstance(values, list) or not values:
                raise ScanError(f"{path}: vary.{key} must be a non-empty list")
        runs = [dict(zip(vary, combo)) for combo in itertools.product(*vary.values())]
    else:
        runs = data["runs"]
        if not isinstance(runs, list) or not runs or not all(isinstance(r, dict) and r for r in runs):
            raise ScanError(f"{path}: [[runs]] needs at least one table with at least one key")

    known = {f.name for f in fields(ShotParams)}
    bad = sorted({key for run in runs for key in run} - known)
    if bad:
        raise ScanError(
            f"{path}: {bad} are not shotfile fields; a scan varies shotfile fields only"
        )
    copy = data.get("copy", [])
    if not isinstance(copy, list) or not all(isinstance(c, str) for c in copy):
        raise ScanError(f"{path}: copy must be a list of file names")

    scan = Scan(
        source=path, base=data["base"], folder=data["folder"], name=data["name"],
        runs=runs, copy=copy, cases=bool(data.get("cases", True)),
        comparison=str(data.get("comparison") or path.stem),
    )
    names = [scan.run_name(run) for run in runs]
    for name in names:
        if not name or "/" in name or name in (".", ".."):
            raise ScanError(f"{path}: name = {scan.name!r} gives the folder name {name!r}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ScanError(
            f"{path}: name = {scan.name!r} gives the same folder for several runs "
            f"({', '.join(repeated)}); put every varied key in it"
        )
    return scan


@dataclass(frozen=True)
class PlannedRun:
    name: str                 #: case name, "<folder>/<run>", as cases.toml spells it
    run_dir: Path
    values: dict[str, Any]
    shotfile_text: str
    #: "new": no shotfile.py there. "same": it holds exactly this shotfile.
    #: "differs": it holds another one, and is left alone unless forced.
    status: str


def plan_scan(scan: Scan, root: Path | str) -> list[PlannedRun]:
    """What the scan would create under the campaign ``root``. Reads only."""
    root = Path(root)
    base = root / scan.base
    try:
        base_text = base.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScanError(f"{scan.source}: base shotfile {base} cannot be read ({exc})") from exc
    for name in scan.copy:
        if not (root / name).is_file():
            raise ScanError(f"{scan.source}: copy lists {name}, but {root / name} is not a file")

    planned = []
    for values in scan.runs:
        run_name = scan.run_name(values)
        try:
            text = shotfile_text_with(base_text, values, source=base)
        except ShotfileError as exc:
            raise ScanError(f"{scan.source}: {exc}") from exc
        run_dir = root / scan.folder / run_name
        existing = run_dir / "shotfile.py"
        if not existing.is_file():
            status = "new"
        elif existing.read_text(encoding="utf-8") == text:
            status = "same"
        else:
            status = "differs"
        planned.append(PlannedRun(
            name=f"{scan.folder}/{run_name}", run_dir=run_dir, values=dict(values),
            shotfile_text=text, status=status,
        ))
    return planned


def apply_plan(scan: Scan, root: Path | str, plan: list[PlannedRun], *, force: bool = False) -> list[PlannedRun]:
    """Create the run folders: shotfile.py and the copied files. Returns the
    runs written. A run already there with this same shotfile is not touched
    (it may be running); one with a different shotfile is skipped unless
    ``force``. Does not prepare or submit anything."""
    root = Path(root)
    written = []
    for run in plan:
        if run.status == "same" or (run.status == "differs" and not force):
            continue
        run.run_dir.mkdir(parents=True, exist_ok=True)
        (run.run_dir / "shotfile.py").write_text(run.shotfile_text, encoding="utf-8")
        for name in scan.copy:
            shutil.copy2(root / name, run.run_dir / Path(name).name)
        written.append(run)
    return written


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def cases_toml_additions(scan: Scan, plan: list[PlannedRun], existing: dict) -> tuple[str, list[str]]:
    """(text to append to cases.toml, notes) for this scan.

    ``existing`` is cases.toml parsed (``{}`` if there is none). A case
    already there is not repeated. A comparison already there is not edited:
    the note gives the lines to add by hand.
    """
    have_cases = existing.get("cases", {})
    have_comparisons = existing.get("comparisons", {})
    blocks, notes = [], []

    new_cases = [run for run in plan if run.name not in have_cases]
    for run in new_cases:
        what = ", ".join(f"{k} = {format_value(v)}" for k, v in run.values.items())
        blocks.append(
            f'[cases."{run.name}"]\n'
            f'note  = "{scan.source.name}: {what}"\n'
            "steps = { first_last = true }\n"
        )

    names = [run.name for run in plan]
    lines = [f"[comparisons.{scan.comparison}]", f'note  = "from {scan.source.name}"',
             "cases = [" + ", ".join(f'"{n}"' for n in names) + "]"]
    varied = scan.varied
    numeric = len(varied) == 1 and all(
        isinstance(run.values.get(varied[0]), (int, float))
        and not isinstance(run.values.get(varied[0]), bool) for run in plan
    )
    if numeric:
        lines.append("x_values = [" + ", ".join(_toml_value(r.values[varied[0]]) for r in plan) + "]")
        lines.append(f'x_label  = "{varied[0]}"')
    if scan.comparison not in have_comparisons:
        blocks.append("\n".join(lines) + "\n")
    elif list(have_comparisons[scan.comparison].get("cases", [])) != names:
        notes.append(
            f"[comparisons.{scan.comparison}] is already in cases.toml and lists other "
            "cases; it was not edited. To match this scan it should read:\n  "
            + "\n  ".join(lines[2:])
        )
    return ("\n".join(blocks), notes)


def update_cases_toml(scan: Scan, root: Path | str, plan: list[PlannedRun]) -> tuple[str, list[str]]:
    """Append this scan's cases and comparison to the campaign's cases.toml
    (created if absent). Returns (the text appended, notes)."""
    path = Path(root) / "cases.toml"
    existing: dict = {}
    current = ""
    if path.is_file():
        current = path.read_text(encoding="utf-8")
        try:
            existing = tomllib.loads(current)
        except tomllib.TOMLDecodeError as exc:
            raise ScanError(f"{path}: not valid TOML, so nothing was added ({exc})") from exc
    text, notes = cases_toml_additions(scan, plan, existing)
    if text:
        separator = "" if not current or current.endswith("\n\n") else ("\n" if current.endswith("\n") else "\n\n")
        path.write_text(current + separator + text, encoding="utf-8")
    return text, notes
