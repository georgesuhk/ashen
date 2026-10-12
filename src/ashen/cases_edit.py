"""Edits to cases.toml that keep the rest of the file as the user wrote it.

tomllib only reads, and writing the parsed file back would drop every
comment. So an edit here changes the text of one value and is then checked:
the new text must parse to exactly the old file with that one value
changed, or nothing is written.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
import tomllib
from pathlib import Path

from ashen.cases import CasesError

__all__ = ["format_steps_table", "set_case_steps"]

_STEPS_KEY = re.compile(r"^(\s*)steps\s*=")


def format_steps_table(steps: dict) -> str:
    """A steps range table as cases.toml writes it: ``{ start = 0, step = 400 }``."""
    if not steps:
        return "{}"
    return "{ " + ", ".join(f"{key} = {int(value)}" for key, value in steps.items()) + " }"


def _header(line: str) -> dict | None:
    """What a ``[table]`` header line opens, or None for any other line."""
    stripped = line.strip()
    if not stripped.startswith("[") or stripped.startswith("[["):
        return None
    try:
        return tomllib.loads(stripped)
    except tomllib.TOMLDecodeError:
        return None


def _value_end(lines: list[str], start: int, stop: int) -> int:
    """The line after the ``steps = ...`` value starting at ``start``: a
    list may run over several lines."""
    for end in range(start + 1, stop + 1):
        try:
            parsed = tomllib.loads("".join(lines[start:end]))
        except tomllib.TOMLDecodeError:
            continue
        if set(parsed) == {"steps"}:
            return end
    raise CasesError("could not find where the case's steps value ends")


def set_case_steps(path: Path | str, name: str, steps: dict) -> str:
    """Set ``steps`` of ``[cases."<name>"]`` in cases.toml to a range table.

    Replaces the case's own ``steps`` (a list over several lines too), adds
    the key to a case that took its steps from ``[defaults]``, or adds the
    case at the end of the file. Every other line is left as it is; a
    comment on the replaced ``steps`` line itself goes with the old value.
    Per-diag ``[cases.NAME.<diag>]`` step overrides are not touched.

    Returns what was done: "replaced", "added to the case" or "added the
    case". Raises CasesError, having written nothing, if the file is not
    valid TOML or the edit would change anything else.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        before = tomllib.loads(text)
    except FileNotFoundError:
        raise CasesError(f"{path}: not found") from None
    except tomllib.TOMLDecodeError as exc:
        raise CasesError(f"{path}: malformed TOML -- {exc}") from exc

    new_line = f"steps = {format_steps_table(steps)}\n"
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    wanted = {"cases": {name: {}}}
    head = next((i for i, line in enumerate(lines) if _header(line) == wanted), None)
    if head is None:
        gap = [] if not lines or not lines[-1].strip() else ["\n"]
        lines += gap + [f"[cases.{json.dumps(name)}]\n", new_line]
        done = "added the case"
    else:
        stop = next(
            (i for i in range(head + 1, len(lines)) if _header(lines[i]) is not None), len(lines)
        )
        at = next((i for i in range(head + 1, stop) if _STEPS_KEY.match(lines[i])), None)
        if at is None:
            lines.insert(head + 1, new_line)
            done = "added to the case"
        else:
            indent = _STEPS_KEY.match(lines[at]).group(1)
            lines[at:_value_end(lines, at, stop)] = [indent + new_line]
            done = "replaced"

    new_text = "".join(lines)
    expected = copy.deepcopy(before)
    expected.setdefault("cases", {}).setdefault(name, {})["steps"] = dict(steps)
    try:
        after = tomllib.loads(new_text)
    except tomllib.TOMLDecodeError:
        after = None
    if after != expected:
        raise CasesError(
            f"{path}: could not set steps of case {name!r} without changing something else "
            "in the file; nothing was written. Edit the case's steps by hand."
        )

    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(new_text)
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return done
