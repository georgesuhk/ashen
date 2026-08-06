"""Reading scalar values out of a JOREK run log.

Ports ``castor3d/util/io.py:220 extract_from_file``, with one deliberate
change: the legacy function swallows every failure (bad path, absent field,
malformed value) and returns ``math.nan`` (``io.py:259-262``), printing a
message that is easy to miss in batch output. The only caller that mattered,
``gather_profiles.py:130 postproc_get_q``, then fed that ``nan`` straight into
a q-profile calculation -- silently poisoning downstream physics rather than
stopping.

Here, failure raises :class:`LogfileError` naming the file and field. This is
what :func:`r_axis` is for: four call sites in the legacy plotting code
hardcode ``R0 = 1.36`` (``data_jorek.py:409,452,497,531``) instead of using
this extraction, which the one correct caller (``postproc_get_q``) already
demonstrates works. A wrong or unreadable log should stop a connection-length
calculation, not quietly scale every result by a guessed constant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

__all__ = ["LogfileError", "extract_from_file", "r_axis"]

T = TypeVar("T")


class LogfileError(RuntimeError):
    """A field could not be found or parsed in a log file."""


def extract_from_file(
    path: Path | str,
    fieldname: str,
    *,
    val_type: Callable[[str], T] = float,
    delimiter: str = "=",
    occurrence: int | str = 1,
) -> T:
    """Find ``fieldname = value`` in ``path`` and parse ``value``.

    A line matches when ``fieldname`` appears on the left of ``delimiter``
    (not merely anywhere in the line, which would also match a value that
    happens to contain the field's name). ``occurrence`` selects which match
    to use, 1-indexed; ``"last"`` takes the final one, which is what a
    restarted run's log needs since the same field is logged once per restart
    stage.

    Raises :class:`LogfileError` if the file is unreadable, the field never
    matches, or the requested occurrence doesn't exist -- never returns a
    placeholder value.
    """
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise LogfileError(f"{path}: cannot be read -- {exc}") from exc

    count = 0
    last_value: str | None = None
    for line in lines:
        if fieldname not in line:
            continue
        parts = line.split(delimiter, 1)
        if len(parts) < 2 or fieldname not in parts[0]:
            continue
        raw = parts[1].strip().split()
        if not raw:
            continue
        value = raw[0]
        if occurrence == "last":
            last_value = value
            continue
        count += 1
        if count == occurrence:
            return _parse(value, val_type, path, fieldname)

    if occurrence == "last" and last_value is not None:
        return _parse(last_value, val_type, path, fieldname)

    raise LogfileError(
        f"{path}: field {fieldname!r} occurrence {occurrence!r} not found"
    )


def _parse(value: str, val_type: Callable[[str], T], path: Path, fieldname: str) -> T:
    try:
        return val_type(value)
    except (TypeError, ValueError) as exc:
        raise LogfileError(
            f"{path}: field {fieldname!r} = {value!r} could not be parsed "
            f"as {getattr(val_type, '__name__', val_type)} -- {exc}"
        ) from exc


def r_axis(log_path: Path | str) -> float:
    """The magnetic axis major radius JOREK logged for this equilibrium.

    Ports the one already-correct extraction in the legacy tree
    (``gather_profiles.py:130 postproc_get_q``, ``extract_from_file(log,
    "R_axis", occurrence=1)``), promoted to a named helper so it can replace
    every hardcoded ``R0`` site rather than being reimplemented at each one.
    """
    return extract_from_file(log_path, "R_axis", occurrence=1)
