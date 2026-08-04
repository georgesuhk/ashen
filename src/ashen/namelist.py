"""Reading and editing Fortran namelists.

Replaces three independent implementations that had drifted apart:

* ``run_jorek_util.py:9   replace_field_in_files``
* ``basics.py:220         update_field_in_file``
* ``io.py:266             update_bracket_value``

plus ``io.py:220 extract_from_file`` on the read side.

Two behaviours are deliberately different from all of them:

**Edits must match exactly once.** The old code replaced every match silently,
so a duplicated key was edited in several places at once with no warning. This
mirrors the guard in JOREK's own ``util/setinput.sh``, which refuses to write
unless the match count is one.

**Numbers are formatted properly.** The old ``f"{v:.16g}".replace("e", ".d")``
produced malformed Fortran whenever the mantissa had a decimal point --
``1.5e20`` became ``1.5.d+20`` -- and silently emitted single-precision
literals (``0.03``) or integers (``100``) into double-precision fields.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "NamelistError",
    "effective_fields",
    "format_boundary_block",
    "fortran_literal",
    "parse_value",
    "read_field",
    "set_boundary_block",
    "set_fields",
    "write_boundary_file",
]

BOUNDARY_ARRAYS = ("r_boundary", "z_boundary", "psi_boundary")


class NamelistError(RuntimeError):
    """Raised when a namelist cannot be parsed or an edit is unsafe."""


# --- value formatting --------------------------------------------------------


def fortran_literal(value: Any) -> str:
    """Render a Python value as a Fortran namelist literal.

    Floats always become double-precision exponent form (``1.5d20``), which is
    what the old ``.replace("e", ".d")`` was reaching for but got wrong for any
    mantissa containing a decimal point.

    Strings pass through untouched: callers legitimately supply raw Fortran
    such as ``.t.`` or ``'in_bnd.dat'``.
    """
    if isinstance(value, bool):
        return ".t." if value else ".f."
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise NamelistError(f"cannot write non-finite value {value!r}")
        # repr() gives the shortest string that round-trips exactly. Formatting
        # to a fixed precision instead would surface representation noise
        # (1e-6 -> 9.9999999999999995e-07) and bloat every diff.
        mantissa, _, exponent = repr(value).partition("e")
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}d{int(exponent or 0)}"
    if isinstance(value, (list, tuple)):
        return ", ".join(fortran_literal(v) for v in value)
    return str(value)


def parse_value(text: str) -> Any:
    """Parse a namelist value into a Python object.

    Handles Fortran doubles (``1.d-6``), logicals (``.t.``), quoted strings and
    comma-separated lists. Anything unrecognised is returned as a stripped
    string, so the parser never loses information it cannot interpret.
    """
    text = text.strip().rstrip(",").strip()
    if not text:
        return ""

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) > 1:
        return tuple(parse_value(p) for p in parts)

    lowered = text.lower()
    if lowered in (".t.", ".true.", "t", "true"):
        return True
    if lowered in (".f.", ".false.", "f", "false"):
        return False

    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]

    try:
        return float(re.sub(r"[dD]", "e", text))
    except ValueError:
        return text


# --- parsing -----------------------------------------------------------------

# A key is a name optionally followed by an index, e.g. `R_boundary(  1)`.
# The value runs until the next `key =` on the same line, or end of line --
# boundary lines carry three assignments each.
_ASSIGNMENT = re.compile(
    r"([A-Za-z_]\w*(?:\s*\([^)]*\))?)\s*=\s*"
    r"(.*?)(?=(?:,\s*)?[A-Za-z_]\w*(?:\s*\([^)]*\))?\s*=|$)"
)


def _normalise_key(key: str) -> str:
    """Fortran is case-insensitive and indifferent to spacing inside indices."""
    return re.sub(r"\s+", "", key).lower()


def _strip_comment(line: str) -> str:
    return line.split("!", 1)[0]


def effective_fields(path: Path | str) -> dict[str, Any]:
    """The key/value mapping a Fortran reader would actually see.

    Duplicate keys resolve **last-wins**, matching Fortran namelist semantics.
    That makes this the right comparator for the golden tests: the shipped
    ``in_eq`` template carries seven stacked ``n_boundary`` blocks, so byte
    comparison is meaningless, while the effective mapping is exact.
    """
    fields: dict[str, Any] = {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = _strip_comment(line).strip()
        if not stripped or stripped.startswith("&"):
            continue
        for match in _ASSIGNMENT.finditer(stripped):
            key = _normalise_key(match.group(1))
            value = match.group(2).strip().rstrip(",").strip()
            if value:
                fields[key] = parse_value(value)
    return fields


def read_field(path: Path | str, field: str, cast: type | None = None) -> Any:
    """Read a single field. Raises if it is absent."""
    fields = effective_fields(path)
    key = _normalise_key(field)
    if key not in fields:
        raise NamelistError(f"{path}: no field named {field!r}")
    value = fields[key]
    return cast(value) if cast is not None else value


# --- editing -----------------------------------------------------------------


def _field_pattern(field: str) -> re.Pattern[str]:
    # Case-insensitive, and the value stops at a comment so it is preserved.
    return re.compile(
        rf"^(\s*{re.escape(field)}\s*=\s*)([^!\n]*?)(\s*!.*)?$",
        flags=re.IGNORECASE,
    )


def _insertion_index(lines: list[str]) -> int:
    """Where a new key goes: before the boundary block, else before ``&end``.

    Preserves the placement logic of ``replace_field_in_files``.
    """
    for i, line in enumerate(lines):
        if "n_boundary" in line.lower():
            return i
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("&end"):
            return i
    return len(lines)


def set_fields(
    paths: Path | str | Iterable[Path | str],
    updates: dict[str, Any],
    *,
    create_missing: bool = False,
) -> None:
    """Set fields in one or more namelists, in place.

    Args:
        paths: a namelist path, or several.
        updates: field name -> value. Values are rendered with
            :func:`fortran_literal`; pass a string to write raw Fortran.
        create_missing: insert absent fields rather than failing. Needed for
            the RE parameters, which the shipped templates do not declare.

    Raises:
        NamelistError: if a field is absent and ``create_missing`` is False, or
            if it appears more than once. A duplicated key is always a mistake
            worth surfacing -- the old code edited every occurrence silently.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    for raw_path in paths:
        path = Path(raw_path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError as exc:
            raise NamelistError(f"{path}: cannot be read -- {exc}") from exc

        for field, value in updates.items():
            literal = fortran_literal(value)
            pattern = _field_pattern(field)

            hits = [i for i, line in enumerate(lines) if pattern.match(line)]

            if len(hits) > 1:
                locations = ", ".join(str(i + 1) for i in hits)
                raise NamelistError(
                    f"{path}: {field!r} appears {len(hits)} times "
                    f"(lines {locations}); refusing to guess which to edit."
                )

            if not hits:
                if not create_missing:
                    raise NamelistError(
                        f"{path}: no field named {field!r}. Pass "
                        "create_missing=True to insert it."
                    )
                index = _insertion_index(lines)
                if index == len(lines) and lines and not lines[-1].endswith("\n"):
                    lines[-1] += "\n"
                lines.insert(index, f" {field} = {literal}\n")
                continue

            index = hits[0]
            match = pattern.match(lines[index])
            assert match is not None
            comment = match.group(3) or ""
            newline = "\n" if lines[index].endswith("\n") else ""
            lines[index] = f"{match.group(1)}{literal}{comment}{newline}"

        path.write_text("".join(lines), encoding="utf-8")


# --- boundary blocks ---------------------------------------------------------


def format_boundary_block(
    R: Sequence[float],
    Z: Sequence[float],
    psi: Sequence[float],
    float_fmt: str = ".6f",
) -> list[str]:
    """Render a boundary block as namelist lines, without the trailing newline.

    Single source of truth for a layout that ``write_boundary_file`` and
    ``write_boundary_to_namelist`` previously produced with two independent
    implementations and two different default precisions.
    """
    if not (len(R) == len(Z) == len(psi)):
        raise NamelistError(
            f"R, Z, psi must have the same length; got {len(R)}, {len(Z)}, {len(psi)}"
        )
    lines = [f" n_boundary = {len(R)}"]
    for i, (r, z, p) in enumerate(zip(R, Z, psi), start=1):
        lines.append(
            f"R_boundary({i:3d}) = {format(r, float_fmt)}, "
            f"Z_boundary({i:3d}) = {format(z, float_fmt)}, "
            f"psi_boundary({i:3d}) = {format(p, float_fmt)}"
        )
    return lines


def _is_boundary_line(line: str) -> bool:
    stripped = _strip_comment(line).strip().lower()
    if not stripped:
        return False
    if stripped.startswith("n_boundary"):
        return True
    return any(stripped.startswith(name + "(") for name in BOUNDARY_ARRAYS)


def set_boundary_block(
    path: Path | str,
    R: Sequence[float],
    Z: Sequence[float],
    psi: Sequence[float],
    float_fmt: str = ".6f",
) -> None:
    """Write a boundary block into a namelist, **replacing** any existing one.

    The old ``write_boundary_to_namelist`` appended, which is why the shipped
    ``template/copy/in_eq`` accumulated seven stacked ``n_boundary`` blocks.
    Fortran's last-wins rule made that harmless but unreadable, and it grew by
    one block every time the template was regenerated from an output.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    kept = [line for line in lines if not _is_boundary_line(line)]

    end_idx = None
    for i, line in enumerate(kept):
        if line.strip().lower().startswith("&end"):
            end_idx = i
            break
    if end_idx is None:
        raise NamelistError(f"{path}: no '&end' marker found")

    block = format_boundary_block(R, Z, psi, float_fmt)
    merged = kept[:end_idx] + block + [""] + kept[end_idx:]
    path.write_text("\n".join(merged) + "\n", encoding="utf-8")


def write_boundary_file(
    path: Path | str,
    R: Sequence[float],
    Z: Sequence[float],
    psi: Sequence[float],
    float_fmt: str = ".2f",
) -> None:
    """Write a standalone boundary file (``in_bnd``).

    Same block as :func:`set_boundary_block`, without the leading space on
    ``n_boundary`` that the namelist form carries.
    """
    block = format_boundary_block(R, Z, psi, float_fmt)
    block[0] = block[0].lstrip()
    Path(path).write_text("\n".join(block) + "\n", encoding="utf-8")
