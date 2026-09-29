"""Restart-step padding: the one place ashen decides how wide a step index is.

JOREK writes a step index into filenames at one of two widths --
``jorek08002.h5`` or ``jorek008002.h5`` -- and the two halves of JOREK are
not symmetric about it (see :data:`JOREK_PAD_WIDTHS`). Every question ashen
asks about a padded filename is one of three kinds, and each has exactly one
answer here:

1. **Finding a file JOREK wrote** (restarts, ``zeroD_quantities_*``,
   ``qprofile_*``, ...): accept either width, the run's own first.
   :func:`find_step_file` / :func:`resolve_step_file`.
2. **Naming a file ashen writes** (the Poincare, profile and jorek2_four
   caches): one width only, the run's own -- ``RunPaths.step_str``. A second
   accepted spelling would be a way to end up with two caches for one step.
3. **Naming a file a JOREK binary will look up by a name it builds itself**
   (e.g. staging restarts for the particle tracer, whose field reader tries
   ``rst_file_ind_fmt(1)`` only): the width *that build* writes, read from
   the source tree it was compiled from -- :func:`source_pad_width`.

Nothing outside this module should format a step index with a literal width
or re-pad a filename by hand.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

__all__ = [
    "DEFAULT_PAD_WIDTH",
    "JOREK_PAD_WIDTHS",
    "PaddingError",
    "detect_pad_width",
    "find_step_file",
    "resolve_step_file",
    "restart_name",
    "restart_steps",
    "source_pad_width",
    "step_name_variants",
    "step_str",
]

#: Only a fallback for synthetic cases. Real runs must sniff the width.
DEFAULT_PAD_WIDTH = 6

#: Step-index widths a JOREK build may use in a filename, preferred first.
#: Mirrors ``rst_file_ind_fmt = (/'(a,i6.6)', '(a,i5.5)'/)``
#: (``communication/mod_import_restart.f90:5``).
#:
#: The two halves of JOREK are not symmetric about this, which is what makes
#: a mismatch so quiet. *Importing* a restart loops over both entries
#: (``mod_import_restart.f90:2686``), so ``jorek08002.h5`` and
#: ``jorek008002.h5`` are equally readable. *Naming an output* takes
#: ``rst_file_ind_fmt(1)`` alone (``step_range_string``,
#: ``exec_commands.f90:1068``) -- one fixed width, whatever the restarts
#: happen to use. So a build reads your run happily and then writes
#: ``..._s008002.dat`` next to ``jorek08002.h5``, and a reader that derived
#: its width from the restart filenames looks for a file that is right there
#: under a name one character longer.
#:
#: Different builds differ in which entry is first, so this is a property of
#: the binary in use, not of the run -- :func:`source_pad_width` reads it.
JOREK_PAD_WIDTHS = (6, 5)

_STEP_DIGITS_RE = re.compile(r"\d+")
_RESTART_RE = re.compile(r"^jorek.*?(\d+)\.h5$")

#: The declaration :func:`source_pad_width` reads, relative to a JOREK tree.
_IMPORT_RESTART_SOURCE = Path("communication") / "mod_import_restart.f90"

#: First entry of ``rst_file_ind_fmt = (/'(a,i6.6)', ...``: the width a build
#: names its own outputs with.
_FIRST_FMT_RE = re.compile(
    r"rst_file_ind_fmt\s*\(\s*\d+\s*\)\s*=\s*\(/\s*['\"]\(\s*a\s*,\s*i(\d+)",
    re.IGNORECASE,
)


class PaddingError(RuntimeError):
    """Raised when a step padding width cannot be determined."""


def step_str(step: int | float, width: int = DEFAULT_PAD_WIDTH) -> str:
    """Zero-pad a step index. Prefer :meth:`RunPaths.step_str`."""
    return f"{int(step):0{width}d}"


def restart_name(
    step: int | float, width: int, prefix: str = "jorek", ext: str = ".h5"
) -> str:
    """A restart's filename at one specific width -- e.g. ``jorek08002.h5``."""
    return f"{prefix}{step_str(step, width)}{ext}"


def step_name_variants(
    name: str, step: int | float, widths: tuple[int, ...] = JOREK_PAD_WIDTHS
) -> list[str]:
    """``name`` with the step index re-padded to each width in ``widths``.

    Only digit runs that *are* the step are touched, so
    ``fluxsurface_at_psi_0.200_s08002.dat`` re-pads the ``08002`` and leaves
    the ``0`` and ``200`` of the psi value alone. A psi value that happened
    to read as the step number would be rewritten too; with psi formatted to
    three decimals and steps in the thousands that cannot arise.

    The original spelling is never included -- these are the *alternatives*
    to whatever the caller already tried.
    """
    target = int(step)
    variants: list[str] = []
    for width in widths:
        padded = step_str(target, width)
        variant = _STEP_DIGITS_RE.sub(
            lambda m: padded if int(m.group()) == target else m.group(), name
        )
        if variant != name and variant not in variants:
            variants.append(variant)
    return variants


def find_step_file(canonical: Path | str, step: int | float) -> Path | None:
    """The file JOREK actually wrote for ``step``, or None if there is none.

    ``canonical`` (the spelling the caller expects) wins whenever it exists;
    only then are the other widths tried, so an exact match is never
    displaced by a variant that also happens to be there. Only the filename
    is re-padded, never a directory component.

    For files **JOREK writes** only -- see the module docstring's rule 1.
    """
    canonical = Path(canonical)
    if canonical.is_file():
        return canonical
    for name in step_name_variants(canonical.name, step):
        alt = canonical.with_name(name)
        if alt.is_file():
            return alt
    return None


def resolve_step_file(canonical: Path | str, step: int | float) -> Path:
    """:func:`find_step_file`, falling back to ``canonical`` itself.

    Returning the canonical name rather than None keeps the result usable
    as a write target, and keeps "expected <path>" messages predictable
    when nothing exists yet.
    """
    canonical = Path(canonical)
    return find_step_file(canonical, step) or canonical


def detect_pad_width(directory: Path | str = ".") -> int:
    """Infer a run's zero-pad width from ``jorek*.h5`` files in ``directory``.

    Takes the majority width, so a stray differently-named file does not
    change the answer.
    """
    widths = [
        len(match.group(1))
        for path in Path(directory).glob("jorek*.h5")
        if (match := _RESTART_RE.match(path.name))
    ]
    if not widths:
        raise PaddingError(
            f"{directory}: no JOREK restart files (jorek*.h5) to infer the "
            "step padding width from."
        )
    return Counter(widths).most_common(1)[0][0]


def restart_steps(directory: Path | str) -> list[int]:
    """Every step with a ``jorek<step>.h5`` in ``directory``, ascending.

    A step present at both widths is listed once -- which file to use for
    it is :func:`find_step_file`'s decision, not this listing's.
    """
    return sorted({
        int(match.group(1))
        for path in Path(directory).glob("jorek*.h5")
        if (match := _RESTART_RE.match(path.name))
    })


def source_pad_width(jorek_tree: Path | str) -> int:
    """The width binaries built from ``jorek_tree`` name their files with.

    Read from the first entry of ``rst_file_ind_fmt`` in
    ``communication/mod_import_restart.f90`` -- the format every output name
    is written with, and the *only* one some readers try (the particle
    tracer's ``read_jorek_fields_interp_linear`` does not fall back). This
    is the width to stage restarts at for such a binary; see the module
    docstring's rule 3.
    """
    source = Path(jorek_tree) / _IMPORT_RESTART_SOURCE
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PaddingError(
            f"cannot read {source} to find this JOREK build's step width: {exc}"
        ) from exc
    match = _FIRST_FMT_RE.search(text)
    if match is None:
        raise PaddingError(
            f"{source}: no rst_file_ind_fmt declaration found -- cannot tell "
            "which step width this JOREK build writes"
        )
    return int(match.group(1))
