"""Cross-case comparisons for ``bin/plot``.

Some figures compare *different runs* against each other rather than
different steps of one run -- e.g. an eta scan's field-line theta-crossing
distributions, one panel per run. :mod:`ashen.cases` has no notion of this:
a ``Case`` is one run folder. A comparison is a named group of already-defined
cases, read from the same ``cases.toml`` rather than a second file, since it
is never independently meaningful -- every member it names must already be a
``[cases.*]`` entry in that file.

Each member's own per-diag step override still does the per-run time-window
selection (e.g. ``[cases.NAME.theta_hist] steps = [...]``) -- a comparison
only supplies which cases to pool and how to label them, reusing
:meth:`ashen.cases.Case.steps_for` rather than inventing a parallel mechanism.

A comparison can also carry an explicit ``x_values`` (parallel to ``cases``)
for plotting a derived scalar against a scan parameter across its members --
e.g. wetted fraction vs. eta (:mod:`ashen.plotting.wetted_fraction`).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ashen.cases import Case, CasesError

__all__ = ["Comparison", "load_comparisons"]


@dataclass(frozen=True)
class Comparison:
    name: str
    #: Member case names, in panel order.
    cases: list[str]
    #: Parallel to `cases`; defaults to the case names themselves if omitted.
    labels: list[str] = field(default_factory=list)
    note: str = ""
    n_cols: int = 4
    #: Parallel to `cases`: an explicit numeric value per member (e.g. each
    #: case's resistivity), for a "derived scalar vs. scan parameter" plot
    #: (ashen.plotting.wetted_fraction). Deliberately not inferred from run
    #: folder names (e.g. parsing "eta1e-3" out of "eta1e-3_RE") -- CLAUDE.md
    #: flags CASTOR3D's directory-name parsing as a hazard this project
    #: exists to not repeat; a folder rename must not silently change a
    #: plotted x-value. None (default) if the comparison isn't used for this
    #: kind of plot.
    x_values: list[float] | None = None
    #: Axis label for `x_values`, e.g. "$\\eta$ [$\\Omega \\cdot$ m]".
    x_label: str = ""

    def labelled_cases(self) -> list[tuple[str, str]]:
        """``[(label, case_name), ...]`` in panel order -- what a renderer
        iterates to build panels."""
        labels = self.labels or self.cases
        return list(zip(labels, self.cases))


def load_comparisons(path: Path | str, cases: dict[str, Case]) -> dict[str, Comparison]:
    """Parse ``[comparisons.*]`` tables from ``cases.toml``.

    ``cases`` is the already-loaded result of :func:`ashen.cases.load_cases`
    against the same file, passed in rather than re-derived here so every
    member name is validated against exactly what the caller resolved --
    ``load_cases`` and this function reading the file independently could
    otherwise disagree if one were extended and the other weren't.

    Returns ``{}`` if the file has no ``[comparisons.*]`` section -- this is
    not an error, since most cases.toml files will not define any.
    """
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CasesError(f"{path}: not found") from None
    except tomllib.TOMLDecodeError as exc:
        raise CasesError(f"{path}: malformed TOML -- {exc}") from exc

    raw_comparisons = data.get("comparisons", {})

    comparisons: dict[str, Comparison] = {}
    for name, raw in raw_comparisons.items():
        unknown = sorted(
            set(raw) - {"cases", "labels", "note", "n_cols", "x_values", "x_label"}
        )
        if unknown:
            raise CasesError(
                f"{path}: comparison {name!r} has unknown key(s): {unknown}"
            )

        members = raw.get("cases")
        if not members:
            raise CasesError(
                f"{path}: comparison {name!r} has no 'cases' (or it is empty)"
            )
        members = [str(c) for c in members]

        missing = [c for c in members if c not in cases]
        if missing:
            raise CasesError(
                f"{path}: comparison {name!r} names undefined case(s) {missing}"
            )

        labels = [str(l) for l in raw.get("labels", [])]
        if labels and len(labels) != len(members):
            raise CasesError(
                f"{path}: comparison {name!r} has {len(labels)} labels for "
                f"{len(members)} cases; labels must be omitted or match cases 1:1"
            )

        x_values = None
        if "x_values" in raw:
            x_values = [float(v) for v in raw["x_values"]]
            if len(x_values) != len(members):
                raise CasesError(
                    f"{path}: comparison {name!r} has {len(x_values)} x_values "
                    f"for {len(members)} cases; x_values must match cases 1:1"
                )

        comparisons[name] = Comparison(
            name=name,
            cases=members,
            labels=labels,
            note=str(raw.get("note", "")),
            n_cols=int(raw.get("n_cols", 4)),
            x_values=x_values,
            x_label=str(raw.get("x_label", "")),
        )

    return comparisons
