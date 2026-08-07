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

A comparison names its members one of two ways, never both:

* ``cases`` (flat) -- one series, one point per member. What every comparison
  used before ``datasets`` existed, and still the only form the theta_hist
  grid comparison understands (one panel per member).
* ``datasets`` (nested ``[comparisons.NAME.datasets.DATASET]`` tables) -- more
  than one *related* scan sharing the same x-axis, e.g. a resistivity scan
  repeated under two different profile assumptions ("normal" vs "rho19").
  Each dataset is its own group of cases with its own points; the
  wetted_fraction plot overlays them on one axes, one colour and legend
  entry per dataset (:func:`ashen.plotting.wetted_fraction.
  plot_wetted_fraction_datasets`). A dataset's own ``x_values``/``labels``
  fall back to the comparison's own, since the whole point of grouping scans
  this way is that they usually share one scan parameter -- set them on a
  dataset only when that one scan's points genuinely differ.

A comparison can also override the analysis parameters that produce that
scalar (``theta_target_psi``, ``theta_bins``, ``theta_psi_n_range``,
``theta_wetted_threshold``) for every member **uniformly**, rather than each
member case needing its own matching copy (or all cases sharing one
``[defaults]``, which would apply to non-comparison uses of those cases too).
A scan is only an apples-to-apples comparison if every point was computed
the same way, so these belong to the comparison, not scattered across its
members. Precedence, most specific wins: CLI flag (e.g.
``--theta_target_psi``) > this comparison's own setting > the member case's
own setting > the diagnostic's built-in default.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ashen.cases import Case, CasesError

__all__ = ["Comparison", "Dataset", "load_comparisons"]


@dataclass(frozen=True)
class Dataset:
    """One named group of cases within a ``datasets``-style comparison --
    e.g. the "rho19" scan, alongside a sibling "normal" scan, both under the
    same comparison. See the module docstring for when to use this instead
    of a comparison's own flat ``cases``.
    """

    name: str
    #: Member case names, in point order.
    cases: list[str]
    #: Parallel to `cases`; defaults to the case names themselves if omitted.
    labels: list[str] = field(default_factory=list)
    #: Parallel to `cases`. Falls back to the parent comparison's `x_values`
    #: if not set here -- see the module docstring.
    x_values: list[float] | None = None
    #: Explicit legend colour for this dataset's series. `None` (default)
    #: means the renderer assigns one from `ashen.plotting.colors.
    #: DISCRETE_PALETTE`, cycling by this dataset's position among its
    #: siblings.
    color: str | None = None

    def labelled_cases(self) -> list[tuple[str, str]]:
        """``[(label, case_name), ...]`` in point order."""
        labels = self.labels or self.cases
        return list(zip(labels, self.cases))


@dataclass(frozen=True)
class Comparison:
    name: str
    #: Member case names, in panel order. Empty when `datasets` is used
    #: instead -- exactly one of the two is ever set, never both.
    cases: list[str] = field(default_factory=list)
    #: Parallel to `cases`; defaults to the case names themselves if omitted.
    labels: list[str] = field(default_factory=list)
    #: Named sub-scans sharing this comparison's x-axis -- see the module
    #: docstring. Empty when the comparison uses flat `cases` instead.
    datasets: dict[str, Dataset] = field(default_factory=dict)
    note: str = ""
    n_cols: int = 4
    #: Parallel to `cases` (flat mode), or the shared default every dataset
    #: falls back to (datasets mode): an explicit numeric value per member
    #: (e.g. each case's resistivity), for a "derived scalar vs. scan
    #: parameter" plot (ashen.plotting.wetted_fraction). Deliberately not
    #: inferred from run folder names (e.g. parsing "eta1e-3" out of
    #: "eta1e-3_RE") -- CLAUDE.md flags CASTOR3D's directory-name parsing as
    #: a hazard this project exists to not repeat; a folder rename must not
    #: silently change a plotted x-value. None (default) if the comparison
    #: isn't used for this kind of plot.
    x_values: list[float] | None = None
    #: Axis label for `x_values`, e.g. "$\\eta$ [$\\Omega \\cdot$ m]".
    x_label: str = ""
    #: Uniform overrides applied to every member case -- see the module
    #: docstring for why these live on the comparison rather than each case.
    #: None (default) falls through to each case's own setting.
    theta_target_psi: float | None = None
    theta_bins: int | None = None
    theta_psi_n_range: list[float] | None = None
    theta_wetted_threshold: float | None = None

    def labelled_cases(self) -> list[tuple[str, str]]:
        """``[(label, case_name), ...]`` in panel order -- what a renderer
        iterates to build panels. Flat-mode (`cases`) only; a `datasets`
        comparison has no single flat panel order -- iterate `datasets`
        instead."""
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
            set(raw) - {
                "cases", "labels", "datasets", "note", "n_cols", "x_values", "x_label",
                "theta_target_psi", "theta_bins", "theta_psi_n_range",
                "theta_wetted_threshold",
            }
        )
        if unknown:
            raise CasesError(
                f"{path}: comparison {name!r} has unknown key(s): {unknown}"
            )

        has_cases = bool(raw.get("cases"))
        has_datasets = bool(raw.get("datasets"))
        if has_cases and has_datasets:
            raise CasesError(
                f"{path}: comparison {name!r} sets both 'cases' and "
                "'datasets' -- use exactly one (see the module docstring)"
            )
        if not has_cases and not has_datasets:
            raise CasesError(
                f"{path}: comparison {name!r} has no 'cases' (or it is "
                "empty) and no 'datasets'"
            )

        # Parsed once, regardless of mode: flat mode validates it against
        # `members` below; datasets mode treats it as each dataset's shared
        # fallback, validated per-dataset instead (see the datasets loop).
        x_values = None
        if "x_values" in raw:
            x_values = [float(v) for v in raw["x_values"]]

        members: list[str] = []
        labels: list[str] = []
        datasets: dict[str, Dataset] = {}

        if has_cases:
            members = [str(c) for c in raw["cases"]]

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

            if x_values is not None and len(x_values) != len(members):
                raise CasesError(
                    f"{path}: comparison {name!r} has {len(x_values)} x_values "
                    f"for {len(members)} cases; x_values must match cases 1:1"
                )
        else:
            if "labels" in raw:
                raise CasesError(
                    f"{path}: comparison {name!r} sets 'labels' but uses "
                    "'datasets' -- put labels on each dataset instead"
                )
            for ds_name, raw_ds in raw["datasets"].items():
                datasets[ds_name] = _parse_dataset(
                    path, name, ds_name, raw_ds, cases, default_x_values=x_values,
                )

        theta_target_psi = None
        if "theta_target_psi" in raw:
            theta_target_psi = float(raw["theta_target_psi"])

        theta_bins = None
        if "theta_bins" in raw:
            theta_bins = int(raw["theta_bins"])

        theta_psi_n_range = None
        if "theta_psi_n_range" in raw:
            spec = raw["theta_psi_n_range"]
            if not (isinstance(spec, list) and len(spec) == 2):
                raise CasesError(
                    f"{path}: comparison {name!r} theta_psi_n_range must be "
                    f"[min, max], got {spec!r}"
                )
            lo, hi = float(spec[0]), float(spec[1])
            if not lo < hi:
                raise CasesError(
                    f"{path}: comparison {name!r} theta_psi_n_range must "
                    f"satisfy min < max, got [{lo}, {hi}]"
                )
            theta_psi_n_range = [lo, hi]

        theta_wetted_threshold = None
        if "theta_wetted_threshold" in raw:
            theta_wetted_threshold = float(raw["theta_wetted_threshold"])
            if theta_wetted_threshold <= 0:
                raise CasesError(
                    f"{path}: comparison {name!r} theta_wetted_threshold "
                    f"must be positive, got {theta_wetted_threshold}"
                )

        comparisons[name] = Comparison(
            name=name,
            cases=members,
            labels=labels,
            datasets=datasets,
            note=str(raw.get("note", "")),
            n_cols=int(raw.get("n_cols", 4)),
            x_values=x_values,
            x_label=str(raw.get("x_label", "")),
            theta_target_psi=theta_target_psi,
            theta_bins=theta_bins,
            theta_psi_n_range=theta_psi_n_range,
            theta_wetted_threshold=theta_wetted_threshold,
        )

    return comparisons


def _parse_dataset(
    path: Path,
    comparison_name: str,
    dataset_name: str,
    raw: dict,
    cases: dict[str, Case],
    *,
    default_x_values: list[float] | None,
) -> Dataset:
    """One ``[comparisons.NAME.datasets.DATASET]`` table."""
    unknown = sorted(set(raw) - {"cases", "labels", "x_values", "color"})
    if unknown:
        raise CasesError(
            f"{path}: dataset {dataset_name!r} of comparison {comparison_name!r} "
            f"has unknown key(s): {unknown}"
        )

    members = raw.get("cases")
    if not members:
        raise CasesError(
            f"{path}: dataset {dataset_name!r} of comparison {comparison_name!r} "
            "has no 'cases' (or it is empty)"
        )
    members = [str(c) for c in members]

    missing = [c for c in members if c not in cases]
    if missing:
        raise CasesError(
            f"{path}: dataset {dataset_name!r} of comparison {comparison_name!r} "
            f"names undefined case(s) {missing}"
        )

    labels = [str(l) for l in raw.get("labels", [])]
    if labels and len(labels) != len(members):
        raise CasesError(
            f"{path}: dataset {dataset_name!r} of comparison {comparison_name!r} "
            f"has {len(labels)} labels for {len(members)} cases; labels must "
            "be omitted or match cases 1:1"
        )

    if "x_values" in raw:
        x_values = [float(v) for v in raw["x_values"]]
        source = f"dataset {dataset_name!r}'s own x_values"
    else:
        x_values = default_x_values
        source = f"comparison {comparison_name!r}'s x_values (dataset {dataset_name!r} sets none of its own)"
    if x_values is not None and len(x_values) != len(members):
        raise CasesError(
            f"{path}: {source} has {len(x_values)} value(s) for "
            f"{len(members)} cases in dataset {dataset_name!r}; x_values must "
            "match cases 1:1"
        )

    color = raw.get("color")
    if color is not None:
        color = str(color)

    return Dataset(
        name=dataset_name, cases=members, labels=labels, x_values=x_values, color=color,
    )
