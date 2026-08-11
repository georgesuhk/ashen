"""Cross-case comparisons for `bin/plot`.

Some figures compare different runs (e.g. an eta scan's theta-crossing
distributions, one panel per run) rather than different steps of one run --
ashen.cases has no notion of this; a Case is one run folder. A comparison is
a named group of already-defined cases, read from the same cases.toml (not
a second file): every member must already be a [cases.*] entry there.

Each member's own per-diag step override still does per-run time-window
selection ([cases.NAME.theta_hist] steps = [...]); a comparison only
supplies which cases to pool and how to label them, via Case.steps_for.

A comparison can carry an explicit x_values (parallel to cases) for
plotting a derived scalar vs. a scan parameter across members -- e.g.
wetted fraction vs. eta (ashen.plotting.wetted_fraction).

A comparison names members one of two ways, never both:
- cases (flat): one series, one point per member. Pre-datasets default;
  still the only form theta_hist's grid comparison understands (one panel
  per member).
- datasets (nested [comparisons.NAME.datasets.DATASET] tables): >1 related
  scans sharing an x-axis, e.g. a resistivity scan under two profile
  assumptions ("normal" vs "rho19"). Each dataset is its own case group;
  wetted_fraction overlays them on one axes, one colour/legend entry per
  dataset (plot_wetted_fraction_datasets). See Dataset for fields.

A comparison can override theta_target_psi/theta_bins/theta_psi_n_range/
theta_wetted_threshold for every member uniformly, rather than duplicating
per case (or a shared [defaults], which would leak into non-comparison
uses) -- an apples-to-apples scan needs every point computed the same way.
Precedence, most specific wins: CLI flag (--theta_target_psi) > comparison
setting > member case setting > diagnostic default.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ashen.cases import Case, CasesError

__all__ = ["Comparison", "Dataset", "load_comparisons"]


@dataclass(frozen=True)
class Dataset:
    """One named case group within a `datasets`-style comparison, e.g.
    "rho19" alongside sibling "normal". See module docstring."""

    name: str
    #: Member case names, in point order.
    cases: list[str]
    #: Per-case labels, parallel to `cases`. Not legend text -- see dataset_label.
    x_tick_labels: list[str] = field(default_factory=list)
    #: Parallel to `cases`; falls back to the comparison's own x_values.
    x_values: list[float] | None = None
    #: Legend colour. None -> assigned from DISCRETE_PALETTE by position
    #: among sibling datasets.
    color: str | None = None
    #: Legend text; falls back to `name` (the TOML table key) if empty.
    dataset_label: str = ""

    def labelled_cases(self) -> list[tuple[str, str]]:
        """[(x_tick_label, case_name), ...] in point order."""
        labels = self.x_tick_labels or self.cases
        return list(zip(labels, self.cases))

    @property
    def series_label(self) -> str:
        """Legend text for this dataset's series: dataset_label if set, else name."""
        return self.dataset_label or self.name


@dataclass(frozen=True)
class Comparison:
    name: str
    #: Member case names, panel order. Empty when `datasets` is used instead
    #: -- exactly one of the two is ever set.
    cases: list[str] = field(default_factory=list)
    #: Per-case labels, parallel to `cases` (e.g. theta_hist panel titles).
    #: Defaults to the case names if omitted.
    x_tick_labels: list[str] = field(default_factory=list)
    #: Named sub-scans sharing this comparison's x-axis; see module
    #: docstring. Empty when flat `cases` is used instead.
    datasets: dict[str, Dataset] = field(default_factory=dict)
    note: str = ""
    n_cols: int = 4
    #: Explicit numeric value per member (e.g. resistivity), for a "scalar
    #: vs. scan parameter" plot (wetted_fraction). Never inferred from run
    #: folder names -- CASTOR3D's directory-name parsing is the hazard this
    #: project exists to avoid (see CLAUDE.md).
    x_values: list[float] | None = None
    #: Axis label for x_values, e.g. "$\\eta$ [$\\Omega \\cdot$ m]".
    x_label: str = ""
    #: Uniform overrides for every member case (see module docstring for
    #: why they live here, not per case). None falls through to the case's
    #: own setting.
    theta_target_psi: float | None = None
    theta_bins: int | None = None
    theta_psi_n_range: list[float] | None = None
    theta_wetted_threshold: float | None = None

    def labelled_cases(self) -> list[tuple[str, str]]:
        """[(x_tick_label, case_name), ...] in panel order. Flat-mode
        (`cases`) only -- a `datasets` comparison has no single flat panel
        order; iterate `datasets` instead."""
        labels = self.x_tick_labels or self.cases
        return list(zip(labels, self.cases))


def load_comparisons(path: Path | str, cases: dict[str, Case]) -> dict[str, Comparison]:
    """Parse [comparisons.*] tables from cases.toml.

    `cases` is load_cases's already-resolved result for the same file,
    passed in rather than re-derived, so member names validate against
    exactly what the caller resolved (avoids drift if one function is
    extended and the other isn't).

    Returns {} if there's no [comparisons.*] section -- not an error, most
    cases.toml files won't define any.
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
                "cases", "x_tick_labels", "datasets", "note", "n_cols", "x_values", "x_label",
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

        # Parsed once regardless of mode: flat mode validates against
        # `members` below; datasets mode treats it as each dataset's shared
        # fallback (validated per-dataset in the loop below).
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

            labels = [str(l) for l in raw.get("x_tick_labels", [])]
            if labels and len(labels) != len(members):
                raise CasesError(
                    f"{path}: comparison {name!r} has {len(labels)} x_tick_labels "
                    f"for {len(members)} cases; x_tick_labels must be omitted or "
                    "match cases 1:1"
                )

            if x_values is not None and len(x_values) != len(members):
                raise CasesError(
                    f"{path}: comparison {name!r} has {len(x_values)} x_values "
                    f"for {len(members)} cases; x_values must match cases 1:1"
                )
        else:
            if "x_tick_labels" in raw:
                raise CasesError(
                    f"{path}: comparison {name!r} sets 'x_tick_labels' but uses "
                    "'datasets' -- put x_tick_labels on each dataset instead"
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
            x_tick_labels=labels,
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
    unknown = sorted(
        set(raw) - {"cases", "x_tick_labels", "x_values", "color", "dataset_label"}
    )
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

    labels = [str(l) for l in raw.get("x_tick_labels", [])]
    if labels and len(labels) != len(members):
        raise CasesError(
            f"{path}: dataset {dataset_name!r} of comparison {comparison_name!r} "
            f"has {len(labels)} x_tick_labels for {len(members)} cases; "
            "x_tick_labels must be omitted or match cases 1:1"
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

    dataset_label = str(raw.get("dataset_label", ""))

    return Dataset(
        name=dataset_name, cases=members, x_tick_labels=labels, x_values=x_values,
        color=color, dataset_label=dataset_label,
    )
