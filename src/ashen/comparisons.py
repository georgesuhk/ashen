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

A comparison can instead name per-run *quantities* for its axes
(x_quantity/y_quantity/c_quantity, ashen.quantities) for a 2D scan map --
one point per member, both axes and a third encoded quantity read from
each run's own namelist and caches. Those need no x_values at all --
nothing parallel to maintain means nothing to drift. x_values and
x_quantity may coexist on one comparison: the same comparison can
legitimately drive a wetted_fraction line (needs x_values) and a scan map
(does not).

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
from ashen.quantities import is_known_quantity

__all__ = ["Comparison", "Dataset", "load_comparisons"]

#: Duplicated from ashen.plotting.scan_map.ENCODINGS, not imported: importing
#: the plotting package here would drag matplotlib into TOML parsing.
#: test_comparisons.py pins the two in sync.
_MAP_ENCODINGS = ("color", "size")


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
    #: Marker shape for this dataset on a `--diag scan_map` figure whose
    #: third quantity is encoded as COLOUR -- colour is spent on the
    #: colourbar there, so shape is what tells the datasets apart. None ->
    #: assigned from plotting.MARKER_CYCLE by position among siblings,
    #: exactly as `color` is from DISCRETE_PALETTE.
    #:
    #: Coexists with `color` rather than replacing it: the two encodings
    #: are mutually exclusive per figure but a dataset is not per figure.
    #: The same [comparisons.X.datasets.rho19] table drives a
    #: wetted_fraction line (uses color), a colour-encoded scan map (uses
    #: marker), and a size-encoded scan map (uses color again). Setting
    #: both is normal and correct; cli/plot notes when a figure ignores
    #: one, so a dataset that looks wrong is traceable.
    marker: str | None = None

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
    #: Axis label for a `y`-quantity scan map, the counterpart of x_label.
    #: Both are *overrides*: a scan map's axis labels default to the
    #: chosen quantity's own Quantity.label, so a comparison only sets
    #: these where it wants different wording. x_label keeps its original
    #: meaning for the x_values-based figures, which is the same meaning
    #: -- "what the x axis says" -- so it is reused rather than doubled.
    y_label: str = ""
    #: Colourbar / size-legend label for the third quantity.
    c_label: str = ""
    #: `--diag scan_map`: the named per-run scalar (ashen.quantities) on
    #: each axis, and the third one encoded as colour or ring size.
    #: x_quantity/y_quantity go together -- both or neither. c_quantity is
    #: optional: without it the map is a plain 2D scatter of runs, which
    #: is a legitimate (if less informative) figure.
    x_quantity: str = ""
    y_quantity: str = ""
    c_quantity: str = ""
    #: "color" (colourbar; datasets -> marker shape) or "size" (ring
    #: radius; datasets -> colour). See plotting.scan_map.ENCODINGS.
    map_encoding: str = "color"
    #: Label each point with its case's x_tick_label. Off by default -- a
    #: twenty-run scan annotates into illegibility.
    map_annotate: bool = False
    #: Axis/encoding scale overrides. None falls through to each chosen
    #: quantity's own Quantity.log_scale (log for eta and the delta_b
    #: family, linear for q95/li/edge_q) -- so the common case needs no
    #: setting at all, and an unusual one is not stuck with it.
    log_x: bool | None = None
    log_y: bool | None = None
    log_c: bool | None = None
    #: psi_n the `edge_q` quantity interpolates the q-profile at. None ->
    #: 1.0 (the separatrix). Lives on the comparison, not the case, for
    #: the same reason theta_target_psi does: a scan is only a comparison
    #: if every point used the same definition.
    edge_q_psi_n: float | None = None
    #: Step every equilibrium-type quantity (edge_q, q95, q99, li) is read
    #: at. None -> each member case's own first plotted step. Explicit
    #: here rather than per case because "q95 at step 200 for this run and
    #: step 3000 for that one" is not a scan.
    equilibrium_step: int | None = None
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
                "y_label", "c_label",
                "x_quantity", "y_quantity", "c_quantity",
                "map_encoding", "map_annotate",
                "log_x", "log_y", "log_c",
                "edge_q_psi_n", "equilibrium_step",
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

        # Scan-map fields (--diag scan_map). Validated at parse time, not
        # mid-figure -- a typo here must not produce an empty PNG.
        x_quantity = str(raw.get("x_quantity", ""))
        y_quantity = str(raw.get("y_quantity", ""))
        c_quantity = str(raw.get("c_quantity", ""))
        if bool(x_quantity) != bool(y_quantity):
            set_one, missing = (
                ("x_quantity", "y_quantity") if x_quantity else ("y_quantity", "x_quantity")
            )
            raise CasesError(
                f"{path}: comparison {name!r} sets {set_one!r} but not "
                f"{missing!r} -- a scan map needs both axes (or neither)"
            )
        for key, value in (
            ("x_quantity", x_quantity), ("y_quantity", y_quantity), ("c_quantity", c_quantity),
        ):
            if value and not is_known_quantity(value):
                raise CasesError(
                    f"{path}: comparison {name!r} {key} = {value!r} is not a "
                    "known quantity; `plot --list-quantities` lists them"
                )

        map_encoding = str(raw.get("map_encoding", "color"))
        if map_encoding not in _MAP_ENCODINGS:
            raise CasesError(
                f"{path}: comparison {name!r} map_encoding must be one of "
                f"{_MAP_ENCODINGS}, got {map_encoding!r}"
            )
        if "map_encoding" in raw and not c_quantity:
            raise CasesError(
                f"{path}: comparison {name!r} sets map_encoding but no "
                "c_quantity -- there is nothing to encode"
            )

        map_annotate = bool(raw.get("map_annotate", False))

        log_x = bool(raw["log_x"]) if "log_x" in raw else None
        log_y = bool(raw["log_y"]) if "log_y" in raw else None
        log_c = bool(raw["log_c"]) if "log_c" in raw else None

        edge_q_psi_n = None
        if "edge_q_psi_n" in raw:
            edge_q_psi_n = float(raw["edge_q_psi_n"])
            if edge_q_psi_n <= 0:
                raise CasesError(
                    f"{path}: comparison {name!r} edge_q_psi_n must be "
                    f"positive, got {edge_q_psi_n}"
                )

        equilibrium_step = None
        if "equilibrium_step" in raw:
            equilibrium_step = int(raw["equilibrium_step"])

        comparisons[name] = Comparison(
            name=name,
            cases=members,
            x_tick_labels=labels,
            datasets=datasets,
            note=str(raw.get("note", "")),
            n_cols=int(raw.get("n_cols", 4)),
            x_values=x_values,
            x_label=str(raw.get("x_label", "")),
            y_label=str(raw.get("y_label", "")),
            c_label=str(raw.get("c_label", "")),
            x_quantity=x_quantity,
            y_quantity=y_quantity,
            c_quantity=c_quantity,
            map_encoding=map_encoding,
            map_annotate=map_annotate,
            log_x=log_x,
            log_y=log_y,
            log_c=log_c,
            edge_q_psi_n=edge_q_psi_n,
            equilibrium_step=equilibrium_step,
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
        set(raw) - {"cases", "x_tick_labels", "x_values", "color", "dataset_label", "marker"}
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

    marker = raw.get("marker")
    if marker is not None:
        marker = str(marker)

    dataset_label = str(raw.get("dataset_label", ""))

    return Dataset(
        name=dataset_name, cases=members, x_tick_labels=labels, x_values=x_values,
        color=color, dataset_label=dataset_label, marker=marker,
    )
