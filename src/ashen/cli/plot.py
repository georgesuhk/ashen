"""``plot`` entry point: figures from data ``analyse`` already gathered.

Deliberately a separate command from ``analyse``. Gathering is slow and
batch (minutes per step, meant to run once and be cached); plotting is fast
and iterative (re-run constantly while tuning a figure) -- keeping them apart
means re-plotting never risks touching the gathering path, and this CLI can
carry flags (``--dpi``, ``--log``/``--linear``, ``--smooth``) that would only
clutter ``analyse``.

Reads the same ``cases.toml`` as ``analyse`` (:mod:`ashen.cases`), and never
runs a ``jorek2_*`` tool itself -- only :mod:`ashen.diagnostics.poincare_cache`
and :mod:`ashen.postproc.read_zeroD` are read.

**In scope this pass**: Poincare puncture plots and the LC/LCTT
connection-length maps -- the two that consume the field-line cache built in
Phase 4b. Everything else the legacy ``data_jorek.py`` / ``gather_profiles.py``
plotted (macroscopic-variable traces, field-line diffusion, radial profiles,
stochastic factor, the never-implemented ``max_fieldline_pos``) is not ported
-- see ``ashen/KNOWN_ISSUES.md`` #4.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np

from ashen.cases import Case, CasesError, load_cases
from ashen.config import SiteConfigError, load_site
from ashen.diagnostics.connection_length import connection_length_matrix
from ashen.diagnostics.four_modes import (
    DELTA_B,
    DELTA_B_OVER_B,
    delta_b_over_b_series,
    delta_b_series,
    format_growth_rates,
    growth_rate_series,
    max_amplitude_series,
    rational_surface_series,
)
from ashen.diagnostics.poincare_cache import read_step
from ashen.diagnostics.profiles import (
    ensure_edge_toroidal_field,
    expand_compound_vars,
    read_profile_series,
)
from ashen.diagnostics.qprofile import rational_surface_matches, read_qprofile
from ashen.jorek2 import Jorek2Error, Jorek2Run
from ashen.logfile import LogfileError, r_axis
from ashen.paths import RunPaths, read_float
from ashen.plotting.connection_length import plot_connection_length_map
from ashen.plotting.four_modes import plot_mode_amplitudes
from ashen.plotting.poincare import plot_poincare_step
from ashen.plotting.profiles import plot_profile_comparison
from ashen.postproc import read_zeroD

DIAG_CHOICES = ("poincare", "connection_length", "four", "profiles")

#: four_vars entries that are derived from "Psi" at plot time rather than
#: read directly from the jorek2_four cache -- see _plot_four_modes.
PSI_DERIVED = {DELTA_B, DELTA_B_OVER_B}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot",
        description="Draw figures from data gathered by `analyse` for cases in cases.toml.",
    )
    parser.add_argument(
        "--cases", type=Path, default=Path("cases.toml"),
        help="path to cases.toml (default: ./cases.toml)",
    )
    parser.add_argument(
        "--case", action="append", dest="selected",
        help="case name to plot (repeatable; default: every case in the file)",
    )
    parser.add_argument("--list", action="store_true", help="list defined cases and exit")
    parser.add_argument(
        "--diag", action="append", dest="diags", choices=DIAG_CHOICES,
        help="which figure(s) to draw (repeatable; default: all)",
    )
    parser.add_argument(
        "--step", type=int, action="append", dest="steps",
        help="restart step to plot (repeatable; applies to every diag drawn "
        "this run, overriding any [cases.NAME.<diag>] steps override; "
        "default: each diag's own steps, or the case's steps)",
    )
    parser.add_argument("--dpi", type=int, default=None, help="override figure DPI")
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="poincare steps rendered concurrently (default: site.toml's "
        "[diagnostics] n_workers)",
    )
    parser.add_argument(
        "--linear", action="store_true",
        help="connection_length: linear colour scale instead of the default log",
    )
    parser.add_argument(
        "--smooth", action="store_true",
        help="connection_length: smooth over time, ignoring confined (inf) cells",
    )
    parser.add_argument(
        "--psi-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
        help="connection_length: restrict the plotted psi_n axis to [MIN, MAX] "
        "(applied on top of the case's lc_psi_n_in, or psi_n_in if unset)",
    )
    parser.add_argument(
        "--four-linear", action="store_true",
        help="four: linear amplitude scale instead of the default log",
    )
    parser.add_argument("--site", type=Path, default=None, help="explicit site.toml")
    parser.add_argument(
        "--show-config", action="store_true",
        help="print where site.toml was found and what each key resolved to",
    )
    return parser


def _render_poincare_step(
    step: int, records, highlight: dict[float, str] | None, *, paths: RunPaths, kwargs: dict
) -> Path:
    """Module-level so it can be pickled for :class:`ProcessPoolExecutor`."""
    out = paths.figures_dir / f"{step}_poincare.png"
    step_kwargs = dict(kwargs)
    if highlight:
        step_kwargs["highlight"] = highlight
    plot_poincare_step(records, out, title=f"t={step}", **step_kwargs)
    return out


def _rational_highlight_for_step(
    case: Case, paths: RunPaths, step: int, records
) -> dict[float, str] | None:
    """``{psi_n: color}`` for the traced field lines nearest each of ``case``'s
    configured rational surfaces at this step -- ``None`` if highlighting is
    off, or this step has no qprofile cache yet.

    **Both sides are already in JOREK's own normalised psi_n**, and are
    compared directly with no ``real_psi_edge`` rescaling:

    * ``LineKey.psi_n`` is the value handed to ``jorek2_postproc``'s
      ``fluxsurface`` command, which rejects anything outside ``[0, 1]`` and
      converts it internally as ``psi_axis + psi_n*(psi_bnd - psi_axis)``
      (``exec_commands.f90:3063-3068``).
    * the q-profile's first column is ``get_psi_n(...)``
      (``exec_commands.f90::qprofile``) -- the exact inverse of that.

    ``real_psi_edge`` converts a *plasma-fraction* psi_n into this JOREK-grid
    psi_n (``boundary.py::extend_psi`` defines it as ``psi_plasma_edge /
    psi_extended_edge``), and ``cli/analyse.py`` already applies it once when
    turning ``case.psi_n_in`` into the traced positions. Dividing again here
    applied it twice and shifted every match by exactly that factor -- the
    same double-normalisation trap ``KNOWN_ISSUES.md`` #7 records against the
    connection-length threshold. It is deliberately absent now.
    """
    if not case.poincare_highlight:
        return None
    q_path = paths.qprofile(step)
    if not q_path.is_file():
        print(f"  step {step}: no qprofile cache, skipping rational-surface highlight "
              "(run analyse --diag poincare with poincare_highlight, or --diag four)")
        return None

    traced_psi_n = sorted({key.psi_n for key in records})
    if not traced_psi_n:
        return None

    psi_n_q, q = read_qprofile(q_path)
    modes = [
        (n, m, color)
        for (m, n), color in zip(case.poincare_highlight_modes, case.poincare_highlight_colors)
    ]
    return rational_surface_matches(psi_n_q, q, modes, traced_psi_n)


def _plot_poincare(
    case: Case, paths: RunPaths, steps: list[int], *, dpi: int | None, n_workers: int = 1
) -> None:
    kwargs = {} if dpi is None else {"dpi": dpi}

    to_render: list[tuple[int, dict, dict[float, str] | None]] = []
    for step in steps:
        records = read_step(paths, step)
        if not records:
            print(f"  step {step}: no Poincare cache, skipped")
            continue
        highlight = _rational_highlight_for_step(case, paths, step, records)
        to_render.append((step, records, highlight))

    if not to_render:
        return

    if n_workers <= 1 or len(to_render) <= 1:
        for step, records, highlight in to_render:
            out = _render_poincare_step(step, records, highlight, paths=paths, kwargs=kwargs)
            print(f"  step {step}: {out}")
        return

    one = partial(_render_poincare_step, paths=paths, kwargs=kwargs)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(one, step, records, highlight): step
            for step, records, highlight in to_render
        }
        outputs: dict[int, Path] = {}
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
    for step, _, _ in to_render:
        print(f"  step {step}: {outputs[step]}")


def _plot_connection_length(
    case: Case,
    paths: RunPaths,
    steps: list[int],
    *,
    log: bool,
    smooth: bool,
    dpi: int | None,
    psi_range: tuple[float, float] | None = None,
) -> None:
    real_psi_edge = read_float(paths.real_psi_edge)
    try:
        R0 = r_axis(paths.log)
    except LogfileError as exc:
        print(f"  error: {exc}")
        return

    psi_n_in = case.lc_psi_n_in if case.lc_psi_n_in is not None else case.psi_n_in
    if psi_range is not None:
        lo, hi = psi_range
        psi_n_in = [p for p in psi_n_in if lo <= p <= hi]
        if not psi_n_in:
            print(f"  error: no psi_n_in within range {psi_range}")
            return

    psi_n_targets = [p * real_psi_edge for p in psi_n_in]
    records_by_step = {step: read_step(paths, step) for step in steps}
    matrix = connection_length_matrix(
        records_by_step, steps, psi_n_targets, real_psi_edge=real_psi_edge, R0=R0
    )

    true_times = _read_true_times(paths, steps)

    kwargs = {} if dpi is None else {"dpi": dpi}
    for plot_true_times in (True, False):
        if plot_true_times and true_times is None:
            print("  skipping LCTT: no zeroD cache for one or more steps (run analyse --diag zerod)")
            continue
        out = plot_connection_length_map(
            matrix, steps, psi_n_in, paths.figures_dir,
            true_times=true_times, plot_true_times=plot_true_times,
            log=log, smooth=smooth, **kwargs,
        )
        print(f"  {out}")


def _read_true_times(paths: RunPaths, steps: list[int]) -> list[float] | None:
    """Each step's true time (seconds) from the zeroD cache, or ``None`` if
    any requested step's cache is missing -- a partial time axis is worse
    than none, so this is all-or-nothing rather than dropping steps."""
    true_times = []
    for step in steps:
        try:
            true_times.append(read_zeroD(paths.zero_d(step))["Time"])
        except (FileNotFoundError, KeyError):
            return None
    return true_times


def _peak_of_variable(series: dict, variable: str) -> float | None:
    """The largest finite value across every mode of ``variable`` in
    ``series`` -- ``None`` if there isn't one (all-nan or no matching key).
    Used to caption a figure with its peak delta-B/delta-B-over-B, a single
    figure-level number that isn't tied to any one mode's legend entry.
    """
    arrays = [np.asarray(v, dtype=float) for (var, _, _), v in series.items() if var == variable]
    if not arrays:
        return None
    values = np.concatenate(arrays)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else None


def _plot_four_modes(
    case: Case, paths: RunPaths, steps: list[int], *, log: bool, dpi: int | None
) -> None:
    want_max = "max" in case.four_quantities
    want_rational = "rational_surface" in case.four_quantities

    # delta_b/delta_b_over_b are derived from the "Psi" cache variable, not a
    # raw jorek2_four output -- only computed when explicitly requested
    # (never picked up by an unrestricted four_vars = [] "plot everything").
    # "Psi" is added to the cache fetch if the caller didn't already ask for
    # it, then stripped back out below so it doesn't appear as its own panel.
    requested_vars = list(case.four_vars) if case.four_vars else None
    requested_derived = set(requested_vars or []) & PSI_DERIVED
    fetch_vars = requested_vars
    if requested_derived:
        fetch_vars = [v for v in requested_vars if v not in PSI_DERIVED]
        if "Psi" not in fetch_vars:
            fetch_vars.append("Psi")

    # case.four_modes entries are [m, n] pairs (user-facing); the diagnostics
    # layer's ModeKey/modes filter is (n, m), matching FourRecord's own
    # (variable, n, m) field order -- swap here, at the one point they meet.
    # Computed unconditionally: it's the only place mode/variable keys are
    # discovered from the cache (cheap -- just reading what's on disk), and
    # is itself the primary series whenever "max" is selected.
    series = max_amplitude_series(
        paths, steps,
        variables=fetch_vars,
        modes=[(n, m) for m, n in case.four_modes] if case.four_modes else None,
    )
    if not series:
        print("  no jorek2_four cache found for any requested step/variable/mode "
              "(run analyse --diag four)")
        return

    # Pin each n!=0 mode's amplitude to its q=m/n rational surface, using
    # whatever q-profile cache `analyse --diag four` already gathered
    # alongside it. Silently empty (not an error) if that cache is missing --
    # e.g. plotting a case gathered before this feature existed -- so a
    # "max"-only plot is unaffected, and a rational-only one just reports
    # nothing to draw (below).
    rational = {}
    if want_rational:
        rational_modes = {(n, m) for _, n, m in series if n != 0}
        if rational_modes:
            rational = rational_surface_series(
                paths, steps, sorted(rational_modes), variables=fetch_vars
            )

    if requested_derived:
        remaining = set(requested_derived)
        try:
            r0 = r_axis(paths.log)
        except LogfileError as exc:
            print(f"  skipping {', '.join(sorted(remaining))}: {exc}")
            remaining = set()

        b_ref = None
        if DELTA_B_OVER_B in remaining:
            jrun = Jorek2Run(
                run_dir=paths.run_dir, exe_dir=paths.run_dir,
                namelist=paths.run_dir / case.namelist, pad_width=paths.pad_width,
            )
            try:
                b_ref = ensure_edge_toroidal_field(jrun, paths)
            except (FileNotFoundError, Jorek2Error) as exc:
                # FileNotFoundError also covers MissingRestartError (a
                # subclass) -- e.g. jorek2_postproc itself isn't symlinked
                # into this run folder. Caught here, not left to `main`'s
                # outer per-case handler, so this only skips
                # delta_b_over_b rather than aborting every other diag
                # this plot invocation was also asked to draw.
                print(f"  skipping {DELTA_B_OVER_B}: {exc}")
                remaining.discard(DELTA_B_OVER_B)
            else:
                if b_ref is None:
                    print(f"  skipping {DELTA_B_OVER_B}: could not gather the Btor "
                          "profile at the plasma edge for step 0 (jorek2_postproc "
                          "produced no output -- see the warning above)")
                    remaining.discard(DELTA_B_OVER_B)

        if remaining:
            psi_keys = {k for k in series if k[0] == "Psi"}
            psi_only = {k: series[k] for k in psi_keys}
            if DELTA_B in remaining:
                series.update(delta_b_series(psi_only, r_axis=r0))
            if DELTA_B_OVER_B in remaining:
                series.update(delta_b_over_b_series(psi_only, r_axis=r0, b_ref=b_ref))
            if rational:
                psi_rational_keys = {k for k in rational if k[0] == "Psi"}
                psi_rational_only = {k: rational[k] for k in psi_rational_keys}
                if DELTA_B in remaining:
                    rational.update(delta_b_series(psi_rational_only, r_axis=r0))
                if DELTA_B_OVER_B in remaining:
                    rational.update(
                        delta_b_over_b_series(psi_rational_only, r_axis=r0, b_ref=b_ref)
                    )

        # Drop the raw Psi keys unless Psi itself was explicitly requested --
        # they were only fetched as the input to the conversion above.
        if requested_vars is not None and "Psi" not in requested_vars:
            series = {k: v for k, v in series.items() if k[0] != "Psi"}
            rational = {k: v for k, v in rational.items() if k[0] != "Psi"}

    # Primary (solid) series is max whenever selected -- both-selected
    # reproduces the original solid-max/dashed-rational-overlay look;
    # rational-only makes rational itself the (solid) primary, with no
    # overlay, labelled to say what it actually is.
    primary_series = series if want_max else rational
    overlay_series = rational if (want_max and want_rational) else None
    if want_max:
        ylabel_suffix, label_suffix = None, ""
    else:
        ylabel_suffix, label_suffix = " @ rational surface", " @ rational surface"

    if not primary_series:
        print("  no rational-surface data to plot (need the qprofile cache "
              "and at least one n != 0 mode; run analyse --diag four)")
        return

    # Both x-axis variants are always written (unlike connection_length's
    # LC/LCTT, which are separate figures too, but this mirrors that split
    # rather than picking one): "step" always works, "time" only if the
    # zeroD cache covers every requested step.
    variants = [("step", list(steps), "Time step")]
    true_times = _read_true_times(paths, steps)
    if true_times is not None:
        variants.append(("time", [t * 1e6 for t in true_times], r"t [$\mu s$]"))
    else:
        print("  skipping time-axis four-mode plots: no zeroD cache for one or "
              "more steps (run analyse --diag zerod)")

    # Growth rate (gamma, 1/s) is a physical quantity fit against real time,
    # so it needs true_times regardless of which x-axis variant it ends up
    # labelled on -- fit against whichever series is primary, since gamma
    # should describe the quantity actually on the plot.
    growth_fits = {}
    if case.four_growth_rate:
        if true_times is None:
            print("  skipping growth-rate fit: no zeroD cache for one or more "
                  "steps (run analyse --diag zerod)")
        else:
            growth_fits = growth_rate_series(
                primary_series, true_times, steps,
                step_range=tuple(case.four_growth_steps) if case.four_growth_steps else None,
            )
            if growth_fits:
                growth_path = paths.four_dir / "growth_rates.txt"
                growth_path.parent.mkdir(parents=True, exist_ok=True)
                growth_path.write_text(format_growth_rates(growth_fits), encoding="utf-8")
                print(f"  {growth_path}")
            else:
                print("  no growth-rate fit possible (need >=2 positive-amplitude "
                      "points per mode in the fit window)")

    kwargs = {} if dpi is None else {"dpi": dpi}
    for suffix, x, xlabel in variants:
        for variable in sorted({var for var, _, _ in primary_series}):
            out = paths.four_dir / f"{variable}_modes_{suffix}.png"
            caption = None
            if variable == DELTA_B_OVER_B:
                ylabel = f"\N{GREEK SMALL LETTER DELTA}B/B{ylabel_suffix or ''}"
                peak = _peak_of_variable(primary_series, variable)
                if peak is not None:
                    caption = f"max \N{GREEK SMALL LETTER DELTA}B/B = {peak:.3g}"
            elif variable == DELTA_B:
                ylabel = f"\N{GREEK SMALL LETTER DELTA}B [T]{ylabel_suffix or ''}"
                peak = _peak_of_variable(primary_series, variable)
                if peak is not None:
                    caption = f"max \N{GREEK SMALL LETTER DELTA}B = {peak:.3g} T"
            else:
                ylabel = f"|{variable}|{ylabel_suffix}" if ylabel_suffix else None
            plot_mode_amplitudes(
                x, primary_series, variable, out, rational_series=overlay_series or None,
                growth_fits=growth_fits or None, log=log, xlabel=xlabel,
                ylabel=ylabel, label_suffix=label_suffix, caption=caption, **kwargs,
            )
            print(f"  {out}")


def _plot_profiles(case: Case, paths: RunPaths, steps: list[int], *, dpi: int | None) -> None:
    """One figure per variable: a panel per ``tor_mode``, a curve per step.

    Colour carries true time where the zeroD cache allows it, falling back to
    step index -- the same all-or-nothing rule as ``_plot_four_modes``' time
    axis, since a colour scale mixing seconds and step indices would be worse
    than one honest unit.
    """
    variables = expand_compound_vars(case.vars)
    if not variables:
        print("  no vars configured for this case; nothing to plot")
        return

    true_times = _read_true_times(paths, steps)
    if true_times is not None:
        color_by = {step: t * 1e6 for step, t in zip(steps, true_times)}
        color_label = r"t [$\mu s$]"
    else:
        color_by = None
        color_label = "Time step"
        print("  no zeroD cache for one or more steps (run analyse --diag zerod); "
              "colouring profiles by step index")

    kwargs = {} if dpi is None else {"dpi": dpi}
    for var in variables:
        series_by_mode = {
            mode: read_profile_series(paths, steps, case.coords_var, var, mode)
            for mode in case.tor_mode
        }
        if not any(series_by_mode.values()):
            print(f"  no cached {var!r} profiles (run analyse --diag profiles)")
            continue

        out = paths.figures_dir / f"{case.coords_var}_{var}_profile.png"
        plot_profile_comparison(
            series_by_mode, var, out,
            color_by=color_by, color_label=color_label,
            xlabel=case.coords_var, **kwargs,
        )
        print(f"  {out}")


def _resolve_n_workers(args) -> int:
    """``--n-workers`` if given, else ``site.toml``'s ``[diagnostics]
    n_workers`` if that's explicitly set, else 1 (serial).

    Unlike ``cli/analyse.py``'s ``_resolve_parallelism``, an *unset*
    ``[diagnostics]`` (or no ``site.toml`` at all) does not derive a worker
    count from the machine's core count: rendering a handful of PNGs is cheap
    enough that auto-parallelising by default would just spawn processes for
    no measurable benefit, so plotting stays serial unless asked otherwise.
    """
    if args.n_workers is not None:
        return max(1, args.n_workers)
    try:
        diagnostics = load_site(args.site).diagnostics
    except SiteConfigError:
        return 1
    return max(1, diagnostics.n_workers) if diagnostics.n_workers else 1


def _run_case(
    case: Case,
    *,
    diags: list[str],
    steps: list[int] | None,
    log: bool,
    smooth: bool,
    dpi: int | None,
    n_workers: int = 1,
    psi_range: tuple[float, float] | None = None,
    four_log: bool = True,
) -> None:
    run_dir = Path.cwd() / case.name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"case {case.name!r}: no such folder {run_dir}")
    paths = RunPaths.detect(run_dir)

    # --step (CLI) outranks everything below it in the default -> case ->
    # case+diag tree -- an explicit runtime request wins over any configured
    # override. Absent that, each diag falls back to its own steps_for(diag)
    # override (a [cases.NAME.<diag>] table) before the case's plain steps.
    if "poincare" in diags:
        _plot_poincare(
            case, paths, steps or case.steps_for("poincare"), dpi=dpi, n_workers=n_workers
        )
    if "connection_length" in diags:
        # --psi-range (if given) further bounds-filters whichever list is
        # already in effect (case.lc_psi_n_in, or case.psi_n_in if unset).
        # connection_length's own steps only select which already-gathered
        # poincare steps to plot (same "no interpolation" rule lc_psi_n_in
        # already has) -- it never gathers new data on its own.
        _plot_connection_length(
            case, paths, steps or case.steps_for("connection_length"),
            log=log, smooth=smooth, dpi=dpi, psi_range=psi_range,
        )
    if "four" in diags:
        _plot_four_modes(case, paths, steps or case.steps_for("four"), log=four_log, dpi=dpi)
    if "profiles" in diags:
        _plot_profiles(case, paths, steps or case.steps_for("profiles"), dpi=dpi)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.show_config:
        try:
            site = load_site(args.site)
        except SiteConfigError as exc:
            print(f"error: {exc}")
            return 1
        print(site.describe())
        return 0

    try:
        cases = load_cases(args.cases)
    except CasesError as exc:
        print(f"error: {exc}")
        return 1

    if args.list:
        for name, case in cases.items():
            note = f" -- {case.note}" if case.note else ""
            print(f"{name} ({len(case.steps)} steps){note}")
        return 0

    selected = args.selected or list(cases)
    unknown = [name for name in selected if name not in cases]
    if unknown:
        print(f"error: unknown case(s) {unknown}; --list to see defined cases")
        return 1

    diags = args.diags or list(DIAG_CHOICES)
    n_workers = _resolve_n_workers(args)
    psi_range = tuple(args.psi_range) if args.psi_range is not None else None

    for name in selected:
        print(f"==== {name} ====")
        try:
            _run_case(
                cases[name],
                diags=diags,
                steps=args.steps,
                log=not args.linear,
                smooth=args.smooth,
                dpi=args.dpi,
                n_workers=n_workers,
                psi_range=psi_range,
                four_log=not args.four_linear,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}")
            return 1

    return 0
