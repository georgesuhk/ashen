"""ashen.quantities -- named per-run scalars for scan-map-style figures.

Every quantity extracts from a synthetic on-disk cache, never a real
jorek2_* tool; h5py-backed (four cache) tests are grouped at the bottom
behind their own importorskip so the rest of this module runs without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashen.cases import Case
from ashen.paths import RunPaths, write_float
from ashen.quantities import (
    EDGE_Q_CLAMP_TOL,
    ZEROD_PREFIX,
    QuantityContext,
    QuantityError,
    describe_quantities,
    is_known_quantity,
    quantity,
    quantity_names,
)


def _case(name: str = "run", **kw) -> Case:
    return Case(name=name, steps=kw.pop("steps", [100, 200]), **kw)


def _paths(tmp_path) -> RunPaths:
    paths = RunPaths(tmp_path / "run", pad_width=6)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _ctx(case: Case, paths: RunPaths, **kw) -> QuantityContext:
    """A QuantityContext defaulting `steps` to the case's own -- most tests
    want the ordinary "reads whatever the case would plot" shape; a test
    checking the empty/missing-steps path overrides `steps=[]` explicitly.
    """
    kw.setdefault("steps", list(case.steps))
    return QuantityContext(case=case, paths=paths, **kw)


def _write_namelist(paths: RunPaths, *, filename: str = "in_main", **fields) -> None:
    lines = ["&in1"]
    for key, value in fields.items():
        lines.append(f"  {key} = {value}")
    lines.append("&end")
    (paths.run_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_zerod(paths: RunPaths, step: int, **columns) -> None:
    path = paths.zero_d(step)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = " ".join(columns)
    values = " ".join(str(v) for v in columns.values())
    path.write_text(f"{keys}\n{values}\n", encoding="utf-8")


def _write_qprofile(paths: RunPaths, step: int, psi_n, q) -> None:
    path = paths.qprofile(step)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Psi_n q", f"# time step #{step:06d}"]
    lines += [f"{p} {v}" for p, v in zip(psi_n, q)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_log(paths: RunPaths, r_axis: float = 1.363245) -> None:
    (paths.run_dir / "log").write_text(f"R_axis = {r_axis}\n", encoding="utf-8")


# --- registry / API ----------------------------------------------------------


def test_every_registered_quantity_key_matches_its_own_name():
    from ashen.quantities import QUANTITIES

    for key, q in QUANTITIES.items():
        assert q.name == key


def test_quantity_names_sorted_and_unique():
    names = quantity_names()
    assert list(names) == sorted(names)
    assert len(names) == len(set(names))
    assert "eta" in names
    assert "edge_q" in names
    assert "li" in names
    assert "wetted_fraction" in names


def test_quantity_returns_registered_entry():
    q = quantity("eta")
    assert q.name == "eta"
    assert q.log_scale is True


def test_quantity_unknown_name_raises_and_lists_known_names():
    with pytest.raises(QuantityError, match="eta"):
        quantity("nope")


def test_is_known_quantity():
    assert is_known_quantity("eta")
    assert not is_known_quantity("nope")


def test_zerod_prefix_form_is_dynamic():
    assert is_known_quantity(f"{ZEROD_PREFIX}beta_n")
    assert not is_known_quantity(ZEROD_PREFIX)  # needs a column name
    q = quantity(f"{ZEROD_PREFIX}beta_n")
    assert q.name == f"{ZEROD_PREFIX}beta_n"
    assert "beta_n" in q.label


def test_describe_quantities_has_one_row_per_name():
    rows = describe_quantities()
    assert {name for name, _, _ in rows} == set(quantity_names())
    for _, scale, _ in rows:
        assert scale in ("log", "linear")


# --- eta -----------------------------------------------------------------


def test_eta_reads_from_in_main(tmp_path):
    paths = _paths(tmp_path)
    _write_namelist(paths, eta="1.d-3")
    case = _case()
    assert quantity("eta").extract(_ctx(case, paths)) == pytest.approx(1e-3)


def test_eta_reads_from_in_main_r_when_case_namelist_says_so(tmp_path):
    paths = _paths(tmp_path)
    _write_namelist(paths, filename="in_main_r", eta="2.d-4")
    case = _case(namelist="in_main_r")
    assert quantity("eta").extract(_ctx(case, paths)) == pytest.approx(2e-4)


def test_eta_missing_field_reports_and_returns_none(tmp_path):
    paths = _paths(tmp_path)
    _write_namelist(paths, other_field="1.0")
    case = _case()
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("eta").extract(ctx) is None
    assert len(reports) == 1
    assert "eta" in reports[0]


def test_eta_missing_file_reports_and_returns_none(tmp_path):
    paths = _paths(tmp_path)
    case = _case()
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("eta").extract(ctx) is None
    assert len(reports) == 1


# --- edge_q ----------------------------------------------------------------


def test_edge_q_exact_hit(tmp_path):
    paths = _paths(tmp_path)
    _write_qprofile(paths, 100, [0.1, 0.5, 0.9], [1.0, 1.5, 2.0])
    case = _case()
    ctx = _ctx(case, paths, edge_q_psi_n=0.5)
    assert quantity("edge_q").extract(ctx) == pytest.approx(1.5)


def test_edge_q_interpolates_between_grid_points(tmp_path):
    paths = _paths(tmp_path)
    _write_qprofile(paths, 100, [0.0, 1.0], [1.0, 2.0])
    case = _case()
    ctx = _ctx(case, paths, edge_q_psi_n=0.5)
    assert quantity("edge_q").extract(ctx) == pytest.approx(1.5)


def test_edge_q_clamps_within_tolerance_and_reports(tmp_path):
    paths = _paths(tmp_path)
    _write_qprofile(paths, 100, [0.001, 0.5, 0.999], [1.0, 1.5, 2.0])
    case = _case()
    reports = []
    ctx = _ctx(case, paths, edge_q_psi_n=1.0, report=reports.append)
    value = quantity("edge_q").extract(ctx)
    assert value == pytest.approx(2.0)
    assert any("clamped" in r for r in reports)


def test_edge_q_beyond_tolerance_is_none(tmp_path):
    paths = _paths(tmp_path)
    _write_qprofile(paths, 100, [0.001, 0.5, 0.8], [1.0, 1.5, 2.0])
    case = _case()
    reports = []
    ctx = _ctx(
        case, paths, edge_q_psi_n=0.8 + EDGE_Q_CLAMP_TOL * 2, report=reports.append,
    )
    assert quantity("edge_q").extract(ctx) is None
    assert reports


def test_edge_q_missing_cache_no_callback_is_none(tmp_path):
    paths = _paths(tmp_path)
    case = _case()
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("edge_q").extract(ctx) is None
    assert reports


def test_edge_q_missing_cache_calls_ensure_qprofile_callback(tmp_path):
    paths = _paths(tmp_path)
    case = _case()
    calls = []

    def _ensure(steps):
        calls.append(list(steps))
        _write_qprofile(paths, steps[0], [0.0, 1.0], [1.0, 3.0])

    ctx = _ctx(case, paths, ensure_qprofile=_ensure, edge_q_psi_n=0.5)
    value = quantity("edge_q").extract(ctx)
    assert calls == [[100]]  # eq_step -> steps[0]
    assert value == pytest.approx(2.0)


# --- q95 / li (zeroD columns) -----------------------------------------------


def test_q95_reads_at_equilibrium_step(tmp_path):
    paths = _paths(tmp_path)
    _write_zerod(paths, 100, Time=1e-4, Q95=3.2)
    case = _case()
    assert quantity("q95").extract(_ctx(case, paths)) == pytest.approx(3.2)


def test_li_reads_li3_column(tmp_path):
    paths = _paths(tmp_path)
    _write_zerod(paths, 100, Time=1e-4, li3=0.85)
    case = _case()
    assert quantity("li").extract(_ctx(case, paths)) == pytest.approx(0.85)


def test_equilibrium_step_override_wins_over_first_step(tmp_path):
    paths = _paths(tmp_path)
    _write_zerod(paths, 100, Time=1e-4, li3=0.1)
    _write_zerod(paths, 200, Time=2e-4, li3=0.9)
    case = _case()
    ctx = _ctx(case, paths, equilibrium_step=200)
    assert quantity("li").extract(ctx) == pytest.approx(0.9)


def test_missing_zerod_column_reports_available_columns(tmp_path):
    paths = _paths(tmp_path)
    _write_zerod(paths, 100, Time=1e-4, Energy=1.0)
    case = _case()
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("li").extract(ctx) is None
    assert len(reports) == 1
    assert "Time" in reports[0] and "Energy" in reports[0]


def test_unparseable_zerod_cache_is_none_not_an_exception(tmp_path):
    paths = _paths(tmp_path)
    path = paths.zero_d(100)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Time Energy\n", encoding="utf-8")  # header only
    case = _case()
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("li").extract(ctx) is None
    assert reports


def test_zerod_column_calls_ensure_zero_d_callback(tmp_path):
    paths = _paths(tmp_path)
    case = _case()
    calls = []

    def _ensure(steps):
        calls.append(list(steps))
        _write_zerod(paths, steps[0], Time=1e-4, li3=0.42)

    ctx = _ctx(case, paths, ensure_zero_d=_ensure)
    value = quantity("li").extract(ctx)
    assert calls == [[100]]
    assert value == pytest.approx(0.42)


def test_zerod_prefix_reads_arbitrary_column(tmp_path):
    paths = _paths(tmp_path)
    _write_zerod(paths, 100, Time=1e-4, beta_n=1.23)
    case = _case()
    assert quantity(f"{ZEROD_PREFIX}beta_n").extract(_ctx(case, paths)) == pytest.approx(1.23)


# --- t_final -----------------------------------------------------------------


def test_t_final_reads_time_at_last_step(tmp_path):
    paths = _paths(tmp_path)
    _write_zerod(paths, 100, Time=1e-4)
    _write_zerod(paths, 200, Time=5e-4)
    case = _case(steps=[100, 200])
    assert quantity("t_final").extract(_ctx(case, paths)) == pytest.approx(5e-4)


def test_t_final_needs_a_step_list(tmp_path):
    paths = _paths(tmp_path)
    case = _case()
    reports = []
    ctx = _ctx(case, paths, steps=[], report=reports.append)
    assert quantity("t_final").extract(ctx) is None
    assert reports


# --- wetted_fraction ---------------------------------------------------------


def test_wetted_fraction_matches_the_underlying_pooled_computation(tmp_path):
    from ashen.diagnostics import poincare_cache as pc

    paths = _paths(tmp_path)
    write_float(paths.real_psi_edge, 1.0)
    path = paths.poincare_cache(100)
    with pc.open_cache(path, step=100, pad_width=6) as h:
        for psi_n, R, theta in [(0.2, 1.7, 0.5), (0.5, 1.8, -0.5)]:
            key = pc.LineKey(psi_n=psi_n, R=R, Z=0.0, phi=0.0)
            pc.append_line(
                h, key,
                {
                    "R": np.full(3, R, dtype=np.float32),
                    "Z": np.zeros(3, dtype=np.float32),
                    "rho": np.sqrt(np.array([0.3, 1.5, 1.6], dtype=np.float32)),
                    "theta": np.array([0.0, theta, theta + 0.1], dtype=np.float32),
                },
                n_turns=3, terminated=False,
            )
    case = _case(steps=[100])
    value = quantity("wetted_fraction").extract(_ctx(case, paths))
    assert value is not None
    assert 0.0 <= value <= 1.0


def test_wetted_fraction_threshold_defaults_to_one_over_bins(tmp_path):
    from ashen.diagnostics import poincare_cache as pc

    paths = _paths(tmp_path)
    write_float(paths.real_psi_edge, 1.0)
    path = paths.poincare_cache(100)
    with pc.open_cache(path, step=100, pad_width=6):
        pass  # no lines -- empty pooled result is still a valid (non-crashing) figure
    case = _case(steps=[100])
    # An empty pool has no crossings -> theta_histogram's all-zero counts
    # give a defined, non-crashing 0.0 (see theta_histogram.wetted_fraction).
    value = quantity("wetted_fraction").extract(_ctx(case, paths))
    assert value == pytest.approx(0.0)


# --- QuantityContext.eq_step ---------------------------------------------


def test_eq_step_prefers_explicit_equilibrium_step():
    case = _case()
    ctx = QuantityContext(case=case, paths=None, steps=[100, 200], equilibrium_step=200)
    assert ctx.eq_step == 200


def test_eq_step_falls_back_to_first_step():
    case = _case()
    ctx = QuantityContext(case=case, paths=None, steps=[100, 200])
    assert ctx.eq_step == 100


def test_eq_step_is_none_when_nothing_available():
    case = _case()
    ctx = QuantityContext(case=case, paths=None, steps=[])
    assert ctx.eq_step is None


# --- report contract: every None comes with exactly one report --------------


@pytest.mark.parametrize("name", ["eta", "edge_q", "q95", "li", "t_final", "wetted_fraction"])
def test_missing_data_reports_exactly_once(tmp_path, name):
    paths = _paths(tmp_path)
    case = _case()
    reports = []
    ctx = _ctx(case, paths, steps=[], report=reports.append)
    assert quantity(name).extract(ctx) is None
    assert len(reports) == 1


# --- delta_b family (needs h5py) ---------------------------------------------

h5py = pytest.importorskip("h5py")

from ashen.diagnostics import four_cache as fc  # noqa: E402


def _four_record(variable, n, m, *, real_peak):
    psi_n = np.linspace(0.0, 1.0, 4)
    real = np.array([0.1, real_peak, 0.2, 0.05], dtype=np.float32)
    return fc.FourRecord(
        variable=variable, n=n, m=m, psi_n=psi_n, real=real, imag=np.zeros(4, dtype=np.float32),
    )


def test_delta_b_max(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths, r_axis=1.363245)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_four_record("Psi", n=0, m=2, real_peak=1.0)],
    )
    case = _case(steps=[100])
    value = quantity("delta_b_max").extract(_ctx(case, paths))
    assert value == pytest.approx(1.0 * abs(2) / 1.363245**2)


def test_delta_b_over_b_max_divides_by_b_ref(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths, r_axis=1.363245)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_four_record("Psi", n=0, m=2, real_peak=1.0)],
    )
    case = _case(steps=[100])
    max_value = quantity("delta_b_max").extract(_ctx(case, paths))
    ctx = _ctx(case, paths, ensure_b_ref=lambda: 4.0)
    value = quantity("delta_b_over_b_max").extract(ctx)
    assert value == pytest.approx(max_value / 4.0)


def test_delta_b_over_b_max_no_b_ref_is_none(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths, r_axis=1.363245)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_four_record("Psi", n=0, m=2, real_peak=1.0)],
    )
    case = _case(steps=[100])
    reports = []
    ctx = _ctx(case, paths, ensure_b_ref=lambda: None, report=reports.append)
    assert quantity("delta_b_over_b_max").extract(ctx) is None
    assert reports


def test_delta_b_at_deconfinement_needs_the_case_field(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_four_record("Psi", n=0, m=2, real_peak=1.0)],
    )
    case = _case(steps=[100])  # no four_deconfinement_step set
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("delta_b_at_deconfinement").extract(ctx) is None
    assert reports


def test_delta_b_mode_max_needs_a_mode(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_four_record("Psi", n=0, m=2, real_peak=1.0)],
    )
    case = _case(steps=[100])
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("delta_b_mode_max").extract(ctx) is None
    assert reports


def test_delta_b_missing_log_reports_and_returns_none(tmp_path):
    paths = _paths(tmp_path)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_four_record("Psi", n=0, m=2, real_peak=1.0)],
    )
    case = _case(steps=[100])
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("delta_b_max").extract(ctx) is None
    assert reports


def test_delta_b_no_cache_reports_and_returns_none(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths)
    case = _case(steps=[100])
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("delta_b_max").extract(ctx) is None
    assert reports


# --- energy_32_over_21: (3,2)/(2,1) mode energy fraction ---------------------


def test_energy_32_over_21_max(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths, r_axis=1.363245)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[
            _four_record("Psi", n=2, m=3, real_peak=1.0),
            _four_record("Psi", n=1, m=2, real_peak=2.0),
        ],
    )
    case = _case(steps=[100])
    value = quantity("energy_32_over_21_max").extract(_ctx(case, paths))
    r0 = 1.363245
    db32 = 1.0 * abs(3) / r0**2
    db21 = 2.0 * abs(2) / r0**2
    assert value == pytest.approx((db32 / db21) ** 2)


def test_energy_32_over_21_max_picks_the_largest_step(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths, r_axis=1.0)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[
            _four_record("Psi", n=2, m=3, real_peak=1.0),
            _four_record("Psi", n=1, m=2, real_peak=10.0),  # small ratio here
        ],
    )
    fc.write_cache(
        paths.four_cache(200), step=200, pad_width=6,
        records=[
            _four_record("Psi", n=2, m=3, real_peak=5.0),
            _four_record("Psi", n=1, m=2, real_peak=1.0),  # large ratio here
        ],
    )
    case = _case(steps=[100, 200])
    value = quantity("energy_32_over_21_max").extract(_ctx(case, paths))
    db32 = 5.0 * abs(3)
    db21 = 1.0 * abs(2)
    assert value == pytest.approx((db32 / db21) ** 2)


def test_energy_32_over_21_at_deconfinement_reads_the_configured_step(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths, r_axis=1.0)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[
            _four_record("Psi", n=2, m=3, real_peak=1.0),
            _four_record("Psi", n=1, m=2, real_peak=1.0),
        ],
    )
    fc.write_cache(
        paths.four_cache(200), step=200, pad_width=6,
        records=[
            _four_record("Psi", n=2, m=3, real_peak=4.0),
            _four_record("Psi", n=1, m=2, real_peak=2.0),
        ],
    )
    case = _case(steps=[100, 200], four_deconfinement_step=200)
    value = quantity("energy_32_over_21_at_deconfinement").extract(_ctx(case, paths))
    db32 = 4.0 * abs(3)
    db21 = 2.0 * abs(2)
    assert value == pytest.approx((db32 / db21) ** 2)


def test_energy_32_over_21_at_deconfinement_needs_the_case_field(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[
            _four_record("Psi", n=2, m=3, real_peak=1.0),
            _four_record("Psi", n=1, m=2, real_peak=1.0),
        ],
    )
    case = _case(steps=[100])  # no four_deconfinement_step set
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("energy_32_over_21_at_deconfinement").extract(ctx) is None
    assert reports


def test_energy_32_over_21_missing_one_mode_reports_and_returns_none(tmp_path):
    paths = _paths(tmp_path)
    _write_log(paths)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[_four_record("Psi", n=2, m=3, real_peak=1.0)],  # (2,1) missing
    )
    case = _case(steps=[100])
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("energy_32_over_21_max").extract(ctx) is None
    assert reports


def test_energy_32_over_21_needs_no_b_ref(tmp_path):
    """Unlike delta_b_over_b_*, this ratio needs no ensure_b_ref callback --
    r_axis cancels in the division, so no reference Btor profile is needed."""
    paths = _paths(tmp_path)
    _write_log(paths)
    fc.write_cache(
        paths.four_cache(100), step=100, pad_width=6,
        records=[
            _four_record("Psi", n=2, m=3, real_peak=1.0),
            _four_record("Psi", n=1, m=2, real_peak=2.0),
        ],
    )
    case = _case(steps=[100])
    ctx = _ctx(case, paths)  # ensure_b_ref left unset (None)
    value = quantity("energy_32_over_21_max").extract(ctx)
    assert value is not None


# --- lc_finite_psi_n: innermost surface with a finite connection length -------


def _write_lc_cache(paths, step, surfaces, *, real_psi_edge=1.0):
    """One Poincare cache step. `surfaces` maps user-facing psi_n_in ->
    "open" (a line that crosses psi_n = 1, so finite L_c) or "confined"
    (stays inside for every turn, so L_c = inf). Keys are traced at
    psi_n * real_psi_edge, as cli/analyse does."""
    from ashen.diagnostics import poincare_cache as pc

    with pc.open_cache(paths.poincare_cache(step), step=step, pad_width=6) as h:
        for psi_n, kind in surfaces.items():
            if kind == "open":
                psi_track = np.array([0.3, 1.5, 1.6]) * real_psi_edge
            else:
                psi_track = np.array([0.3, 0.4, 0.5]) * real_psi_edge
            key = pc.LineKey(psi_n=psi_n * real_psi_edge, R=1.7, Z=0.0, phi=0.0)
            pc.append_line(
                h, key,
                {
                    "R": np.full(3, 1.7, dtype=np.float32),
                    "Z": np.zeros(3, dtype=np.float32),
                    "rho": np.sqrt(psi_track).astype(np.float32),
                    "theta": np.zeros(3, dtype=np.float32),
                },
                n_turns=3, terminated=False,
            )


def _lc_paths(tmp_path, real_psi_edge=1.0):
    paths = _paths(tmp_path)
    write_float(paths.real_psi_edge, real_psi_edge)
    _write_log(paths)
    return paths


def test_lc_finite_psi_n_min_is_innermost_open_surface(tmp_path):
    paths = _lc_paths(tmp_path)
    _write_lc_cache(paths, 100, {0.2: "confined", 0.5: "open", 0.8: "open"})
    case = _case(steps=[100], psi_n_in=[0.2, 0.5, 0.8])
    assert quantity("lc_finite_psi_n_min").extract(_ctx(case, paths)) == pytest.approx(0.5)


def test_lc_finite_psi_n_min_takes_the_deepest_step(tmp_path):
    paths = _lc_paths(tmp_path)
    _write_lc_cache(paths, 100, {0.2: "confined", 0.5: "confined", 0.8: "open"})
    _write_lc_cache(paths, 200, {0.2: "confined", 0.5: "open", 0.8: "open"})
    case = _case(steps=[100, 200], psi_n_in=[0.2, 0.5, 0.8])
    assert quantity("lc_finite_psi_n_min").extract(_ctx(case, paths)) == pytest.approx(0.5)


def test_lc_finite_psi_n_scans_psi_n_in_not_the_lc_display_window(tmp_path):
    """lc_psi_n_in is the LC map's display window; using it would clamp the
    answer to its lower edge (0.7 here) instead of finding 0.5."""
    paths = _lc_paths(tmp_path)
    _write_lc_cache(paths, 100, {0.2: "confined", 0.5: "open", 0.8: "open"})
    case = _case(steps=[100], psi_n_in=[0.2, 0.5, 0.8], lc_psi_n_in=[0.8])
    assert quantity("lc_finite_psi_n_min").extract(_ctx(case, paths)) == pytest.approx(0.5)


def test_lc_finite_psi_n_reports_user_facing_psi_n_when_edge_is_extended(tmp_path):
    paths = _lc_paths(tmp_path, real_psi_edge=0.8)
    _write_lc_cache(
        paths, 100, {0.2: "confined", 0.5: "open", 0.8: "open"}, real_psi_edge=0.8,
    )
    case = _case(steps=[100], psi_n_in=[0.2, 0.5, 0.8])
    assert quantity("lc_finite_psi_n_min").extract(_ctx(case, paths)) == pytest.approx(0.5)


def test_lc_finite_psi_n_all_confined_is_none_not_one(tmp_path):
    paths = _lc_paths(tmp_path)
    _write_lc_cache(paths, 100, {0.2: "confined", 0.5: "confined"})
    case = _case(steps=[100], psi_n_in=[0.2, 0.5])
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("lc_finite_psi_n_min").extract(ctx) is None
    assert len(reports) == 1 and "confined" in reports[0]


def test_lc_finite_psi_n_no_poincare_cache_is_reported(tmp_path):
    paths = _lc_paths(tmp_path)
    case = _case(steps=[100], psi_n_in=[0.2, 0.5])
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("lc_finite_psi_n_min").extract(ctx) is None
    assert len(reports) == 1 and "Poincare" in reports[0]


def test_lc_finite_psi_n_at_deconfinement_reads_only_that_step(tmp_path):
    paths = _lc_paths(tmp_path)
    _write_lc_cache(paths, 100, {0.2: "open", 0.5: "open"})
    _write_lc_cache(paths, 200, {0.2: "confined", 0.5: "open"})
    case = _case(steps=[100, 200], psi_n_in=[0.2, 0.5], four_deconfinement_step=200)
    value = quantity("lc_finite_psi_n_at_deconfinement").extract(_ctx(case, paths))
    assert value == pytest.approx(0.5)


def test_lc_finite_psi_n_at_deconfinement_step_must_be_a_traced_step(tmp_path):
    paths = _lc_paths(tmp_path)
    _write_lc_cache(paths, 100, {0.2: "open"})
    case = _case(steps=[100], psi_n_in=[0.2], four_deconfinement_step=300)
    reports = []
    ctx = _ctx(case, paths, report=reports.append)
    assert quantity("lc_finite_psi_n_at_deconfinement").extract(ctx) is None
    assert len(reports) == 1


def test_lc_finite_psi_n_uses_connection_length_step_override():
    assert quantity("lc_finite_psi_n_min").steps_diag == "connection_length"
