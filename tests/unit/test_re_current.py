"""ashen.diagnostics.re_current -- the runaway / thermal split of the current."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ashen.diagnostics import re_current as rc  # noqa: E402
from ashen.paths import RunPaths  # noqa: E402

STEPS = [0, 100, 200]


def _folder(tmp_path, *, sign="1", re_share=(1.0, 0.6, 0.2)):
    """A run folder with both profiles and a zeroD file per step. The total
    current is 400 kA throughout; the runaways carry ``re_share`` of it."""
    run = tmp_path / "run"
    (run / "postproc").mkdir(parents=True)
    if sign is not None:
        (run / "in_main").write_text(f" &in1\n vpar_re_sign = {sign}\n&end\n")
    paths = RunPaths(run, pad_width=6)
    psi_n = np.linspace(0.0, 1.0, 41)
    j_total = 2e6 * (1 - psi_n**2)
    for step, share in zip(STEPS, re_share):
        np.savez(paths.profile_cache("Psi_N", "currdens", step, rc.TOR_MODE), x=psi_n, y=j_total)
        # written back to front: the reader must not rely on the order
        np.savez(paths.profile_cache("Psi_N", "recurrdens", step, rc.TOR_MODE),
                 x=psi_n[::-1], y=(share * j_total)[::-1])
        ipre = -float(sign or 1) * share * 4e5       # JOREK's own sign for Ipre_tot
        paths.zero_d(step).write_text(
            f"Time Ip_tot Ipre_tot\n{step * 1e-6} 4.0e5 {ipre}\n"
        )
    return run, paths


def test_thermal_density_is_total_minus_runaway(tmp_path):
    _, paths = _folder(tmp_path)
    series = rc.current_density_series(paths, STEPS)
    assert sorted(series) == STEPS
    for step, share in zip(STEPS, (1.0, 0.6, 0.2)):
        d = series[step]
        assert np.all(np.diff(d.psi_n) > 0)
        np.testing.assert_allclose(d.re, share * d.total)
        np.testing.assert_allclose(d.thermal, (1 - share) * d.total, atol=1e-6)
    assert series[0].total[0] == pytest.approx(2e6)


def test_a_step_missing_one_profile_is_left_out(tmp_path):
    _, paths = _folder(tmp_path)
    paths.profile_cache("Psi_N", "recurrdens", 100, rc.TOR_MODE).unlink()
    assert sorted(rc.current_density_series(paths, STEPS)) == [0, 200]


@pytest.mark.parametrize("sign", ["1", "-1"])
def test_total_runaway_current_is_put_in_ip_tot_s_sign_convention(tmp_path, sign):
    """Ipre_tot carries -sign(vpar_re_sign) relative to Ip_tot (see the
    module docstring): a run that is all runaway current has re == total,
    whichever way the runaways go."""
    _, paths = _folder(tmp_path, sign=sign)
    assert rc.vpar_re_sign(paths) == float(sign)
    totals = rc.current_totals(paths, STEPS, rc.vpar_re_sign(paths))
    np.testing.assert_allclose(totals.total, 4e5)
    np.testing.assert_allclose(totals.re, [4e5, 2.4e5, 0.8e5])
    np.testing.assert_allclose(totals.thermal, [0.0, 1.6e5, 3.2e5], atol=1e-6)
    np.testing.assert_allclose(totals.time, [0.0, 1e-4, 2e-4])


def test_the_real_run_s_numbers():
    """qa3.3_g3.2/eta1e-3_adv0.1, step 26000 (vpar_re_sign = 1): Ip_tot =
    439827.87 A, Ipre_tot = -437498.51 A. The runaways carry all but 2.3 kA."""
    assert rc.total_re_current(-437498.5103886531, 1.0) == pytest.approx(437498.51, abs=0.01)
    assert 439827.8748655455 - rc.total_re_current(-437498.5103886531, 1.0) == pytest.approx(
        2329.36, abs=0.01)


def test_without_vpar_re_sign_the_runaway_total_is_not_guessed(tmp_path):
    _, paths = _folder(tmp_path, sign=None)
    assert rc.vpar_re_sign(paths) is None
    totals = rc.current_totals(paths, STEPS, None)
    assert np.isfinite(totals.total).all()
    assert np.isnan(totals.re).all() and np.isnan(totals.thermal).all()


def test_a_step_without_zerod_is_nan(tmp_path):
    _, paths = _folder(tmp_path)
    paths.zero_d(100).unlink()
    totals = rc.current_totals(paths, STEPS, 1.0)
    assert np.isnan(totals.total[1]) and np.isfinite(totals.total[[0, 2]]).all()


def test_gather_asks_for_both_expressions_on_the_outer_midplane(tmp_path, monkeypatch):
    asked = {}

    def fake(run, paths, steps, variables, **kwargs):
        asked.update(steps=steps, variables=variables, **kwargs)
        return {rc.TOR_MODE: 6}

    monkeypatch.setattr(rc, "gather_profiles", fake)
    assert rc.gather_re_current("run", "paths", STEPS, n_points=80, n_workers=1) == 6
    assert asked["variables"] == ["currdens", "recurrdens"]
    assert (asked["coords_var"], asked["tor_modes"], asked["n_points"]) == (
        "Psi_N", ["midplane outer"], 80)


# --- figure, CLI, viewer ----------------------------------------------------------


def test_figure_draws_three_species_in_both_panels(tmp_path):
    from matplotlib.figure import Figure

    from ashen.plotting.re_current import COLORS, plot_re_current, re_current_figure

    _, paths = _folder(tmp_path)
    totals = rc.current_totals(paths, STEPS, 1.0)
    densities = rc.current_density_series(paths, STEPS)
    fig = re_current_figure(Figure(), totals, densities, step=100, real_psi_edge=0.8)
    ax_t, ax_j = fig.axes
    for ax in (ax_t, ax_j):
        colours = {line.get_color() for line in ax.get_lines()}
        assert set(COLORS.values()) <= colours
    assert "step 100" in ax_j.get_title(loc="left") and ax_t.get_xlabel() == "t [ms]"
    thermal = next(l for l in ax_j.get_lines() if l.get_color() == COLORS["thermal"])
    np.testing.assert_allclose(thermal.get_ydata(), densities[100].thermal / 1e6)

    out = plot_re_current(totals, densities, tmp_path / "figs" / "re_current.png")
    assert out.is_file() and out.stat().st_size > 1000


def test_figure_says_what_is_missing(tmp_path):
    from matplotlib.figure import Figure

    from ashen.plotting.re_current import re_current_figure

    _, paths = _folder(tmp_path, sign=None)
    fig = re_current_figure(Figure(), rc.current_totals(paths, STEPS, None), {})
    ax_t, ax_j = fig.axes
    assert "vpar_re_sign" in ax_t.texts[0].get_text()
    assert "no currdens / recurrdens profile" in ax_j.texts[0].get_text()


def test_viewer_section(tmp_path, monkeypatch):
    w = pytest.importorskip("ipywidgets")
    pytest.importorskip("h5py")
    from ashen import viewer

    run, paths = _folder(tmp_path)
    assert viewer.re_current_steps(run) == STEPS
    assert len(viewer.re_current_figure(run, 100).axes) == 2

    button, _, holder = viewer.re_current_view(run).children
    assert button.description == "Run analyse --diag re_current"
    step, canvas, _ = holder.children[0].children
    before = bytes(canvas.children[0].value)
    step.value = 0
    assert bytes(canvas.children[0].value) != before
    assert any("re_current" in title for title, _, _ in viewer._SECTIONS)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert isinstance(viewer.re_current_view(empty).children[2].children[0], w.HTML)


# --- thermal / runaway against time and psi_N ---------------------------------------


def test_ratio_map_is_thermal_over_runaway_on_one_grid(tmp_path):
    _, paths = _folder(tmp_path, re_share=(0.8, 0.5, 0.2))
    ratio_map = rc.current_ratio_map(rc.current_density_series(paths, STEPS), n_psi=50)
    assert ratio_map.steps == STEPS and ratio_map.ratio.shape == (3, 50)
    inside = ratio_map.psi_n < 0.9
    for row, share in zip(ratio_map.ratio, (0.8, 0.5, 0.2)):
        np.testing.assert_allclose(row[inside], (1 - share) / share)
    # j_total = 0 at psi_N = 1: no current to take a ratio of
    assert np.isnan(ratio_map.ratio[:, -1]).all()


def test_ratio_map_special_values():
    psi_n = np.linspace(0, 1, 11)
    one = np.ones(11)
    densities = {
        0: rc.CurrentDensity(psi_n, one, 0 * one, one),                  # no runaways at all
        1: rc.CurrentDensity(psi_n, one, 2 * one, -one),                 # thermal runs against
        2: rc.CurrentDensity(psi_n[:6], one[:6], one[:6] / 2, one[:6] / 2),   # reaches 0.5 only
    }
    ratio_map = rc.current_ratio_map(densities, n_psi=11)
    assert np.isposinf(ratio_map.ratio[0]).all()
    np.testing.assert_allclose(ratio_map.ratio[1], -0.5)
    np.testing.assert_allclose(ratio_map.ratio[2, :6], 1.0)
    assert np.isnan(ratio_map.ratio[2, 6:]).all()                        # not extrapolated
    assert rc.current_ratio_map({}).ratio.shape[0] == 0


def test_ratio_map_figure_colours_red_for_runaway_and_green_for_thermal(tmp_path):
    from matplotlib.figure import Figure

    from ashen.plotting.re_current import (
        RATIO_RANGE, plot_current_ratio_map, ratio_map_figure,
    )

    _, paths = _folder(tmp_path, re_share=(0.999, 0.5, 0.001))
    ratio_map = rc.current_ratio_map(rc.current_density_series(paths, STEPS), n_psi=40)
    fig = ratio_map_figure(Figure(), ratio_map, [0.0, 1e-4, 2e-4], real_psi_edge=0.8)
    ax = fig.axes[0]
    mesh = ax.collections[0]
    assert (mesh.norm.vmin, mesh.norm.vmax) == RATIO_RANGE and ax.get_xlabel() == "t [ms]"
    colour = lambda ratio: mesh.cmap(mesh.norm(ratio))        # noqa: E731
    r, g, _, _ = colour(1e-3)
    assert r > 0.6 and g < 0.3                                # all runaway: red
    r, g, _, _ = colour(1e3)
    assert g > 0.3 and r < 0.2                                # all thermal: green
    values = mesh.get_array().reshape(40, 3)
    assert values[5, 0] < 0.01 and values[5, 1] == pytest.approx(1.0) and values[5, 2] > 100

    out = plot_current_ratio_map(ratio_map, tmp_path / "f" / "ratio.png", time=None)
    assert out.is_file()

    one_step = rc.current_ratio_map({0: rc.current_density_series(paths, STEPS)[0]})
    assert "two steps" in ratio_map_figure(Figure(), one_step).axes[0].texts[0].get_text()


def test_viewer_shows_the_ratio_map_under_the_profiles(tmp_path):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("h5py")
    from ashen import viewer

    run, _ = _folder(tmp_path)
    assert len(viewer.re_current_ratio_figure(run).axes) == 3          # map, l_i, colour bar
    _, _, holder = viewer.re_current_view(run).children
    assert len(holder.children[0].children) == 3
    assert bytes(holder.children[0].children[2].children[0].value[:4]) == b"\x89PNG"


# --- l_i under the ratio map --------------------------------------------------------


def test_li_series_reads_li3_and_the_plasma_s_own_where_it_can(tmp_path):
    from ashen.diagnostics.equilibrium import achieved_q_li, li_series

    run = tmp_path / "run"
    (run / "postproc").mkdir(parents=True)
    paths = RunPaths(run, pad_width=6)
    psi_n = np.linspace(0.01, 0.99, 50)
    for step, li3 in ((0, 1.5), (100, 1.2)):
        paths.zero_d(step).write_text(f"psi_axis psi_bnd R_axis li3\n-1.0 0.0 1.5 {li3}\n")
    lines = ["# Psi_n q", "# time step #000000"] + [f"{p} {1 + 2.5 * p}" for p in psi_n]
    paths.qprofile(0).write_text("\n".join(lines) + "\n")       # step 100 has no q-profile

    li = li_series(paths, [0, 100, 200], f0=3.0)
    np.testing.assert_allclose(li.li3[:2], [1.5, 1.2])
    assert np.isnan(li.li3[2])                                  # no zeroD
    assert li.li_plasma[0] == pytest.approx(achieved_q_li(paths, 0, f0=3.0).li)
    assert np.isnan(li.li_plasma[1:]).all()
    assert np.isnan(li_series(paths, [0], f0=None).li_plasma).all()   # no F0: li3 only


def test_ratio_map_figure_puts_li_under_the_map_on_the_same_time_axis(tmp_path):
    from matplotlib.figure import Figure

    from ashen.diagnostics.equilibrium import LiSeries
    from ashen.plotting.re_current import plot_current_ratio_map, ratio_map_figure

    _, paths = _folder(tmp_path)
    ratio_map = rc.current_ratio_map(rc.current_density_series(paths, STEPS))
    li = LiSeries(STEPS, np.array([1.5, 1.4, 1.2]), np.array([1.3, np.nan, 1.1]))
    fig = ratio_map_figure(Figure(layout="constrained"), ratio_map, [0.0, 1e-4, 2e-4], li=li)
    ax_map, ax_li, _bar = fig.axes
    assert ax_li.get_xlabel() == "t [ms]" and ax_map.get_xlabel() == ""
    assert ax_map.get_shared_x_axes().joined(ax_map, ax_li)
    by_label = {line.get_label(): line for line in ax_li.get_lines()}
    assert len(by_label) == 2
    li3 = next(l for name, l in by_label.items() if "zeroD" in name)
    np.testing.assert_allclose(li3.get_ydata(), [1.5, 1.4, 1.2])
    np.testing.assert_allclose(li3.get_xdata(), [0.0, 0.1, 0.2])

    only_li3 = LiSeries(STEPS, np.array([1.5, 1.4, 1.2]), np.full(3, np.nan))
    fig = ratio_map_figure(Figure(layout="constrained"), ratio_map, None, li=only_li3)
    assert len(fig.axes[1].get_lines()) == 1 and fig.axes[1].get_xlabel() == "Time step"

    assert plot_current_ratio_map(ratio_map, tmp_path / "f" / "r.png", li=li).is_file()
    assert len(ratio_map_figure(Figure(), ratio_map).axes) == 2      # without li: as before
