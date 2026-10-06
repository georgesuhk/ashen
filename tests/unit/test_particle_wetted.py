"""ashen.diagnostics.particle_wetted and `plot --diag particle_wetted`."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ashen.cli import plot as plot_cli
from ashen.diagnostics.particle_exits import ParticleHistory
from ashen.diagnostics.particle_wetted import Wall, WallHits, wall_hits, wetted_area

h5py = pytest.importorskip("h5py")

R0, A = 1.7, 0.5  # a circular wall: centre R0, radius A


def circle(n=720, clockwise=False):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if clockwise:
        theta = theta[::-1]
    return np.column_stack([R0 + A * np.cos(theta), A * np.sin(theta)])


@pytest.fixture
def wall():
    return Wall.from_points(circle())


# --- the wall ------------------------------------------------------------------------


def test_wall_starts_at_the_outboard_midplane_counter_clockwise():
    for clockwise in (False, True):
        wall = Wall.from_points(circle(clockwise=clockwise))
        assert (wall.R[0], wall.Z[0]) == pytest.approx((R0 + A, 0.0))
        assert wall.Z[5] > 0  # going up from the outboard midplane first


def test_wall_length_and_area(wall):
    assert wall.length == pytest.approx(2 * np.pi * A, rel=1e-4)
    assert wall.area == pytest.approx(4 * np.pi**2 * A * R0, rel=1e-4)
    assert wall.int_R_dl(wall.length) == pytest.approx(wall.area / (2 * np.pi))
    # The upper half is half the area (up-down symmetric); its outboard
    # quarter is more than a quarter: int_0^(pi/2) (R0 + A cos) A = A (R0 pi/2 + A).
    total = wall.int_R_dl(wall.length)
    assert wall.int_R_dl(wall.length / 2) / total == pytest.approx(0.5, rel=1e-6)
    assert wall.int_R_dl(wall.length / 4) / total == pytest.approx(
        0.25 + A / (2 * np.pi * R0), rel=1e-4)


def test_closed_input_is_accepted():
    pts = circle()
    assert Wall.from_points(np.vstack([pts, pts[:1]])).length == pytest.approx(
        Wall.from_points(pts).length)


def test_crossing_and_nearest(wall):
    # Straight up from the centre meets the wall at theta = pi/2.
    frac, l = wall.crossing(np.array([R0, 0.0]), np.array([R0, 2 * A]))
    assert frac == pytest.approx(0.5, abs=1e-3)
    assert l == pytest.approx(A * np.pi / 2, rel=1e-3)
    assert wall.crossing(np.array([R0, 0.0]), np.array([R0, 0.1])) is None
    assert wall.nearest(np.array([R0 - 2 * A, 0.0])) == pytest.approx(A * np.pi, rel=1e-3)


# --- where the particles hit --------------------------------------------------------


def history(R, Z, phi, lost):
    R, Z, phi, lost = (np.asarray(x, dtype=float) for x in (R, Z, phi, lost))
    return ParticleHistory(path=None, time=np.arange(R.shape[0]) * 1e-6, psi_n=np.zeros_like(R),
                           R=R, Z=Z, phi=phi, theta=None, lost=lost > 0)


def test_hits_are_interpolated_crossings(wall):
    # Particle 0 goes straight up, from r = 0.4 to r = 0.6 between rows 1 and 2,
    # phi 1 -> 2: it crosses r = 0.5 halfway, at theta = pi/2, phi 1.5.
    # Particle 1 stays inside; particle 2 starts outside and is not counted.
    h = history(
        R=[[R0, R0, R0 + 0.7], [R0, R0 + 0.1, R0 + 0.7], [R0, R0 + 0.1, R0 + 0.7]],
        Z=[[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]],
        phi=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        lost=np.zeros((3, 3)),
    )
    hits = wall_hits(h, wall)
    assert (hits.n, hits.n_crossed, hits.n_left_grid, hits.n_considered) == (1, 1, 0, 2)
    assert hits.l[0] == pytest.approx(A * np.pi / 2, rel=1e-3)
    assert hits.phi[0] == pytest.approx(1.5, rel=1e-3)


def test_leaving_the_grid_first_is_placed_at_the_nearest_wall_point(wall):
    h = history(R=[[R0 - 0.45], [0.0]], Z=[[0.0], [0.0]], phi=[[7.0], [0.0]], lost=[[0], [1]])
    hits = wall_hits(h, wall)
    assert (hits.n_crossed, hits.n_left_grid) == (0, 1)
    assert hits.l[0] == pytest.approx(A * np.pi, rel=1e-3)
    assert hits.phi[0] == pytest.approx(7.0 - 2 * np.pi)


# --- how widely ----------------------------------------------------------------------


def hits_at(l, phi):
    return WallHits(np.asarray(l, float), np.asarray(phi, float), len(l), 0, len(l))


def test_hits_everywhere_wet_everything(wall):
    rng = np.random.default_rng(1)
    n = 200_000
    # Uniform per unit area: phi uniform, l weighted by R.
    l = rng.uniform(0, wall.length, 4 * n)
    keep = rng.uniform(0, R0 + A, l.size) < np.interp(l, wall.cum_l[:-1], wall.R)
    l = l[keep][:n]
    r = wetted_area(hits_at(l, rng.uniform(0, 2 * np.pi, l.size)), wall,
                    n_l=12, n_phi=12, n_boot=20)
    assert (r.f_pol, r.f_tor, r.f_tot, r.s) == pytest.approx((1, 1, 1, 1), abs=0.01)
    assert r.area == pytest.approx(wall.area, rel=0.01)
    assert 0 < r.f_tot_err < 0.01


def test_one_cell_wets_one_cell(wall):
    r = wetted_area(hits_at([0.01] * 50, [0.01] * 50), wall, n_l=10, n_phi=8, n_boot=0)
    assert r.f_tor == pytest.approx(1 / 8)
    assert r.f_tot == pytest.approx(r.cell_area[0, 0] / wall.area)
    assert r.s == pytest.approx(1.0)
    assert np.isnan(r.f_tot_err)


def test_separable_footprint_has_s_one_and_helical_below(wall):
    n = 16
    l_mid = (np.arange(n) + 0.5) * wall.length / n
    phi_mid = (np.arange(n) + 0.5) * 2 * np.pi / n
    grid_l, grid_phi = np.meshgrid(l_mid[:4], phi_mid[:8], indexing="ij")
    band = wetted_area(hits_at(grid_l.ravel(), grid_phi.ravel()), wall,
                       n_l=n, n_phi=n, n_boot=0)
    assert band.s == pytest.approx(1.0)
    helix = wetted_area(hits_at(l_mid, phi_mid), wall, n_l=n, n_phi=n, n_boot=0)
    # Every poloidal and toroidal bin once, but only n of n^2 cells.
    assert helix.f_tor == pytest.approx(1.0)
    assert helix.s < 0.1


# --- plot --diag particle_wetted -----------------------------------------------------

CASES = """
[cases.run]
steps             = [3000]
ptrace_exe        = "./exe/ptrace_gc"
ptrace_start_step = 3000
ptrace_wetted_bins = [8, 6]
"""


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "jorek03000.h5").write_bytes(b"")
    (tmp_path / "cases.toml").write_text(CASES, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASHEN_SITE", raising=False)
    return tmp_path


def write_diag(path, R, Z, phi, lost):
    path.parent.mkdir(parents=True, exist_ok=True)
    R = np.asarray(R, float)
    with h5py.File(path, "w") as f:
        g = f.create_group("groups/001")
        g["t"] = np.arange(R.shape[0], dtype=np.float32) * 1e-6
        for name, data in dict(psi_n=np.zeros_like(R), R=R, Z=Z, phi=phi, lost=lost).items():
            g[name] = np.asarray(data)


def test_plot_writes_the_figure_and_the_numbers(campaign, capsys):
    np.savetxt(campaign / "run" / "original_bnd.dat", circle(200))
    folder = campaign / "run" / "ptrace" / "ptrace_gc"
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    R = [R0 + 0.3 * np.cos(angles), R0 + 0.7 * np.cos(angles)]
    Z = [0.3 * np.sin(angles), 0.7 * np.sin(angles)]
    write_diag(folder / "ptrace_diag.h5", R, Z, [angles, angles + 0.2], np.zeros((2, 20), int))
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "20 of 20 particles hit the wall" in out
    assert (folder / "particle_wetted.png").is_file()
    numbers = json.loads((folder / "particle_wetted.json").read_text(encoding="utf-8"))
    assert numbers["n_hits"] == 20 and numbers["bins"] == [8, 6]
    assert 0 < numbers["f_tot"] <= numbers["f_pol"] <= 1
    # how much of the trace the hits were collected over: two rows, 1 µs apart
    assert "20 of 20 particles hit the wall, over 1 µs of trace" in out
    assert numbers["duration_microseconds"] == pytest.approx(1.0, rel=1e-5)
    assert numbers["duration"] == pytest.approx(numbers["t_end"] - numbers["t_start"])
    assert numbers["t_start"] == 0.0


def test_no_original_boundary_is_skipped(campaign, capsys):
    folder = campaign / "run" / "ptrace" / "ptrace_gc"
    write_diag(folder / "ptrace_diag.h5", [[R0]], [[0.0]], [[0.0]], [[0]])
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted"]) == 0
    out = capsys.readouterr().out
    assert "no original_bnd.dat" in out and "the wall is the boundary before extend_bnd" in out


def test_wetted_bins_key(tmp_path):
    from ashen.cases import CasesError, load_cases

    path = tmp_path / "cases.toml"
    path.write_text(CASES.replace("[8, 6]", "24"), encoding="utf-8")
    assert load_cases(path)["run"].ptrace_wetted_bins == [24, 24]
    path.write_text(CASES.replace("[8, 6]", "[8, 0]"), encoding="utf-8")
    with pytest.raises(CasesError, match="ptrace_wetted_bins must be"):
        load_cases(path)


def test_wetted_density_range_key(tmp_path):
    from ashen.cases import CasesError, load_cases

    path = tmp_path / "cases.toml"
    path.write_text(CASES, encoding="utf-8")
    assert load_cases(path)["run"].ptrace_wetted_density_range is None
    path.write_text(CASES + "ptrace_wetted_density_range = [0, 2]\n", encoding="utf-8")
    assert load_cases(path)["run"].ptrace_wetted_density_range == [0.0, 2.0]
    for bad in ("[2, 1]", "[-1, 1]", "[1]", "3", '["a", 1]'):
        path.write_text(CASES + f"ptrace_wetted_density_range = {bad}\n", encoding="utf-8")
        with pytest.raises(CasesError, match="ptrace_wetted_density_range must be"):
            load_cases(path)


def test_density_range_sets_the_colour_scale_and_the_label_is_per_area(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from ashen.plotting.particle_wetted import DENSITY_LABEL, plot_wetted_area

    assert DENSITY_LABEL == r"(fraction of total hits) / m$^2$"
    wall = Wall.from_points(np.array([[3.5, -0.5], [4.5, -0.5], [4.5, 0.5], [3.5, 0.5]]))
    rng = np.random.default_rng(1)
    hits = WallHits(rng.uniform(0, wall.length, 200), rng.uniform(0, 2 * np.pi, 200), 200, 0, 200)
    result = wetted_area(hits, wall, n_l=6, n_phi=6, n_boot=2)
    monkeypatch.setattr(plt, "close", lambda *args: None)
    for density_range, expected in (((0.0, 0.001), (0.0, 0.001)), (None, None)):
        plot_wetted_area(result, hits, tmp_path / "w.png", density_range=density_range, dpi=40)
        fig = plt.gcf()
        mesh = fig.axes[0].collections[0]
        if expected is not None:
            assert mesh.get_clim() == expected
        assert any(ax.get_ylabel() == DENSITY_LABEL for ax in fig.axes)
    monkeypatch.undo()
    plt.close("all")


# --- only the markers that started in a psi_n range -------------------------------


def _two_shells(campaign, span=None):
    """20 markers starting at psi_n 0.3 and 20 at 0.7; only the outer 20
    reach the wall."""
    np.savetxt(campaign / "run" / "original_bnd.dat", circle(200))
    folder = campaign / "run" / "ptrace" / "ptrace_gc"
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    inner = R0 + 0.2 * np.cos(angles), 0.2 * np.sin(angles)
    outer0 = R0 + 0.3 * np.cos(angles), 0.3 * np.sin(angles)
    outer1 = R0 + 0.7 * np.cos(angles), 0.7 * np.sin(angles)
    R = [np.concatenate([inner[0], outer0[0]]), np.concatenate([inner[0], outer1[0]])]
    Z = [np.concatenate([inner[1], outer0[1]]), np.concatenate([inner[1], outer1[1]])]
    phi = [np.concatenate([angles, angles])] * 2
    write_diag(folder / "ptrace_diag.h5", R, Z, phi, np.zeros((2, 40), int))
    with h5py.File(folder / "ptrace_diag.h5", "a") as f:
        f["groups/001/psi_n"][...] = np.tile(np.r_[np.full(20, 0.3), np.full(20, 0.7)], (2, 1))
    if span is not None:
        (campaign / "cases.toml").write_text(
            CASES + f"ptrace_initial_psi_n_range = {span}\n", encoding="utf-8")
    return folder


def test_wetted_by_the_markers_that_started_in_a_psi_range(campaign, capsys):
    folder = _two_shells(campaign, "[0.6, 0.8]")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "20 of 20 particles hit the wall" in out
    assert "markers starting at psi_n 0.6 to 0.8: 20 of 40" in out
    # its own files, beside the all-marker ones rather than over them
    assert (folder / "particle_wetted_psi0.6-0.8.png").is_file()
    assert not (folder / "particle_wetted.png").exists()
    numbers = json.loads((folder / "particle_wetted_psi0.6-0.8.json").read_text(encoding="utf-8"))
    assert numbers["initial_psi_n_range"] == [0.6, 0.8]
    assert (numbers["n_markers"], numbers["n_selected"], numbers["n_hits"]) == (40, 20, 20)


def test_wetted_without_a_range_counts_every_marker(campaign, capsys):
    folder = _two_shells(campaign)
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "20 of 40 particles hit the wall" in out and "markers starting" not in out
    numbers = json.loads((folder / "particle_wetted.json").read_text(encoding="utf-8"))
    assert numbers["initial_psi_n_range"] is None
    assert (numbers["n_markers"], numbers["n_selected"]) == (40, 40)


def test_wetted_range_whose_markers_never_reach_the_wall(campaign, capsys):
    _two_shells(campaign, "[0.2, 0.4]")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert ("none of 20 particles hit the wall (markers starting at psi_n 0.2 to 0.4: "
            "20 of 40), skipped") in capsys.readouterr().out


def test_wetted_range_no_marker_started_in(campaign, capsys):
    _two_shells(campaign, "[0.9, 0.95]")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert "none of 40 markers start with psi_n in 0.9..0.95" in capsys.readouterr().out


# --- a trace that stopped before its end -----------------------------------------


def test_wetted_of_an_unfinished_trace_records_how_far_it_got(campaign, capsys):
    """Two rows 1 us apart, of a trace given restarts out to 4 us: a quarter."""
    folder = _two_shells(campaign)
    (folder / "ptrace.log").write_text(
        "ptrace_gc: restart step 3000 at t =   0.000000E+00 s\n"
        "ptrace_gc: restart step 3400 at t =   4.000000E-06 s\n", encoding="utf-8")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "particle_wetted: the trace stops at t = 0.001 ms, 25 % of the way" in out
    assert "trace unfinished: 25 % of the way to the last restart, step 3400" in out
    numbers = json.loads((folder / "particle_wetted.json").read_text(encoding="utf-8"))
    assert numbers["trace_fraction"] == pytest.approx(0.25)
    assert numbers["duration_microseconds"] == pytest.approx(1.0, rel=1e-5)


def test_wetted_of_a_finished_trace_has_no_trace_fraction(campaign):
    folder = _two_shells(campaign)
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    numbers = json.loads((folder / "particle_wetted.json").read_text(encoding="utf-8"))
    assert numbers["trace_fraction"] is None


# --- counts instead of fractions ------------------------------------------------------


def test_counts_mode_draws_numbers_of_particles(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from ashen.plotting.particle_wetted import COUNT_LABEL, plot_wetted_area

    wall = Wall.from_points(np.array([[3.5, -0.5], [4.5, -0.5], [4.5, 0.5], [3.5, 0.5]]))
    rng = np.random.default_rng(2)
    hits = WallHits(rng.uniform(0, wall.length, 300), rng.uniform(0, 2 * np.pi, 300), 300, 0, 300)
    result = wetted_area(hits, wall, n_l=6, n_phi=5, n_boot=2)
    monkeypatch.setattr(plt, "close", lambda *args: None)
    plot_wetted_area(result, hits, tmp_path / "w.png", counts=True, dpi=40)
    fig = plt.gcf()
    ax_map, ax_tor, ax_pol = fig.axes[:3]
    # the map: hits per cell, as counted
    np.testing.assert_array_equal(ax_map.collections[0].get_array().filled(0).reshape(6, 5),
                                  result.counts)
    assert any(ax.get_ylabel() == COUNT_LABEL for ax in fig.axes)
    # the profiles: hits per bin, adding up to every hit
    assert ax_tor.get_ylabel() == "particles" and ax_pol.get_xlabel() == "particles"
    tor_heights = ax_tor.patches[0].get_path().vertices[:, 1].max()
    assert tor_heights == result.counts.sum(axis=0).max()
    monkeypatch.undo()
    plt.close("all")


def test_counts_from_the_flag_or_the_case_key_beside_the_fractions(campaign, capsys):
    folder = _two_shells(campaign)
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert (folder / "particle_wetted.png").is_file()
    assert not (folder / "particle_wetted_counts.png").exists()

    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--wetted-counts",
                          "--dpi", "40"]) == 0
    assert f"-> {folder / 'particle_wetted_counts.png'}" in capsys.readouterr().out
    (folder / "particle_wetted_counts.png").unlink()

    (campaign / "cases.toml").write_text(CASES + "ptrace_wetted_counts = true\n", encoding="utf-8")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert (folder / "particle_wetted_counts.png").is_file()
    # the numbers are the same either way
    numbers = json.loads((folder / "particle_wetted.json").read_text(encoding="utf-8"))
    assert numbers["n_hits"] == 20


def test_counts_with_a_psi_range_carry_both_in_the_name(campaign):
    folder = _two_shells(campaign, "[0.6, 0.8]")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--wetted-counts",
                          "--dpi", "40"]) == 0
    assert (folder / "particle_wetted_counts_psi0.6-0.8.png").is_file()


def test_invalid_counts_key(campaign, capsys):
    _two_shells(campaign)
    (campaign / "cases.toml").write_text(CASES + "ptrace_wetted_counts = 1\n", encoding="utf-8")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted"]) == 1
    assert "ptrace_wetted_counts must be true or false" in capsys.readouterr().err


def test_count_range_sets_the_counts_map_apart_from_the_density_range(campaign, monkeypatch):
    from ashen.cases import CasesError, load_cases

    path = campaign / "cases.toml"
    assert load_cases(path)["run"].ptrace_wetted_count_range is None
    for bad in ("[2, 1]", "[-1, 1]", "[1]", '["a", 1]'):
        path.write_text(CASES + f"ptrace_wetted_count_range = {bad}\n", encoding="utf-8")
        with pytest.raises(CasesError, match="ptrace_wetted_count_range must be .* in particles"):
            load_cases(path)

    _two_shells(campaign)
    path.write_text(CASES + "ptrace_wetted_density_range = [0, 2]\n"
                    "ptrace_wetted_count_range = [0, 30]\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(plot_cli, "plot_wetted_area",
                        lambda *args, counts, density_range, **kw: seen.append(
                            (counts, density_range)) or args[2])
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--wetted-counts",
                          "--dpi", "40"]) == 0
    assert seen == [(False, [0.0, 2.0]), (True, [0.0, 30.0])]
    # without a count range the counts map takes the data's own, not the density's
    seen.clear()
    path.write_text(CASES + "ptrace_wetted_density_range = [0, 2]\n", encoding="utf-8")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--wetted-counts",
                          "--dpi", "40"]) == 0
    assert seen == [(True, None)]


# --- the wall-hit cache ---------------------------------------------------------------


def test_replotting_reuses_the_cached_hits_with_the_same_numbers(campaign, capsys, monkeypatch):
    folder = _two_shells(campaign)
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    first = capsys.readouterr().out
    assert "wall hits from" not in first
    assert (folder / "particle_wetted_cache.npz").is_file()
    numbers = (folder / "particle_wetted.json").read_text(encoding="utf-8")

    # a replot neither reads the diagnostics file nor finds the hits again
    def boom(*args, **kwargs):
        raise AssertionError("recomputed")

    monkeypatch.setattr(plot_cli, "read_particle_diag", boom)
    monkeypatch.setattr(plot_cli, "particle_hits", boom)
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "particle_wetted: wall hits from particle_wetted_cache.npz (ptrace_diag.h5 unchanged)" in out
    assert (folder / "particle_wetted.json").read_text(encoding="utf-8") == numbers
    # the selection, counts and bins are made from the cache too
    (campaign / "cases.toml").write_text(
        CASES.replace("[8, 6]", "[4, 3]") + "ptrace_initial_psi_n_range = [0.6, 0.8]\n",
        encoding="utf-8")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--wetted-counts",
                          "--dpi", "40"]) == 0
    selected = json.loads((folder / "particle_wetted_psi0.6-0.8.json").read_text(encoding="utf-8"))
    assert (selected["n_markers"], selected["n_selected"], selected["n_hits"]) == (40, 20, 20)
    assert selected["bins"] == [4, 3]


def test_cached_selection_matches_computing_it_afresh(campaign, monkeypatch):
    """The numbers from a selection taken from the cache are those of a
    first run with that selection, which reads and selects the history."""
    folder = _two_shells(campaign, "[0.6, 0.8]")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    fresh = (folder / "particle_wetted_psi0.6-0.8.json").read_text(encoding="utf-8")
    (folder / "particle_wetted_psi0.6-0.8.json").unlink()
    (folder / "particle_wetted_cache.npz").unlink()
    (campaign / "cases.toml").write_text(CASES, encoding="utf-8")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    (campaign / "cases.toml").write_text(
        CASES + "ptrace_initial_psi_n_range = [0.6, 0.8]\n", encoding="utf-8")
    monkeypatch.setattr(plot_cli, "read_particle_diag", lambda *a: pytest.fail("recomputed"))
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert (folder / "particle_wetted_psi0.6-0.8.json").read_text(encoding="utf-8") == fresh


def test_a_changed_trace_or_wall_is_recomputed(campaign, capsys):
    import os

    folder = _two_shells(campaign)
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    capsys.readouterr()
    diag = folder / "ptrace_diag.h5"
    # the trace moved on: a newer file
    stat = diag.stat()
    os.utime(diag, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert "wall hits from" not in capsys.readouterr().out
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert "wall hits from" in capsys.readouterr().out
    # another wall
    np.savetxt(campaign / "run" / "original_bnd.dat", circle(100))
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert "wall hits from" not in capsys.readouterr().out
    # another window
    (folder / "ptrace.log").write_text(
        "ptrace_gc: restart step 3000 at t =   0.000000E+00 s\n"
        "ptrace_gc: restart step 3400 at t =   4.000000E-06 s\n", encoding="utf-8")
    (campaign / "cases.toml").write_text(CASES + "ptrace_end_step = 3400\n", encoding="utf-8")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert "wall hits from" not in capsys.readouterr().out
    # an unreadable cache is recomputed, not trusted
    (folder / "particle_wetted_cache.npz").write_bytes(b"not a cache")
    assert plot_cli.main(["--case", "run", "--diag", "particle_wetted", "--dpi", "40"]) == 0
    assert "wall hits from" not in capsys.readouterr().out


def test_particle_hits_selected_equal_wall_hits_of_the_selection(wall):
    rng = np.random.default_rng(5)
    n_t, n = 30, 200
    r = np.clip(np.cumsum(rng.normal(0.02, 0.05, (n_t, n)), axis=0) + 0.3, 0, None)
    a = rng.uniform(0, 2 * np.pi, n) + np.cumsum(rng.normal(0, 0.2, (n_t, n)), axis=0)
    lost = np.zeros((n_t, n), int)
    lost[20:, ::7] = 1
    h = history(R0 + r * np.cos(a), r * np.sin(a), np.cumsum(rng.uniform(0, 1, (n_t, n)), 0), lost)
    from ashen.diagnostics.particle_wetted import particle_hits

    per = particle_hits(h, wall)
    mask = rng.random(n) < 0.5
    a_hits, b_hits = per.wall_hits(mask), wall_hits(h.select(mask), wall)
    assert a_hits.n > 10 and a_hits.n_left_grid > 0
    np.testing.assert_array_equal(a_hits.l, b_hits.l)
    np.testing.assert_array_equal(a_hits.phi, b_hits.phi)
    assert (a_hits.n_crossed, a_hits.n_left_grid, a_hits.n_considered) == (
        b_hits.n_crossed, b_hits.n_left_grid, b_hits.n_considered)
