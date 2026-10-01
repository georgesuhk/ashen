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
