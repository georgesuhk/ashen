"""ashen.viewer -- figures and notebook views of a run folder."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ashen import current_profile as cur  # noqa: E402
from ashen import viewer  # noqa: E402
from ashen.namelist import write_boundary_file  # noqa: E402
from ashen.paths import RunPaths, write_float  # noqa: E402

h5py = pytest.importorskip("h5py")

from ashen.diagnostics import four_cache as fc  # noqa: E402

STARWALL = """&PARAMS
  n_harm = 1,
/
&PARAMS_WALL
  mn_w       = 2,
  n_w        =    0,      0,         ! Toroidal mode numbers
  m_w        =    0,      1,         ! Poloidal mode numbers
  rc_w       =    1.4,   0.7,       ! R cos-mode
  rs_w       =    0.,     0.,
  zc_w       =    0.,     0.,
  zs_w       =    0.,     0.9d0,
/
"""


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _ellipse(R0=1.5, a=0.5, kappa=1.2, n=120, scale=1.0):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((R0 + scale * a * np.cos(theta), scale * a * kappa * np.sin(theta)))


@pytest.fixture
def run_dir(tmp_path):
    """A run folder with every file the viewer reads, for a circular-ish
    plasma whose psi is quadratic in minor radius."""
    run = tmp_path / "run"
    (run / "postproc").mkdir(parents=True)
    np.savetxt(run / "original_bnd.dat", _ellipse())
    domain = _ellipse(scale=1.2, n=50)
    write_boundary_file(run / "in_bnd", domain[:, 0], domain[:, 1], np.zeros(50), ".6f")
    rows = ["     50     50      1      1      2"]
    rows += [f"  {i} {i} {i + 1}  {r:.8E}  {z:.8E}  0.0 0.0 1.0 1.0" for i, (r, z) in enumerate(domain, 1)]
    rows += ["     59    117    118      2"]
    (run / "boundary.txt").write_text("\n".join(rows) + "\n")
    (run / "input_starwall").write_text(STARWALL)
    (run / "in_eq").write_text(" &in1\n F0 = 3.0\n&end\n")
    write_float(run / "real_psi_edge.dat", 1 / 1.2)

    # restart: nodes on rings, psi = rho^2 (rho = 1 at the domain edge), zj < 0 inside the plasma
    rho, theta = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 2 * np.pi, 24, endpoint=False))
    rho, theta = rho.ravel(), theta.ravel()
    R, Z = 1.5 + 0.6 * rho * np.cos(theta), 0.72 * rho * np.sin(theta)
    values = np.zeros((8, 4, 1, R.size))
    values[0, 0, 0] = rho**2 - 1.0
    values[2, 0, 0] = -5.0 * np.clip(1 - rho**2 * 1.2, 0, None)
    x = np.zeros((2, 4, 1, R.size))
    x[0, 0, 0], x[1, 0, 0] = R, Z
    for step in (0, 100):
        with h5py.File(run / f"jorek{step:06d}.h5", "w") as f:
            f["values"], f["x"] = values, x

    paths = RunPaths(run, pad_width=6)
    psi_n = np.linspace(0.01, 0.99, 50)
    for step in (0, 100):
        lines = ["# Psi_n q", f"# time step #{step:06d}"] + [f"{p} {1 + 2.5 * p}" for p in psi_n]
        paths.qprofile(step).write_text("\n".join(lines) + "\n")
        paths.zero_d(step).write_text("psi_axis psi_bnd R_axis li3 q95\n-1.0 0.0 1.5 1.5 3.4\n")
        grid = np.linspace(0, 1, 20)
        records = [
            fc.FourRecord("Psi", n, m, grid, (step + 1) * 1e-4 * m * grid * (1 - grid) + 1e-9,
                          np.zeros(20))
            for n, m in ((1, 1), (1, 2), (1, 3))
        ]
        fc.write_cache(paths.four_cache(step), step=step, pad_width=6, records=records)
        np.savez(paths.profile_cache("Psi_N", "currdens", step, "midplane outer"),
                 x=grid, y=(1 - grid) * (1 + step / 100))
    return run


# --- readers -----------------------------------------------------------------------


def test_starwall_wall_reads_the_fourier_wall(run_dir):
    wall = viewer.starwall_wall(run_dir / "input_starwall")
    assert wall[:, 0].min() == pytest.approx(0.7, abs=1e-3)
    assert wall[:, 0].max() == pytest.approx(2.1, abs=1e-3)
    assert wall[:, 1].max() == pytest.approx(0.9, abs=1e-3)      # the 0.9d0
    assert viewer.starwall_wall(run_dir / "nope") is None


def test_grid_boundary_reads_the_element_rows_only(run_dir):
    grid = viewer.grid_boundary(run_dir / "boundary.txt")
    assert grid.shape == (50, 2)
    np.testing.assert_allclose(grid, _ellipse(scale=1.2, n=50), atol=1e-7)


def test_case_boundaries_finds_what_is_there(run_dir, tmp_path):
    assert set(viewer.case_boundaries(run_dir)) == {
        "plasma", "domain (in_bnd)", "JOREK grid", "STARWALL wall"
    }
    assert viewer.case_boundaries(tmp_path) == {}


def test_plasma_geometry_from_the_run_folder(run_dir):
    g = viewer.plasma_geometry(run_dir)
    assert (g.R0, g.a, g.kappa, g.B0) == pytest.approx((1.5, 0.5, 1.2, 2.0), rel=1e-3)


def test_profiles_available_parses_cache_names(run_dir):
    key = viewer.ProfileKey("Psi_N", "currdens", "midplane outer")
    assert viewer.profiles_available(run_dir) == {key: [0, 100]}


def test_four_steps(run_dir):
    assert viewer.four_steps(run_dir) == [0, 100]


# --- figures -----------------------------------------------------------------------


def test_tuner_figure_draws_three_panels_and_the_numbers():
    g = cur.PlasmaGeometry(1.5, 0.5, 1.2, 2.0)
    fig = viewer.tuner_figure(1.1, 1.2, 3.0, g)
    assert len(fig.axes) == 3
    assert "q0 = 1.100" in fig._suptitle.get_text() and "Ip" in fig._suptitle.get_text()
    assert len(fig.axes[0].lines) == 1


def test_tuner_figure_outside_the_window_shows_the_message():
    fig = viewer.tuner_figure(0.2, 1.2, 3.0, cur.PlasmaGeometry(1.5, 0.5, 1.2, 2.0))
    assert "outside what l_i" in fig._suptitle.get_text()
    assert not fig.axes[0].lines


def test_tuner_figure_overlays_what_jorek_achieved(run_dir):
    from ashen.diagnostics.equilibrium import achieved_q_li

    g = viewer.plasma_geometry(run_dir)
    achieved = achieved_q_li(RunPaths(run_dir, pad_width=6), 0, f0=3.0)
    fig = viewer.tuner_figure(1.1, 1.2, 3.0, g, achieved=achieved)
    assert "JOREK:" in fig._suptitle.get_text() and "li3 = 1.500" in fig._suptitle.get_text()
    assert len(fig.axes[1].lines) == 2


def test_boundary_figure(run_dir, tmp_path):
    fig = viewer.boundary_figure(run_dir)
    assert len(fig.axes[0].lines) == 4
    assert viewer.boundary_figure(tmp_path).axes[0].texts          # says nothing is there


def test_equilibrium_figure_marks_the_plasma_edge(run_dir):
    fig = viewer.equilibrium_figure(run_dir, 0)
    ax_map, ax_q, ax_j = fig.axes
    labels = [t.get_text() for t in ax_map.get_legend().get_texts()]
    assert any("0.833" in label for label in labels)
    assert len(ax_q.lines) == 2                # q and the edge marker
    # j_phi = -zj / (mu_0 R): positive where zj is negative
    assert ax_j.collections[0].get_offsets()[:, 1].max() > 0


def test_equilibrium_figure_without_a_qprofile_says_so(run_dir):
    RunPaths(run_dir, pad_width=6).qprofile(100).unlink()
    fig = viewer.equilibrium_figure(run_dir, 100)
    assert "no q-profile cache" in fig.axes[1].texts[0].get_text()


def test_four_figure_draws_the_largest_modes_in_matching_colours(run_dir):
    fig = viewer.four_figure(run_dir, "Psi", step=100, n_modes=2)
    ax_t, ax_r = fig.axes
    left = {l.get_label(): l.get_color() for l in ax_t.get_lines() if l.get_label().startswith("n=")}
    right = {l.get_label(): l.get_color() for l in ax_r.get_lines() if l.get_label().startswith("n=")}
    assert set(left) == set(right) == {"n=1, m=2", "n=1, m=3"}
    assert left == right


def test_four_figure_without_caches(tmp_path):
    (tmp_path / "jorek000000.h5").write_bytes(b"")
    assert viewer.four_figure(tmp_path).axes[0].texts


def test_profiles_figure_highlights_a_step(run_dir):
    key = viewer.ProfileKey("Psi_N", "currdens", "midplane outer")
    fig = viewer.profiles_figure(run_dir, key, step=100)
    assert any(l.get_label() == "step 100" for l in fig.axes[0].lines)


# --- notebook wrappers -----------------------------------------------------------


def test_views_build(run_dir):
    pytest.importorskip("ipywidgets")
    for build in (viewer.boundary_view, viewer.equilibrium_view, viewer.four_view,
                  viewer.profiles_view):
        assert build(run_dir) is not None


def test_views_say_when_there_is_nothing(tmp_path):
    w = pytest.importorskip("ipywidgets")
    for build in (viewer.equilibrium_view, viewer.four_view, viewer.profiles_view):
        assert isinstance(build(tmp_path), w.HTML)


SHOTFILE = """qa = 2.1
g = 2.3
eta = 1e-3
tstep_n = [0.03]
nstep_n = [10]
nout = 1
exe = "jorek_test_exe"
jobscript = "23h"
rho_const = 1e18
ffprime_method = "q_li"
T_method = "const"
T_const = 100.0
rho_method = "const"
bnd_method = "file"
bnd_file = "bnd.dat"
bnd_file_is_plasma = True
current_q0 = 1.1
current_li = 1.2
current_q_edge = 3.0
"""


def _tuner_children(box):
    sliders, buttons = box.children[0].children, box.children[1].children
    return sliders, buttons


def test_profile_tuner_saves_and_regenerates(synthetic_campaign, tmp_path, symlinks_maybe_bypassed, capsys):
    pytest.importorskip("ipywidgets")
    from ashen.physics import MU_0
    from ashen.shotfile import load_shotfile

    site, template_dir, _ = synthetic_campaign
    in_eq = template_dir / "copy" / "in_eq"
    in_eq.write_text(in_eq.read_text().replace("&end", " F0 = 3.0\n&end", 1))
    run = tmp_path / "newrun"
    run.mkdir()
    (run / "shotfile.py").write_text(SHOTFILE)
    np.savetxt(run / "bnd.dat", _ellipse())

    box = viewer.profile_tuner(run, site=site)
    (q0, li, q_edge), (save, regenerate) = _tuner_children(box)
    status = box.children[2]
    assert (q0.value, li.value, q_edge.value) == (1.1, 1.2, 3.0)
    assert "have not been made" in status.value            # a new folder

    q0.value = 1.25
    assert "not saved" in status.value
    save.click()
    assert load_shotfile(run / "shotfile.py").current_q0 == 1.25
    assert "not saved" not in status.value and "Regenerate inputs" in status.value

    regenerate.click()
    assert status.value == ""                              # all three agree
    q0.value = 1.3
    save.click()
    assert "out of date" in status.value and "q0 = 1.25" in status.value
    q0.value = 1.25
    save.click()
    assert status.value == ""
    j0 = 2.0 * (1 + 1.2**2) / (MU_0 * 1.5 * 1.2 * 1.25)
    assert np.loadtxt(run / "ffprime_prof.dat")[0, 1] == pytest.approx(-MU_0 * 1.5 * j0, rel=1e-6)
    assert "q0 = 1.25" in (run / "j_prof.dat").read_text()
    assert "run_jorek shotfile.py --run_eq" in capsys.readouterr().out


def test_profile_tuner_does_not_save_an_unreachable_q0(synthetic_campaign, tmp_path, capsys):
    pytest.importorskip("ipywidgets")
    site, template_dir, _ = synthetic_campaign
    in_eq = template_dir / "copy" / "in_eq"
    in_eq.write_text(in_eq.read_text().replace("&end", " F0 = 3.0\n&end", 1))
    run = tmp_path / "newrun"
    run.mkdir()
    (run / "shotfile.py").write_text(SHOTFILE)
    np.savetxt(run / "bnd.dat", _ellipse())

    box = viewer.profile_tuner(run, site=site)
    (q0, _, _), (save, _) = _tuner_children(box)
    q0.value = 0.3
    save.click()

    assert (run / "shotfile.py").read_text() == SHOTFILE
    assert "not saved" in capsys.readouterr().out


# --- tuner_status ------------------------------------------------------------------


def _status_folder(tmp_path, *, method="q_li", generated=(1.1, 1.2, 3.0)):
    run = tmp_path / "statusrun"
    run.mkdir()
    (run / "shotfile.py").write_text(SHOTFILE.replace('"q_li"', f'"{method}"'))
    if generated is not None:
        (run / "j_prof.dat").write_text(
            "# made by run_jorek\n"
            f"# requested: q0 = {generated[0]:g}, l_i = {generated[1]:g}, q_edge = {generated[2]:g}\n"
            "0.0 1.0\n1.0 0.0\n"
        )
    return run


def test_tuner_status_is_empty_when_everything_agrees(tmp_path):
    assert viewer.tuner_status(_status_folder(tmp_path), 1.1, 1.2, 3.0) == []


def test_tuner_status_unsaved_sliders(tmp_path):
    notes = viewer.tuner_status(_status_folder(tmp_path), 1.1, 1.35, 3.0)
    assert len(notes) == 1 and "not saved" in notes[0] and "l_i = 1.2" in notes[0]


def test_tuner_status_saved_but_not_regenerated(tmp_path):
    notes = viewer.tuner_status(_status_folder(tmp_path, generated=(0.9, 1.2, 3.0)), 1.1, 1.2, 3.0)
    assert len(notes) == 1
    assert "out of date" in notes[0] and "q0 = 0.9" in notes[0] and "q0 = 1.1" in notes[0]


def test_tuner_status_both_at_once(tmp_path):
    notes = viewer.tuner_status(_status_folder(tmp_path, generated=(0.9, 1.2, 3.0)), 1.4, 1.2, 3.0)
    assert ["not saved" in notes[0], "out of date" in notes[1]] == [True, True]


def test_tuner_status_inputs_never_generated(tmp_path):
    notes = viewer.tuner_status(_status_folder(tmp_path, generated=None), 1.1, 1.2, 3.0)
    assert len(notes) == 1 and "have not been made" in notes[0]


def test_tuner_status_a_hand_made_j_prof_counts_as_not_generated(tmp_path):
    run = _status_folder(tmp_path, generated=None)
    (run / "j_prof.dat").write_text("0.0 1.0\n1.0 0.0\n")
    assert "have not been made" in viewer.tuner_status(run, 1.1, 1.2, 3.0)[0]


def test_tuner_status_other_ffprime_method(tmp_path):
    run = tmp_path / "castorrun"
    run.mkdir()
    (run / "shotfile.py").write_text(
        SHOTFILE.replace('ffprime_method = "q_li"', 'ffprime_method = "current"\ncurrent_file = "j.dat"')
    )
    notes = viewer.tuner_status(run, 1.1, 1.2, 3.0)
    assert len(notes) == 1 and "ffprime_method = 'current'" in notes[0]
