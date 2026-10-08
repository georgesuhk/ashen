"""ashen.viewer -- figures and notebook views of a run folder."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sys  # noqa: E402

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
    assert isinstance(viewer.equilibrium_view(tmp_path), w.HTML)
    for build, label in ((viewer.four_view, "four"), (viewer.profiles_view, "profiles")):
        button, _, holder = build(tmp_path).children          # the gather button stays
        assert button.description == f"Run analyse --diag {label}"
        assert isinstance(holder.children[0], w.HTML)


SHOTFILE = """qa = 2.1
g = 2.3
eta = 1e-3
tstep_n = [0.03]
nstep_n = [10]
nout = 1
exe = "jorek_test_exe"
jobscript = "23h"
freeboundary = False
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
current_qa = 3.0
"""


def _tuner_children(box):
    sliders, buttons = box.children[0].children, box.children[1].children
    return sliders, buttons[:2]              # save, regenerate; the third is reset


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
            f"# requested: q0 = {generated[0]:g}, l_i = {generated[1]:g}, qa = {generated[2]:g}\n"
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


def test_profile_tuner_on_a_campaign_boundary_before_anything_is_prepared(
    synthetic_campaign, tmp_path, symlinks_maybe_bypassed
):
    pytest.importorskip("ipywidgets")
    from ashen.shotfile import load_shotfile

    site, template_dir, _ = synthetic_campaign
    in_eq = template_dir / "copy" / "in_eq"
    in_eq.write_text(in_eq.read_text().replace("&end", " F0 = 3.0\n&end", 1))
    (template_dir / "symlink" / "boundary").mkdir(parents=True)
    np.savetxt(template_dir / "symlink" / "boundary" / "boundary1.dat", _ellipse())
    run = tmp_path / "newrun"
    run.mkdir()
    (run / "shotfile.py").write_text(
        SHOTFILE.replace('bnd_method = "file"\nbnd_file = "bnd.dat"\nbnd_file_is_plasma = True',
                         'bnd_method = "template"\nbnd_file = "boundary1.dat"')
        .replace("qa = 2.1\ng = 2.3\n", "")
    )
    lines = (run / "shotfile.py").read_text().splitlines()
    assert not any(line.startswith(("qa =", "g =")) for line in lines)

    g = viewer.plasma_geometry(run, site, load_shotfile(run / "shotfile.py"))
    assert (g.R0, g.a, g.kappa) == pytest.approx((1.5, 0.5, 1.2), rel=1e-3)

    box = viewer.profile_tuner(run, site=site)
    (_, _, _), (_, regenerate) = _tuner_children(box)
    regenerate.click()
    assert (run / "ffprime_prof.dat").is_file() and (run / "boundary1.dat").is_file()
    assert box.children[2].value == ""


def test_tuner_status_reads_a_j_prof_written_before_the_rename(tmp_path):
    """j_prof.dat headers used to say q_edge; they still count."""
    run = _status_folder(tmp_path, generated=None)
    (run / "j_prof.dat").write_text(
        "# requested: q0 = 1.1, l_i = 1.2, q_edge = 3\n0.0 1.0\n1.0 0.0\n"
    )
    assert viewer.tuner_status(run, 1.1, 1.2, 3.0) == []


# --- folder_name_mismatches -----------------------------------------------------------


@pytest.mark.parametrize("folder, values, expected", [
    ("qa2.1_li1.0_q01.0/eta1e-3", (1.0, 1.0, 2.1), []),
    ("qa2.1_li1.0_q01.0/eta1e-3", (1.04, 0.96, 2.14), []),              # round to the name
    ("qa2.1_li1.0_q01.0/eta1e-3", (1.0, 1.2, 2.1), ["l_i = 1.0", ]),
    ("qa2.1_li1.0_q01.0/eta1e-3", (1.3, 1.2, 2.8), ["qa = 2.1", "l_i = 1.0", "q0 = 1.0"]),
    ("qa2.1_li1.25_q01/eta1e-3", (1.4, 1.25, 2.1), []),                 # q01: no decimals
    ("qa2.1_li1.25_q01/eta1e-3", (1.6, 1.254, 2.1), ["q0 = 1"]),
    ("scan/qa2.8_li0.85", (1.5, 0.85, 2.8), []),                        # in the run folder's own name
    ("scan/qa2.8_li0.85", (1.5, 0.95, 2.8), ["l_i = 0.85"]),
    ("demo_shot/q_li_demo", (1.0, 1.2, 3.0), []),                       # q_li_demo names no value
    ("equality/quality0.5", (1.0, 1.2, 3.0), []),
    ("qa3.3_g3.2/eta1e-3_adv0.1", (0.9, 1.4, 3.3), []),                 # g is not a profile value
])
def test_folder_name_mismatches(tmp_path, folder, values, expected):
    q0, li, qa = values
    notes = viewer.folder_name_mismatches(tmp_path / folder, q0, li, qa)
    assert len(notes) == len(expected)
    for note, fragment in zip(notes, expected):
        assert f"says {fragment} " in note and "but the profile has" in note


def test_profile_tuner_shows_a_red_band_for_a_misnamed_folder(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    pytest.importorskip("ipywidgets")
    site, template_dir, _ = synthetic_campaign
    in_eq = template_dir / "copy" / "in_eq"
    in_eq.write_text(in_eq.read_text().replace("&end", " F0 = 3.0\n&end", 1))
    run = tmp_path / "qa3.0_li1.2_q01.1" / "eta1e-3"
    run.mkdir(parents=True)
    (run / "shotfile.py").write_text(SHOTFILE)
    np.savetxt(run / "bnd.dat", _ellipse())

    box = viewer.profile_tuner(run, site=site)
    (q0, li, qa), _ = _tuner_children(box)
    status = box.children[2]
    assert "Folder name says" not in status.value               # 1.1, 1.2, 3.0 match the name

    li.value = 1.45
    assert "#f8d7da" in status.value                            # red
    assert "Folder name says l_i = 1.2" in status.value and "l_i = 1.45" in status.value
    assert status.value.index("Folder name says") < status.value.index("not saved")   # red first

    li.value = 1.2
    assert "Folder name says" not in status.value


def test_profile_tuner_reset_puts_the_sliders_back_to_the_shotfile(
    synthetic_campaign, tmp_path, symlinks_maybe_bypassed, capsys
):
    pytest.importorskip("ipywidgets")
    from ashen.shotfile import set_shotfile_values

    site, template_dir, _ = synthetic_campaign
    in_eq = template_dir / "copy" / "in_eq"
    in_eq.write_text(in_eq.read_text().replace("&end", " F0 = 3.0\n&end", 1))
    run = tmp_path / "newrun"
    run.mkdir()
    (run / "shotfile.py").write_text(SHOTFILE)
    np.savetxt(run / "bnd.dat", _ellipse())

    box = viewer.profile_tuner(run, site=site)
    (q0, li, qa), _ = _tuner_children(box)
    reset = box.children[1].children[2]
    status = box.children[2]
    assert reset.description == "Reset to shotfile"

    q0.value, li.value, qa.value = 1.4, 1.0, 4.5
    assert "not saved" in status.value
    reset.click()
    assert (q0.value, li.value, qa.value) == (1.1, 1.2, 3.0)
    assert "not saved" not in status.value
    assert "sliders back to shotfile.py" in capsys.readouterr().out
    assert (run / "shotfile.py").read_text() == SHOTFILE          # reset writes nothing

    # it reads the file as it is now, even a value beyond the slider's travel
    set_shotfile_values(run / "shotfile.py", {"current_qa": 12.0, "current_li": 2.3})
    reset.click()
    assert (qa.value, li.value) == (12.0, 2.3)


def test_profile_tuner_reset_without_saved_values_says_so(synthetic_campaign, tmp_path, capsys):
    pytest.importorskip("ipywidgets")
    site, template_dir, _ = synthetic_campaign
    in_eq = template_dir / "copy" / "in_eq"
    in_eq.write_text(in_eq.read_text().replace("&end", " F0 = 3.0\n&end", 1))
    run = tmp_path / "newrun"
    run.mkdir()
    np.savetxt(run / "bnd.dat", _ellipse())
    np.savetxt(run / "j.dat", np.column_stack(([0.0, 1.0], [1e6, 0.0])))
    (run / "shotfile.py").write_text(
        SHOTFILE.replace('ffprime_method = "q_li"', 'ffprime_method = "current"\ncurrent_file = "j.dat"')
        .replace("current_q0 = 1.1\ncurrent_li = 1.2\ncurrent_qa = 3.0\n", "")
    )

    box = viewer.profile_tuner(run, site=site)
    (q0, _, _), _ = _tuner_children(box)
    q0.value = 1.3
    box.children[1].children[2].click()

    assert q0.value == 1.3
    assert "not reset" in capsys.readouterr().out


# --- the folder's own FF' profile, whatever made it -------------------------------------


def _folder_with_ffprime(tmp_path, q0=0.95, li=1.26, qa=2.96, real_psi_edge=1 / 1.2):
    """A run folder holding only what input_profile reads: an ffprime_prof.dat
    on the extended grid, as run_jorek writes it."""
    run = tmp_path / "castorrun"
    run.mkdir()
    g = cur.PlasmaGeometry(1.5, 0.5, 1.2, 2.0)
    made = cur.current_from_q_li(q0, li, qa, g)
    x = np.linspace(0, 1, 200)
    ffprime = np.where(
        x <= real_psi_edge,
        np.interp(x / real_psi_edge, made.psi_n, cur.ffprime_from_current(made.j, g.R0)), 0.0,
    )
    np.savetxt(run / "ffprime_prof.dat", np.column_stack((x, ffprime)))
    write_float(run / "real_psi_edge.dat", real_psi_edge)
    return run, g


def test_input_profile_reads_the_plasma_part_of_ffprime_prof(tmp_path):
    run, g = _folder_with_ffprime(tmp_path)
    got = viewer.input_profile(run, g)
    assert (got.q[0], got.li, got.q[-1]) == pytest.approx((0.95, 1.26, 2.96), rel=5e-3)
    assert viewer.input_profile(tmp_path, g) is None                 # no file


def test_tuner_figure_draws_the_folder_s_profile_on_all_three_panels(tmp_path):
    run, g = _folder_with_ffprime(tmp_path)
    inputs = viewer.input_profile(run, g)
    fig = viewer.tuner_figure(1.2, 1.1, 3.4, g, inputs=inputs, inputs_label="ffprime_prof.dat (castor)")
    assert [len(ax.lines) for ax in fig.axes] == [2, 2, 2]
    title = fig._suptitle.get_text()
    assert "ffprime_prof.dat (castor):  q0 = 0.9" in title and "(cylinder estimate)" in title

    # the sliders outside the window: the folder's profile is still drawn
    fig = viewer.tuner_figure(0.2, 1.2, 3.0, g, inputs=inputs)
    assert [len(ax.lines) for ax in fig.axes] == [1, 1, 1]


def test_profile_tuner_on_a_castor_shotfile_shows_and_starts_from_its_profile(
    synthetic_campaign, tmp_path, monkeypatch
):
    pytest.importorskip("ipywidgets")
    site, template_dir, params = synthetic_campaign
    run, g = _folder_with_ffprime(tmp_path)
    np.savetxt(run / "original_bnd.dat", _ellipse())
    (run / "in_eq").write_text(" &in1\n F0 = 3.0\n&end\n")
    (run / "shotfile.py").write_text(
        "qa = 2.1\ng = 2.3\neta = 1e-3\ntstep_n = [0.03]\nnstep_n = [10]\nnout = 1\n"
        "exe = 'jorek_test_exe'\njobscript = '23h'\nrho_const = 1e18\n"
        "ffprime_method = 'castor'\nT_method = 'castor'\nrho_method = 'const'\nbnd_method = 'castor'\n"
        f"castor_suffix = 'TEST'\ncastor_params = {params.castor_params!r}\n"
    )
    drawn = []
    original = viewer.tuner_figure
    monkeypatch.setattr(viewer, "tuner_figure",
                        lambda *a, **k: (drawn.append(k), original(*a, **k))[1])

    box = viewer.profile_tuner(run, site=site)

    (q0, li, qa), _ = _tuner_children(box)
    assert (q0.value, li.value, qa.value) == pytest.approx((0.95, 1.26, 2.95), abs=0.02)
    assert drawn[-1]["inputs"] is not None
    assert drawn[-1]["inputs_label"] == "ffprime_prof.dat (castor)"
    assert "ffprime_method = &#x27;castor&#x27;" in box.children[2].value     # the existing warning


# --- gathering from the viewer ---------------------------------------------------------


def _campaign_with_case(tmp_path, listed=True):
    root = tmp_path / "camp"
    run = root / "qa2.1_li1.0_q01.0" / "eta1e-3"
    run.mkdir(parents=True)
    (root / "cases.toml").write_text(
        '[cases."qa2.1_li1.0_q01.0/eta1e-3"]\nsteps = { first_last = true }\n' if listed
        else '[cases."something/else"]\nsteps = [1]\n'
    )
    return root, run


def test_analyse_command_names_the_case_by_its_path_below_cases_toml(tmp_path):
    root, run = _campaign_with_case(tmp_path)
    command, cwd = viewer.analyse_command(run, ["four", "zerod"])
    assert cwd == root
    assert command[command.index("--case") + 1] == "qa2.1_li1.0_q01.0/eta1e-3"
    assert command[command.index("--cases") + 1] == str(root / "cases.toml")
    assert command[-4:] == ["--diag", "four", "--diag", "zerod"]


def test_analyse_command_gives_the_lines_to_add_for_an_unlisted_run(tmp_path):
    from ashen.shotfile import ShotfileError

    root, run = _campaign_with_case(tmp_path, listed=False)
    with pytest.raises(ShotfileError) as excinfo:
        viewer.analyse_command(run, ["four"])
    assert '[cases."qa2.1_li1.0_q01.0/eta1e-3"]' in str(excinfo.value)
    assert "first_last" in str(excinfo.value)
    with pytest.raises(ShotfileError, match="no cases.toml above"):
        viewer.analyse_command(tmp_path, ["four"])


def test_run_analyse_really_runs_analyse_for_this_case(tmp_path, capfd):
    """A real subprocess: the run has no restarts, so analyse has nothing to
    do, but it must start, find the case and say so."""
    root, run = _campaign_with_case(tmp_path)
    status = viewer.run_analyse(run, ["zerod"])
    out = capfd.readouterr().out
    assert isinstance(status, int)
    assert "analyse --case qa2.1_li1.0_q01.0/eta1e-3 --diag zerod" in out
    assert "qa2.1_li1.0_q01.0/eta1e-3" in out.split("\n", 1)[1]        # analyse's own output
    assert "ModuleNotFoundError" not in out and "Traceback" not in out


def test_four_view_button_gathers_then_shows_the_view(run_dir, monkeypatch):
    w = pytest.importorskip("ipywidgets")
    paths = RunPaths(run_dir, pad_width=6)
    kept = {step: fc.read_cache(paths.four_cache(step)) for step in (0, 100)}
    for step in (0, 100):
        paths.four_cache(step).unlink()
    asked = []

    def fake_analyse(folder, diags):
        asked.append((Path(folder), diags))
        for step, records in kept.items():
            fc.write_cache(paths.four_cache(step), step=step, pad_width=6, records=list(records.values()))
        return 0

    monkeypatch.setattr(viewer, "run_analyse", fake_analyse)
    button, _, holder = viewer.four_view(run_dir).children
    assert isinstance(holder.children[0], w.HTML)

    button.click()

    assert asked == [(run_dir, ["four"])]
    assert isinstance(holder.children[0], w.VBox) and not button.disabled


def test_gather_button_reports_a_failure_and_keeps_the_view(run_dir, monkeypatch, capsys):
    pytest.importorskip("ipywidgets")

    def broken(folder, diags):
        raise RuntimeError("jorek2_four is not in this folder")

    monkeypatch.setattr(viewer, "run_analyse", broken)
    button, _, holder = viewer.profiles_view(run_dir).children
    button.click()
    assert "jorek2_four is not in this folder" in capsys.readouterr().out
    assert holder.children and not button.disabled


def test_gather_step_caches_runs_only_what_is_missing(run_dir, monkeypatch):
    import ashen.diagnostics.qprofile as qprofile_mod
    import ashen.jorek2 as jorek2_mod

    calls = []
    monkeypatch.setattr(qprofile_mod, "run_qprofile_step", lambda run, step, paths: calls.append(("q", step)))
    monkeypatch.setattr(jorek2_mod, "run_zero_d", lambda run, step, paths: calls.append(("zeroD", step)))

    assert viewer.gather_step_caches(run_dir, 0) == [
        "q-profile of step 0: already there", "zeroD of step 0: already there"
    ]
    RunPaths(run_dir, pad_width=6).qprofile(100).unlink()
    notes = viewer.gather_step_caches(run_dir, 100)
    assert calls == [("q", 100)]
    assert notes == ["q-profile of step 100: gathered", "zeroD of step 100: already there"]


def test_gather_step_caches_reports_a_failing_tool(run_dir, monkeypatch):
    import ashen.diagnostics.qprofile as qprofile_mod

    def no_exe(run, step, paths):
        raise FileNotFoundError("jorek2_postproc")

    monkeypatch.setattr(qprofile_mod, "run_qprofile_step", no_exe)
    RunPaths(run_dir, pad_width=6).qprofile(100).unlink()
    notes = viewer.gather_step_caches(run_dir, 100)
    assert "failed (FileNotFoundError: jorek2_postproc)" in notes[0]


def test_tuner_gather_button_brings_in_what_jorek_achieved(run_dir, monkeypatch):
    pytest.importorskip("ipywidgets")
    paths = RunPaths(run_dir, pad_width=6)
    q_text = paths.qprofile(0).read_text()
    paths.qprofile(0).unlink()
    (run_dir / "shotfile.py").write_text(
        SHOTFILE.replace('bnd_file = "bnd.dat"', 'bnd_file = "original_bnd.dat"')
    )
    drawn = []
    original = viewer.tuner_figure
    monkeypatch.setattr(viewer, "tuner_figure",
                        lambda *a, **k: (drawn.append(k.get("achieved")), original(*a, **k))[1])
    monkeypatch.setattr(viewer, "gather_step_caches",
                        lambda folder, step: (paths.qprofile(step).write_text(q_text), ["gathered"])[1])

    box = viewer.profile_tuner(run_dir, step=0)
    assert drawn[-1] is None
    gather = box.children[1].children[3]
    assert gather.description == "Gather JOREK's q-profile (step 0)"

    gather.click()

    assert drawn[-1] is not None and drawn[-1].q0 == pytest.approx(1.0 + 2.5 * 0.01, abs=1e-6)


def test_case_viewer_draws_every_section_and_survives_a_broken_one(run_dir, monkeypatch):
    pytest.importorskip("ipywidgets")
    page = viewer.case_viewer(run_dir)
    headings = [c.value for c in page.children if getattr(c, "value", "").startswith("<h3>")]
    assert len(headings) == len(viewer._SECTIONS) == (len(page.children) - 1) // 2

    def broken(run_dir, step):
        raise RuntimeError("no such <file>")

    monkeypatch.setattr(viewer, "_SECTIONS", [("Broken", "", broken), viewer._SECTIONS[1]])
    page = viewer.case_viewer(run_dir)
    assert "RuntimeError: no such &lt;file&gt;" in page.children[2].value
    assert len(page.children) == 5


def test_views_draw_into_image_widgets_and_leave_no_figure_open(run_dir):
    """Not through an Output widget: VS Code repeats what one captures while
    the cell runs, so every figure appeared twice."""
    w = pytest.importorskip("ipywidgets")
    import matplotlib.pyplot as plt

    canvas = viewer.boundary_view(run_dir)
    (image,) = canvas.children
    assert isinstance(image, w.Image) and bytes(image.value[:4]) == b"\x89PNG"
    assert plt.get_fignums() == []

    def broken():
        raise RuntimeError("no <file>")

    viewer._show(canvas, broken)
    assert "RuntimeError: no &lt;file&gt;" in canvas.children[0].value


def test_default_four_modes_stop_one_past_the_last_rational_surface():
    assert viewer.default_four_modes(2.8) == [(1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (2, 6)]
    assert viewer.default_four_modes(3.3, n_values=(1,)) == [(1, 1), (1, 2), (1, 3), (1, 4)]
    assert viewer.default_four_modes(3.0, n_values=(1,))[-1] == (1, 4)   # 3/1 sits at the edge


def test_four_view_modes_follow_the_shotfile_qa_else_the_largest(run_dir):
    # no shotfile: the largest modes in the cache
    assert viewer.shotfile_qa(run_dir) is None
    assert viewer.four_view_modes(run_dir, "Psi", n_fallback=2) == [(1, 2), (1, 3)]

    (run_dir / "shotfile.py").write_text(SHOTFILE.replace("current_qa = 3.0", "current_qa = 1.4"))
    assert viewer.shotfile_qa(run_dir) == 1.4
    assert viewer.four_view_modes(run_dir, "Psi") == [(1, 1), (1, 2)]   # 3/2 is not cached

    # any other ffprime_method: the shotfile's qa
    (run_dir / "shotfile.py").write_text(
        SHOTFILE.replace('ffprime_method = "q_li"', 'ffprime_method = "current"\ncurrent_file = "j.dat"')
    )
    assert viewer.shotfile_qa(run_dir) == 2.1
    assert viewer.four_view_modes(run_dir, "Psi") == [(1, 1), (1, 2), (1, 3)]


def test_four_figure_radial_axis_is_linear_unless_asked(run_dir):
    fig = viewer.four_figure(run_dir, "Psi", step=100, modes=[(1, 2)], colors={(1, 2): "#123456"})
    ax_t, ax_r = fig.axes
    assert ax_t.get_yscale() == "log" and ax_r.get_yscale() == "linear"
    assert {l.get_color() for l in ax_r.get_lines() if l.get_label() == "n=1, m=2"} == {"#123456"}
    assert viewer.four_figure(run_dir, "Psi", step=100, log_radial=True).axes[1].get_yscale() == "log"
    viewer.four_figure(run_dir, "Psi", step=100, modes=[])            # nothing ticked still draws


def _four_parts(run_dir):
    _, _, holder = viewer.four_view(run_dir).children
    controls, mode_row, figure = holder.children[0].children
    return controls.children, mode_row.children[1].children, figure.children[0]


def _png(run_dir, monkeypatch):
    """four_view without ipympl: the figure is a PNG Image widget."""
    monkeypatch.setitem(sys.modules, "ipympl.backend_nbagg", None)
    controls, boxes, shown = _four_parts(run_dir)
    return controls, boxes, shown.children[0]


def test_four_view_has_a_checkbox_per_mode_that_removes_it(run_dir, monkeypatch):
    pytest.importorskip("ipywidgets")
    controls, boxes, image = _png(run_dir, monkeypatch)
    assert [b.description for b in boxes] == ["1/1", "2/1", "3/1"] and all(b.value for b in boxes)
    assert [c.description for c in controls[2:]] == [
        "log amplitudes", "log radial structure", "Reset view"]
    assert controls[3].value is False
    before = bytes(image.value)
    boxes[2].value = False
    assert bytes(image.value) != before


def test_four_view_controls_update_the_figure_without_reading_the_caches_again(run_dir, monkeypatch):
    pytest.importorskip("ipywidgets")
    reads = []
    real = viewer.load_four_data
    monkeypatch.setattr(viewer, "load_four_data", lambda *a: reads.append(a[1]) or real(*a))
    controls, boxes, image = _png(run_dir, monkeypatch)
    assert reads == ["Psi"]

    seen = {bytes(image.value)}
    controls[1].value = 0            # step
    seen.add(bytes(image.value))
    controls[3].value = True         # log radial structure
    seen.add(bytes(image.value))
    boxes[0].value = False
    seen.add(bytes(image.value))
    controls[4].click()              # reset view
    assert len(seen) == 4 and reads == ["Psi"]


def test_four_view_uses_an_ipympl_canvas_when_there_is_one(run_dir):
    pytest.importorskip("ipywidgets")
    nbagg = pytest.importorskip("ipympl.backend_nbagg")
    _, _, holder = viewer.four_view(run_dir).children
    assert isinstance(holder.children[0].children[2].children[0], nbagg.Canvas)


def test_four_plot_updates_its_lines_in_place(run_dir):
    from matplotlib.figure import Figure

    data = viewer.load_four_data(run_dir, "Psi", [(1, 1), (1, 3), (2, 9)])
    assert data.steps == [0, 100] and data.radial[(2, 9)] == {}       # not cached: empty
    assert np.isnan(data.amplitude[(2, 9)]).all()
    np.testing.assert_allclose(data.amplitude[(1, 3)][1], data.radial[(1, 3)][100][1].max())

    plot = viewer.FourPlot(Figure(), data, step=100)
    line = plot.radial_lines[(1, 3)]
    at_100 = line.get_ydata().copy()
    plot.set_step(0)
    assert plot.radial_lines[(1, 3)] is line and not np.allclose(line.get_ydata(), at_100)
    assert list(plot.marker.get_xdata()) == [0, 0]

    plot.set_visible([(1, 1)])
    assert not line.get_visible() and not plot.amplitude_lines[(1, 3)].get_visible()
    assert [t.get_text() for t in plot.ax_r.get_legend().get_texts()] == ["n=1, m=1"]
    plot.set_visible([])
    assert plot.ax_r.get_legend() is None

    # a hand-set range survives a change of step; Reset view lets it follow the data again
    plot.set_visible([(1, 1), (1, 3)])
    plot.ax_r.set_ylim(0.0, 123.0)
    plot.set_step(100)
    assert plot.ax_r.get_ylim() == (0.0, 123.0)
    plot.reset_view()
    assert plot.ax_r.get_ylim()[1] < 1.0
    plot.set_log(False, True)
    assert (plot.ax_t.get_yscale(), plot.ax_r.get_yscale()) == ("linear", "log")
