"""ashen.demo -- the made-up campaign the viewer's demo notebook uses."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from ashen import demo, viewer  # noqa: E402
from ashen.diagnostics.equilibrium import achieved_q_li  # noqa: E402
from ashen.paths import RunPaths  # noqa: E402
from ashen.shotfile import load_shotfile, set_shotfile_values  # noqa: E402


@pytest.fixture
def demo_run(tmp_path, symlinks_maybe_bypassed):
    return demo.make_demo_campaign(tmp_path / "campaign")


def test_demo_run_has_inputs_and_pretend_outputs(demo_run):
    for name in ("shotfile.py", "j_prof.dat", "ffprime_prof.dat", "in_eq", "in_bnd",
                 "original_bnd.dat", "boundary.txt", "input_starwall", "real_psi_edge.dat"):
        assert (demo_run / name).is_file(), name
    assert viewer.four_steps(demo_run) == list(demo.DEMO_STEPS)
    assert set(viewer.case_boundaries(demo_run)) == {
        "plasma", "domain (in_bnd)", "JOREK grid", "STARWALL wall"
    }
    assert {k.var for k in viewer.profiles_available(demo_run)} == {"Btor", "currdens"}


def test_demo_equilibrium_is_a_little_off_the_request(demo_run):
    params = load_shotfile(demo_run / "shotfile.py")
    got = achieved_q_li(RunPaths.detect(demo_run), 0, f0=3.7)
    assert got.q0 == pytest.approx(params.current_q0 + 0.05, abs=5e-3)
    assert got.li == pytest.approx(params.current_li + 0.06, abs=1e-2)
    assert got.q_edge == pytest.approx(params.current_q_edge - 0.12, abs=1e-2)
    assert got.real_psi_edge == pytest.approx(1 / 1.2)


def test_demo_figures_draw(demo_run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viewer.equilibrium_figure(demo_run, 0)
    viewer.four_figure(demo_run, "Psi", step=1200)
    viewer.boundary_figure(demo_run)
    plt.close("all")


def test_demo_keeps_a_saved_shotfile_and_follows_it(demo_run):
    set_shotfile_values(demo_run / "shotfile.py", {"current_q0": 1.3})
    before = np.loadtxt(demo_run / "ffprime_prof.dat")[0, 1]

    again = demo.make_demo_campaign(demo_run.parents[1])

    assert again == demo_run
    assert load_shotfile(demo_run / "shotfile.py").current_q0 == 1.3
    after = np.loadtxt(demo_run / "ffprime_prof.dat")[0, 1]
    assert after / before == pytest.approx(1.05 / 1.3, rel=1e-6)      # j0 ~ 1 / q0
