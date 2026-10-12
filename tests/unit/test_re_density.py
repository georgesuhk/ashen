"""The runaway density read from a restart (diagnostics.re_density) and
its view in the case viewer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

h5py = pytest.importorskip("h5py")

from ashen.diagnostics.re_density import read_re_density  # noqa: E402
from ashen.physics import ELEMENTARY_CHARGE, MU_0  # noqa: E402

C = 299792458.0
AMU = 1.66053906660e-27
N_PEAK = 5.0e16            # m^-3, on axis
CENTRAL_DENSITY, CENTRAL_MASS = 0.01, 2.0


def _restart(path, *, index=0, lost=0.0, sign=-1.0, n_tor=3, with_elements=True):
    """A restart on a 21 x 32 polar grid of quads around (1.5, 0): n_RE =
    N_PEAK (1 - lost) (1 - rho^2), stored JOREK's way (normalised, times
    ``sign``), with an n = 1 cos part of 1e-3 of it."""
    n_rho, n_theta = 21, 32
    rho, theta = np.meshgrid(np.linspace(0, 1, n_rho), np.linspace(0, 2 * np.pi, n_theta, endpoint=False),
                             indexing="ij")
    R, Z = (1.5 + 0.4 * rho * np.cos(theta)).ravel(), (0.4 * rho * np.sin(theta)).ravel()
    node = np.arange(n_rho * n_theta).reshape(n_rho, n_theta)
    quads = np.array([
        [node[i, j], node[i + 1, j], node[i + 1, (j + 1) % n_theta], node[i, (j + 1) % n_theta]]
        for i in range(n_rho - 1) for j in range(n_theta)
    ])
    rho_0 = CENTRAL_DENSITY * 1e20 * CENTRAL_MASS * AMU
    n_si = N_PEAK * (1 - lost) * (1 - rho.ravel() ** 2)
    stored = sign * n_si * ELEMENTARY_CHARGE * R / np.sqrt(rho_0 / MU_0)
    values = np.zeros((8, 4, n_tor, R.size))
    values[0, 0, 0] = rho.ravel() ** 2 - 1.0
    values[2, 0, 0] = -5.0 * (1 - rho.ravel() ** 2) - 1e-6
    values[7, 0, 0] = stored
    if n_tor > 1:
        values[7, 0, 1] = 1e-3 * stored
    x = np.zeros((2, 4, 1, R.size))
    x[0, 0, 0], x[1, 0, 0] = R, Z
    with h5py.File(path, "w") as f:
        f["values"], f["x"] = values, x
        f["central_density"], f["central_mass"] = [CENTRAL_DENSITY], [CENTRAL_MASS]
        f["index_now"], f["n_period"] = [index], [1]
        f["t_now"], f["sqrt_mu0_rho0"] = [float(index)], [2.0e-9]
        if index:
            trace = np.zeros(index)
            trace[-1] = -4.2e5
            f["Ipre_tot_t"] = trace
        if with_elements:
            f["vertex"], f["n_elements"] = quads.T + 1, [len(quads)]
    return n_si


def test_density_is_in_si_and_positive_whatever_sign_jorek_stored(tmp_path):
    n_si = _restart(tmp_path / "jorek00000.h5")
    d = read_re_density(tmp_path / "jorek00000.h5")
    np.testing.assert_allclose(d.n0, n_si, rtol=1e-12, atol=1.0)
    assert d.n0.max() == pytest.approx(N_PEAK) and d.index == 0 and np.isnan(d.ipre_tot)

    _restart(tmp_path / "jorek00100.h5", index=100, sign=+1.0)
    other = read_re_density(tmp_path / "jorek00100.h5")
    np.testing.assert_allclose(other.n0, n_si, rtol=1e-12, atol=1.0)
    assert other.time == pytest.approx(100 * 2.0e-9) and other.ipre_tot == pytest.approx(4.2e5)


def test_harmonics_follow_jorek_s_toroidal_basis(tmp_path):
    n_si = _restart(tmp_path / "jorek00000.h5")
    d = read_re_density(tmp_path / "jorek00000.h5")
    # harmonic 2 is cos(phi): +1e-3 at phi = 0, nothing at 90 degrees, -1e-3 at 180
    np.testing.assert_allclose(d.at(0.0), 1.001 * n_si, rtol=1e-12, atol=1.0)
    np.testing.assert_allclose(d.at(np.pi / 2), n_si, rtol=1e-12, atol=1.0)
    np.testing.assert_allclose(d.at(np.pi, perturbation=True), -1e-3 * n_si, rtol=1e-9, atol=1.0)
    assert d.mode_amplitudes() == pytest.approx({0: N_PEAK, 1: 1e-3 * N_PEAK})


def test_current_is_e_c_times_the_density_integral(tmp_path):
    _restart(tmp_path / "jorek00000.h5")
    d = read_re_density(tmp_path / "jorek00000.h5")
    # the integral of N (1 - rho^2) over a disc of radius a is N pi a^2 / 2
    exact = ELEMENTARY_CHARGE * C * N_PEAK * np.pi * 0.4**2 / 2
    assert d.current() == pytest.approx(exact, rel=0.02)       # linear on a coarse grid
    assert len(d.triangles) == 2 * 20 * 32

    _restart(tmp_path / "jorek00200.h5", index=200, lost=0.25)
    assert read_re_density(tmp_path / "jorek00200.h5").current() == pytest.approx(0.75 * d.current())


def test_a_file_without_the_normalisation_says_so(tmp_path):
    with h5py.File(tmp_path / "jorek00000.h5", "w") as f:
        f["values"], f["x"] = np.zeros((8, 4, 1, 5)), np.zeros((2, 4, 1, 5))
    with pytest.raises(ValueError, match="no central_density, central_mass"):
        read_re_density(tmp_path / "jorek00000.h5")


@pytest.fixture
def run(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "in_main").write_text(" &in1\n vpar_re_sign = 1\n&end\n")
    _restart(folder / "jorek00000.h5")
    _restart(folder / "jorek00200.h5", index=200, lost=0.25)
    return folder


def test_figure_shows_the_map_the_profile_and_what_was_lost(run):
    from ashen import viewer

    fig = viewer.re_density_figure(run, 200)
    ax_map, ax_n, ax_modes = fig.axes[:3]
    assert "step 200" in ax_map.get_title(loc="left") and "t = 0.0004 ms" in ax_map.get_title(loc="left")
    assert "-25.00% against step 0" in fig._suptitle.get_text()
    assert "Ipre_tot at this step: 420.00 kA" in fig._suptitle.get_text()
    assert [t.get_text() for t in ax_n.get_legend().get_texts()] == ["step 0", "step 200"]
    now = ax_n.collections[-1].get_offsets()
    assert now[:, 1].max() == pytest.approx(0.75 * 5.0)          # in units of 1e16 m^-3
    assert [t.get_text() for t in ax_modes.get_xticklabels()] == ["0", "1"]

    part = viewer.re_density_figure(run, 200, phi=180, perturbation=True)
    assert "n > 0 part" in part.axes[3].get_ylabel() and "180°" in part.axes[0].get_title(loc="left")
    # the first restart against itself: no reference curve, no percentage
    first = viewer.re_density_figure(run, 0)
    assert "against" not in first._suptitle.get_text() and len(first.axes[1].collections) == 1


def test_view_redraws_on_each_control_and_refuses_a_run_without_the_re_fluid(run):
    w = pytest.importorskip("ipywidgets")
    from ashen import viewer

    controls, canvas = viewer.re_density_view(run).children
    step, phi, part = controls.children
    assert step.value == 200 and bytes(canvas.children[0].value[:4]) == b"\x89PNG"
    for control, value in ((step, 0), (phi, 90), (part, True)):
        before = bytes(canvas.children[0].value)
        control.value = value
        assert bytes(canvas.children[0].value) != before
    assert any(title.startswith("Runaway density") for title, _, _ in viewer._SECTIONS)

    (run / "in_main").write_text(" &in1\n eta = 1e-6\n&end\n")
    refused = viewer.re_density_view(run)
    assert isinstance(refused, w.HTML) and "not a run with the RE fluid" in refused.value
    with pytest.raises(ValueError, match="not a run with the RE fluid"):
        viewer.re_density_figure(run, 0)
