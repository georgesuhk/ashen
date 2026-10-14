"""The case viewer's Poincare panel (viewer.poincare_view)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("h5py")

from ashen import viewer  # noqa: E402
from ashen.diagnostics import poincare_cache as pc  # noqa: E402
from ashen.paths import RunPaths  # noqa: E402


def _cache(run, step, *, n=12):
    paths = RunPaths(run, pad_width=6)
    with pc.open_cache(paths.poincare_cache(step), step=step, pad_width=6) as handle:
        for psi_n, radius in ((0.2, 0.1), (0.6, 0.25)):
            theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + step * 1e-3
            pc.append_line(
                handle, pc.LineKey(psi_n=psi_n, R=1.5 + radius, Z=0.0, phi=0.0),
                {"R": 1.5 + radius * np.cos(theta), "Z": radius * np.sin(theta),
                 "rho": np.full(n, np.sqrt(psi_n)), "theta": theta},
                n_turns=n, terminated=False,
            )


@pytest.fixture
def run(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    for step in (100, 200, 300):
        (folder / f"jorek{step:06d}.h5").write_bytes(b"")
    for step in (100, 300):
        _cache(folder, step)
    postproc = folder / "postproc"
    postproc.mkdir()
    (postproc / "zeroD_quantities_s000300.dat").write_text("Time Ip_tot\n2.5e-5 1.0\n")
    return folder


def test_steps_are_the_cached_ones_and_the_figure_says_what_it_holds(run):
    assert viewer.poincare_steps(run) == [100, 300]
    fig = viewer.poincare_figure(run, 300, point_size=0.5)
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "Poincaré, step 300, t = 0.025 ms"
    assert "2 field lines from 2 psi_N, 24 punctures" in ax.texts[0].get_text()
    assert sum(len(c.get_offsets()) for c in ax.collections) == 24
    assert all(size == 0.5 for c in ax.collections for size in c.get_sizes())
    assert "t =" not in viewer.poincare_figure(run, 100).axes[0].get_title(loc="left")
    with pytest.raises(ValueError, match="no Poincare cache for step 200"):
        viewer.poincare_figure(run, 200)


def test_view_steps_through_with_the_buttons_and_slider(run):
    w = pytest.importorskip("ipywidgets")
    button, _, holder = viewer.poincare_view(run).children
    assert button.description == "Run analyse --diag poincare"
    controls, canvas = holder.children[0].children
    back, step, forward, size = controls.children
    assert step.options == (100, 300) and step.value == 300
    image = lambda: bytes(canvas.children[0].value)
    shown = image()
    forward.click()                                   # already at the last step
    assert step.value == 300 and image() == shown
    back.click()
    assert step.value == 100 and image() != shown
    back.click()                                      # already at the first
    assert step.value == 100
    shown = image()
    size.value = 1.0
    assert image() != shown
    assert any(title.startswith("Poincaré plots") for title, _, _ in viewer._SECTIONS)

    empty = run.parent / "empty"
    empty.mkdir()
    assert isinstance(viewer.poincare_view(empty).children[2].children[0], w.HTML)
