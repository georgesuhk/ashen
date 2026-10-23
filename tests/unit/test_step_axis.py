"""Step <-> time: live_data.read_step_times and plotting.step_axis, and the
second x axis the case viewer's time-series figures carry."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ashen.diagnostics.live_data import LIVE_DATA_FILE, StepTimes, read_step_times  # noqa: E402
from ashen.plotting.step_axis import add_step_time_axis  # noqa: E402

HEADER = """@times_xlabel: time step
@times_xlabel_si: time step
@times_ylabel: normalized time
@times_ylabel_si: time [ms]
@times_x2si:   1.000000000E+00
@times_y2si:   2.000000000E-03
@times: "step"     "time"
@magnetic_energies_xlabel_si: time [ms]
@magnetic_energies_x2si:   2.000000000E-03
@magnetic_energies: %"time"           "E_{mag,00}" "E_{mag,01}"
"""


def _file(tmp_path, rows):
    """rows: (step, normalised time) in the order JOREK wrote them."""
    lines = [HEADER]
    for step, t in rows:
        lines.append(f"@times:  {step}   {t:.9E}\n")
        lines.append(f"@magnetic_energies:  {t:.9E}  5.0D-02  1.0E-12\n")
    path = tmp_path / LIVE_DATA_FILE
    path.write_text("".join(lines))
    return path


def test_steps_and_times_in_ms_with_a_restart_overwriting_rows(tmp_path):
    # ran steps 1..4 at dt 0.5, restarted from step 2 with dt 1.0 to step 4
    path = _file(tmp_path, [(1, 0.5), (2, 1.0), (3, 1.5), (4, 2.0), (3, 2.0), (4, 3.0)])
    st = read_step_times(path)
    np.testing.assert_array_equal(st.steps, [1, 2, 3, 4])
    np.testing.assert_allclose(st.time, [1e-3, 2e-3, 4e-3, 6e-3])     # the later run stands
    assert st.time_label == "time [ms]"


def test_top_axis_maps_through_the_runs_own_steps():
    # dt changes at step 10: time is not a fixed multiple of the step
    st = StepTimes(steps=np.array([0.0, 10.0, 20.0]), time=np.array([0.0, 1.0, 3.0]),
                   time_label="time [ms]")
    fig = Figure()
    ax = fig.subplots()
    top = add_step_time_axis(ax, st, x_is="time")
    assert top.get_xlabel() == "time step"
    forward, inverse = top._functions
    np.testing.assert_allclose(forward(np.array([0.5, 2.0])), [5.0, 15.0])
    np.testing.assert_allclose(inverse(np.array([15.0])), [2.0])

    ax2 = Figure().subplots()
    assert add_step_time_axis(ax2, st, x_is="step").get_xlabel() == "time [ms]"
    with pytest.raises(ValueError):
        add_step_time_axis(ax2, st, x_is="steps")


def test_a_single_step_gives_no_axis():
    st = StepTimes(steps=np.array([5.0]), time=np.array([1.0]), time_label="time [ms]")
    assert add_step_time_axis(Figure().subplots(), st, x_is="time") is None


def test_viewer_energies_figure_has_the_step_on_top(tmp_path):
    pytest.importorskip("h5py")
    from ashen import viewer

    _file(tmp_path, [(1, 0.5), (2, 1.0), (3, 1.5)])
    ax = viewer.energies_figure(tmp_path).axes[0]
    assert [child.get_xlabel() for child in ax.child_axes] == ["time step"]
    import io

    viewer.energies_figure(tmp_path).savefig(io.BytesIO(), format="png")   # renders with it


def test_viewer_step_axis_skipped_without_live_data(tmp_path):
    pytest.importorskip("h5py")
    from ashen import viewer

    ax = Figure().subplots()
    viewer._step_time_axis(ax, tmp_path, "step")                   # no macroscopic_vars.dat
    assert ax.child_axes == []
