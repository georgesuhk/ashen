"""plot --diag particle_loss: the lost fraction of the traced particles,
by the psi_n they started at, over time."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ashen.cli import plot as plot_cli
from ashen.diagnostics.particle_exits import exit_angles, loss_map, read_particle_diag
from ashen.plotting.particle_loss import LOSS_LABEL, plot_loss_map
from test_particle_exits import CASES, campaign, write_diag  # noqa: F401  (fixture)

#: Six particles over five times. Two start at psi_n 0.2 and stay; two at
#: 0.6, of which one crosses psi_n 1 at the third time; two at 0.9: one
#: crosses at the second time, the other is lost from the grid at the fourth.
T = [0.0, 1e-6, 2e-6, 3e-6, 4e-6]
PSI = [[0.2, 0.2, 0.6, 0.6, 0.9, 0.9],
       [0.2, 0.2, 0.7, 0.6, 1.1, 0.9],
       [0.2, 0.2, 1.2, 0.6, 1.3, 0.9],
       [0.2, 0.2, 1.4, 0.6, 1.5, 0.0],
       [0.2, 0.2, 1.6, 0.6, 1.7, 0.0]]
LOST = [[0] * 6, [0] * 6, [0] * 6, [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]
ONES = np.ones((5, 6))


@pytest.fixture
def history(tmp_path):
    return read_particle_diag(write_diag(
        tmp_path / "d.h5", t=T, psi_n=PSI, R=ONES, Z=ONES * 0, phi=ONES * 0, lost=LOST,
        theta=ONES * 0,
    ))


def test_lost_fraction_per_starting_psi_over_time(history):
    result = loss_map(history, psi_n=1.0, n_psi=3, psi_range=(0.0, 0.9))
    np.testing.assert_allclose(result.psi_edges, [0.0, 0.3, 0.6, 0.9])
    np.testing.assert_allclose(result.time, T)
    assert result.counts.tolist() == [2, 0, 4]
    # started at 0.2: never lost
    np.testing.assert_allclose(result.fraction[0], 0.0)
    # nobody started in 0.3..0.6: no data
    assert np.isnan(result.fraction[1]).all()
    # started at 0.6 and 0.9 (the top edge counts): 1 of 4 gone by the second
    # time, 3 of 4 by the third -- the one lost from the grid at the fourth
    # counts from its last time on it -- and the fourth stays
    np.testing.assert_allclose(result.fraction[2], [0.0, 0.25, 0.75, 0.75, 0.75])
    assert (result.n_lost, result.n_considered) == (3, 6)


def test_it_only_ever_rises(history):
    fraction = loss_map(history, psi_n=1.0, n_psi=9).fraction
    rows = fraction[~np.isnan(fraction).any(axis=1)]
    assert (np.diff(rows, axis=1) >= 0).all()


def test_same_rule_as_particle_exits(history):
    """Whoever particle_exits counts as exited is lost here, at that time."""
    exits = exit_angles(history, psi_n=1.0)
    result = loss_map(history, psi_n=1.0, n_psi=1)
    assert result.n_lost == exits.n_exited and result.n_considered == exits.n_considered
    np.testing.assert_allclose(
        result.fraction[0],
        [np.count_nonzero(exits.time <= t) / exits.n_considered for t in T],
    )


def test_default_range_runs_to_the_outermost_start(history):
    result = loss_map(history, psi_n=1.0, n_psi=9)
    np.testing.assert_allclose(result.psi_edges[[0, -1]], [0.0, 0.9])
    assert result.counts.sum() == 6


def test_particles_off_the_grid_from_the_start_are_not_counted(tmp_path):
    lost = np.array(LOST)
    lost[:, 0] = 1
    history = read_particle_diag(write_diag(
        tmp_path / "d.h5", t=T, psi_n=PSI, R=ONES, Z=ONES * 0, phi=ONES * 0, lost=lost,
        theta=ONES * 0,
    ))
    assert loss_map(history, psi_n=1.0, n_psi=3).n_considered == 5


def test_long_histories_are_thinned_evenly(tmp_path):
    n = 2001
    t = np.arange(n) * 1e-8
    psi = np.tile([[0.5, 0.5]], (n, 1))
    psi[1000:, 0] = 1.5
    history = read_particle_diag(write_diag(
        tmp_path / "d.h5", t=t, psi_n=psi, R=np.ones((n, 2)), Z=np.zeros((n, 2)),
        phi=np.zeros((n, 2)), lost=np.zeros((n, 2), int), theta=np.zeros((n, 2)),
    ))
    result = loss_map(history, psi_n=1.0, n_psi=1, max_times=101)
    assert result.time.size == 101
    assert (result.time[0], result.time[-1]) == pytest.approx((0.0, 2e-5))
    # half the particles, from the time the first one crosses
    np.testing.assert_allclose(result.fraction[0][result.time < 0.99e-5], 0.0)
    np.testing.assert_allclose(result.fraction[0][result.time >= 1e-5], 0.5)


def test_the_figure_matches_the_connection_length_map(history, tmp_path, monkeypatch):
    """Lost is the green end, as a short connection length is; no data is black."""
    import matplotlib.colors as mcolors

    monkeypatch.setattr(plt, "close", lambda *args: None)
    result = loss_map(history, psi_n=1.0, n_psi=3, psi_range=(0.0, 0.9))
    out = plot_loss_map(result, tmp_path / "loss.png", caption="c", dpi=40)
    assert out.is_file()
    fig = plt.gcf()
    mesh = fig.axes[0].collections[0]
    assert mesh.get_clim() == (0.0, 1.0)
    lost_all, lost_none = mesh.cmap(1.0), mesh.cmap(0.0)
    assert lost_all[1] > lost_all[0] and lost_none[0] > lost_none[1]      # green, red
    assert mcolors.same_color(mesh.cmap(np.ma.masked), "black")
    assert fig.axes[0].get_xlabel() == r"t [$\mu s$]"
    assert any(ax.get_ylabel() == LOSS_LABEL for ax in fig.axes)
    monkeypatch.undo()
    plt.close("all")


# --- from the command line ---------------------------------------------------------


def _traced(campaign, extra=""):  # noqa: F811
    folder = campaign / "run" / "ptrace" / "ptrace_gc"
    write_diag(folder / "ptrace_diag.h5", t=T, psi_n=PSI, R=ONES, Z=ONES * 0, phi=ONES * 0,
               lost=LOST, theta=ONES * 0)
    (campaign / "cases.toml").write_text(
        CASES.replace("ptrace_exit_psi_n = 1.3", "ptrace_exit_psi_n = 1.0" + extra),
        encoding="utf-8")
    return folder


def test_plot_particle_loss(campaign, capsys):  # noqa: F811
    folder = _traced(campaign)
    assert plot_cli.main(["--case", "run", "--diag", "particle_loss", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert ("particle_loss: 3 of 6 particles lost (past psi_n = 1 or off the grid) "
            "over 4 µs of trace") in out
    assert (folder / "particle_loss.png").is_file()


def test_exit_psi_n_flag_and_bins_key(campaign, capsys, monkeypatch):  # noqa: F811
    _traced(campaign, "\nptrace_loss_bins = 7")
    seen = {}
    monkeypatch.setattr(plot_cli, "plot_loss_map",
                        lambda result, out, **kw: seen.update(result=result) or out)
    assert plot_cli.main(["--case", "run", "--diag", "particle_loss", "--exit-psi-n", "1.45"]) == 0
    assert seen["result"].counts.size == 7
    # only the two that pass 1.45, and the one lost from the grid
    assert "3 of 6 particles lost (past psi_n = 1.45" in capsys.readouterr().out


def test_initial_psi_range_sets_the_axis_and_the_name(campaign, capsys):  # noqa: F811
    folder = _traced(campaign, "\nptrace_initial_psi_n_range = [0.5, 1.0]")
    assert plot_cli.main(["--case", "run", "--diag", "particle_loss", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "3 of 4 particles lost" in out and "markers starting at psi_n 0.5 to 1: 4 of 6" in out
    assert (folder / "particle_loss_psi0.5-1.png").is_file()
    assert not (folder / "particle_loss.png").exists()


def test_no_trace_yet(campaign, capsys):  # noqa: F811
    assert plot_cli.main(["--case", "run", "--diag", "particle_loss"]) == 0
    assert "run bin/ptrace first" in capsys.readouterr().out


def test_invalid_loss_bins(campaign, capsys):  # noqa: F811
    _traced(campaign, "\nptrace_loss_bins = 0")
    assert plot_cli.main(["--case", "run", "--diag", "particle_loss"]) == 1
    assert "ptrace_loss_bins must be a whole number >= 1" in capsys.readouterr().err
