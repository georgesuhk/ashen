"""Particle snapshots: reading JOREK particle files (ashen.diagnostics.
particles), drawing them on R-Z (ashen.plotting.particles), and
`plot --diag particles` end to end."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: this suite never opens a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ashen.cases import load_cases
from ashen.cli import plot as plot_cli
from ashen.paths import RunPaths
from ashen.diagnostics.particles import (
    ParticleFileError,
    find_snapshots,
    read_snapshot,
)
from ashen.plotting.particles import (
    RZPanel,
    animate_rz_panels,
    draw_particles,
    particle_panels,
    plot_rz_panels,
    snapshot_label,
)

h5py = pytest.importorskip("h5py")


def write_particle_file(path, *, time, groups):
    """A particle file laid out as write_simulation_hdf5 writes it: /time,
    and per group x as (n, 3) -- Fortran's x(3, n) seen row-major -- and
    i_elm. `groups` maps a group id to (R, Z, phi, i_elm) sequences."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["time"] = time
        for gid, (R, Z, phi, i_elm) in groups.items():
            g = f.create_group(f"groups/{gid}")
            g["x"] = np.column_stack([R, Z, phi])
            g["i_elm"] = np.asarray(i_elm, dtype=np.int32)
            g["weight"] = np.ones(len(R))
    return path


def simple_file(path, time, R, i_elm=None):
    R = np.asarray(R, dtype=float)
    return write_particle_file(path, time=time, groups={
        "01": (R, np.zeros_like(R), np.zeros_like(R),
               i_elm if i_elm is not None else np.ones(R.size, dtype=int)),
    })


# --- reading -------------------------------------------------------------------


def test_read_snapshot_positions_time_and_lost(tmp_path):
    path = simple_file(tmp_path / "part_restart.h5", 2.5e-3, [3.6, 3.7, 3.8], i_elm=[4, 0, -1])
    snap = read_snapshot(path)
    assert snap.time == 2.5e-3
    np.testing.assert_allclose(snap.R, [3.6, 3.7, 3.8])
    np.testing.assert_allclose(snap.Z, 0.0)
    assert snap.lost.tolist() == [False, True, True]
    assert (snap.n, snap.n_lost) == (3, 2)


def test_groups_are_concatenated(tmp_path):
    path = write_particle_file(tmp_path / "p.h5", time=1.0, groups={
        "a": ([1.0, 2.0], [0.1, 0.2], [0.0, 0.0], [1, 1]),
        "b": ([3.0], [0.3], [0.0], [1]),
    })
    snap = read_snapshot(path)
    np.testing.assert_allclose(snap.R, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(snap.Z, [0.1, 0.2, 0.3])


def test_transposed_positions_are_accepted(tmp_path):
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as f:
        f["time"] = 0.0
        g = f.create_group("groups/01")
        g["x"] = np.array([[1.0, 2.0], [0.5, 0.6], [0.0, 0.0]])  # (3, n)
        g["i_elm"] = np.array([1, 1])
    snap = read_snapshot(path)
    np.testing.assert_allclose(snap.R, [1.0, 2.0])
    np.testing.assert_allclose(snap.Z, [0.5, 0.6])


def test_missing_weight_defaults_to_one(tmp_path):
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as f:
        f["time"] = 0.0
        g = f.create_group("groups/01")
        g["x"] = np.zeros((2, 3))
        g["i_elm"] = np.array([1, 1])
    assert read_snapshot(path).weight.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("build, message", [
    (lambda f: f.create_group("groups/01"), "no /time"),
    (lambda f: f.__setitem__("time", 0.0), "no particle groups"),
])
def test_malformed_files(tmp_path, build, message):
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as f:
        build(f)
    with pytest.raises(ParticleFileError, match=message):
        read_snapshot(path)


def test_group_without_positions(tmp_path):
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as f:
        f["time"] = 0.0
        f.create_group("groups/01")["i_elm"] = np.array([1])
    with pytest.raises(ParticleFileError, match="has no 'x'"):
        read_snapshot(path)


def test_positions_that_do_not_match_i_elm(tmp_path):
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as f:
        f["time"] = 0.0
        g = f.create_group("groups/01")
        g["x"] = np.zeros((4, 3))
        g["i_elm"] = np.array([1, 1])
    with pytest.raises(ParticleFileError, match="x has shape"):
        read_snapshot(path)


def test_unreadable_file(tmp_path):
    path = tmp_path / "part_restart.h5"
    path.write_bytes(b"not hdf5")
    with pytest.raises(ParticleFileError, match="cannot open"):
        read_snapshot(path)


def test_find_snapshots_sorts_by_time_and_drops_duplicates(tmp_path):
    """The file order by name is not the time order; and a snapshot on the
    final step duplicates part_restart.h5."""
    simple_file(tmp_path / "part_restart.h5", 3e-3, [3.9])
    simple_file(tmp_path / "part_restart000.00300000.h5", 3e-3, [3.9])
    simple_file(tmp_path / "part_restart000.00250000.h5", 2.5e-3, [3.7])
    simple_file(tmp_path / "part_restart000.00275000.h5", 2.75e-3, [3.8])
    (tmp_path / "diag.h5").write_bytes(b"not a particle file")
    times = [s.time for s in find_snapshots(tmp_path)]
    assert times == [2.5e-3, 2.75e-3, 3e-3]


def test_find_snapshots_empty_folder(tmp_path):
    assert find_snapshots(tmp_path) == []


# --- drawing -------------------------------------------------------------------


@pytest.fixture
def snapshots(tmp_path):
    first = read_snapshot(simple_file(tmp_path / "a.h5", 1e-3, [3.6, 3.7, 3.8]))
    later = read_snapshot(simple_file(tmp_path / "b.h5", 2e-3, [3.65, 3.75, 4.5], i_elm=[1, 1, 0]))
    return first, later


def test_draw_particles_layers_start_current_and_lost(snapshots):
    first, later = snapshots
    fig, ax = plt.subplots()
    draw_particles(ax, later, reference=first)
    reference, alive, lost = ax.collections
    assert len(reference.get_offsets()) == 3
    assert len(alive.get_offsets()) == 2
    np.testing.assert_allclose(lost.get_offsets(), [[4.5, 0.0]])
    # Above anything a layer drew first (e.g. a Poincare plot).
    assert min(c.get_zorder() for c in ax.collections) > 1
    plt.close(fig)


def test_draw_particles_no_reference_on_the_first_panel(snapshots):
    first, _ = snapshots
    fig, ax = plt.subplots()
    draw_particles(ax, first, reference=first)
    assert len(ax.collections) == 1
    plt.close(fig)


def test_snapshot_label(snapshots):
    first, later = snapshots
    assert snapshot_label(first) == "t = 1 ms"
    assert snapshot_label(first, start=first.time) == "t = 1 ms"
    assert snapshot_label(later) == "t = 2 ms, 1/3 lost"
    assert snapshot_label(later, start=first.time) == "t = 2 ms (+1 ms), 1/3 lost"
    assert snapshot_label(later, start=first.time + 0.998e-3).startswith(r"t = 2 ms (+2 $\mu$s)")


def test_layers_draw_in_order(snapshots, tmp_path):
    """The seam for overlaying a Poincare plot: a layer listed first is
    drawn first, underneath the particles."""
    first, _ = snapshots
    drawn = []
    panel = RZPanel(title="x", layers=[
        lambda ax: drawn.append("poincare"),
        lambda ax: (drawn.append("particles"), draw_particles(ax, first)),
    ])
    plot_rz_panels([panel], tmp_path / "out.png")
    assert drawn == ["poincare", "particles"]


def test_grid_shares_limits_across_panels(snapshots, tmp_path, monkeypatch):
    seen = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy(fig, *args, **kwargs):
        seen["limits"] = {(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes if ax.get_visible()}
        return real_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    out = plot_rz_panels(particle_panels(list(snapshots)), tmp_path / "particles.png")
    assert out.is_file()
    assert len(seen["limits"]) == 1
    (xlim, _), = seen["limits"]
    assert xlim[0] < 3.6 and xlim[1] > 4.5  # holds every panel's particles


def test_animation_needs_two_frames(snapshots, tmp_path):
    first, _ = snapshots
    assert animate_rz_panels(particle_panels([first]), tmp_path / "a.gif") is None
    assert not (tmp_path / "a.gif").exists()


def test_animation_writes_a_gif(snapshots, tmp_path):
    out = animate_rz_panels(particle_panels(list(snapshots)), tmp_path / "a.gif", dpi=40)
    assert out.is_file() and out.read_bytes()[:3] == b"GIF"


# --- plot --diag particles --------------------------------------------------------

CASES = """
[cases.run]
steps            = [3000]
trace_exe        = "./exe/ex7_jorek"
trace_start_step = 3000

[cases.untraced]
steps = [3000]
"""


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    for name in ("run", "untraced"):
        run = tmp_path / name
        run.mkdir()
        (run / "jorek03000.h5").write_bytes(b"")
    (tmp_path / "cases.toml").write_text(CASES, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASHEN_SITE", raising=False)
    return tmp_path


def test_plot_particles_from_the_trace_folder(campaign, capsys):
    folder = campaign / "run" / "trace" / "ex7_jorek"
    simple_file(folder / "part_restart000.00250000.h5", 2.5e-3, [3.6, 3.7])
    simple_file(folder / "part_restart.h5", 2.6e-3, [3.62, 3.9], i_elm=[1, 0])

    assert plot_cli.main(["--case", "run", "--diag", "particles", "--animate", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert f"particles: 2 snapshot(s) -> {folder / 'particles.png'}" in out
    assert (folder / "particles.png").is_file()
    assert (folder / "particles.gif").is_file()


def test_single_snapshot_explains_why(campaign, capsys):
    simple_file(campaign / "run" / "trace" / "ex7_jorek" / "part_restart.h5", 2.6e-3, [3.6])
    assert plot_cli.main(["--case", "run", "--diag", "particles", "--dpi", "40"]) == 0
    assert "only one snapshot" in capsys.readouterr().out


def test_no_trace_output_yet(campaign, capsys):
    assert plot_cli.main(["--case", "run", "--diag", "particles"]) == 0
    assert "run bin/trace first" in capsys.readouterr().out


def test_untraced_case_is_quiet_unless_asked(campaign, capsys):
    """A default run (no --diag) asks every case for every diag; only an
    explicit --diag particles says why a case was skipped."""
    assert plot_cli.main(["--case", "untraced", "--diag", "particles"]) == 0
    assert "sets no trace_exe" in capsys.readouterr().out

    # What a default run does for this diag (the rest of a default run needs
    # JOREK's tools, which this synthetic folder lacks).
    case = load_cases(campaign / "cases.toml")["untraced"]
    paths = RunPaths.detect(campaign / "untraced")
    plot_cli._plot_particles(case, paths, dpi=None, n_cols=None, animate=False, explicit=False)
    assert capsys.readouterr().out == ""


def test_bad_particle_file_fails_the_case(campaign, capsys):
    folder = campaign / "run" / "trace" / "ex7_jorek"
    folder.mkdir(parents=True)
    (folder / "part_restart.h5").write_bytes(b"not hdf5")
    assert plot_cli.main(["--case", "run", "--diag", "particles"]) == 1
    assert "cannot open" in capsys.readouterr().err
