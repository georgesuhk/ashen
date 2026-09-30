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
from ashen.diagnostics import poincare_cache as pc
from ashen.diagnostics.particles import (
    BoundaryExits,
    ParticleFileError,
    exits_from_snapshots,
    find_snapshots,
    freeze_exited,
    inside_polygon,
    named_step,
    read_snapshot,
)
from ashen.plotting.particles import (
    PoincareOverlay,
    RZPanel,
    animate_rz_panels,
    draw_particles,
    particle_caption,
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


def test_find_snapshots_reads_ptrace_gc_names(tmp_path):
    """ptrace_gc names snapshots part_restart_s<step>_t<time>.h5; the order
    still comes from the time inside each file."""
    simple_file(tmp_path / "part_restart_s003200_t2.600000E-03.h5", 2.6e-3, [3.8])
    simple_file(tmp_path / "part_restart_s003000_t2.500000E-03.h5", 2.5e-3, [3.7])
    simple_file(tmp_path / "part_restart.h5", 2.7e-3, [3.9])
    assert [s.time for s in find_snapshots(tmp_path)] == [2.5e-3, 2.6e-3, 2.7e-3]


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
    assert snapshot_label(first) == "t = 1 ms, 0/3 escaped"
    assert snapshot_label(first, start=first.time) == "t = 1 ms, 0/3 escaped"
    assert snapshot_label(later) == "t = 2 ms, 1/3 escaped"
    assert snapshot_label(later, start=first.time) == "t = 2 ms (+1 ms), 1/3 escaped"
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
ptrace_exe        = "./exe/ex7_jorek"
ptrace_start_step = 3000

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
    folder = campaign / "run" / "ptrace" / "ex7_jorek"
    simple_file(folder / "part_restart000.00250000.h5", 2.5e-3, [3.6, 3.7])
    simple_file(folder / "part_restart.h5", 2.6e-3, [3.62, 3.9], i_elm=[1, 0])

    assert plot_cli.main(["--case", "run", "--diag", "particles", "--animate", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert f"particles: 2 snapshot(s) -> {folder / 'particles.png'}" in out
    assert (folder / "particles.png").is_file()
    assert (folder / "particles.gif").is_file()


def test_single_snapshot_explains_why(campaign, capsys):
    simple_file(campaign / "run" / "ptrace" / "ex7_jorek" / "part_restart.h5", 2.6e-3, [3.6])
    assert plot_cli.main(["--case", "run", "--diag", "particles", "--dpi", "40"]) == 0
    assert "only one snapshot" in capsys.readouterr().out


def test_no_trace_output_yet(campaign, capsys):
    assert plot_cli.main(["--case", "run", "--diag", "particles"]) == 0
    assert "run bin/ptrace first" in capsys.readouterr().out


def test_untraced_case_is_quiet_unless_asked(campaign, capsys):
    """A default run (no --diag) asks every case for every diag; only an
    explicit --diag particles says why a case was skipped."""
    assert plot_cli.main(["--case", "untraced", "--diag", "particles"]) == 0
    assert "sets no ptrace_exe" in capsys.readouterr().out

    # What a default run does for this diag (the rest of a default run needs
    # JOREK's tools, which this synthetic folder lacks).
    case = load_cases(campaign / "cases.toml")["untraced"]
    paths = RunPaths.detect(campaign / "untraced")
    plot_cli._plot_particles(case, paths, dpi=None, n_cols=None, animate=False, explicit=False)
    assert capsys.readouterr().out == ""


def test_bad_particle_file_fails_the_case(campaign, capsys):
    folder = campaign / "run" / "ptrace" / "ex7_jorek"
    folder.mkdir(parents=True)
    (folder / "part_restart.h5").write_bytes(b"not hdf5")
    assert plot_cli.main(["--case", "run", "--diag", "particles"]) == 1
    assert "cannot open" in capsys.readouterr().err


# --- original boundary and Poincare overlay -------------------------------------

#: A square plasma boundary, R 3.5..4.0, Z -0.25..0.25.
SQUARE = np.array([[3.5, -0.25], [4.0, -0.25], [4.0, 0.25], [3.5, 0.25]])


def test_named_step():
    assert named_step("part_restart_s003200_t2.500000E-03.h5") == 3200
    assert named_step("part_restart.h5") is None
    assert named_step("part_restart000.00250000.h5") is None


def test_inside_polygon():
    inside = inside_polygon([3.6, 4.5, 3.9, 3.7], [0.0, 0.0, 0.3, -0.2], SQUARE)
    assert inside.tolist() == [True, False, False, True]


#: The square, shrunk to R 3.5..3.72.
SHRUNK = SQUARE - [[0, 0], [0.28, 0], [0.28, 0], [0, 0]]


def test_exits_from_snapshots_first_time_outside_on_the_grid(snapshots):
    first, later = snapshots  # first: R 3.6, 3.7, 3.8; later: 3.65, 3.75, 4.5 (lost)
    exits = exits_from_snapshots([first, later], SHRUNK)
    # Particle 1 leaves at the later snapshot, particle 2 at the first (3.8);
    # being lost later doesn't move its exit.
    np.testing.assert_allclose(exits.time, [np.inf, 2e-3, 1e-3])
    np.testing.assert_allclose(exits.R[1:], [3.75, 3.8])


def test_freeze_exited_stops_tracking_at_the_exit(snapshots):
    _, later = snapshots
    exits = BoundaryExits(
        time=np.array([np.inf, 1.5e-3, 1e-3]), R=np.array([np.nan, 3.71, 3.8]),
        Z=np.array([np.nan, 0.1, 0.0]),
    )
    frozen, exited = freeze_exited(later, exits)
    assert exited.tolist() == [False, True, True]
    np.testing.assert_allclose(frozen.R, [3.65, 3.71, 3.8])
    np.testing.assert_allclose(frozen.Z, [0.0, 0.1, 0.0])
    # Particle 2 left the boundary before it was lost: an exit, not a loss.
    assert frozen.n_lost == 0
    untouched, exited = freeze_exited(later, BoundaryExits(
        time=np.full(3, 2.5e-3), R=np.zeros(3), Z=np.zeros(3)))
    assert untouched is later and not exited.any()


def test_freeze_exited_needs_the_same_particles(snapshots):
    _, later = snapshots
    with pytest.raises(ParticleFileError, match="exits are for 2"):
        freeze_exited(later, BoundaryExits(time=np.zeros(2), R=np.zeros(2), Z=np.zeros(2)))


def test_panels_freeze_particles_that_left_the_boundary(snapshots):
    first, later = snapshots
    panels = particle_panels([first, later], boundary=SHRUNK)
    assert panels[0].title == "t = 1 ms, 1/3 escaped"
    assert panels[1].title == "t = 2 ms (+1 ms), 2/3 escaped"
    fig, ax = plt.subplots()
    for layer in panels[1].layers:
        layer(ax)
    _, tracked, left = ax.collections  # grey start, black, magenta crosses
    np.testing.assert_allclose(tracked.get_offsets(), [[3.65, 0.0]])
    np.testing.assert_allclose(left.get_offsets(), [[3.75, 0.0], [3.8, 0.0]])
    plt.close(fig)


def test_outside_particles_are_magenta_crosses(snapshots):
    _, later = snapshots
    fig, ax = plt.subplots()
    outside = np.array([False, True, False])
    draw_particles(ax, later, outside=outside)
    inside, lost, out = ax.collections
    assert len(inside.get_offsets()) == 1
    np.testing.assert_allclose(lost.get_offsets(), [[4.5, 0.0]])
    np.testing.assert_allclose(out.get_offsets(), [[3.75, 0.0]])
    plt.close(fig)


def test_label_counts_particles_outside(snapshots):
    _, later = snapshots
    # Lost from the grid and out of the boundary both count as escaped.
    assert snapshot_label(later, n_outside=1) == "t = 2 ms, 2/3 escaped"


def _record(psi_n, n):
    key = pc.LineKey(psi_n, 3.7, 0.0, 0.0)
    R = np.linspace(3.6, 3.9, n)
    return key, pc.LineRecord(
        key=key, n_turns=n, terminated=False, n_segments=1,
        R=R, Z=np.zeros(n), rho=np.full(n, np.sqrt(psi_n)), theta=np.zeros(n),
    )


def test_select_lines_by_psi_n_and_turns():
    records = dict([_record(0.5, 10), _record(0.9, 10)])
    chosen, missing = pc.select_lines(records, [0.9, 0.7], n_turns=4)
    assert [key.psi_n for key in chosen] == [0.9]
    assert next(iter(chosen.values())).R.size == 4
    assert missing == [0.7]
    everything, missing = pc.select_lines(records)
    assert len(everything) == 2 and missing == []


def test_panels_stack_poincare_particles_boundary(snapshots, tmp_path):
    first, later = snapshots
    overlay = PoincareOverlay(step=3000, records=dict([_record(0.5, 20)]))
    panels = particle_panels([first, later], boundary=SQUARE, poincare=[overlay, overlay])
    assert panels[1].title.endswith("\nPoincare: step 3000")
    fig, ax = plt.subplots()
    for layer in panels[1].layers:
        layer(ax)
    punctures = ax.collections[0]
    assert punctures.get_alpha() < 1 and punctures.get_zorder() < 2
    assert len(ax.lines) == 1  # the boundary, closed
    np.testing.assert_allclose(ax.lines[0].get_xydata()[[0, -1]], [SQUARE[0], SQUARE[0]])
    plt.close(fig)
    caption = particle_caption([first], boundary=True, poincare=True)
    assert "magenta" in caption and "Poincare" in caption


def test_panels_need_one_overlay_per_snapshot(snapshots):
    with pytest.raises(ValueError, match="1 Poincare overlays for 2 snapshots"):
        particle_panels(list(snapshots), poincare=[None])


def _write_poincare_cache(run_dir, step, lines):
    paths = RunPaths.detect(run_dir)
    with pc.open_cache(paths.poincare_cache(step), step=step, pad_width=paths.pad_width) as h:
        for psi_n, n in lines:
            key, record = _record(psi_n, n)
            pc.append_line(h, key, {a: getattr(record, a) for a in ("R", "Z", "rho", "theta")},
                           n_turns=n, terminated=False)


def test_plot_particles_with_poincare_and_original_boundary(campaign, capsys, monkeypatch):
    run = campaign / "run"
    (run / "jorek03200.h5").write_bytes(b"")
    (run / "real_psi_edge.dat").write_text("0.5\n", encoding="utf-8")
    np.savetxt(run / "original_bnd.dat", SQUARE)
    # Cached lines at JOREK-grid psi_n 0.25 and 0.45 = psi_n_in 0.5, 0.9.
    _write_poincare_cache(run, 3000, [(0.25, 30), (0.45, 30)])
    _write_poincare_cache(run, 3200, [(0.25, 30)])
    folder = run / "ptrace" / "ex7_jorek"
    simple_file(folder / "part_restart_s003000_t2.500000E-03.h5", 2.5e-3, [3.6, 3.7])
    simple_file(folder / "part_restart_s003200_t2.600000E-03.h5", 2.6e-3, [3.62, 4.2])
    (campaign / "cases.toml").write_text(CASES.replace(
        "ptrace_start_step = 3000",
        "ptrace_start_step = 3000\nptrace_poincare_psi_n = [0.9]\n"
        "ptrace_poincare_n_turns = 10\nptrace_original_boundary = true",
    ), encoding="utf-8")

    drawn = []
    real = plot_cli.particle_panels

    def spy(snapshots, **kwargs):
        drawn.append(kwargs)
        return real(snapshots, **kwargs)

    monkeypatch.setattr(plot_cli, "particle_panels", spy)
    assert plot_cli.main(["--case", "run", "--diag", "particles", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert (folder / "particles.png").is_file()
    (kwargs,) = drawn
    np.testing.assert_allclose(kwargs["boundary"], SQUARE)
    first, second = kwargs["poincare"]
    # Each snapshot gets the step in its name; only psi_n_in 0.9, cut to 10 turns.
    assert (first.step, second.step) == (3000, 3200)
    assert [k.psi_n for k in first.records] == [pytest.approx(0.45)]
    assert all(r.n_points == 10 for r in first.records.values())
    assert second.records == {}
    assert "step 3200's Poincare cache has no line at psi_n [0.9] (it has [0.5])" in out


def test_poincare_overlay_without_a_cache_says_how_to_get_one(campaign, capsys):
    simple_file(campaign / "run" / "ptrace" / "ex7_jorek" / "part_restart.h5", 2.6e-3, [3.6])
    (campaign / "cases.toml").write_text(CASES.replace(
        "ptrace_start_step = 3000", "ptrace_start_step = 3000\nptrace_poincare = true",
    ), encoding="utf-8")
    assert plot_cli.main(["--case", "run", "--diag", "particles", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "no Poincare cache for the traced steps (3000 on)" in out
    assert "step 3000 in the case's poincare steps and run `analyse --case run --diag poincare`" in out


def test_original_boundary_missing_is_a_note(campaign, capsys):
    simple_file(campaign / "run" / "ptrace" / "ex7_jorek" / "part_restart.h5", 2.6e-3, [3.6])
    (campaign / "cases.toml").write_text(CASES.replace(
        "ptrace_start_step = 3000", "ptrace_start_step = 3000\nptrace_original_boundary = true",
    ), encoding="utf-8")
    assert plot_cli.main(["--case", "run", "--diag", "particles", "--dpi", "40"]) == 0
    assert "no original_bnd.dat" in capsys.readouterr().out


def test_particles_are_red_by_default_and_the_colour_can_be_set(snapshots):
    import matplotlib.colors as mcolors

    first, later = snapshots
    for color, expected in ((None, "red"), ("tab:orange", "tab:orange")):
        fig, ax = plt.subplots()
        kwargs = {} if color is None else {"color": color}
        draw_particles(ax, later, **kwargs)
        alive, lost = ax.collections
        assert mcolors.same_color(alive.get_facecolor()[0][:3], expected)
        assert mcolors.same_color(lost.get_edgecolor()[0], "black")  # lost crosses stay apart
        plt.close(fig)
    assert particle_caption([first], color="tab:orange").startswith("tab:orange: particles now")


def test_animation_has_no_caption(campaign, monkeypatch):
    folder = campaign / "run" / "ptrace" / "ex7_jorek"
    simple_file(folder / "part_restart000.00250000.h5", 2.5e-3, [3.6, 3.7])
    simple_file(folder / "part_restart.h5", 2.6e-3, [3.62, 3.9], i_elm=[1, 0])
    seen = {}
    real = plot_cli.animate_rz_panels

    def spy(frames, out, **kwargs):
        seen.update(kwargs)
        return real(frames, out, **kwargs)

    monkeypatch.setattr(plot_cli, "animate_rz_panels", spy)
    assert plot_cli.main(["--case", "run", "--diag", "particles", "--animate", "--dpi", "40"]) == 0
    assert "caption" not in seen
