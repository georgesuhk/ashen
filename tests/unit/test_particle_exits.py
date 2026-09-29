"""ashen.diagnostics.particle_exits and `plot --diag particle_exits`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ashen.cli import plot as plot_cli
from ashen.diagnostics.particle_exits import exit_angles, exits_from_history, read_particle_diag
from ashen.diagnostics.particles import ParticleFileError
from ashen.plotting.particle_exits import exit_caption

h5py = pytest.importorskip("h5py")


def write_diag(path, *, t, psi_n, R, Z, phi, lost, theta=None, transposed=False):
    """A write_particle_diagnostics file: per group, t and each variable as
    (n_times, n_particles) -- Fortran's (n_particles, n_times) seen row-major."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("groups/001")
        g["t"] = np.asarray(t, dtype=np.float32)
        columns = dict(psi_n=psi_n, R=R, Z=Z, phi=phi, lost=lost)
        if theta is not None:
            columns["theta"] = theta
        for name, data in columns.items():
            data = np.asarray(data)
            g[name] = data.T if transposed else data
    return path


#: Three diagnostics times, four particles:
#: 0 crosses psi_n 1 at the second time; 1 stays inside; 2 leaves the grid
#: at the third time without a time showing it past 1; 3 is off the grid
#: from the start.
T = [0.0, 1e-6, 2e-6]
PSI = [[0.5, 0.3, 0.8, 0.0],
       [1.2, 0.4, 0.9, 0.0],
       [1.5, 0.5, 0.0, 0.0]]
LOST = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 1]]
R = [[1.5, 1.6, 1.7, 0.0], [2.0, 1.6, 2.1, 0.0], [2.1, 1.6, 0.0, 0.0]]
Z = [[0.0, 0.0, 0.0, 0.0], [0.3, 0.0, -0.4, 0.0], [0.3, 0.0, 0.0, 0.0]]
PHI = [[0.0, 0.0, 0.0, 0.0], [7.0, 1.0, -1.0, 0.0], [8.0, 2.0, 0.0, 0.0]]
THETA = [[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 4.0, 0.0], [0.6, 0.0, 0.0, 0.0]]


@pytest.fixture
def diag(tmp_path):
    return write_diag(tmp_path / "ptrace_diag.h5", t=T, psi_n=PSI, R=R, Z=Z,
                      phi=PHI, lost=LOST, theta=THETA)


def test_exits_by_crossing_and_by_leaving_the_grid(diag):
    result = exit_angles(read_particle_diag(diag), psi_n=1.0)
    assert (result.n_crossed, result.n_left_grid, result.n_considered) == (1, 1, 3)
    # Particle 0 at the second time; particle 2 at its last on-grid time
    # (the second), theta 4 wrapped into (-pi, pi], phi -1 into [0, 2 pi).
    np.testing.assert_allclose(result.theta, [0.5, 4.0 - 2 * np.pi])
    np.testing.assert_allclose(result.phi, [7.0 - 2 * np.pi, 2 * np.pi - 1.0])
    np.testing.assert_allclose(result.time, [1e-6, 1e-6])


def test_threshold_moves_the_exit(diag):
    history = read_particle_diag(diag)
    later = exit_angles(history, psi_n=1.3)
    np.testing.assert_allclose(later.theta[0], 0.6)
    inner = exit_angles(history, psi_n=0.45)
    assert inner.n_crossed == 3  # particle 1 too, at its third time


def test_transposed_layout_reads_the_same(tmp_path):
    a = read_particle_diag(write_diag(tmp_path / "a.h5", t=T, psi_n=PSI, R=R, Z=Z,
                                      phi=PHI, lost=LOST, theta=THETA))
    b = read_particle_diag(write_diag(tmp_path / "b.h5", t=T, psi_n=PSI, R=R, Z=Z,
                                      phi=PHI, lost=LOST, theta=THETA, transposed=True))
    np.testing.assert_array_equal(a.psi_n, b.psi_n)


def test_theta_from_the_axis_when_the_file_has_none(tmp_path):
    history = read_particle_diag(write_diag(tmp_path / "d.h5", t=T, psi_n=PSI, R=R, Z=Z,
                                            phi=PHI, lost=LOST))
    assert history.theta is None
    with pytest.raises(ParticleFileError, match="no theta"):
        exit_angles(history, psi_n=1.0)
    result = exit_angles(history, psi_n=1.0, axis=(1.7, 0.0))
    np.testing.assert_allclose(result.theta, [np.arctan2(0.3, 0.3), np.arctan2(-0.4, 0.4)])


def test_missing_variables(tmp_path):
    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as f:
        f["groups/001/t"] = np.zeros(2)
        f["groups/001/psi_n"] = np.zeros((2, 1))
    with pytest.raises(ParticleFileError, match=r"no \['R', 'Z', 'phi', 'lost'\]"):
        read_particle_diag(path)


#: A boundary R 1.45..1.95, |Z| <= 0.5: particle 1 (R 1.6) stays inside;
#: particle 0 is outside at the second time (R 2.0), before psi_n 1.3 has it
#: past; particle 2 is outside at the second time (2.1), before leaving the grid.
BOUNDARY = np.array([[1.45, -0.5], [1.95, -0.5], [1.95, 0.5], [1.45, 0.5]])


def test_boundary_exit_comes_first(diag):
    history = read_particle_diag(diag)
    result = exit_angles(history, psi_n=1.3, boundary=BOUNDARY)
    assert (result.n_outside_boundary, result.n_crossed, result.n_left_grid) == (2, 0, 0)
    np.testing.assert_allclose(result.theta, [0.5, 4.0 - 2 * np.pi])
    # Past psi_n on the same diagnostics time counts as the boundary.
    assert exit_angles(history, psi_n=1.0, boundary=BOUNDARY).n_outside_boundary == 2


def test_exits_from_history(diag):
    exits = exits_from_history(read_particle_diag(diag), BOUNDARY)
    np.testing.assert_allclose(exits.time, [1e-6, np.inf, 1e-6, np.inf], rtol=1e-6)
    np.testing.assert_allclose(exits.R[[0, 2]], [2.0, 2.1])
    # Particle 1 at R 1.6 = inside; particle 3 never on the grid.
    assert np.isnan(exits.R[[1, 3]]).all()
    assert exits.source == "ptrace_diag.h5 (every diag_step)"


def test_caption(diag):
    result = exit_angles(read_particle_diag(diag), psi_n=1.0)
    assert exit_caption(result, psi_n=1.0) == (
        "2 of 3 particles exit past psi_n = 1 (1 left the grid first -- "
        "drawn at their last position on it)"
    )


# --- plot --diag particle_exits ----------------------------------------------------

CASES = """
[cases.run]
steps             = [3000]
ptrace_exe        = "./exe/ptrace_gc"
ptrace_start_step = 3000
ptrace_exit_psi_n = 1.3

[cases.other]
steps             = [3000]
ptrace_exe        = "./exe/my_tracer"
ptrace_start_step = 3000
"""


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    for name in ("run", "other"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "jorek03000.h5").write_bytes(b"")
    (tmp_path / "cases.toml").write_text(CASES, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASHEN_SITE", raising=False)
    return tmp_path


def test_plot_uses_the_case_threshold_and_the_flag_overrides_it(campaign, capsys):
    folder = campaign / "run" / "ptrace" / "ptrace_gc"
    write_diag(folder / "ptrace_diag.h5", t=T, psi_n=PSI, R=R, Z=Z, phi=PHI,
               lost=LOST, theta=THETA)
    assert plot_cli.main(["--case", "run", "--diag", "particle_exits", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "2 of 3 particles exit past psi_n = 1.3" in out
    assert (folder / "particle_exits.png").is_file()

    assert plot_cli.main(["--case", "run", "--diag", "particle_exits", "--dpi", "40",
                          "--exit-psi-n", "0.45"]) == 0
    assert "3 of 3 particles exit past psi_n = 0.45" in capsys.readouterr().out


def test_no_theta_uses_the_logged_axis(campaign, capsys):
    folder = campaign / "run" / "ptrace" / "ptrace_gc"
    write_diag(folder / "ptrace_diag.h5", t=T, psi_n=PSI, R=R, Z=Z, phi=PHI, lost=LOST)
    assert plot_cli.main(["--case", "run", "--diag", "particle_exits", "--dpi", "40"]) == 0
    assert "gives no R_axis/Z_axis" in capsys.readouterr().out

    from ashen.paths import RunPaths
    RunPaths.detect(campaign / "run").log.write_text(
        " R_axis             =   1.70000E+00\n Z_axis             =   0.00000E+00\n",
        encoding="utf-8",
    )
    assert plot_cli.main(["--case", "run", "--diag", "particle_exits", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "computing it about the logged axis (R, Z) = (1.7, 0) m" in out
    assert (folder / "particle_exits.png").is_file()


def test_plot_with_the_original_boundary(campaign, capsys):
    folder = campaign / "run" / "ptrace" / "ptrace_gc"
    write_diag(folder / "ptrace_diag.h5", t=T, psi_n=PSI, R=R, Z=Z, phi=PHI,
               lost=LOST, theta=THETA)
    np.savetxt(campaign / "run" / "original_bnd.dat", BOUNDARY)
    (campaign / "cases.toml").write_text(
        CASES.replace("ptrace_exit_psi_n = 1.3", "ptrace_exit_psi_n = 1.3\nptrace_original_boundary = true"),
        encoding="utf-8",
    )
    assert plot_cli.main(["--case", "run", "--diag", "particle_exits", "--dpi", "40"]) == 0
    assert ("2 of 3 particles exit past psi_n = 1.3 or the plasma boundary before "
            "extension (2 at the boundary, 0 at psi_n)") in capsys.readouterr().out


def test_particles_plot_takes_boundary_exits_from_the_diag_file(campaign, capsys, monkeypatch):
    from test_particles import simple_file

    folder = campaign / "run" / "ptrace" / "ptrace_gc"
    write_diag(folder / "ptrace_diag.h5", t=T, psi_n=PSI, R=R, Z=Z, phi=PHI,
               lost=LOST, theta=THETA)
    simple_file(folder / "part_restart_s003000_t0.000000E+00.h5", 0.0, [1.5, 1.6, 1.7, 0.0],
                i_elm=[1, 1, 1, 0])
    simple_file(folder / "part_restart.h5", 2e-6, [2.1, 1.6, 0.0, 0.0], i_elm=[1, 1, 0, 0])
    np.savetxt(campaign / "run" / "original_bnd.dat", BOUNDARY)
    (campaign / "cases.toml").write_text(
        CASES.replace("ptrace_exit_psi_n = 1.3", "ptrace_original_boundary = true"),
        encoding="utf-8",
    )
    seen = {}
    real = plot_cli.particle_panels

    def spy(snapshots, **kwargs):
        seen.update(kwargs)
        return real(snapshots, **kwargs)

    monkeypatch.setattr(plot_cli, "particle_panels", spy)
    assert plot_cli.main(["--case", "run", "--diag", "particles", "--dpi", "40"]) == 0
    # Particle 0 left at the diag time between the two snapshots, at R 2.0.
    np.testing.assert_allclose(seen["exits"].time[:3], [1e-6, np.inf, 1e-6], rtol=1e-6)
    np.testing.assert_allclose(seen["exits"].R[0], 2.0)


def test_not_traced_yet_and_unknown_programs(campaign, capsys):
    assert plot_cli.main(["--case", "run", "--diag", "particle_exits"]) == 0
    assert "run bin/ptrace first" in capsys.readouterr().out
    assert plot_cli.main(["--case", "other", "--diag", "particle_exits"]) == 0
    assert "doesn't know which diagnostics file my_tracer writes" in capsys.readouterr().out
