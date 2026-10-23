"""ashen.tstep_history -- each job's steps, time steps and GMRES iterations."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from ashen.log_archive import apply_archive, plan_archive
from ashen.tstep_history import (
    RECORD_FILE, JobHistory, find_histories, harvest, linear_growth, load_history,
    parse_job_log,
)
from support import jorek_log_text


def test_steps_times_iterations_and_refactorisations(tmp_path):
    path = tmp_path / "log"
    path.write_text(jorek_log_text([(0.5, 25, True), (0.5, 26, False), (0.5, 41, True)],
                                   first_index=961, t0=960.0))
    job = parse_job_log(path)
    assert job.index.tolist() == [961, 962, 963]
    np.testing.assert_allclose(job.dt, 0.5)
    np.testing.assert_allclose(job.t_start, [960.0, 960.5, 961.0])   # the state each solve began from
    assert job.t_end == pytest.approx(961.5)
    assert job.iters.tolist() == [25, 26, 41]
    assert job.refactor.tolist() == [True, False, True]
    assert job.max_iter == 200 and job.outcome is None


def test_a_run_from_the_start_begins_at_t_zero(tmp_path):
    path = tmp_path / "log"
    path.write_text(jorek_log_text([(1.0, 2, True)] * 3))
    np.testing.assert_allclose(parse_job_log(path).t_start, [0.0, 1.0, 2.0])


def test_the_settings_come_from_the_jobs_own_log(tmp_path):
    path = tmp_path / "log"
    path.write_text(jorek_log_text([(1.0, 2, True)], eta=1e-3, iter_precon=20, max_iter=300))
    params = parse_job_log(path).params
    assert params["eta"] == pytest.approx(1e-3)
    assert params["iter_precon"] == 20 and params["gmres_max_iter"] == 300
    assert params["gmres_tol"] == pytest.approx(1e-8)
    assert params["solver"] == "strumpack" and params["time_evol_scheme"] == "Gears"
    assert params["n_tor"] == 5 and params["n_radial"] == 60
    assert params["sqrt_mu0_rho0"] == pytest.approx(6.4836e-8)
    assert params["git"].startswith("v2.25")


def test_a_failed_step_is_kept_and_a_step_killed_midway_is_not(tmp_path):
    path = tmp_path / "log"
    path.write_text(jorek_log_text([(1.0, 100, True), (1.0, 200, True)], ending="gmresfail"))
    job = parse_job_log(path)
    assert job.outcome == "gmresfail" and job.failed_step() == 1 and job.iters[-1] == 200

    path.write_text(jorek_log_text([(1.0, 3, True)] * 2, killed_mid_step=True))
    job = parse_job_log(path, outcome="walltime")
    assert len(job.iters) == 2 and job.outcome == "walltime"


def test_a_log_two_jobs_wrote_is_read_up_to_where_the_second_begins(tmp_path):
    path = tmp_path / "log"
    first = jorek_log_text([(0.5, 94, True)] * 4, first_index=3401, t0=2180.0)
    other = jorek_log_text([(0.5, 99, True)] * 2, first_index=3856, ending="gmresfail")
    path.write_text(first + other[other.index(" *   time step"):])
    job = parse_job_log(path, outcome="gmresfail")
    assert job.index.tolist() == [3401, 3402, 3403, 3404]
    assert job.params["spliced_after"] == 3404
    assert job.outcome is None                  # neither the other job's abort nor the file's


def test_record_round_trips(tmp_path):
    path = tmp_path / "log"
    path.write_text(jorek_log_text([(1.0, 2, True)] * 3 + [(0.5, 7, False)] * 2,
                                   first_index=11, t0=10.0, ending="nan"))
    job = parse_job_log(path, sha256="abc")
    back = JobHistory.from_json(json.loads(json.dumps(job.to_json())))
    for name in ("index", "dt", "iters", "refactor"):
        np.testing.assert_array_equal(getattr(back, name), getattr(job, name))
    assert (back.t0, back.outcome, back.params, back.sha256) == \
        (job.t0, job.outcome, job.params, "abc")


# --- the run's record ----------------------------------------------------------


def _write_live(path, gamma=0.036, t_end=1000.0, dt=1.0):
    """macroscopic_vars.dat with an n = 1 mode growing at gamma from 1e-30 of
    n = 0 until it saturates at 1e-3, and its growth rate, as JOREK writes
    them (in ms and 1/s, with the factors back to JOREK units)."""
    x2si, y2si = 6.48e-5, 1.54e7
    lines = [
        "@magnetic_energies_x2si:   1.0E+00\n",
        '@magnetic_energies: %"time"   "E_{mag,00}" "E_{mag,01}"\n',
        f"@magnetic_growth_rates_x2si:   {x2si:.9E}\n",
        f"@magnetic_growth_rates_y2si:   {y2si:.9E}\n",
        '@magnetic_growth_rates: %"time"   "G_{mag,00}" "G_{mag,01}"\n',
    ]
    for t in np.arange(dt, t_end + dt / 2, dt):
        e1 = min(1e-30 * np.exp(2 * gamma * t), 1e-3)
        g1 = gamma if e1 < 1e-3 else 0.0
        lines.append(f"@magnetic_energies:  {t:.9E}  1.0E+00  {e1:.9E}\n")
        lines.append(f"@magnetic_growth_rates:  {t:.9E}  0.0E+00  {g1:.9E}\n")
    path.write_text("".join(lines))


def test_linear_growth_and_onset(tmp_path):
    _write_live(tmp_path / "macroscopic_vars.dat", gamma=0.036)
    growth = linear_growth(tmp_path / "macroscopic_vars.dat")
    assert growth["gamma"] == pytest.approx(0.036, rel=1e-6) and growth["n"] == 1
    # E_1 / E_0 = 1e-30 exp(0.072 t) passes 1e-5 at t = ln(1e25) / 0.072 = 799.4
    assert growth["t_onset"] == pytest.approx(800.0)
    assert linear_growth(tmp_path / "nope.dat") is None


def test_harvest_reads_every_kept_log_once(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "jorek00000.h5").touch()
    crashed = jorek_log_text([(1.0, 2, True)] * 3 + [(1.0, 200, True)], ending="gmresfail")
    restart = jorek_log_text([(0.5, 25, True)] * 4, first_index=3, t0=2.0)
    (run / "a.1.out").write_text(crashed)
    (action,) = plan_archive(run, [run / "a.1.out"])
    apply_archive(action)
    (run / "a.1.out").unlink()                       # only the archive is left
    (run / "a.2.out").write_text(restart)
    (run / "log").write_text(restart)                # the same job: read once
    (run / "shotfile.py").write_text("eta = 1e-5\nn0 = 1e18\nrho_const = n0\n"
                                     "tstep_n = [0.5] # [1.0]\nnstep_n = [4]\n")
    _write_live(run / "macroscopic_vars.dat")

    history = harvest(run)
    assert len(history.jobs) == 2
    assert [j.outcome for j in history.jobs] == ["gmresfail", None]
    assert history.jobs[0].source == "log_s00001-s00004_gmresfail.gz"
    assert history.shot == {"eta": 1e-5, "rho_const": 1e18}
    assert history.hints == ["tstep_n = [0.5] # [1.0]"]
    assert history.growth["gamma"] == pytest.approx(0.036, rel=1e-6)

    assert (run / RECORD_FILE).is_file()
    saved = load_history(run)
    assert [len(j.iters) for j in saved.jobs] == [4, 4]
    assert find_histories(tmp_path) == [run]


def test_a_saved_record_from_an_older_parser_is_read_again(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "log").write_text(jorek_log_text([(1.0, 5, True)] * 2))
    harvest(run)
    with gzip.open(run / RECORD_FILE, "rt") as f:
        data = json.load(f)
    data["version"] = 0
    data["jobs"][0]["iters"] = [999, 999]             # what an old parser got wrong
    with gzip.open(run / RECORD_FILE, "wt") as f:
        json.dump(data, f)
    assert harvest(run).jobs[0].iters.tolist() == [5, 5]


def test_a_job_whose_log_is_gone_stays_in_the_record(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "a.1.out").write_text(jorek_log_text([(1.0, 200, True)], ending="gmresfail"))
    harvest(run)
    (run / "a.1.out").unlink()                       # deleted without being archived
    (run / "log").write_text(jorek_log_text([(0.5, 25, True)] * 2))
    history = harvest(run)
    assert sorted(j.outcome or "" for j in history.jobs) == ["", "gmresfail"]


def test_nothing_to_record_writes_nothing(tmp_path):
    harvest(tmp_path)
    assert not (tmp_path / RECORD_FILE).exists()
