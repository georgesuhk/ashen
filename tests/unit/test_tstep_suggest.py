"""ashen.tstep_suggest -- the time step from the GMRES iterations on record."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ashen.tstep_history import harvest, parse_job_log
from ashen.tstep_suggest import (
    MAX_GROWTH, Q_ASSUMED, Settings, _round_down, _step_limit, exponent_range, find_pairs,
    format_suggestion, suggest,
)
from support import jorek_log_text

#: qa2.1_g1.8/eta1e-5_adv0.1 (Columbia/NL_kinks), the same states stepped
#: twice: GMRES iterations at t = 961 ... 986, with dt = 1 (job 5229995,
#: refactorised every step, aborted at 200 on the last) and dt = 0.5 (job
#: 5230509, restarted from t = 960; refactorised from t = 984 on).
REAL_DT1 = [69, 72, 74, 76, 79, 82, 86, 90, 92, 96, 99, 103, 108, 112, 118, 123, 131, 137,
            143, 154, 161, 170, 176, 183, 194, 200]
REAL_DT05 = [25, 26, 27, 28, 29, 30, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 37, 38,
             39, 40, 41, 42, 43, 44]
#: and the dt = 0.5 steps in between (t = 960.5, 961.5, ...), which have no
#: dt = 1 partner. The first, from the restart, refactorised.
REAL_DT05_BETWEEN = [25, 25, 26, 27, 28, 29, 30, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                     37, 38, 39, 40, 41, 42, 43]


def _real_jobs(tmp_path):
    crashed = tmp_path / "crashed"
    crashed.write_text(jorek_log_text([(1.0, n, True) for n in REAL_DT1],
                                      first_index=962, t0=961.0, ending="gmresfail"))
    steps = []
    for k, (between, paired) in enumerate(zip(REAL_DT05_BETWEEN, REAL_DT05)):
        steps.append((0.5, between, k == 0))
        steps.append((0.5, paired, k >= 23))
    restart = tmp_path / "restart"
    restart.write_text(jorek_log_text(steps, first_index=962, t0=960.5))
    return [parse_job_log(crashed, sha256="a"), parse_job_log(restart, sha256="b")]


def test_same_time_pairs_from_the_real_crash_and_restart(tmp_path):
    jobs = _real_jobs(tmp_path)
    pairs = find_pairs(jobs)
    # the failed step (t = 986, 200 iterations) is no pair: it only says N >= 200
    assert len(pairs) == 25
    assert all(p.dt_a == 1.0 and p.dt_b == 0.5 for p in pairs)
    fresh = [p for p in pairs if p.fresh]
    assert [p.t for p in fresh] == [984.0, 985.0]
    (q_low, q_high), source = exponent_range(pairs)
    assert "both refactorised" in source
    # 183/42 and 194/43: N ~ dt^2.1..2.2 near the limit
    assert q_low == pytest.approx(math.log2(183 / 42)) and q_high == pytest.approx(math.log2(194 / 43))
    assert 2.0 < q_low < q_high < 2.3


def test_without_fresh_pairs_every_pair_is_used(tmp_path):
    jobs = _real_jobs(tmp_path)
    pairs = [p for p in find_pairs(jobs) if not p.fresh]
    (q_low, q_high), source = exponent_range(pairs)
    assert "reused" in source and q_low < 1.6       # a staler preconditioner on one side


def test_step_limit_is_on_the_safe_side():
    q = (1.5, 2.2)
    # shrinking: the smallest q, the larger cut
    assert _step_limit(1.0, 200, 100, q, failed=True) == pytest.approx(0.5 ** (1 / 1.5))
    # growing: the largest q, the smaller rise
    assert _step_limit(0.5, 44, 100, q, failed=False) == pytest.approx(0.5 * (100 / 44) ** (1 / 2.2))
    # one or two iterations are at the floor: counted as MIN_ITERS
    assert _step_limit(0.03, 1, 100, q, failed=False) == _step_limit(0.03, 3, 100, q, failed=False)


def test_round_down_two_figures():
    assert _round_down(0.4189) == pytest.approx(0.41)
    assert _round_down(0.06) == pytest.approx(0.06)
    assert _round_down(1.39) == pytest.approx(1.3)


# --- whole suggestions from a run folder -----------------------------------------


def _run(tmp_path, name, logs, *, in_main_r=True, max_iter=200):
    run = tmp_path / name
    run.mkdir(parents=True)
    for i, text in enumerate(logs):
        (run / f"job.{i + 1}.out").write_text(text)
    (run / "in_main_r").write_text(
        f"&in1\n gmres_max_iter = {max_iter}\n iter_precon = 40\n time_evol_scheme = 'Gears'\n"
        " use_strumpack = .true.\n n_radial = 60\n n_pol = 60\n&end\n")
    (run / "in_main").write_text((run / "in_main_r").read_text())
    (run / "shotfile.py").write_text("eta = 1e-5\nrho_const = 1e18\nqa = 2.1\nwith_refluid = True\n")
    return run


def test_restart_before_a_recorded_crash_steps_below_it(tmp_path):
    crashed = jorek_log_text([(1.0, 2, True)] * 50 + [(1.0, n, True) for n in REAL_DT1],
                             ending="gmresfail")
    run = _run(tmp_path, "run", [crashed])
    s = suggest(run, t_now=40.0)
    assert s.binding.kind == "iterations"
    assert "did not converge" in s.binding.why
    # 200 iterations at dt = 1 with q = 1 (assumed, no pairs): half the step
    assert s.q == Q_ASSUMED
    assert s.dt == pytest.approx(_round_down(1.0 * 0.5 ** (1 / Q_ASSUMED[0])))
    assert s.dt < 1.0


def test_the_frontier_says_nothing_is_known_ahead(tmp_path):
    job = jorek_log_text([(0.5, 95 + k % 5, True) for k in range(100)])
    run = _run(tmp_path, "run", [job])
    s = suggest(run, t_now=50.0)
    assert any("nothing is known of the time ahead" in note for note in s.notes)
    assert s.binding.kind == "iterations"
    # ~99 of a 100 target: about where it is
    assert 0.45 < s.dt <= 0.5


def test_never_more_than_max_growth_times_the_step_run(tmp_path):
    job = jorek_log_text([(0.03, 1, k == 0) for k in range(200)])
    run = _run(tmp_path, "run", [job])
    s = suggest(run, t_now=6.0)
    assert s.binding.kind == "growth"
    assert s.dt == pytest.approx(MAX_GROWTH * 0.03)


def test_a_nan_crash_ahead_halves_its_step(tmp_path):
    job = jorek_log_text([(0.2, 5, True)] * 20, ending="nan")
    run = _run(tmp_path, "run", [job])
    s = suggest(run, t_now=1.0)
    assert s.binding.kind == "nan" and s.dt == pytest.approx(0.1)
    assert "heuristic" in s.binding.why


def test_accuracy_caps_a_step_the_solver_would_manage(tmp_path):
    from test_tstep_history import _write_live

    job = jorek_log_text([(1.0, 3, True)] * 100)
    run = _run(tmp_path, "run", [job])
    _write_live(run / "macroscopic_vars.dat", gamma=0.25, t_end=100.0)
    s = suggest(run, t_now=50.0, settings=Settings(accuracy=0.1))
    assert s.binding.kind == "accuracy"
    assert s.dt == pytest.approx(0.4)                 # 0.1 / 0.25


def test_a_new_run_takes_the_hardest_step_of_similar_runs(tmp_path):
    campaign = tmp_path / "campaign"
    near = _run(campaign, "eta1e-5", [jorek_log_text([(0.5, 2, True)] * 10 + [(0.5, 130, True)])])
    far = _run(campaign, "eta1e-3", [jorek_log_text([(0.03, 20, True)] * 10)])
    for run in (near, far):
        harvest(run)
    (far / "shotfile.py").write_text("eta = 1e-3\nrho_const = 1e18\nqa = 3.3\nwith_refluid = True\n")
    harvest(far)
    new = _run(campaign, "new", [])
    s = suggest(new, campaign, t_now=None)
    neighbours = [lim for lim in s.limits if lim.kind == "neighbours"]
    assert len(neighbours) == 1 and s.binding.kind in ("neighbours", "growth")
    assert sum("similar run" in note for note in s.notes) == 2
    # the far run's 20 iterations at 0.03 allow less than the near run's 130 at 0.5
    assert "eta1e-3" in neighbours[0].why


def test_similar_runs_are_only_context_for_a_run_with_its_own_history(tmp_path):
    campaign = tmp_path / "campaign"
    other = _run(campaign, "other", [jorek_log_text([(0.03, 20, True)] * 10)])
    harvest(other)
    mine = _run(campaign, "mine", [jorek_log_text([(0.5, 95, True)] * 20)])
    s = suggest(mine, campaign, t_now=5.0)
    assert not any(lim.kind == "neighbours" for lim in s.limits)
    assert any("similar run other" in note for note in s.notes)


def test_runs_whose_solver_settings_differ_are_left_out(tmp_path):
    job = jorek_log_text([(0.5, 95, True)] * 20, iter_precon=20)
    run = _run(tmp_path, "run", [job])                 # in_main_r: iter_precon 40
    s = suggest(run, t_now=5.0)
    assert s.dt is None
    assert any("iter_precon" in note for note in s.notes)


def test_the_planned_cap_sets_the_target(tmp_path):
    job = jorek_log_text([(0.5, 95, True)] * 20)
    run = _run(tmp_path, "run", [job], max_iter=400)
    s = suggest(run, t_now=5.0)
    assert s.n_target == pytest.approx(200) and s.dt > 0.5


def test_format_names_the_binding_limit(tmp_path):
    run = _run(tmp_path, "run", [jorek_log_text([(0.5, 95, True)] * 20)])
    text = format_suggestion(suggest(run, t_now=5.0))
    assert "suggested: tstep =" in text and "(set by: iterations)" in text
    assert "N ~ dt^q" in text and "aim: at most 100 GMRES iterations" in text
