"""ashen.ptracing.trace_summary: what a finished trace covered, read from
ptrace_gc's log in the formats ptrace_gc.f90 writes it."""

from __future__ import annotations

from ashen.ptracing import LOG_FILE, trace_summary

TABLE = """ptrace_gc: settings from ptrace_settings.nml
ptrace_gc: start time   1.000000E-03 s
ptrace_gc: restart step 3000 at t =   1.000000E-03 s
ptrace_gc: restart step 4000 at t =   1.010000E-03 s
ptrace_gc: restart step 5000 at t =   1.018500E-03 s
"""
TO_LAST = "ptrace_gc: tracing to the last restart, step 5000, t =  1.01850E-03 s\n"


def _folder(tmp_path, log, settings=None):
    (tmp_path / LOG_FILE).write_text(log, encoding="utf-8")
    if settings is not None:
        (tmp_path / "ptrace_settings.nml").write_text(settings, encoding="utf-8")
    return tmp_path


def test_a_trace_that_stopped_early_on_stalled_losses(tmp_path):
    log = (TABLE + TO_LAST
           + "ptrace_gc: NOTE: losses stalled -- stopped at t =  1.01750E-03 s,  94.59 % of the "
             "way through the trace: the loss rate is   4.20 % of its peak, with 640 of 1000 "
             "markers gone\n"
           + "ptrace_gc: done at t =  1.01750E-03 s, 640 of 1000 markers lost\n")
    lines = trace_summary(_folder(tmp_path, log))
    assert lines[0] == "markers: 1000, 640 lost (64 %)"
    # 17.5 of 18.5 us; step 4000 at 10 us, 5000 at 18.5 us: 17.5 us is ~4882
    assert lines[1] == "traced 17.5 / 18.5 us (94.6 %), JOREK steps 3000 -> ~4882 / 5000"
    assert lines[2] == "stopped early: losses stalled"


def test_a_trace_that_got_to_its_end(tmp_path):
    log = TABLE + TO_LAST + "ptrace_gc: done at t =  1.01850E-03 s, 12 of 400 markers lost\n"
    assert trace_summary(_folder(tmp_path, log)) == [
        "markers: 400, 12 lost (3 %)",
        "traced 18.5 / 18.5 us (100 %), JOREK steps 3000 -> 5000 / 5000",
    ]


def test_a_t_span_past_the_last_restart_has_no_step_for_its_end(tmp_path):
    log = TABLE + "ptrace_gc: done at t =  1.03000E-03 s, 0 of 10 markers lost\n"
    settings = "&ptrace\n  t_span = 3.0d-05,\n  n_markers = 10\n/\n"
    assert trace_summary(_folder(tmp_path, log, settings))[1] == "traced 30 / 30 us (100 %)"


def test_a_t_span_cut_back_to_the_last_restart(tmp_path):
    log = (TABLE + "ptrace_gc: NOTE: stopped early -- t_span runs to t =  1.03000E-03 s, but "
           "the last restart (step 5000) is at t =  1.01850E-03 s; traced ...\n"
           + "ptrace_gc: done at t =  1.01850E-03 s, 3 of 10 markers lost\n")
    settings = "&ptrace\n  t_span = 3.0d-05\n/\n"
    assert trace_summary(_folder(tmp_path, log, settings))[1] == (
        "traced 18.5 / 18.5 us (100 %), JOREK steps 3000 -> 5000 / 5000")


def test_every_marker_off_the_grid(tmp_path):
    log = (TABLE + TO_LAST
           + "ptrace_gc: NOTE: every marker is off the grid -- stopped at t =  1.01000E-03 s,  "
             "54.05 % of the way through the trace\n"
           + "ptrace_gc: done at t =  1.01000E-03 s, 10 of 10 markers lost\n")
    lines = trace_summary(_folder(tmp_path, log))
    assert lines[1] == "traced 10 / 18.5 us (54.1 %), JOREK steps 3000 -> 4000 / 5000"
    assert lines[2] == "stopped early: every marker off the grid"


def test_nothing_without_a_finished_ptrace_gc_log(tmp_path):
    assert trace_summary(tmp_path) == []                                    # no log
    assert trace_summary(_folder(tmp_path, TABLE + TO_LAST)) == []          # never got to done
    assert trace_summary(_folder(tmp_path, "ex7_jorek: some other program\n")) == []
