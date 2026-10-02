"""ashen.particle_programs -- ptrace_gc's settings: the defaults, what a
case's keys resolve to, and the ptrace_settings.nml ashen writes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ashen.particle_programs import (
    MAX_MARKERS,
    PTRACE_DEFAULTS,
    PTRACE_SETTINGS,
    PtraceSettingsError,
    describe_settings,
    energy_text,
    resolve_settings,
    settings_namelist,
)

FORTRAN = Path(__file__).resolve().parents[2] / "fortran" / "ptrace_gc.f90"

PDF = {"initialiser": "current_pdf_simple", "n_markers": 1000, "E_kin_eV": [1e7],
       "cos_pitch": [0.9]}
MARKERS = {"n_markers": 3, "R0": [1.5, 1.6, 1.7], "Z0": [0.0], "E_kin_eV": [1e7],
           "cos_pitch": [0.9]}


def _fortran_declarations() -> str:
    text = FORTRAN.read_text(encoding="utf-8")
    return text[text.index("! --- &ptrace namelist ---"):text.index("namelist /ptrace/")]


def _fortran_default(name: str):
    """`name`'s initial value in ptrace_gc.f90's declarations, as Python."""
    match = re.search(rf"\b{name}(?:\([A-Z_]+\))?\s*=\s*('[^']*'|[^,\s!]+)", _fortran_declarations())
    assert match, f"{name} is not declared with a default in {FORTRAN.name}"
    raw = match.group(1)
    if raw.startswith("'"):
        return raw.strip("'")
    if raw in (".true.", ".false."):
        return raw == ".true."
    return float(raw.replace("d", "e")) if re.search(r"[.d]", raw) else int(raw)


def test_every_setting_is_in_the_fortran_namelist():
    text = FORTRAN.read_text(encoding="utf-8")
    block = text[text.index("namelist /ptrace/"):text.index("!> Two times closer")]
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", block)) - {"namelist", "ptrace"}
    assert set(PTRACE_SETTINGS) <= names
    # the two ashen never sets from a case key
    assert names - set(PTRACE_SETTINGS) == {"restart_index", "pdf_n_phi"}


@pytest.mark.parametrize("name", sorted(PTRACE_DEFAULTS))
def test_ashen_and_fortran_defaults_agree(name):
    """The file ashen writes always carries every setting, so the binary's
    own defaults never apply under ashen -- but they are documented as the
    same, and a hand-run binary uses them."""
    ours = PTRACE_DEFAULTS[name]
    assert _fortran_default(name) == (ours[0] if isinstance(ours, list) else ours)


def test_settings_without_a_default_have_none_usable_in_fortran():
    assert not {"n_markers", "E_kin_eV", "cos_pitch", "R0", "Z0"} & set(PTRACE_DEFAULTS)
    assert _fortran_default("n_markers") == 0 and _fortran_default("E_kin_eV") == 0.0
    # outside -1..1: ptrace_gc refuses a cos_pitch left unset
    assert abs(_fortran_default("cos_pitch")) > 1


@pytest.mark.parametrize("missing", ["n_markers", "E_kin_eV", "cos_pitch"])
def test_required_settings(missing):
    settings = {k: v for k, v in PDF.items() if k != missing}
    with pytest.raises(PtraceSettingsError, match=f"ptrace_{missing} is not set, and has no default"):
        resolve_settings(settings)


# --- snapshots: about n_snapshots of them, unless a spacing is set ------------------


def test_snapshots_are_spaced_by_the_tracer_by_default():
    """snapshot_step 0 with n_snapshots > 0: ptrace_gc spaces them over
    whatever time it traces, which only it knows."""
    resolved = resolve_settings(PDF)
    assert (resolved["snapshot_step"], resolved["n_snapshots"]) == (0.0, 100)
    assert resolve_settings({**PDF, "n_snapshots": 40})["n_snapshots"] == 40


@pytest.mark.parametrize("step", [1e-6, 0.0])
def test_a_snapshot_step_the_case_sets_is_used_instead(step):
    """Written as n_snapshots = 0, so the file means the same to ptrace_gc
    (whose own n_snapshots default would otherwise space them); a step of 0
    is then only the final snapshot."""
    resolved = resolve_settings({**PDF, "snapshot_step": step, "n_snapshots": 40})
    assert (resolved["snapshot_step"], resolved["n_snapshots"]) == (step, 0)
    assert "  n_snapshots = 0\n" in settings_namelist(resolved)


def test_negative_n_snapshots():
    with pytest.raises(PtraceSettingsError, match="ptrace_n_snapshots must be >= 0"):
        resolve_settings({**PDF, "n_snapshots": -1})


def test_describe_says_which_snapshot_setting_is_in_force():
    def lines(given):
        return {l.split("=")[0].strip(): l.split("=", 1)[1].strip()
                for l in describe_settings(resolve_settings(given), given)}

    by_default = lines(PDF)
    assert by_default["snapshot_step"] == "from n_snapshots   (default)"
    assert by_default["n_snapshots"] == "about 100, at round times   (default)"
    assert lines({**PDF, "n_snapshots": 40})["n_snapshots"] == "about 40, at round times"
    stepped = lines({**PDF, "snapshot_step": 1e-6})
    assert stepped["snapshot_step"] == "1e-06"
    assert stepped["n_snapshots"] == "not used   (snapshot_step is set)"
    assert lines({**PDF, "n_snapshots": 0})["snapshot_step"] == "0.0: only the final snapshot   (default)"


def _tidy_step(raw: float) -> float:
    """ptrace_gc's tidy_step, transcribed -- kept in step with the Fortran
    by test_tidy_step_is_the_fortran_one."""
    import math

    slack = 1.0 + 1e-9
    base = 10.0 ** math.floor(math.log10(raw))
    mantissa = raw / base
    for factor in (1.0, 2.0, 5.0):
        if mantissa <= factor * slack:
            return factor * base
    return 10.0 * base


def test_tidy_step_is_the_fortran_one():
    text = FORTRAN.read_text(encoding="utf-8")
    body = text[text.index("pure function tidy_step"):text.index("end function tidy_step")]
    for line in ("base     = 10.d0**floor(log10(raw))", "mantissa = raw / base",
                 "if (mantissa .le. 1.d0 * SLACK) then", "else if (mantissa .le. 2.d0 * SLACK) then",
                 "else if (mantissa .le. 5.d0 * SLACK) then", "step = 10.d0 * base"):
        assert line in body
    assert "SLACK = 1.d0 + 1.d-9" in body


@pytest.mark.parametrize("traced, n, step, count", [
    (123e-6, 100, 2e-6, 62),     # 1.23 us -> 2 us
    (100e-6, 100, 1e-6, 101),    # already round: every one of them
    (1e-3, 100, 1e-5, 101),
    (20e-6, 100, 2e-7, 101),
    (4.9e-5, 100, 5e-7, 99),
    (5.1e-5, 100, 1e-6, 52),     # just past 5: up to the next decade
    (123e-6, 10, 2e-5, 7),
])
def test_tidy_spacing_gives_half_to_all_of_n_snapshots(traced, n, step, count):
    """The rounding only goes up, by less than 2.5x: between 0.4 n and n
    snapshots (plus the one at the start), at round times."""
    got = _tidy_step(traced / n)
    assert got == pytest.approx(step, rel=1e-12)
    assert int(traced / got * (1 + 1e-9)) + 1 == count
    assert 0.4 * n <= count - 1 <= n


def test_max_markers_matches_fortran():
    match = re.search(r"MAX_MARKERS\s*=\s*(\d+)", FORTRAN.read_text(encoding="utf-8"))
    assert int(match.group(1)) == MAX_MARKERS


# --- resolve_settings ---------------------------------------------------------------


def test_resolved_settings_are_complete_and_in_order():
    resolved = resolve_settings(PDF)
    assert list(resolved) == [name for name in PTRACE_SETTINGS if name in resolved]
    assert resolved["dt"] == 1e-10 and resolved["diag_step"] == 1e-8
    assert resolved["snapshot_step"] == 0.0 and resolved["field_mode"] == "evolving"
    assert (resolved["E_kin_eV"], resolved["cos_pitch"], resolved["charge"]) == ([1e7], [0.9], [-1])
    # every setting but the positions the pdf initialiser chooses itself
    assert set(PTRACE_SETTINGS) - set(resolved) == {"R0", "Z0", "phi0"}


def test_a_case_key_wins_over_the_default():
    assert resolve_settings({**PDF, "dt": 5e-11})["dt"] == 5e-11
    assert resolve_settings({**PDF, "seed": 7})["seed"] == 7


def test_markers_single_values_stand_for_every_marker():
    """ptrace_gc would leave markers 2.. at E_kin_eV(k) = 0 otherwise."""
    resolved = resolve_settings(MARKERS)
    assert resolved["R0"] == [1.5, 1.6, 1.7]
    assert resolved["Z0"] == [0.0] * 3 and resolved["phi0"] == [0.0] * 3
    assert resolved["E_kin_eV"] == [1e7] * 3 and resolved["charge"] == [-1] * 3


def test_resolving_does_not_touch_the_case_settings():
    before = {k: list(v) if isinstance(v, list) else v for k, v in MARKERS.items()}
    resolve_settings(MARKERS)
    assert MARKERS == before


@pytest.mark.parametrize("change, message", [
    ({"E_kin_eV": [1e7, 2e7]}, "ptrace_E_kin_eV has 2 values for ptrace_n_markers = 3"),
    ({"E_kin_eV": [0.0]}, "ptrace_E_kin_eV must be > 0"),
    ({"cos_pitch": [1.5]}, "ptrace_cos_pitch must be within -1..1"),
    ({"dt": 0.0}, "ptrace_dt must be > 0"),
    ({"snapshot_step": -1.0}, "ptrace_snapshot_step must be >= 0"),
    ({"pdf_n_sub": 0}, "ptrace_pdf_n_sub must be >= 1"),
    ({"n_markers": MAX_MARKERS + 1}, "ptrace_n_markers must be in"),
])
def test_invalid_settings(change, message):
    with pytest.raises(PtraceSettingsError, match=message):
        resolve_settings({**MARKERS, **change})


# --- the file, and what is printed ---------------------------------------------------


def test_namelist_carries_every_setting():
    text = settings_namelist(resolve_settings(PDF), header=("case x: steps 1..2",))
    lines = text.splitlines()
    assert lines[0] == "! case x: steps 1..2" and lines[-1] == "/"
    body = lines[lines.index("&ptrace") + 1:-1]
    assert body[0] == "  restart_index = 0"
    for line in ("  field_mode = 'evolving'", "  hold_last_field = .false.", "  t_span = 0.0",
                 "  dt = 1d-10", "  diag_step = 1d-08", "  snapshot_step = 0.0",
                 "  n_snapshots = 100",
                 "  initialiser = 'current_pdf_simple'", "  n_markers = 1000",
                 "  E_kin_eV = 10000000.0", "  cos_pitch = 0.9", "  charge = -1",
                 "  pdf_n_sub = 4", "  seed = 1"):
        assert line in body
    assert not any(line.lstrip().startswith(("R0", "Z0", "phi0")) for line in body)


def test_namelist_per_marker_lists():
    settings = {"n_markers": 14, "R0": [1.0 + 0.1 * k for k in range(14)], "Z0": [0.0],
                "E_kin_eV": [1e7], "cos_pitch": [0.9]}
    text = settings_namelist(resolve_settings(settings))
    # one value for all: a repeat count, as a Fortran namelist reads it
    assert "  E_kin_eV = 14*10000000.0\n" in text and "  charge = 14*-1\n" in text
    # a real list: wrapped, every value there
    block = text[text.index("  R0 = "):text.index("  Z0 = ")]
    assert len(block.splitlines()) == 3
    assert len(re.findall(r"\d+\.\d+", block)) == 14


def test_describe_marks_the_defaults():
    lines = describe_settings(resolve_settings(PDF), PDF)
    by_name = {line.split("=")[0].strip(): line for line in lines}
    assert by_name["n_markers"].endswith("= 1000")
    assert by_name["E_kin_eV"].endswith("= 10 MeV")
    assert by_name["dt"].endswith("= 1e-10   (default)")
    assert by_name["hold_last_field"].endswith("= false   (default)")
    assert len({line.index("=") for line in lines}) == 1


def test_describe_keeps_long_lists_short():
    settings = {"n_markers": 50, "R0": [1.0 + 0.01 * k for k in range(50)], "Z0": [0.0],
                "E_kin_eV": [1e7], "cos_pitch": [0.9]}
    by_name = {line.split("=")[0].strip(): line
               for line in describe_settings(resolve_settings(settings), settings)}
    assert by_name["R0"].endswith("... (50 values)")
    assert by_name["Z0"].endswith("= 50 x 0.0") and by_name["E_kin_eV"].endswith("= 50 x 10 MeV")


@pytest.mark.parametrize("energy, text", [
    (1e7, "10 MeV"), (2.5e6, "2.5 MeV"), (1e6, "1 MeV"), (5e5, "500 keV"), (1e3, "1 keV"),
    (999.0, "999 eV"), (30.0, "30 eV"), (1.2e9, "1.2 GeV"), (2e7, "20 MeV"), (1.234567e7, "12.3457 MeV"),
])
def test_energy_text(energy, text):
    assert energy_text(energy) == text


def test_energies_are_printed_with_their_unit_and_written_in_eV():
    given = {**MARKERS, "E_kin_eV": [5e5, 1e7, 2.5e6]}
    resolved = resolve_settings(given)
    line = next(l for l in describe_settings(resolved, given) if l.startswith("E_kin_eV"))
    assert line.endswith("= 500 keV, 10 MeV, 2.5 MeV")
    # the file keeps plain eV: it is what ptrace_gc reads
    assert "  E_kin_eV = 500000.0, 10000000.0, 2500000.0\n" in settings_namelist(resolved)


# --- stopping a trace early ---------------------------------------------------------


def test_stall_rule_is_off_and_hidden_by_default():
    resolved = resolve_settings(PDF)
    assert resolved["stop_when_stalled"] is False
    assert (resolved["stall_rate_fraction"], resolved["stall_min_lost"],
            resolved["stall_min_time"], resolved["stall_window"]) == (0.05, 0.10, 0.5, 0.0)
    shown = "\n".join(describe_settings(resolved, PDF))
    assert "stop_when_stalled" in shown and "stall_" not in shown.replace("stop_when_stalled", "")
    # the file still records every one of them
    text = settings_namelist(resolved)
    for line in ("  stop_when_stalled = .false.", "  stall_rate_fraction = 0.05",
                 "  stall_min_lost = 0.1", "  stall_min_time = 0.5", "  stall_window = 0.0"):
        assert line in text.splitlines()


def test_stall_rule_on_is_printed_with_its_settings():
    given = {**PDF, "stop_when_stalled": True, "stall_rate_fraction": 0.1}
    lines = {l.split("=")[0].strip(): l.split("=", 1)[1].strip()
             for l in describe_settings(resolve_settings(given), given)}
    assert lines["stop_when_stalled"] == "true"
    assert lines["stall_rate_fraction"] == "0.1"
    assert lines["stall_min_lost"] == "0.1   (default)"
    assert lines["stall_min_time"] == "0.5   (default)"
    assert lines["stall_window"] == "a tenth of the trace   (default)"
    given["stall_window"] = 2e-5
    assert "2e-05" in "\n".join(describe_settings(resolve_settings(given), given))


@pytest.mark.parametrize("change, message", [
    ({"stall_rate_fraction": 0.0}, "ptrace_stall_rate_fraction must be between 0 and 1"),
    ({"stall_rate_fraction": 1.0}, "ptrace_stall_rate_fraction must be between 0 and 1"),
    ({"stall_min_lost": 1.5}, "ptrace_stall_min_lost is a fraction, 0 to 1"),
    ({"stall_min_time": -0.1}, "ptrace_stall_min_time is a fraction, 0 to 1"),
    ({"stall_window": -1e-6}, "ptrace_stall_window must be >= 0"),
])
def test_invalid_stall_settings(change, message):
    with pytest.raises(PtraceSettingsError, match=message):
        resolve_settings({**PDF, **change})


def test_boundary_file_is_what_the_fortran_reads():
    """A line with n, then n lines of R Z -- read_boundary's list-directed reads."""
    from ashen.particle_programs import BOUNDARY_FILE, boundary_file_text

    text = boundary_file_text([(3.5, -0.25), (4.0, -0.25), (4.0, 0.25)])
    assert text == "3\n3.5 -0.25\n4.0 -0.25\n4.0 0.25\n"
    fortran = FORTRAN.read_text(encoding="utf-8")
    assert f"BOUNDARY_FILE = '{BOUNDARY_FILE}'" in fortran
    body = fortran[fortran.index("subroutine read_boundary"):fortran.index("end subroutine read_boundary")]
    assert "read(unit, *, iostat=io) n" in body and "read(unit, *, iostat=io) R(i), Z(i)" in body


def test_fortran_point_in_polygon_is_the_plots_one():
    """Both sides must agree on who has left: the same even-odd ray cast."""
    fortran = FORTRAN.read_text(encoding="utf-8")
    body = fortran[fortran.index("logical function inside_boundary"):
                   fortran.index("end function inside_boundary")]
    assert "if ((bnd_Z(i) .gt. Z) .neqv. (bnd_Z(j) .gt. Z)) then" in body
    assert ("if (R .lt. bnd_R(i) + (Z - bnd_Z(i)) * (bnd_R(j) - bnd_R(i)) / "
            "(bnd_Z(j) - bnd_Z(i)))") in body


def _stall_stop(times, n_left, n_markers, *, rate_fraction=0.05, min_lost=0.10, min_time=0.5,
                window=None, samples_per_window=4):
    """ptrace_gc's check_stop stall rule, transcribed: the time it stops the
    trace at, or None. times are where the push loop stops (every
    diag_step); n_left the markers gone by each. Kept in step with the
    Fortran by test_stall_rule_is_the_fortran_one."""
    tick = 1e-12
    t_start, t_stop = times[0], times[-1]
    window = window or (t_stop - t_start) / 10
    sample_t, sample_n = [], []
    t_check, max_rate = t_start, 0.0
    for t, n in zip(times, n_left):
        if t < t_check - tick:
            continue
        sample_t.append(t)
        sample_n.append(n)
        t_check = t + window / samples_per_window
        back = next((i for i in range(len(sample_t) - 2, -1, -1)
                     if sample_t[i] <= t - window + tick), None)
        if back is None:
            continue
        rate = (sample_n[-1] - sample_n[back]) / (sample_t[-1] - sample_t[back])
        max_rate = max(max_rate, rate)
        if max_rate <= 0:
            continue
        if n < min_lost * n_markers and t - t_start < min_time * (t_stop - t_start):
            continue
        if rate > rate_fraction * max_rate:
            continue
        return t
    return None


def test_stall_rule_is_the_fortran_one():
    fortran = FORTRAN.read_text(encoding="utf-8")
    body = fortran[fortran.index("subroutine check_stop"):fortran.index("end subroutine check_stop")]
    for line in (
        "if (sim%time .lt. t_check - SNAP_TICK) return",
        "t_check = sim%time + stall_window / SAMPLES_PER_WINDOW",
        "if (sample_t(i) .le. sim%time - stall_window + SNAP_TICK) then",
        "loss_rate = (sample_n(n_samples) - sample_n(j)) / (sample_t(n_samples) - sample_t(j))",
        "max_loss_rate = max(max_loss_rate, loss_rate)",
        "if (max_loss_rate .le. 0.d0) return",
        "if (n_left .lt. stall_min_lost * n_markers .and. &",
        "sim%time - t_start .lt. stall_min_time * (t_stop - t_start)) return",
        "if (loss_rate .gt. stall_rate_fraction * max_loss_rate) return",
    ):
        assert line in body, line
    assert "SAMPLES_PER_WINDOW = 4" in fortran
    assert "if (stall_window .le. 0.d0) stall_window = (t_stop - t_start) / 10.d0" in fortran


def _loss_curve(n_markers, pieces, n_times=1201, length=120e-6):
    """Markers gone over a trace: `pieces` are (start, end, count) -- count
    markers leaving evenly between those fractions of the trace."""
    import numpy as np

    times = np.linspace(0.0, length, n_times)
    frac = times / length
    gone = np.zeros(n_times)
    for start, end, count in pieces:
        gone += count * np.clip((frac - start) / (end - start), 0.0, 1.0)
    return times, np.floor(gone + 1e-9).astype(int), n_markers


def test_stalled_after_the_losses_saturate():
    """Losses from 20 % to 50 % of the trace, then nothing: stopped soon
    after -- a window (a tenth of the trace) of no loss is what shows it."""
    times, gone, n = _loss_curve(1000, [(0.2, 0.5, 600)])
    stopped = _stall_stop(times, gone, n)
    assert 0.55 <= stopped / times[-1] <= 0.65


def test_never_stopped_while_the_losses_go_on():
    times, gone, n = _loss_curve(1000, [(0.1, 1.0, 800)])
    assert _stall_stop(times, gone, n) is None


def test_never_stopped_when_nothing_has_left():
    """No loss at all is a trace whose losses have not begun, not one whose
    losses are over -- even past the half-time gate."""
    times, gone, n = _loss_curve(1000, [])
    assert _stall_stop(times, gone, n) is None


def test_a_tail_of_slow_loss_keeps_it_going_until_under_the_fraction():
    """A fast loss, then a tail at a tenth of that rate: 10 % of the peak is
    above the 5 % mark, so it runs on; with the mark at 20 % it stops."""
    times, gone, n = _loss_curve(10000, [(0.1, 0.3, 4000), (0.3, 1.0, 1400)])
    assert _stall_stop(times, gone, n) is None
    assert _stall_stop(times, gone, n, rate_fraction=0.2) is not None


def test_few_losses_stop_only_after_half_the_trace():
    """3 % lost early, then quiet: under the 10 % gate, so not looked at
    until half the trace has passed -- and stopped then."""
    times, gone, n = _loss_curve(1000, [(0.05, 0.15, 30)])
    stopped = _stall_stop(times, gone, n)
    assert 0.5 <= stopped / times[-1] <= 0.55


def test_the_half_time_gate_can_cut_off_a_late_main_loss():
    """The known hazard: a little early loss, a quiet gap, and the main loss
    after half-time. The rule cannot see the field; at half-time it finds
    the losses stalled and stops, missing the main loss. Raising
    stall_min_time past where the main loss starts keeps the trace going
    through it (here, to the end)."""
    times, gone, n = _loss_curve(1000, [(0.05, 0.15, 30), (0.7, 0.9, 600)])
    assert 0.5 <= _stall_stop(times, gone, n) / times[-1] <= 0.55
    assert _stall_stop(times, gone, n, min_time=1.0) / times[-1] >= 0.95


def test_the_ten_percent_gate_opens_it_before_half_time():
    """Losses done by 30 % of the trace with far more than a tenth gone:
    stopped well before half-time."""
    times, gone, n = _loss_curve(1000, [(0.1, 0.3, 500)])
    assert 0.35 <= _stall_stop(times, gone, n) / times[-1] <= 0.45
