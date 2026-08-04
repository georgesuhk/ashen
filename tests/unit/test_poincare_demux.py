"""Mapping jorek2_poincare's output blocks back onto field lines.

This is the one genuinely dangerous part of tracing a whole batch in a single
invocation. All lines share one ``poinc_R-Z.dat``, and the blocks are written
inside ``!$omp critical`` in **thread-completion order** (jorek2_poincare.f90
:450-455), so with more than one OpenMP thread the Nth block is not the Nth
line. Assigning a trace to the wrong starting position would corrupt results
without producing any error, so every check here is about refusing to guess.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashen.diagnostics.poincare import _demux_blocks, _split_blocks, _stpts
from ashen.diagnostics.poincare_cache import LineKey, LineWork
from ashen.jorek2 import Jorek2Error


def msg(line_no: int, n_points: int) -> str:
    """One '=> Line' progress message, as jorek2_poincare.f90:449 formats it."""
    return f" => Line{line_no:6d}:{n_points:6d} points"


def block(value: float, n: int) -> str:
    return "\n".join(f"  {value:18.8e}  {value + 1:18.8e}" for _ in range(n))


def output_file(tmp_path, blocks: list[str], name="poinc_R-Z.dat"):
    path = tmp_path / name
    body = " #  R                 Z\n" + "\n\n\n".join(blocks) + "\n\n\n"
    path.write_text(body, encoding="utf-8")
    return path


# --- splitting ----------------------------------------------------------------------


def test_splits_on_blank_lines_and_skips_comments(tmp_path):
    path = output_file(tmp_path, [block(1.0, 3), block(2.0, 5)])
    blocks = _split_blocks(path)
    assert [len(b) for b in blocks] == [3, 5]
    assert blocks[0].shape == (3, 2)


def test_a_single_line_still_splits(tmp_path):
    assert [len(b) for b in _split_blocks(output_file(tmp_path, [block(1.0, 7)]))] == [7]


# --- demuxing -----------------------------------------------------------------------


def test_completion_order_is_not_line_order(tmp_path):
    """The core case: three lines finishing out of order. Block position says
    2, 0, 1; only the log says which is which."""
    blocks = [block(20.0, 20), block(5.0, 5), block(11.0, 11)]
    stdout = "\n".join([msg(3, 20), msg(1, 5), msg(2, 11)])

    got = _demux_blocks(stdout, _split_blocks(output_file(tmp_path, blocks)), 3, "R-Z")

    assert len(got[1]) == 5 and got[1][0, 0] == pytest.approx(5.0)
    assert len(got[2]) == 11 and got[2][0, 0] == pytest.approx(11.0)
    assert len(got[3]) == 20 and got[3][0, 0] == pytest.approx(20.0)


def test_in_order_output_works_too(tmp_path):
    blocks = [block(1.0, 4), block(2.0, 6)]
    stdout = "\n".join([msg(1, 4), msg(2, 6)])
    got = _demux_blocks(stdout, _split_blocks(output_file(tmp_path, blocks)), 2, "R-Z")
    assert got[1][0, 0] == pytest.approx(1.0)
    assert got[2][0, 0] == pytest.approx(2.0)


def test_interleaved_progress_chatter_is_ignored(tmp_path):
    """Real stdout also carries per-turn 'Line N: turn M of K' lines
    (.f90:239-243) and startup banners."""
    stdout = "\n".join([
        " nperiod :  1",
        " Line     1 started at   1.700   0.000",
        " Line     1: turn     1 of  1000",
        msg(1, 4),
        " Line     2: turn   500 of  1000",
        msg(2, 6),
    ])
    got = _demux_blocks(
        stdout, _split_blocks(output_file(tmp_path, [block(1.0, 4), block(2.0, 6)])), 2, "R-Z"
    )
    assert set(got) == {1, 2}


# --- refusing to guess ----------------------------------------------------------------


def test_no_messages_at_all_raises(tmp_path):
    with pytest.raises(Jorek2Error, match="cannot be matched"):
        _demux_blocks("", _split_blocks(output_file(tmp_path, [block(1.0, 3)])), 1, "R-Z")


def test_fewer_messages_than_blocks_raises(tmp_path):
    blocks = _split_blocks(output_file(tmp_path, [block(1.0, 3), block(2.0, 4)]))
    with pytest.raises(Jorek2Error, match="reported 1 traced lines but wrote 2"):
        _demux_blocks(msg(1, 3), blocks, 2, "R-Z")


def test_more_messages_than_blocks_raises(tmp_path):
    blocks = _split_blocks(output_file(tmp_path, [block(1.0, 3)]))
    stdout = "\n".join([msg(1, 3), msg(2, 4)])
    with pytest.raises(Jorek2Error, match="reported 2 traced lines but wrote 1"):
        _demux_blocks(stdout, blocks, 2, "R-Z")


def test_point_count_disagreement_raises(tmp_path):
    """The strongest cross-check: the log says 3 points, the block has 5."""
    blocks = _split_blocks(output_file(tmp_path, [block(1.0, 5)]))
    with pytest.raises(Jorek2Error, match="reported 3 points for line 1"):
        _demux_blocks(msg(1, 3), blocks, 1, "R-Z")


def test_a_line_number_out_of_range_raises(tmp_path):
    blocks = _split_blocks(output_file(tmp_path, [block(1.0, 3)]))
    with pytest.raises(Jorek2Error, match="outside 1..1"):
        _demux_blocks(msg(7, 3), blocks, 1, "R-Z")


def test_a_repeated_line_number_raises(tmp_path):
    blocks = _split_blocks(output_file(tmp_path, [block(1.0, 3), block(2.0, 3)]))
    stdout = "\n".join([msg(1, 3), msg(1, 3)])
    with pytest.raises(Jorek2Error, match="reported line 1 twice"):
        _demux_blocks(stdout, blocks, 2, "R-Z")


def test_messages_survive_i6_field_overflow(tmp_path):
    """.f90:449 uses i6 fields with no separators, so a 7-digit line count
    runs the numbers into the ':' -- the regex must still match."""
    stdout = " => Line1234567:   100 points"
    blocks = _split_blocks(output_file(tmp_path, [block(1.0, 100)]))
    got = _demux_blocks(stdout, blocks, 2_000_000, "R-Z")
    assert list(got) == [1234567]


# --- the stpts file ---------------------------------------------------------------------


def work(R, Z, n_turns, phi=0.0, resume=None) -> LineWork:
    return LineWork(
        key=LineKey(psi_n=0.5, R=R, Z=Z, phi=phi),
        n_turns=n_turns,
        resume_from=resume,
    )


def test_stpts_declares_the_whole_batch():
    """n_lines > 1 is what lets the tool's own OpenMP loop do the work -- the
    legacy code always wrote 1."""
    text = _stpts([work(1.7, 0.0, 100), work(1.8, 0.1, 100), work(1.9, 0.2, 100)])
    lines = text.splitlines()
    assert lines[1].strip() == "3"
    assert len(lines) == 6  # 2 comments + n_lines + 3 rows


def test_stpts_rows_are_complete_and_ascending():
    """A gap in nr would be silently *interpolated* by the tool
    (.f90:132-137), inventing starting points that were never requested."""
    text = _stpts([work(1.7, 0.0, 10), work(1.8, 0.0, 20), work(1.9, 0.0, 30)])
    numbers = [int(line.split()[0]) for line in text.splitlines()[3:]]
    assert numbers == [1, 2, 3]


def test_stpts_carries_per_line_turn_counts():
    """Mixed turn counts in one batch are what an incremental cache produces:
    new lines want the full count, resumed ones only the shortfall."""
    text = _stpts([work(1.7, 0.0, 1000), work(1.8, 0.0, 250, resume=(1.75, 0.03))])
    rows = text.splitlines()[3:]
    assert int(rows[0].split()[-1]) == 1000
    assert int(rows[1].split()[-1]) == 250


def test_stpts_uses_the_resume_point_when_extending():
    text = _stpts([work(1.7, 0.0, 250, resume=(1.752345678901, 0.031234567890))])
    row = text.splitlines()[3].split()
    assert float(row[1]) == pytest.approx(1.752345678901)
    assert float(row[2]) == pytest.approx(0.031234567890)


def test_stpts_does_not_quantise_start_points():
    """stpts is read list-directed (.f90:113), so the format is free. The
    legacy '{:10.6f}' threw away everything below ~1um for no reason."""
    R = 1.812345678901234
    text = _stpts([work(R, 0.0, 100)])
    written = float(text.splitlines()[3].split()[1])
    assert written != pytest.approx(round(R, 6), abs=0)
    assert written == pytest.approx(R, rel=1e-11)


def test_stpts_preserves_phi_start():
    text = _stpts([work(1.7, 0.0, 100, phi=0.7853981634)])
    assert float(text.splitlines()[3].split()[3]) == pytest.approx(0.7853981634)


def test_empty_batch_is_refused():
    with pytest.raises(ValueError, match="no field lines"):
        _stpts([])
