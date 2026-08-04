"""Tests for run-folder population: copying and symlinking.

Symlink creation needs elevated privileges on Windows without Developer Mode
enabled. Rather than skip the whole module, each symlink-dependent test probes
capability first and skips itself if unsupported -- copy_all_files tests
always run everywhere. The ``require_symlinks`` fixture lives in conftest.py
so test_runner.py can share it.
"""

from __future__ import annotations

import pytest

from ashen.fs import copy_all_files, symlink_dir, symlink_file, symlink_files_in

# --- copy_all_files (no symlinks involved, runs everywhere) --------------------


def test_copies_only_files_not_subdirectories(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")
    (src / "b.txt").write_text("b")
    (src / "subdir").mkdir()
    (src / "subdir" / "c.txt").write_text("c")

    dst = tmp_path / "dst"
    copy_all_files(src, dst)

    assert (dst / "a.txt").read_text() == "a"
    assert (dst / "b.txt").read_text() == "b"
    assert not (dst / "subdir").exists()


def test_creates_destination_directory(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")

    dst = tmp_path / "does" / "not" / "exist" / "yet"
    copy_all_files(src, dst)

    assert (dst / "a.txt").exists()


def test_missing_source_raises(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        copy_all_files(tmp_path / "nope", tmp_path / "dst")


def test_copy_is_a_real_copy_not_a_link(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("original")
    dst = tmp_path / "dst"

    copy_all_files(src, dst)
    (dst / "a.txt").write_text("modified")

    assert (src / "a.txt").read_text() == "original"


# --- symlink_dir -----------------------------------------------------------------


def test_symlink_dir_points_at_source(tmp_path, require_symlinks):
    target = tmp_path / "target"
    target.mkdir()
    (target / "inside.txt").write_text("hi")

    link = symlink_dir(target, tmp_path / "dst", link_name="exe")

    assert link.is_symlink()
    assert (link / "inside.txt").read_text() == "hi"


def test_symlink_dir_defaults_link_name_to_source_name(tmp_path, require_symlinks):
    target = tmp_path / "myexe"
    target.mkdir()

    link = symlink_dir(target, tmp_path / "dst")

    assert link.name == "myexe"


def test_symlink_dir_replaces_an_existing_link(tmp_path, require_symlinks):
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    (second / "marker.txt").write_text("second")

    symlink_dir(first, tmp_path / "dst", link_name="exe")
    link = symlink_dir(second, tmp_path / "dst", link_name="exe")

    assert (link / "marker.txt").read_text() == "second"


# --- symlink_files_in --------------------------------------------------------------


def test_symlink_files_in_links_every_file(tmp_path, require_symlinks):
    src = tmp_path / "src"
    src.mkdir()
    (src / "submit_jorek.sh").write_text("#!/bin/sh")
    (src / "stpts").write_text("data")

    created = symlink_files_in(src, tmp_path / "dst")

    assert len(created) == 2
    assert all(p.is_symlink() for p in created)


def test_symlink_files_in_skips_subdirectories(tmp_path, require_symlinks):
    src = tmp_path / "src"
    src.mkdir()
    (src / "file.txt").write_text("x")
    (src / "subdir").mkdir()

    created = symlink_files_in(src, tmp_path / "dst")

    assert len(created) == 1
    assert created[0].name == "file.txt"


# --- symlink_file -----------------------------------------------------------------


def test_symlink_file_target_is_relative(tmp_path, require_symlinks):
    """Deliberately different from symlink_dir's absolute target -- preserved
    as-is from the original io.py, not a new inconsistency introduced here."""
    src = tmp_path / "a" / "starwall-response.dat"
    src.parent.mkdir()
    src.write_text("data")
    dst = tmp_path / "b" / "starwall-response.dat"
    dst.parent.mkdir()

    symlink_file(src, dst)

    import os
    raw_target = os.readlink(dst)
    assert not os.path.isabs(raw_target)


def test_symlink_file_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        symlink_file(tmp_path / "nope.dat", tmp_path / "dst.dat")


def test_symlink_file_reads_through_to_content(tmp_path, require_symlinks):
    src = tmp_path / "a" / "data.dat"
    src.parent.mkdir()
    src.write_text("hello")
    dst = tmp_path / "b" / "data.dat"
    dst.parent.mkdir()

    symlink_file(src, dst)

    assert dst.read_text() == "hello"
