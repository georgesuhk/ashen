"""Populating a run folder: copying and symlinking template files.

Ports ``castor3d/util/io.py:23-138`` (``copy_all_files``, ``symlink_folder``,
``symlink_folder_files``, ``symlink_file``).

**One asymmetry preserved from the original, not fixed:** :func:`symlink_dir`
creates an **absolute** symlink target; :func:`symlink_file` creates a
**relative** one (via `os.path.relpath`). Nothing in the refactor plan flags
this as a bug to fix, so it is kept exactly as-is rather than invented as a
new behaviour change.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["copy_all_files", "symlink_dir", "symlink_files_in", "symlink_file"]


def copy_all_files(src_dir: Path | str, dst_dir: Path | str) -> None:
    """Copy every file (not subdirectory) from ``src_dir`` into ``dst_dir``."""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {src_dir}")

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for entry in src_dir.iterdir():
        if entry.is_file():
            shutil.copy2(entry, dst_dir / entry.name)


def symlink_dir(src_dir: Path | str, dst_dir: Path | str, link_name: str | None = None) -> Path:
    """Symlink ``src_dir`` into ``dst_dir`` (absolute target)."""
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    link_path = dst_dir / (link_name or src_dir.name)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(src_dir, target_is_directory=True)
    return link_path


def symlink_files_in(src_dir: Path | str, dst_dir: Path | str) -> list[Path]:
    """Symlink every file in ``src_dir`` into ``dst_dir``, same names.

    Failures on individual files are collected and raised together at the end
    rather than only printed, unlike the old ``symlink_folder_files`` which
    swallowed exceptions with a bare ``print``.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    errors: list[str] = []
    for entry in src_dir.iterdir():
        if not entry.is_file():
            continue
        target = dst_dir / entry.name
        try:
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(entry.resolve())
            created.append(target)
        except OSError as exc:
            errors.append(f"{entry} -> {target}: {exc}")

    if errors:
        raise OSError("failed to symlink:\n  " + "\n  ".join(errors))
    return created


def symlink_file(src_file: Path | str, dst_file: Path | str) -> Path:
    """Symlink a single file (relative target)."""
    src_file = Path(src_file)
    dst_file = Path(dst_file)

    if not src_file.is_file():
        raise FileNotFoundError(f"Source file does not exist: {src_file}")

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if dst_file.exists() or dst_file.is_symlink():
        dst_file.unlink()
    dst_file.symlink_to(os.path.relpath(src_file, dst_file.parent))
    return dst_file
