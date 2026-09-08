"""Lazy h5py import, shared by the HDF5-backed diagnostic caches.

Kept out of module import so the rest of ashen stays importable on a machine
without h5py -- CLAUDE.md's rule that importability must not depend on
``pip install``. Each cache module raises its own error type, so the caller's
``except PoincareCacheError`` / ``except FourCacheError`` still works; only
the import and the message are shared.
"""

from __future__ import annotations

from typing import Type

__all__ = ["require_h5py"]


def require_h5py(what: str, error: Type[Exception]):
    """Return the ``h5py`` module, or raise ``error`` explaining how to get it.

    ``what`` names the caller's cache in the message, e.g. "the Poincare cache".
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise error(
            f"{what} needs h5py, which is not importable here. It is present "
            "in the HPC environment (JOREK's own util scripts use it); on a "
            "dev clone, install it or use an environment that has it."
        ) from exc
    return h5py
