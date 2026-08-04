"""Ashen -- a wrapper around JOREK for preparing, running and analysing shots.

Import rules that keep this package usable from any prefix:

* no module here may call ``sys.path.append``
* no module here may contain an absolute machine path

Machine paths come from ``site.toml`` via :mod:`ashen.config`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["Site", "SiteConfigError", "find_site_file", "load_site"]

from ashen.config import Site, SiteConfigError, find_site_file, load_site
