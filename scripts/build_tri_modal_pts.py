#!/usr/bin/env python3
"""Deprecated. The staged graph-then-manifest builder has been removed.

Use ``scripts/build_tri_modal_pts_direct.py`` with ``config/extract_tri_model.yaml``
to build API+Graph+Manifest tri-modal ``.pt`` files in a single pass.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/build_tri_modal_pts.py is deprecated. "
        "Use scripts/build_tri_modal_pts_direct.py with config/extract_tri_model.yaml."
    )


if __name__ == "__main__":
    main()
