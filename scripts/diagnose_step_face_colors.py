#!/usr/bin/env python3
"""CLI diagnostic: read STEP face colours via XCAF (does not alter the app pipeline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from nutella_scraper.cad_import.step_face_color_diagnostics import (  # noqa: E402
    TARGET_RGB_255,
    diagnose_step_face_colors,
    format_step_face_color_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose whether a STEP file exposes face colours RGB(85,255,255).",
    )
    parser.add_argument(
        "step_path",
        type=Path,
        help="Path to the FreeCAD-coloured STEP file",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help="Absolute RGB channel tolerance (default: 2)",
    )
    args = parser.parse_args(argv)

    diagnostic = diagnose_step_face_colors(
        args.step_path,
        target_rgb_255=TARGET_RGB_255,
        tolerance_255=args.tolerance,
    )
    print(format_step_face_color_report(diagnostic))
    return 0 if diagnostic.color_information_available else 2


if __name__ == "__main__":
    raise SystemExit(main())
