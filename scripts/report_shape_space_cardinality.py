#!/usr/bin/env python3
"""Print the discrete shape-space cardinality report. No CoverageSimulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.unit.engines.compute.coverage_catalog_fixtures import (  # noqa: E402
    load_generated_catalog,
)

from nutella_scraper.engines.visualization.scraper_control_cage import (  # noqa: E402
    build_control_cage_overlay,
)
from nutella_scraper.engines.visualization.scraper_shape_space import (  # noqa: E402
    lattice_from_cage,
)
from nutella_scraper.engines.visualization.shape_space_cardinality import (  # noqa: E402
    format_shape_space_report,
    shape_space_cardinality_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "output" / "coverage" / "shape_space_cardinality.txt",
    )
    args = parser.parse_args()
    surface, _params, reference, catalog = load_generated_catalog(count=1000)
    lattice = lattice_from_cage(
        build_control_cage_overlay(reference.design_path, surface),
        surface,
    )
    report = shape_space_cardinality_report(
        lattice,
        generated_count=1000,
        evaluated_count=100,
        catalog=catalog,
    )
    text = format_shape_space_report(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    print(text)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
