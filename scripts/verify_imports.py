"""Verify module imports when dependencies are installed.

Run after: pip install -e ".[dev]"
"""

from __future__ import annotations

import sys


def main() -> int:
    modules = [
        "nutella_scraper",
        "nutella_scraper.domain.models",
        "nutella_scraper.domain.protocols",
        "nutella_scraper.cad_import",
        "nutella_scraper.engines.compute",
        "nutella_scraper.engines.visualization",
        "nutella_scraper.engines.optimization",
        "nutella_scraper.application.container",
        "nutella_scraper.io.config_loader",
        "nutella_scraper.api.app",
        "nutella_scraper.cli.main",
    ]
    optional = [
        ("OCP", "cadquery-ocp (STEP B-Rep) — pip install -e \".[dev]\""),
    ]
    failed: list[str] = []
    for name in modules:
        try:
            __import__(name)
            print(f"OK  {name}")
        except Exception as exc:
            print(f"FAIL {name}: {exc}")
            failed.append(name)
    for name, hint in optional:
        try:
            __import__(name)
            print(f"OK  {name}")
        except Exception as exc:
            print(f"WARN {name}: {exc} ({hint})")
            failed.append(name)
    if failed:
        return 1
    print(f"\nAll {len(modules)} modules imported successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
