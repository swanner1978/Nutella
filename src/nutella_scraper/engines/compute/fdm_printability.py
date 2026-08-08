"""FDM printability constraint checker."""

from __future__ import annotations

from dataclasses import dataclass

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.contact import Violation


@dataclass(frozen=True)
class FDMConfig:
    min_wall_thickness_mm: float = 1.2
    max_overhang_angle_deg: float = 45.0
    clearance_compensation_mm: float = 0.15


class FDMPrintabilityChecker:
    """Checks FDM manufacturing constraints on scraper 3D model."""

    def __init__(self, config: FDMConfig | None = None) -> None:
        self._config = config or FDMConfig()

    def check(self, scraper: CanonicalModel3D) -> tuple[bool, tuple[Violation, ...]]:
        raise NotImplementedError("FDMPrintabilityChecker.check not implemented")
