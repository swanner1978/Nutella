"""Protocols for future manual pose editing — no UI implementation yet."""

from __future__ import annotations

from typing import Protocol

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.envelope import PoseValidationResult
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose


class IManualPoseValidator(Protocol):
    """Validate a scraper pose against jar constraints before simulation."""

    def validate(
        self,
        jar: CanonicalModel3D,
        geometry: ScraperGeometry,
        pose: ScraperPose,
        config: ContactSimulationConfig,
    ) -> PoseValidationResult:
        """Check whether a manually chosen pose is physically plausible."""
        ...


class IManualPoseEditor(Protocol):
    """
    Future manual mode entry point.

    Callers update ScraperPose only; geometry stays parametric via ScraperBuilder.
    """

    def set_pose(self, pose: ScraperPose) -> PoseValidationResult:
        """Apply a new pose and return whether it remains valid."""
        ...
