"""Residual Nutella volume estimation."""

from __future__ import annotations

from nutella_scraper.domain.models.canonical import CanonicalModel3D, JarCanonicalModel


class ResidualVolumeEstimator:
    """Estimates unreachable Nutella volume from 3D geometry."""

    def estimate(
        self,
        scraper: CanonicalModel3D,
        jar: JarCanonicalModel,
    ) -> float:
        """Return residual volume in millilitres."""
        raise NotImplementedError("ResidualVolumeEstimator.estimate not implemented")
