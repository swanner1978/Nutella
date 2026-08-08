"""Generates profile (XY) and top (XZ) views from CanonicalModel3D — visualization only."""

from __future__ import annotations

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.views import ViewProjectionCache


class ViewProjectionGenerator:
    """
    Generates 2D views for user display.

    Must never compute contact metrics or coverage scores.
    """

    def generate(self, model: CanonicalModel3D) -> ViewProjectionCache:
        raise NotImplementedError("ViewProjectionGenerator.generate not implemented")
