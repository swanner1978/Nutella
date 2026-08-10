"""View projection models — visualization only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nutella_scraper.domain.models.common import Provenance


@dataclass(frozen=True)
class ProjectionMetadata:
    """Camera and scale metadata for a 2D projection."""

    plane: str
    camera: dict[str, float]
    scale: float
    width_px: int
    height_px: int


@dataclass(frozen=True)
class ProjectedView:
    """Single 2D projected view (side/profile or top)."""

    plane: str
    asset_path: Path | None
    svg_content: str | None
    metadata: ProjectionMetadata


@dataclass(frozen=True)
class ViewProjectionCache:
    """
    Cached 2D views for user visualization only.

    @visualization_only — must never feed ContactSimulator or OptimizationEngine.
    """

    model_id: str
    profile_view: ProjectedView
    top_view: ProjectedView
    projection_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = "visualization_projection"


@dataclass(frozen=True)
class SvgLayer:
    """Single SVG layer for overlay rendering."""

    id: str
    z_index: int
    svg_fragment: str
    layer_type: str


@dataclass(frozen=True)
class ViewOverlayPayload:
    """
    Composed overlay for UI display.

    coverage_score_display is a read-only copy of ContactResult.coverage_score.
    """

    model_id: str
    profile_layers: tuple[SvgLayer, ...]
    top_layers: tuple[SvgLayer, ...]
    coverage_score_display: float
    left_layers: tuple[SvgLayer, ...] = ()
    right_layers: tuple[SvgLayer, ...] = ()
    provenance: Provenance = "visualization_projection"


@dataclass(frozen=True)
class RenderedFrame:
    """Fully rendered frame for UI consumption."""

    profile_svg: str
    top_svg: str
    coverage_score_display: float
