"""Visualization engine public API."""

from nutella_scraper.engines.visualization.contact_metrics_panel import ContactMetricsPanel
from nutella_scraper.engines.visualization.contact_result_projector import (
    LAYER_COLLISION_FACES,
    LAYER_COLLISION_POINTS,
    LAYER_CONTACT_COVERED,
    LAYER_CONTACT_POINTS,
    LAYER_CONTACT_UNCOVERED,
    LAYER_DISTANCE_MAP,
    ContactResultProjector,
)
from nutella_scraper.engines.visualization.engine import VisualizationEngine
from nutella_scraper.engines.visualization.overlay_renderer import OverlayRenderer
from nutella_scraper.engines.visualization.view_projection_generator import ViewProjectionGenerator

__all__ = [
    "ContactMetricsPanel",
    "ContactResultProjector",
    "LAYER_COLLISION_FACES",
    "LAYER_COLLISION_POINTS",
    "LAYER_CONTACT_COVERED",
    "LAYER_CONTACT_POINTS",
    "LAYER_CONTACT_UNCOVERED",
    "LAYER_DISTANCE_MAP",
    "OverlayRenderer",
    "ViewProjectionGenerator",
    "VisualizationEngine",
]
