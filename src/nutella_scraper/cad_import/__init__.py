"""CAD import pipeline public API.

See docs/cad_import_api.md for interface documentation.
"""

from nutella_scraper.cad_import.exceptions import (
    CadImportError,
    InvalidGeometryError,
    StepReadError,
    UnsupportedFormatError,
)
from nutella_scraper.cad_import.geometry_normalizer import GeometryNormalizer
from nutella_scraper.cad_import.geometry_validator import GeometryValidator, ValidationConfig
from nutella_scraper.cad_import.mesh_loader import IMeshLoader
from nutella_scraper.cad_import.model_store import CadReferenceNotAvailableError, ModelStore
from nutella_scraper.cad_import.pipeline import ImportPipeline
from nutella_scraper.cad_import.trimesh_loader import TrimeshLoader
from nutella_scraper.cad_import.view_cache_store import ViewCacheStore
from nutella_scraper.cad_import.view_projection_generator import (
    ViewProjectionConfig,
    ViewProjectionGenerator,
)

__all__ = [
    "CadImportError",
    "CadReferenceNotAvailableError",
    "GeometryNormalizer",
    "GeometryValidator",
    "IMeshLoader",
    "ImportPipeline",
    "InvalidGeometryError",
    "ModelStore",
    "StepReadError",
    "TrimeshLoader",
    "UnsupportedFormatError",
    "ValidationConfig",
    "ViewCacheStore",
    "ViewProjectionConfig",
    "ViewProjectionGenerator",
]
