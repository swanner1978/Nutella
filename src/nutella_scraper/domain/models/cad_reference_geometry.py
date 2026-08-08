"""CAD B-Rep reference geometry — source of truth for 2D reference views."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CadBoundingBox:
    min_x_mm: float
    min_y_mm: float
    min_z_mm: float
    max_x_mm: float
    max_y_mm: float
    max_z_mm: float


@dataclass(frozen=True)
class ProjectedPolyline2D:
    """One connected 2D polyline in a reference projection plane."""

    points_mm: tuple[tuple[float, float], ...]
    is_closed: bool = False


@dataclass(frozen=True)
class CadProjectedContour:
    """Exact projected inner-cavity contour from B-Rep HLR — no mesh."""

    plane: str
    view_axis: str
    polylines: tuple[ProjectedPolyline2D, ...]
    edge_count: int
    source: str = "opencascade_hlr"


@dataclass(frozen=True)
class CadReferenceGeometry:
    """
    Reference geometry extracted from STEP B-Rep (OpenCascade).

    Used exclusively for profile/top views, interior contour and envelope display.
    Computational contact/collision continues to use CanonicalModel3D mesh.
    """

    model_id: str
    step_path: str
    step_sha256: str
    bounding_box: CadBoundingBox
    inner_face_count: int
    outer_face_count: int
    profile_contour: CadProjectedContour | None = None
    top_contour: CadProjectedContour | None = None
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)
