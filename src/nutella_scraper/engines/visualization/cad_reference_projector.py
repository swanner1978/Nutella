"""Project CadReferenceGeometry B-Rep contours to SVG — no mesh contour extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.cad_import.brep_contour_extractor import PLANE_PROFILE, PLANE_TOP_XZ
from nutella_scraper.domain.models.cad_reference_geometry import (
    CadProjectedContour,
    CadReferenceGeometry,
    ProjectedPolyline2D,
)
from nutella_scraper.domain.models.views import SvgLayer
from nutella_scraper.engines.visualization.projection_math import fit_to_viewport

LAYER_INTERIOR_PROFILE = "interior-envelope"
PROFILE_STROKE = "#a855f7"
PROFILE_STROKE_WIDTH = 1.1
POT_STROKE = "#ffffff"
POT_STROKE_WIDTH = 2.0


def _coords_from_polylines(polylines: tuple[ProjectedPolyline2D, ...]) -> np.ndarray:
    if not polylines:
        return np.empty((0, 2), dtype=np.float64)
    chunks = [np.asarray(polyline.points_mm, dtype=np.float64) for polyline in polylines]
    return np.vstack(chunks)


def _polyline_path(
    polyline: ProjectedPolyline2D,
    *,
    scale: float,
    offset: np.ndarray,
    css_class: str,
    stroke: str,
    stroke_width: float,
) -> str:
    if len(polyline.points_mm) < 2:
        return ""
    coords = np.asarray(polyline.points_mm, dtype=np.float64)
    projected = coords * scale + offset
    commands: list[str] = []
    for index, (x_value, y_value) in enumerate(projected):
        prefix = "M" if index == 0 else "L"
        commands.append(f"{prefix}{x_value:.2f},{y_value:.2f}")
    if polyline.is_closed:
        commands.append("Z")
    return (
        f'<path class="{css_class}" d="{" ".join(commands)}" fill="none" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" '
        f'stroke-linejoin="round" stroke-linecap="round" '
        f'vector-effect="non-scaling-stroke"/>'
    )


def contour_to_svg_fragment(
    contour: CadProjectedContour,
    *,
    reference_coords: NDArray[np.float64] | None = None,
    css_class: str = "contour",
    stroke: str = POT_STROKE,
    stroke_width: float = POT_STROKE_WIDTH,
) -> str:
    """
    Render contour polylines to SVG path fragments.

    ``reference_coords`` must be the same 2D projected points used to fit the
    corresponding mesh view (``fit_to_viewport``), so overlays superimpose exactly.
    """
    coords = _coords_from_polylines(contour.polylines)
    if len(coords) == 0:
        return ""
    if reference_coords is not None and len(reference_coords) > 0:
        scale, offset = fit_to_viewport(reference_coords)
    else:
        scale, offset = fit_to_viewport(coords)
    parts = [
        _polyline_path(
            polyline,
            scale=scale,
            offset=offset,
            css_class=css_class,
            stroke=stroke,
            stroke_width=stroke_width,
        )
        for polyline in contour.polylines
    ]
    return "".join(part for part in parts if part)


def contour_layers_for_plane(
    contour: CadProjectedContour,
    *,
    view_key: str,
    reference_coords: NDArray[np.float64] | None = None,
    css_class: str = "interior-profile-contour",
    stroke: str = PROFILE_STROKE,
    stroke_width: float = PROFILE_STROKE_WIDTH,
    layer_type: str = LAYER_INTERIOR_PROFILE,
) -> tuple[SvgLayer, ...]:
    fragment = contour_to_svg_fragment(
        contour,
        reference_coords=reference_coords,
        css_class=css_class,
        stroke=stroke,
        stroke_width=stroke_width,
    )
    if not fragment:
        return ()
    return (
        SvgLayer(
            id=f"{view_key}-{layer_type}",
            z_index=15,
            svg_fragment=fragment,
            layer_type=layer_type,
        ),
    )


def cad_reference_layers(
    geometry: CadReferenceGeometry,
    *,
    clearance_mm: float = 0.0,
    jar_vertices: NDArray[np.float64] | None = None,
) -> tuple[tuple[SvgLayer, ...], tuple[SvgLayer, ...]]:
    from nutella_scraper.cad_import.brep_contour_extractor import offset_contour
    from nutella_scraper.engines.visualization.projection_math import project_vertices

    profile = geometry.profile_contour
    top = geometry.top_contour
    if profile is None or top is None:
        return (), ()

    if clearance_mm > 0.0:
        profile = offset_contour(profile, clearance_mm)
        top = offset_contour(top, clearance_mm)

    profile_ref = None
    top_ref = None
    if jar_vertices is not None and len(jar_vertices) > 0:
        profile_ref, _, _ = project_vertices(jar_vertices, PLANE_PROFILE)
        top_ref, _, _ = project_vertices(jar_vertices, PLANE_TOP_XZ)

    return (
        contour_layers_for_plane(profile, view_key="profile", reference_coords=profile_ref),
        contour_layers_for_plane(top, view_key="top", reference_coords=top_ref),
    )


def cad_reference_projection(
    geometry: CadReferenceGeometry,
    *,
    plane: str,
    clearance_mm: float = 0.0,
    jar_vertices: NDArray[np.float64] | None = None,
) -> str:
    from nutella_scraper.cad_import.brep_contour_extractor import offset_contour
    from nutella_scraper.engines.visualization.projection_math import project_vertices

    if plane == PLANE_PROFILE:
        contour = geometry.profile_contour
    elif plane == PLANE_TOP_XZ:
        contour = geometry.top_contour
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")
    if contour is None:
        return ""
    if clearance_mm > 0.0:
        contour = offset_contour(contour, clearance_mm)
    reference_coords = None
    if jar_vertices is not None and len(jar_vertices) > 0:
        reference_coords, _, _ = project_vertices(jar_vertices, plane)
    return contour_to_svg_fragment(contour, reference_coords=reference_coords)
