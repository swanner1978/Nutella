"""Projection of STEP faces selected by colour — visualization only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from nutella_scraper.cad_import.step_face_color_diagnostics import TARGET_RGB_255
from nutella_scraper.domain.models.views import SvgLayer
from nutella_scraper.engines.visualization.cad_reference_projector import (
    LAYER_INTERIOR_PROFILE,
)
from nutella_scraper.engines.visualization.projection_math import (
    PLANE_BOTTOM,
    PLANE_LEFT,
    PLANE_RIGHT,
    PLANE_SIDE,
    PLANE_TOP,
    fit_to_viewport,
    project_vertices,
)

LAYER_TARGET_FACE_COLORS = "target-face-colors"


@dataclass(frozen=True)
class TargetFaceColorProjection:
    profile_layers: tuple[SvgLayer, ...]
    top_layers: tuple[SvgLayer, ...]
    left_layers: tuple[SvgLayer, ...]
    right_layers: tuple[SvgLayer, ...]
    bottom_layers: tuple[SvgLayer, ...] = ()


class TargetFaceColorProjector:
    """Project colour-selected B-Rep faces into viewer SVG overlays.

    Viewport scale/offset always comes from the jar mesh (same as the main model),
    never from an independent fit of the selected faces alone.
    """

    def project(
        self,
        *,
        face_vertices: NDArray[np.float64],
        face_triangles: NDArray[np.int64],
        jar_vertices: NDArray[np.float64],
        target_face_count: int,
        target_area_mm2: float,
        fill_rgb_255: tuple[int, int, int] = TARGET_RGB_255,
        layer_type: str = LAYER_INTERIOR_PROFILE,
        include_labels: bool = False,
    ) -> TargetFaceColorProjection:
        vertices = np.asarray(face_vertices, dtype=np.float64)
        faces = np.asarray(face_triangles, dtype=np.int64)
        jar = np.asarray(jar_vertices, dtype=np.float64)
        if vertices.size == 0 or faces.size == 0:
            empty = self._empty_layers(
                target_face_count=target_face_count,
                target_area_mm2=target_area_mm2,
                fill_rgb_255=fill_rgb_255,
                layer_type=layer_type,
                include_labels=include_labels,
            )
            return TargetFaceColorProjection(*empty)

        return TargetFaceColorProjection(
            profile_layers=self._layers_for_plane(
                vertices=vertices,
                faces=faces,
                jar_vertices=jar,
                plane=PLANE_SIDE,
                view_key="profile",
                target_face_count=target_face_count,
                target_area_mm2=target_area_mm2,
                fill_rgb_255=fill_rgb_255,
                layer_type=layer_type,
                include_labels=include_labels,
            ),
            top_layers=self._layers_for_plane(
                vertices=vertices,
                faces=faces,
                jar_vertices=jar,
                plane=PLANE_TOP,
                view_key="top",
                target_face_count=target_face_count,
                target_area_mm2=target_area_mm2,
                fill_rgb_255=fill_rgb_255,
                layer_type=layer_type,
                include_labels=include_labels,
            ),
            left_layers=self._layers_for_plane(
                vertices=vertices,
                faces=faces,
                jar_vertices=jar,
                plane=PLANE_LEFT,
                view_key="left",
                target_face_count=target_face_count,
                target_area_mm2=target_area_mm2,
                fill_rgb_255=fill_rgb_255,
                layer_type=layer_type,
                include_labels=include_labels,
            ),
            right_layers=self._layers_for_plane(
                vertices=vertices,
                faces=faces,
                jar_vertices=jar,
                plane=PLANE_RIGHT,
                view_key="right",
                target_face_count=target_face_count,
                target_area_mm2=target_area_mm2,
                fill_rgb_255=fill_rgb_255,
                layer_type=layer_type,
                include_labels=include_labels,
            ),
            bottom_layers=self._layers_for_plane(
                vertices=vertices,
                faces=faces,
                jar_vertices=jar,
                plane=PLANE_BOTTOM,
                view_key="bottom",
                target_face_count=target_face_count,
                target_area_mm2=target_area_mm2,
                fill_rgb_255=fill_rgb_255,
                layer_type=layer_type,
                include_labels=include_labels,
            ),
        )

    def _empty_layers(
        self,
        *,
        target_face_count: int,
        target_area_mm2: float,
        fill_rgb_255: tuple[int, int, int],
        layer_type: str,
        include_labels: bool,
    ) -> tuple[
        tuple[SvgLayer, ...],
        tuple[SvgLayer, ...],
        tuple[SvgLayer, ...],
        tuple[SvgLayer, ...],
        tuple[SvgLayer, ...],
    ]:
        fragment = (
            _label_fragment(target_face_count, target_area_mm2, fill_rgb_255)
            if include_labels
            else ""
        )
        layers = {
            key: (
                (
                    SvgLayer(
                        id=f"{key}-{layer_type}",
                        z_index=28,
                        svg_fragment=fragment,
                        layer_type=layer_type,
                    ),
                )
                if fragment
                else ()
            )
            for key in ("profile", "top", "left", "right", "bottom")
        }
        return (
            layers["profile"],
            layers["top"],
            layers["left"],
            layers["right"],
            layers["bottom"],
        )

    @staticmethod
    def _layers_for_plane(
        *,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        jar_vertices: NDArray[np.float64],
        plane: str,
        view_key: str,
        target_face_count: int,
        target_area_mm2: float,
        fill_rgb_255: tuple[int, int, int],
        layer_type: str,
        include_labels: bool,
    ) -> tuple[SvgLayer, ...]:
        jar_coords, _, _ = project_vertices(jar_vertices, plane)
        scale, offset = fit_to_viewport(jar_coords)
        face_coords, _, _ = project_vertices(vertices, plane)
        projected = face_coords * scale + offset
        fill = f"rgb({fill_rgb_255[0]},{fill_rgb_255[1]},{fill_rgb_255[2]})"
        path = _filled_triangles_path(projected, faces)
        labels = (
            _label_fragment(target_face_count, target_area_mm2, fill_rgb_255)
            if include_labels
            else ""
        )
        fragment = (
            f'<path d="{path}" fill="{fill}" fill-opacity="0.55" stroke="{fill}" '
            'stroke-width="0.6" stroke-opacity="0.9" '
            'vector-effect="non-scaling-stroke" fill-rule="evenodd" '
            f'class="{layer_type}"/>'
            f"{labels}"
        )
        return (
            SvgLayer(
                id=f"{view_key}-{layer_type}",
                z_index=28,
                svg_fragment=fragment,
                layer_type=layer_type,
            ),
        )


def _label_fragment(
    target_face_count: int,
    target_area_mm2: float,
    fill_rgb_255: tuple[int, int, int],
) -> str:
    fill = f"rgb({fill_rgb_255[0]},{fill_rgb_255[1]},{fill_rgb_255[2]})"
    return (
        f'<text x="16" y="28" fill="{fill}" font-size="14" '
        'font-family="Consolas, Monaco, monospace" '
        f'class="{LAYER_TARGET_FACE_COLORS}-label">'
        f"Target faces: {target_face_count}</text>"
        f'<text x="16" y="48" fill="{fill}" font-size="14" '
        'font-family="Consolas, Monaco, monospace" '
        f'class="{LAYER_TARGET_FACE_COLORS}-label">'
        f"Target area: {target_area_mm2:.3f} mm²</text>"
    )


def _filled_triangles_path(
    projected_vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
) -> str:
    polygons: list[Polygon] = []
    for face in faces:
        coordinates = projected_vertices[face]
        if len(coordinates) != 3:
            continue
        polygon = Polygon([(float(x), float(y)) for x, y in coordinates])
        if not polygon.is_empty and polygon.area > 1e-9:
            polygons.append(polygon)
    if not polygons:
        return ""
    merged = unary_union(polygons)
    if isinstance(merged, Polygon):
        merged_polygons = (merged,)
    elif isinstance(merged, MultiPolygon):
        merged_polygons = tuple(merged.geoms)
    else:
        return ""
    return " ".join(_polygon_path(polygon) for polygon in merged_polygons)


def _polygon_path(polygon: Polygon) -> str:
    rings = [polygon.exterior, *polygon.interiors]
    commands: list[str] = []
    for ring in rings:
        coordinates = list(ring.coords)
        commands.append(
            " ".join(
                f"{'M' if index == 0 else 'L'}{x:.3f},{y:.3f}"
                for index, (x, y) in enumerate(coordinates)
            )
            + " Z"
        )
    return " ".join(commands)
