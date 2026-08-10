"""Exact orthographic projection of the posed Scraper3D mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.domain.models.views import SvgLayer
from nutella_scraper.engines.compute.internal_jar_surface_builder import internal_mesh_to_trimesh
from nutella_scraper.engines.visualization.projection_math import (
    PLANE_LEFT,
    PLANE_RIGHT,
    PLANE_SIDE,
    PLANE_TOP,
    fit_to_viewport,
    project_vertices,
)

LAYER_SCRAPER_VOLUME = "scraper-volume"
LAYER_SCRAPER_CONTOUR = "scraper-contour"


@dataclass(frozen=True)
class ScraperProjection:
    """Scraper layers aligned to the canonical jar projection viewport."""

    profile_layers: tuple[SvgLayer, ...]
    top_layers: tuple[SvgLayer, ...]
    vertex_count: int
    face_count: int
    left_layers: tuple[SvgLayer, ...] = ()
    right_layers: tuple[SvgLayer, ...] = ()


class ScraperResultProjector:
    """Project the exact posed mesh used by ContactSimulationEngine."""

    def project(
        self,
        *,
        scraper_vertices: NDArray[np.float64],
        scraper_faces: NDArray[np.int64],
        internal: InternalJarSurface,
    ) -> ScraperProjection:
        jar_mesh = internal_mesh_to_trimesh(internal)
        jar_vertices = np.asarray(jar_mesh.vertices, dtype=np.float64)
        vertices = np.asarray(scraper_vertices, dtype=np.float64)
        faces = np.asarray(scraper_faces, dtype=np.int64)
        return ScraperProjection(
            profile_layers=self._layers_for_plane(
                vertices=vertices,
                faces=faces,
                jar_vertices=jar_vertices,
                plane=PLANE_SIDE,
                view_key="profile",
            ),
            top_layers=self._layers_for_plane(
                vertices=vertices,
                faces=faces,
                jar_vertices=jar_vertices,
                plane=PLANE_TOP,
                view_key="top",
            ),
            left_layers=self._layers_for_plane(
                vertices=vertices,
                faces=faces,
                jar_vertices=jar_vertices,
                plane=PLANE_LEFT,
                view_key="left",
            ),
            right_layers=self._layers_for_plane(
                vertices=vertices,
                faces=faces,
                jar_vertices=jar_vertices,
                plane=PLANE_RIGHT,
                view_key="right",
            ),
            vertex_count=len(vertices),
            face_count=len(faces),
        )

    @staticmethod
    def _layers_for_plane(
        *,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        jar_vertices: NDArray[np.float64],
        plane: str,
        view_key: str,
    ) -> tuple[SvgLayer, ...]:
        jar_coords, _, _ = project_vertices(jar_vertices, plane)
        scale, offset = fit_to_viewport(jar_coords)
        scraper_coords, _, _ = project_vertices(vertices, plane)
        projected = scraper_coords * scale + offset
        silhouette = _silhouette_path(projected, faces)
        return (
            SvgLayer(
                id=f"{view_key}-{LAYER_SCRAPER_VOLUME}",
                z_index=45,
                svg_fragment=(
                    f'<path d="{silhouette}" fill="#f2c94c" fill-opacity="0.35" '
                    'stroke="none" fill-rule="evenodd" class="scraper-volume"/>'
                ),
                layer_type=LAYER_SCRAPER_VOLUME,
            ),
            SvgLayer(
                id=f"{view_key}-{LAYER_SCRAPER_CONTOUR}",
                z_index=55,
                svg_fragment=(
                    f'<path d="{silhouette}" fill="none" stroke="#ffe082" '
                    'stroke-width="2" vector-effect="non-scaling-stroke" '
                    'fill-rule="evenodd" class="scraper-contour"/>'
                ),
                layer_type=LAYER_SCRAPER_CONTOUR,
            ),
        )


def _silhouette_path(
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
        raise ValueError("La projection du Scraper3D ne produit aucune silhouette")
    merged = unary_union(polygons)
    if isinstance(merged, Polygon):
        merged_polygons = (merged,)
    elif isinstance(merged, MultiPolygon):
        merged_polygons = tuple(merged.geoms)
    else:
        raise ValueError(f"Silhouette Scraper3D inattendue : {merged.geom_type}")
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
