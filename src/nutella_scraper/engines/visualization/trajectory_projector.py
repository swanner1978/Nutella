"""Orthographic projection of the scraper trajectory — visualization only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.domain.models.views import SvgLayer
from nutella_scraper.engines.compute.internal_jar_surface_builder import internal_mesh_to_trimesh
from nutella_scraper.engines.visualization.projection_math import (
    PLANE_BOTTOM,
    PLANE_LEFT,
    PLANE_RIGHT,
    PLANE_SIDE,
    PLANE_TOP,
    fit_to_viewport,
    project_vertices,
)

LAYER_TRAJECTORY = "scraper-trajectory"
TRAJECTORY_STROKE = "#fbbf24"
TRAJECTORY_STROKE_WIDTH = 1.2


@dataclass(frozen=True)
class TrajectoryProjection:
    profile_layers: tuple[SvgLayer, ...]
    top_layers: tuple[SvgLayer, ...]
    left_layers: tuple[SvgLayer, ...] = ()
    right_layers: tuple[SvgLayer, ...] = ()
    bottom_layers: tuple[SvgLayer, ...] = ()


class TrajectoryProjector:
    """Project the scraper centre path followed during simulation."""

    def project(
        self,
        positions_mm: tuple[tuple[float, float, float], ...],
        internal: InternalJarSurface,
    ) -> TrajectoryProjection:
        if len(positions_mm) < 2:
            return TrajectoryProjection(profile_layers=(), top_layers=())
        jar_mesh = internal_mesh_to_trimesh(internal)
        jar_vertices = np.asarray(jar_mesh.vertices, dtype=np.float64)
        points = np.asarray(positions_mm, dtype=np.float64)
        return TrajectoryProjection(
            profile_layers=self._layers_for_plane(
                points=points,
                jar_vertices=jar_vertices,
                plane=PLANE_SIDE,
                view_key="profile",
            ),
            top_layers=self._layers_for_plane(
                points=points,
                jar_vertices=jar_vertices,
                plane=PLANE_TOP,
                view_key="top",
            ),
            left_layers=self._layers_for_plane(
                points=points,
                jar_vertices=jar_vertices,
                plane=PLANE_LEFT,
                view_key="left",
            ),
            right_layers=self._layers_for_plane(
                points=points,
                jar_vertices=jar_vertices,
                plane=PLANE_RIGHT,
                view_key="right",
            ),
            bottom_layers=self._layers_for_plane(
                points=points,
                jar_vertices=jar_vertices,
                plane=PLANE_BOTTOM,
                view_key="bottom",
            ),
        )

    @staticmethod
    def _layers_for_plane(
        *,
        points: NDArray[np.float64],
        jar_vertices: NDArray[np.float64],
        plane: str,
        view_key: str,
    ) -> tuple[SvgLayer, ...]:
        jar_coords, _, _ = project_vertices(jar_vertices, plane)
        scale, offset = fit_to_viewport(jar_coords)
        coords, _, _ = project_vertices(points, plane)
        projected = coords * scale + offset
        commands = [
            f"{'M' if index == 0 else 'L'}{projected[index, 0]:.2f},{projected[index, 1]:.2f}"
            for index in range(len(projected))
        ]
        path = (
            f'<path class="scraper-trajectory" fill="none" '
            f'stroke="{TRAJECTORY_STROKE}" stroke-width="{TRAJECTORY_STROKE_WIDTH}" '
            f'stroke-linejoin="round" stroke-linecap="round" '
            f'vector-effect="non-scaling-stroke" d="{" ".join(commands)}"/>'
        )
        return (
            SvgLayer(
                id=f"{view_key}-{LAYER_TRAJECTORY}",
                z_index=18,
                svg_fragment=path,
                layer_type=LAYER_TRAJECTORY,
            ),
        )
