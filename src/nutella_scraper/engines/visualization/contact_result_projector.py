"""Projects 3D ContactResult onto 2D views — read-only display."""

from __future__ import annotations

import base64
import io
import time
from collections.abc import MutableMapping
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from nutella_scraper.domain.models.canonical import CanonicalModel3D, JarCanonicalModel
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.engines.compute.internal_jar_surface_builder import (
    internal_mesh_to_trimesh,
    resolve_internal_jar_surface,
)
from nutella_scraper.domain.models.contact import ContactResult
from nutella_scraper.domain.models.views import SvgLayer, ViewOverlayPayload, ViewProjectionCache
from nutella_scraper.engines.visualization.projection_math import (
    PLANE_SIDE,
    PLANE_TOP,
    VIEW_HEIGHT,
    VIEW_WIDTH,
    canonical_to_trimesh,
    distance_to_color,
    fit_to_viewport,
    project_vertices,
    world_to_svg,
)

LAYER_CONTACT_COVERED = "contact-covered"
LAYER_CONTACT_UNCOVERED = "contact-uncovered"
LAYER_DISTANCE_MAP = "distance-map"
LAYER_CONTACT_POINTS = "contact-points"
LAYER_COLLISION_FACES = "collision-faces"
LAYER_COLLISION_POINTS = "collision-points"

_LAYER_Z_INDEX = {
    LAYER_DISTANCE_MAP: 20,
    LAYER_CONTACT_UNCOVERED: 30,
    LAYER_CONTACT_COVERED: 40,
    LAYER_COLLISION_FACES: 50,
    LAYER_CONTACT_POINTS: 60,
    LAYER_COLLISION_POINTS: 70,
}


class ContactResultProjector:
    """
    Maps touched/untouched 3D faces to 2D overlay layers.

    Does not recompute coverage_score — copies from ContactResult.
    """

    def project(
        self,
        contact: ContactResult,
        views: ViewProjectionCache | None,
        jar: JarCanonicalModel | CanonicalModel3D,
        *,
        internal: InternalJarSurface | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> ViewOverlayPayload:
        if contact.overlay is None:
            raise ValueError("ContactResult.overlay is required for visualization")
        del views

        started = time.perf_counter()
        mesh = self._resolve_mesh(jar, internal=internal)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        profile_layers, profile_stats = self._layers_for_plane(
            contact=contact,
            plane=PLANE_SIDE,
            view_key="profile",
            vertices=vertices,
            faces=faces,
        )
        top_layers, top_stats = self._layers_for_plane(
            contact=contact,
            plane=PLANE_TOP,
            view_key="top",
            vertices=vertices,
            faces=faces,
        )

        payload = ViewOverlayPayload(
            model_id=contact.model_id,
            profile_layers=profile_layers,
            top_layers=top_layers,
            coverage_score_display=contact.coverage_score,
        )
        if profile is not None:
            profile.update(
                {
                    "construction_ms": (time.perf_counter() - started) * 1000.0,
                    "face_count": int(len(faces)),
                    "face_projections_processed": int(len(faces) * 2),
                    "contact_point_count": int(len(contact.overlay.contact_points)),
                    "collision_point_count": int(
                        len(contact.collision.collision_points)
                        if contact.collision is not None
                        else 0
                    ),
                    "graphic_element_count": int(
                        profile_stats["graphic_element_count"]
                        + top_stats["graphic_element_count"]
                    ),
                    "svg_bytes": int(profile_stats["svg_bytes"] + top_stats["svg_bytes"]),
                    "views": {
                        "side": profile_stats,
                        "top": top_stats,
                    },
                }
            )
        return payload

    def _resolve_mesh(
        self,
        jar: JarCanonicalModel | CanonicalModel3D,
        *,
        internal: InternalJarSurface | None = None,
    ):
        if isinstance(jar, CanonicalModel3D):
            return internal_mesh_to_trimesh(resolve_internal_jar_surface(jar, cached=internal))
        from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder

        return JarMeshBuilder().from_profile(jar)

    def _layers_for_plane(
        self,
        *,
        contact: ContactResult,
        plane: str,
        view_key: str,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[tuple[SvgLayer, ...], dict[str, int | float]]:
        started = time.perf_counter()
        overlay = contact.overlay
        assert overlay is not None
        collision = contact.collision

        coords, _, _ = project_vertices(vertices, plane)
        scale, offset = fit_to_viewport(coords)
        projected_vertices = coords * scale + offset

        finite_distances = [
            float(value)
            for value in overlay.min_distance_per_face_mm
            if np.isfinite(value)
        ]
        max_distance = max(finite_distances) if finite_distances else 1.0

        covered_image = Image.new("RGBA", (VIEW_WIDTH, VIEW_HEIGHT), (0, 0, 0, 0))
        uncovered_image = Image.new("RGBA", (VIEW_WIDTH, VIEW_HEIGHT), (0, 0, 0, 0))
        distance_image = Image.new("RGBA", (VIEW_WIDTH, VIEW_HEIGHT), (0, 0, 0, 0))
        collision_image = Image.new("RGBA", (VIEW_WIDTH, VIEW_HEIGHT), (0, 0, 0, 0))
        covered_draw = ImageDraw.Draw(covered_image)
        uncovered_draw = ImageDraw.Draw(uncovered_image)
        distance_draw = ImageDraw.Draw(distance_image)
        collision_draw = ImageDraw.Draw(collision_image)
        covered_count = 0
        uncovered_count = 0
        collision_face_count = 0

        colliding_faces = collision.colliding_face_ids if collision is not None else frozenset()

        for face_id, face in enumerate(faces):
            points = projected_vertices[face]
            if len(points) != 3:
                continue
            polygon = [(float(x), float(y)) for x, y in points]

            is_covered = overlay.face_coverage[face_id]
            distance_mm = overlay.min_distance_per_face_mm[face_id]

            if is_covered:
                covered_draw.polygon(polygon, fill=(0, 204, 102, 115))
                covered_count += 1
            else:
                uncovered_draw.polygon(polygon, fill=(204, 51, 51, 115))
                uncovered_count += 1

            distance_color = distance_to_color(distance_mm, max_distance_mm=max_distance)
            distance_draw.polygon(polygon, fill=_color_to_rgba(distance_color, 89))

            if face_id in colliding_faces:
                collision_draw.polygon(
                    polygon,
                    fill=(255, 68, 68, 89),
                    outline=(255, 136, 136, 255),
                    width=1,
                )
                collision_face_count += 1

        covered_fragment = _raster_layer_svg(covered_image, "covered-face")
        uncovered_fragment = _raster_layer_svg(uncovered_image, "uncovered-face")
        distance_fragment = _raster_layer_svg(distance_image, "distance-face")
        collision_fragment = _raster_layer_svg(collision_image, "collision-face")

        contact_point_paths, contact_point_count = self._contact_points_svg(
            overlay.contact_points,
            plane=plane,
            scale=scale,
            offset=offset,
        )
        collision_point_paths, collision_point_count = self._collision_points_svg(
            collision.collision_points if collision is not None else (),
            plane=plane,
            scale=scale,
            offset=offset,
        )

        layers = (
            SvgLayer(
                id=f"{view_key}-{LAYER_DISTANCE_MAP}",
                z_index=_LAYER_Z_INDEX[LAYER_DISTANCE_MAP],
                svg_fragment=distance_fragment,
                layer_type=LAYER_DISTANCE_MAP,
            ),
            SvgLayer(
                id=f"{view_key}-{LAYER_CONTACT_UNCOVERED}",
                z_index=_LAYER_Z_INDEX[LAYER_CONTACT_UNCOVERED],
                svg_fragment=uncovered_fragment,
                layer_type=LAYER_CONTACT_UNCOVERED,
            ),
            SvgLayer(
                id=f"{view_key}-{LAYER_CONTACT_COVERED}",
                z_index=_LAYER_Z_INDEX[LAYER_CONTACT_COVERED],
                svg_fragment=covered_fragment,
                layer_type=LAYER_CONTACT_COVERED,
            ),
            SvgLayer(
                id=f"{view_key}-{LAYER_COLLISION_FACES}",
                z_index=_LAYER_Z_INDEX[LAYER_COLLISION_FACES],
                svg_fragment=collision_fragment,
                layer_type=LAYER_COLLISION_FACES,
            ),
            SvgLayer(
                id=f"{view_key}-{LAYER_CONTACT_POINTS}",
                z_index=_LAYER_Z_INDEX[LAYER_CONTACT_POINTS],
                svg_fragment=contact_point_paths,
                layer_type=LAYER_CONTACT_POINTS,
            ),
            SvgLayer(
                id=f"{view_key}-{LAYER_COLLISION_POINTS}",
                z_index=_LAYER_Z_INDEX[LAYER_COLLISION_POINTS],
                svg_fragment=collision_point_paths,
                layer_type=LAYER_COLLISION_POINTS,
            ),
        )
        fragments = [layer.svg_fragment for layer in layers]
        stats: dict[str, int | float] = {
            "construction_ms": (time.perf_counter() - started) * 1000.0,
            "faces_processed": int(len(faces)),
            "contact_points_rendered": contact_point_count,
            "collision_points_rendered": collision_point_count,
            "graphic_element_count": sum(
                fragment.count("<path") + fragment.count("<image")
                for fragment in fragments
            ),
            "svg_bytes": sum(len(fragment.encode("utf-8")) for fragment in fragments),
            "covered_face_count": covered_count,
            "uncovered_face_count": uncovered_count,
            "collision_face_count": collision_face_count,
        }
        return layers, stats

    @staticmethod
    def _contact_points_svg(
        contact_points: tuple[object, ...],
        *,
        plane: str,
        scale: float,
        offset: np.ndarray,
    ) -> tuple[str, int]:
        if not contact_points:
            return '<g class="contact-point"/>', 0
        coordinates: set[tuple[float, float]] = set()
        for point in contact_points:
            x, y = world_to_svg(point.position_mm, plane=plane, scale=scale, offset=offset)
            coordinates.add((round(x, 3), round(y, 3)))
        path = " ".join(_circle_path(x, y, 3.5) for x, y in sorted(coordinates))
        return (
            f'<path d="{path}" fill="#ffffff" stroke="#00ffaa" '
            'stroke-width="1.2" class="contact-point"/>',
            len(coordinates),
        )

    @staticmethod
    def _collision_points_svg(
        collision_points: tuple[object, ...],
        *,
        plane: str,
        scale: float,
        offset: np.ndarray,
    ) -> tuple[str, int]:
        if not collision_points:
            return '<g class="collision-point"/>', 0
        coordinates: set[tuple[float, float]] = set()
        for point in collision_points:
            x, y = world_to_svg(point.position_mm, plane=plane, scale=scale, offset=offset)
            coordinates.add((round(x, 3), round(y, 3)))
        path = " ".join(_circle_path(x, y, 4.5) for x, y in sorted(coordinates))
        return (
            f'<path d="{path}" fill="#ff4444" stroke="#ffffff" '
            'stroke-width="1.2" class="collision-point"/>',
            len(coordinates),
        )


def _raster_layer_svg(image: Image.Image, css_class: str) -> str:
    """Encode a face raster as one SVG image element."""
    if image.getbbox() is None:
        return f'<g class="{css_class}"/>'
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<image x="0" y="0" width="{VIEW_WIDTH}" height="{VIEW_HEIGHT}" '
        f'href="data:image/png;base64,{encoded}" class="{css_class}"/>'
    )


def _color_to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    if color.startswith("rgb("):
        red, green, blue = (
            int(component) for component in color.removeprefix("rgb(").removesuffix(")").split(",")
        )
        return red, green, blue, alpha
    if color.startswith("#") and len(color) == 7:
        return (
            int(color[1:3], 16),
            int(color[3:5], 16),
            int(color[5:7], 16),
            alpha,
        )
    raise ValueError(f"Couleur SVG non prise en charge : {color}")


def _circle_path(x: float, y: float, radius: float) -> str:
    """Represent one circle as a subpath so many points share one SVG element."""
    diameter = radius * 2.0
    return (
        f"M{x - radius:.3f},{y:.3f}"
        f"a{radius:.3f},{radius:.3f} 0 1,0 {diameter:.3f},0"
        f"a{radius:.3f},{radius:.3f} 0 1,0 -{diameter:.3f},0"
    )
