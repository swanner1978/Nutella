"""Parametric scraper solid swept along the cyan interior active edge."""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import (
    EnvelopeStation,
    ScraperEnvelopePath,
    ScraperEnvelopePathBuilder,
)


class ScraperGeometryGenerator:
    """
    Build a jar-frame scraper solid from ScraperParameters + interior surface.

    Pipeline:
      InteriorSurfaceReference
        → ordered/smooth active-edge path
        → parametric section
        → loft
    """

    _MIN_FACE_AREA_MM2 = 1e-6

    def generate(
        self,
        parameters: ScraperParameters,
        surface: InteriorSurfaceReference,
        *,
        path: ScraperEnvelopePath | None = None,
    ) -> trimesh.Trimesh:
        from nutella_scraper.engines.compute.scraper_envelope_path import (
            assert_no_inverted_station_pairs,
        )

        envelope_path = path or ScraperEnvelopePathBuilder().build(surface, parameters)
        if len(envelope_path.stations) < 2:
            raise ValueError("Envelope path needs at least two stations")
        assert_no_inverted_station_pairs(envelope_path.stations)

        sections: list[NDArray[np.float64]] = []
        for station in envelope_path.stations:
            sections.append(self._section_at_station(parameters, station))

        mesh = self._loft_sections(sections)
        if mesh.is_empty or len(mesh.faces) == 0:
            raise ValueError("ScraperGeometryGenerator produced an empty mesh")
        self._assert_loft_quality(mesh)
        if not bool(mesh.is_watertight):
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                process=True,
            )
            self._assert_loft_quality(mesh)
        return mesh

    @classmethod
    def _assert_loft_quality(cls, mesh: trimesh.Trimesh) -> None:
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        if len(areas) == 0:
            raise ValueError("Loft produced no faces")
        if float(np.min(areas)) < cls._MIN_FACE_AREA_MM2:
            raise ValueError(
                f"Loft has degenerate faces (min area {float(np.min(areas)):.3e} mm²)"
            )
        if not bool(getattr(mesh, "is_winding_consistent", True)):
            raise ValueError("Loft winding is inconsistent")

    def _section_at_station(
        self,
        parameters: ScraperParameters,
        station: EnvelopeStation,
    ) -> NDArray[np.float64]:
        tip = np.asarray(station.tip_points_mm, dtype=np.float64)
        normals = np.asarray(station.inward_normals, dtype=np.float64)
        thickness = float(parameters.thickness_mm)
        bevel = float(np.deg2rad(parameters.bevel_angle_deg))
        relief = float(np.deg2rad(parameters.relief_angle_deg))
        helix = float(np.deg2rad(parameters.helix_rate_deg_per_mm * station.s_mm))
        n_w = len(tip)
        mid = n_w // 2
        tangent = np.asarray(station.tangent_length, dtype=np.float64)

        if bevel > 1e-9:
            bevel_run = min(thickness * 0.85, thickness / max(np.tan(bevel), 1e-6))
        else:
            bevel_run = 0.0
        if relief > 1e-9:
            relief_run = min(thickness * 0.5, thickness / max(np.tan(relief), 1e-6))
        else:
            relief_run = 0.0

        tip_depth = np.zeros(n_w, dtype=np.float64)
        if bevel_run > 1e-9:
            side = np.abs(np.linspace(-1.0, 1.0, n_w))
            tip_depth = bevel_run * side * 0.35

        back_depth = np.full(n_w, thickness, dtype=np.float64)
        if relief_run > 1e-9:
            side = np.abs(np.linspace(-1.0, 1.0, n_w))
            back_depth = thickness - relief_run * (1.0 - side) * 0.5

        tip_pts = tip + normals * tip_depth[:, None]
        back_pts = tip + normals * back_depth[:, None]

        if abs(helix) > 1e-12:
            tip_pts = self._rotate_around_axis(tip_pts, tip[mid], tangent, helix)
            back_pts = self._rotate_around_axis(back_pts, tip[mid], tangent, helix)

        return np.vstack([tip_pts, back_pts[::-1]])

    @staticmethod
    def _rotate_around_axis(
        points: NDArray[np.float64],
        origin: NDArray[np.float64],
        axis: NDArray[np.float64],
        angle_rad: float,
    ) -> NDArray[np.float64]:
        axis_u = axis / max(float(np.linalg.norm(axis)), 1e-9)
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))
        shifted = points - origin
        dotted = shifted @ axis_u
        cross = np.cross(axis_u, shifted)
        rotated = (
            shifted * cos_a
            + cross * sin_a
            + axis_u[None, :] * dotted[:, None] * (1.0 - cos_a)
        )
        return origin + rotated

    @staticmethod
    def _loft_sections(sections: list[NDArray[np.float64]]) -> trimesh.Trimesh:
        if len(sections) < 2:
            raise ValueError("Need at least two sections to loft a scraper")
        n_pts = len(sections[0])
        if any(len(section) != n_pts for section in sections):
            raise ValueError("All scraper sections must share the same vertex count")

        vertices = np.vstack(sections)
        faces: list[tuple[int, int, int]] = []

        for ring in range(len(sections) - 1):
            base = ring * n_pts
            nxt = (ring + 1) * n_pts
            for i in range(n_pts):
                j = (i + 1) % n_pts
                faces.append((base + i, base + j, nxt + j))
                faces.append((base + i, nxt + j, nxt + i))

        def _cap(ring_index: int, reverse: bool) -> None:
            base = ring_index * n_pts
            indices = list(range(n_pts))
            if reverse:
                indices = list(reversed(indices))
            for i in range(1, n_pts - 1):
                faces.append((base + indices[0], base + indices[i], base + indices[i + 1]))

        _cap(0, reverse=True)
        _cap(len(sections) - 1, reverse=False)

        return trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
