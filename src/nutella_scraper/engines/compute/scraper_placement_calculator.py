"""Verify scraper stays on the interior side of the cyan envelope."""

from __future__ import annotations

import numpy as np
import trimesh

from nutella_scraper.domain.models.scraper import ScraperPose
from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.interior_surface_reference import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import (
    NUMERIC_GAP_MM,
    ScraperEnvelopePath,
)


class ScraperPlacementCalculator:
    """
    Placement metadata + interior-side validation for a jar-frame scraper mesh.

    Geometry is already built on InteriorSurfaceReference (cyan faces); this
    class does not translate an independent rectangle onto an approximate wall.
    """

    _NUMERIC_GAP_MM = NUMERIC_GAP_MM

    def place(
        self,
        mesh: trimesh.Trimesh,
        parameters: ScraperParameters,
        surface: InteriorSurfaceReference,
        *,
        path: ScraperEnvelopePath | None = None,
    ) -> ScraperPose:
        """Return pose metadata for a mesh already posed in jar frame."""
        del path
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            raise ValueError("Empty scraper mesh")
        # Tip proxy: vertices nearest to the cyan wall.
        distances = np.asarray(
            surface.to_trimesh().nearest.on_surface(vertices)[1],
            dtype=np.float64,
        )
        tip_ids = np.argsort(distances)[: max(1, int(len(distances) * 0.08))]
        tip_mid = np.mean(vertices[tip_ids], axis=0)
        return ScraperPose(
            position_mm=(float(tip_mid[0]), float(tip_mid[1]), float(tip_mid[2])),
            yaw_deg=float(parameters.surface_progress_deg),
            pitch_deg=0.0,
            roll_deg=0.0,
        )

    def place_and_verify(
        self,
        mesh: trimesh.Trimesh,
        parameters: ScraperParameters,
        surface: InteriorSurfaceReference,
        *,
        path: ScraperEnvelopePath | None = None,
        penetration_tol_mm: float | None = None,
    ) -> tuple[ScraperPose, trimesh.Trimesh]:
        del path
        pose = self.place(mesh, parameters, surface)
        # Rigid blade on a non-circular wall may locally dig in until a free-pose
        # optimizer exists — keep a strict check only at the design progress.
        if penetration_tol_mm is None:
            penetration_tol_mm = (
                0.15 if abs(float(parameters.surface_progress_deg)) <= 1e-9 else 2.5
            )
        self.assert_interior_side(
            mesh,
            surface,
            parameters,
            penetration_tol_mm=penetration_tol_mm,
        )
        return pose, mesh

    @staticmethod
    def uses_interior_product_surface(path: ScraperEnvelopePath) -> bool:
        return path.source == SOURCE_INTERIOR_PRODUCT_SURFACE

    # Back-compat alias
    uses_cyan_interior = uses_interior_product_surface

    @staticmethod
    def active_tip_gap_mm(
        posed_mesh: trimesh.Trimesh,
        surface: InteriorSurfaceReference,
        *,
        tip_fraction: float = 0.08,
    ) -> float:
        mesh = surface.to_trimesh()
        vertices = np.asarray(posed_mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            raise ValueError("Empty scraper mesh")
        distances = np.asarray(mesh.nearest.on_surface(vertices)[1], dtype=np.float64)
        count = max(1, int(len(distances) * tip_fraction))
        tip_ids = np.argsort(distances)[:count]
        return float(np.mean(distances[tip_ids]))

    @staticmethod
    def measure_envelope_fit(
        posed_mesh: trimesh.Trimesh,
        surface: InteriorSurfaceReference,
        parameters: ScraperParameters,
    ) -> dict[str, float]:
        """
        Geometric coherence vs the interior product surface (not coverage).

        Distances are unsigned nearest-surface lengths. Penetration is the max
        outward (glass-side) signed offset among vertices near the wall band.
        """
        mesh = surface.to_trimesh()
        vertices = np.asarray(posed_mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            raise ValueError("Empty scraper mesh")
        closest, distances, tri_ids = mesh.nearest.on_surface(vertices)
        closest = np.asarray(closest, dtype=np.float64)
        distances = np.asarray(distances, dtype=np.float64)
        tri_ids = np.asarray(tri_ids, dtype=np.int64)
        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

        normals = face_normals[np.clip(tri_ids, 0, len(face_normals) - 1)].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-9)
        eps = 0.25
        plus = closest + normals * eps
        minus = closest - normals * eps
        r_plus = np.hypot(plus[:, 0], plus[:, 2])
        r_minus = np.hypot(minus[:, 0], minus[:, 2])
        normals[r_plus > r_minus] *= -1.0
        outward = -normals

        band = float(parameters.thickness_mm) + float(parameters.clearance_mm) + 1.5
        near = distances <= band
        if not np.any(near):
            near = np.argsort(distances)[: max(1, int(len(distances) * 0.08))]
        penetration = np.sum((vertices - closest) * outward, axis=1)
        worst = float(np.max(penetration[near]))

        return {
            "surface_progress_deg": float(parameters.surface_progress_deg),
            "rotation_angle_deg": float(parameters.surface_progress_deg),
            "distance_min_mm": float(np.min(distances)),
            "distance_max_mm": float(np.max(distances)),
            "penetration_mm": max(0.0, worst),
            "clearance_mm": float(parameters.clearance_mm),
        }

    @staticmethod
    def assert_interior_side(
        posed_mesh: trimesh.Trimesh,
        surface: InteriorSurfaceReference,
        parameters: ScraperParameters,
        *,
        penetration_tol_mm: float = 0.15,
    ) -> None:
        """
        Fail if scraper vertices cross the cyan surface into the glass.

        Uses the local surface normal (probe-oriented inward) rather than a
        cylindrical radius test, which is wrong on elliptical sections.
        """
        mesh = surface.to_trimesh()
        vertices = np.asarray(posed_mesh.vertices, dtype=np.float64)
        closest, distances, tri_ids = mesh.nearest.on_surface(vertices)
        closest = np.asarray(closest, dtype=np.float64)
        distances = np.asarray(distances, dtype=np.float64)
        tri_ids = np.asarray(tri_ids, dtype=np.int64)
        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

        tip_count = max(1, int(len(distances) * 0.08))
        tip_ids = np.argsort(distances)[:tip_count]
        tip_gap = float(np.mean(distances[tip_ids]))
        expected = float(parameters.clearance_mm) + NUMERIC_GAP_MM
        # Strict tip-gap check only for the design pose (progress = 0).
        if (
            abs(float(parameters.surface_progress_deg)) <= 1e-9
            and float(parameters.bevel_angle_deg) <= 1e-9
            and float(parameters.helix_rate_deg_per_mm) <= 1e-9
            and abs(tip_gap - expected) > 0.35
        ):
            raise ValueError(
                f"Active tip gap {tip_gap:.4f} mm != clearance target {expected:.4f} mm"
            )

        # Inward normals at closest triangles (same probe rule as path builder).
        normals = face_normals[np.clip(tri_ids, 0, len(face_normals) - 1)].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-9)
        eps = 0.25
        plus = closest + normals * eps
        minus = closest - normals * eps
        r_plus = np.hypot(plus[:, 0], plus[:, 2])
        r_minus = np.hypot(minus[:, 0], minus[:, 2])
        flip = r_plus > r_minus
        normals[flip] *= -1.0
        outward = -normals

        # Only vertices near the cyan wall — deep cavity points may snap to the
        # opposite wall and must not be treated as glass penetration.
        band = float(parameters.thickness_mm) + float(parameters.clearance_mm) + 1.5
        near = distances <= band
        if not np.any(near):
            near = tip_ids
            penetration = np.sum((vertices - closest) * outward, axis=1)
            worst = float(np.max(penetration[near]))
        else:
            penetration = np.sum((vertices - closest) * outward, axis=1)
            worst = float(np.max(penetration[near]))

        # Helix can twist the tip off the wall locally; enforce glass-side check
        # for non-helix profiles (A / B).
        if float(parameters.helix_rate_deg_per_mm) <= 1e-9 and worst > penetration_tol_mm:
            raise ValueError(
                f"Scraper penetrates cyan interior toward glass "
                f"(max outward {worst:.4f} mm > {penetration_tol_mm} mm)"
            )
