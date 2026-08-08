"""Read-only metrics panel derived exclusively from ContactResult / CollisionResult."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nutella_scraper.domain.models.contact import CollisionResult, ContactResult


@dataclass(frozen=True)
class ContactMetricsPanel:
    """
    UI-facing metrics — every value is copied or aggregated from ContactResult fields.

    No jar mesh or 2D projection data is consulted.
    """

    coverage_score_percent: float
    covered_surface_mm2: float | None
    uncovered_surface_mm2: float | None
    covered_face_count: int
    total_face_count: int
    contact_point_count: int
    mean_distance_mm: float | None
    max_distance_mm: float | None
    has_collision: bool
    max_penetration_depth_mm: float
    simulation_duration_ms: float | None

    @classmethod
    def from_contact_result(cls, contact: ContactResult) -> ContactMetricsPanel:
        collision = contact.collision or CollisionResult(
            has_collision=False,
            penetration_depth_mm=0.0,
            collision_points=(),
            colliding_face_ids=frozenset(),
        )
        overlay = contact.overlay
        diagnostics = contact.diagnostics

        covered_faces = len(contact.touched_face_ids)
        uncovered_faces = len(contact.untouched_face_ids)
        total_faces = covered_faces + uncovered_faces

        contact_point_count = len(overlay.contact_points) if overlay is not None else 0
        if "contact_point_count" in diagnostics:
            contact_point_count = int(diagnostics["contact_point_count"])

        finite_distances = contact.contact_distance_map[
            np.isfinite(contact.contact_distance_map)
        ]
        mean_distance = float(np.mean(finite_distances)) if finite_distances.size else None
        max_distance = float(np.max(finite_distances)) if finite_distances.size else None

        covered_surface, uncovered_surface = _surface_areas_from_diagnostics(
            contact.coverage_score,
            diagnostics,
        )

        duration_raw = diagnostics.get("simulation_duration_ms")
        simulation_duration_ms = float(duration_raw) if duration_raw is not None else None

        return cls(
            coverage_score_percent=contact.coverage_score * 100.0,
            covered_surface_mm2=covered_surface,
            uncovered_surface_mm2=uncovered_surface,
            covered_face_count=covered_faces,
            total_face_count=total_faces,
            contact_point_count=contact_point_count,
            mean_distance_mm=mean_distance,
            max_distance_mm=max_distance,
            has_collision=collision.has_collision,
            max_penetration_depth_mm=collision.penetration_depth_mm,
            simulation_duration_ms=simulation_duration_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON API responses."""
        uncovered_percent = 100.0 - self.coverage_score_percent
        return {
            "coverage_score_percent": round(self.coverage_score_percent, 2),
            "covered_surface_mm2": _round_optional(self.covered_surface_mm2),
            "uncovered_surface_mm2": _round_optional(self.uncovered_surface_mm2),
            "covered_surface_percent": round(self.coverage_score_percent, 2),
            "uncovered_surface_percent": round(uncovered_percent, 2),
            "covered_face_count": self.covered_face_count,
            "total_face_count": self.total_face_count,
            "contact_point_count": self.contact_point_count,
            "mean_distance_mm": _round_optional(self.mean_distance_mm),
            "max_distance_mm": _round_optional(self.max_distance_mm),
            "has_collision": self.has_collision,
            "collision_label": "Oui" if self.has_collision else "Non",
            "max_penetration_depth_mm": round(self.max_penetration_depth_mm, 4),
            "simulation_duration_ms": _round_optional(self.simulation_duration_ms),
        }


def _surface_areas_from_diagnostics(
    coverage_score: float,
    diagnostics: dict[str, Any],
) -> tuple[float | None, float | None]:
    covered = diagnostics.get("covered_area_mm2")
    uncovered = diagnostics.get("uncovered_area_mm2")
    total = diagnostics.get("total_inner_surface_mm2")

    if covered is not None and uncovered is not None:
        return float(covered), float(uncovered)
    if covered is not None and total is not None:
        return float(covered), float(total) - float(covered)
    if total is not None:
        covered_area = float(total) * coverage_score
        return covered_area, float(total) - covered_area
    return None, None


def _round_optional(value: float | None, *, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)
