"""Coverage score computation from 3D contact results."""

from __future__ import annotations

import trimesh

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.contact import ContactResult
from nutella_scraper.engines.compute.mesh_utils import face_areas, mesh_data_to_trimesh


class CoverageScorer:
    """Computes coverage score from 3D face contact data."""

    def score(
        self,
        touched_face_ids: frozenset[int],
        untouched_face_ids: frozenset[int],
        jar_mesh: trimesh.Trimesh | CanonicalModel3D,
    ) -> float:
        """Return coverage score in [0, 1] weighted by jar face areas."""
        mesh = (
            jar_mesh
            if isinstance(jar_mesh, trimesh.Trimesh)
            else mesh_data_to_trimesh(jar_mesh.mesh)
        )
        areas = face_areas(mesh)
        relevant = touched_face_ids | untouched_face_ids
        if not relevant:
            return 0.0

        total_area = float(sum(areas[face_id] for face_id in relevant))
        if total_area <= 0.0:
            return 0.0

        touched_area = float(sum(areas[face_id] for face_id in touched_face_ids))
        return max(0.0, min(1.0, touched_area / total_area))

    def from_contact_result(self, result: ContactResult) -> float:
        """Extract or recompute coverage from ContactResult."""
        return result.coverage_score
