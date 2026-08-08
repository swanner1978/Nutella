"""Build CadReferenceGeometry from STEP B-Rep — reference views only."""

from __future__ import annotations

import logging
from pathlib import Path

from nutella_scraper.cad_import.brep_contour_extractor import extract_inner_contours
from nutella_scraper.cad_import.inner_face_selector import build_compound, select_inner_outer_faces
from nutella_scraper.cad_import.step_brep_loader import (
    load_step_shape,
    shape_bounding_box,
    step_file_sha256,
)
from nutella_scraper.domain.models.cad_reference_geometry import CadReferenceGeometry

_LOG = logging.getLogger(__name__)


class CadReferenceGeometryBuilder:
    """Extract inner-cavity B-Rep contours for 2D reference views."""

    def from_step(
        self,
        step_path: Path,
        *,
        model_id: str,
    ) -> CadReferenceGeometry:
        path = step_path.resolve()
        shape = load_step_shape(path)
        bbox = shape_bounding_box(shape)
        inner_faces, outer_faces = select_inner_outer_faces(shape)
        if not inner_faces:
            raise ValueError(f"No inner cavity faces detected in STEP: {path}")

        inner_shape = build_compound(inner_faces)
        center = (
            0.5 * (bbox.min_x_mm + bbox.max_x_mm),
            0.5 * (bbox.min_y_mm + bbox.max_y_mm),
            0.5 * (bbox.min_z_mm + bbox.max_z_mm),
        )
        profile_contour, top_contour = extract_inner_contours(
            inner_shape,
            bounding_box_center=center,
        )

        _LOG.info(
            "[cad_reference] model=%s inner_faces=%d outer_faces=%d profile_edges=%d top_edges=%d",
            model_id,
            len(inner_faces),
            len(outer_faces),
            profile_contour.edge_count,
            top_contour.edge_count,
        )

        return CadReferenceGeometry(
            model_id=model_id,
            step_path=str(path),
            step_sha256=step_file_sha256(path),
            bounding_box=bbox,
            inner_face_count=len(inner_faces),
            outer_face_count=len(outer_faces),
            profile_contour=profile_contour,
            top_contour=top_contour,
            metadata={
                "builder": "CadReferenceGeometryBuilder",
                "source": "opencascade_brep",
            },
        )
