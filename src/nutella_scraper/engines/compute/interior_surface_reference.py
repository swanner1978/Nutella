"""Interior product-surface mesh — same geometry Contour intérieur visualizes.

RGB(85,255,255) is only the FreeCAD/viewer identification marker. Scraper math
uses the resulting triangle mesh (vertices/faces), never colour channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray

from nutella_scraper.cad_import.step_face_color_diagnostics import (
    TARGET_RGB_255,
    TargetFaceMesh,
    extract_target_face_mesh,
)

# Provenance: same TargetFaceMesh pipeline as Contour intérieur (viewer).
SOURCE_INTERIOR_PRODUCT_SURFACE = "interior_product_surface_mesh"
CACHE_NAME = "interior_product_surface.npz"


@dataclass(frozen=True)
class InteriorSurfaceReference:
    """
    Triangle mesh of the jar interior product surface.

    Loaded via the existing Contour intérieur extractor (face selection already
    implemented there). Downstream code consumes ``vertices`` / ``faces`` only.
    """

    model_id: str
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    matching_face_count: int
    source: str = SOURCE_INTERIOR_PRODUCT_SURFACE
    # Identification metadata only (not used in geometric calculations).
    identification_rgb_255: tuple[int, int, int] = TARGET_RGB_255

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def face_count(self) -> int:
        return int(len(self.faces))

    @property
    def y_min_mm(self) -> float:
        return float(np.min(self.vertices[:, 1])) if len(self.vertices) else 0.0

    @property
    def y_max_mm(self) -> float:
        return float(np.max(self.vertices[:, 1])) if len(self.vertices) else 0.0

    # Back-compat alias used by earlier scraper payloads.
    @property
    def target_rgb_255(self) -> tuple[int, int, int]:
        return self.identification_rgb_255

    def to_trimesh(self) -> trimesh.Trimesh:
        if self.vertex_count == 0 or self.face_count == 0:
            raise ValueError("InteriorSurfaceReference has no triangles")
        return trimesh.Trimesh(
            vertices=np.asarray(self.vertices, dtype=np.float64),
            faces=np.asarray(self.faces, dtype=np.int64),
            process=False,
        )

    @classmethod
    def from_target_face_mesh(
        cls,
        target: TargetFaceMesh,
        *,
        model_id: str,
    ) -> InteriorSurfaceReference:
        vertices = np.asarray(target.vertices, dtype=np.float64)
        faces = np.asarray(target.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            raise ValueError(
                "Interior product surface mesh is empty "
                "(Contour intérieur TargetFaceMesh has no triangles)"
            )
        return cls(
            model_id=model_id,
            vertices=vertices,
            faces=faces,
            matching_face_count=int(target.diagnostic.matching_face_count),
            source=SOURCE_INTERIOR_PRODUCT_SURFACE,
            identification_rgb_255=tuple(
                int(c) for c in target.diagnostic.target_rgb_255
            ),
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        model_id: str,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        matching_face_count: int = 0,
        source: str = SOURCE_INTERIOR_PRODUCT_SURFACE,
        identification_rgb_255: tuple[int, int, int] = TARGET_RGB_255,
    ) -> InteriorSurfaceReference:
        return cls(
            model_id=model_id,
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            matching_face_count=matching_face_count,
            source=source,
            identification_rgb_255=identification_rgb_255,
        )


def load_interior_surface_reference(
    *,
    models_root: Path,
    model_id: str,
    step_path: Path | None = None,
    use_cache: bool = True,
) -> InteriorSurfaceReference:
    """
    Load the same interior mesh Contour intérieur projects.

    Calls ``extract_target_face_mesh`` with the same defaults as
    ``build_interior_contour_response`` (no alternate tessellation).
    """
    model_dir = Path(models_root) / model_id
    cache_path = model_dir / CACHE_NAME
    if use_cache and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        rgb = tuple(int(c) for c in payload["identification_rgb_255"].tolist())
        return InteriorSurfaceReference.from_arrays(
            model_id=model_id,
            vertices=np.asarray(payload["vertices"], dtype=np.float64),
            faces=np.asarray(payload["faces"], dtype=np.int64),
            matching_face_count=int(payload["matching_face_count"][0]),
            source=SOURCE_INTERIOR_PRODUCT_SURFACE,
            identification_rgb_255=rgb,
        )

    resolved_step = Path(step_path) if step_path is not None else model_dir / "reference.step"
    if not resolved_step.exists():
        raise FileNotFoundError(
            f"reference.step introuvable pour le modèle {model_id} ({resolved_step})"
        )
    # Identical entry point + defaults as Contour intérieur visualization.
    target = extract_target_face_mesh(resolved_step)
    reference = InteriorSurfaceReference.from_target_face_mesh(target, model_id=model_id)
    if use_cache:
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            vertices=reference.vertices,
            faces=reference.faces,
            matching_face_count=np.asarray(
                [reference.matching_face_count], dtype=np.int64
            ),
            identification_rgb_255=np.asarray(
                reference.identification_rgb_255, dtype=np.int64
            ),
        )
    return reference


# Back-compat alias for imports that still use the old constant name.
SOURCE_STEP_FACE_COLOR = SOURCE_INTERIOR_PRODUCT_SURFACE
