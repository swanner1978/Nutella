"""View projection generator — visualization only, never used for computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from nutella_scraper.domain.models.canonical import CanonicalModel3D, MeshData
from nutella_scraper.domain.models.views import (
    ProjectedView,
    ProjectionMetadata,
    ViewProjectionCache,
)


@dataclass(frozen=True)
class ViewProjectionConfig:
    """Rendering parameters for 2D projection views."""

    width_px: int = 800
    height_px: int = 600
    padding_px: int = 40
    stroke_color: str = "#cccccc"
    fill_color: str = "#666666"
    fill_opacity: float = 0.35


class ViewProjectionGenerator:
    """
    Generates profile (XY) and top (XZ) views from CanonicalModel3D (Y-up jar).

    @visualization_only — output must never feed ContactSimulator or OptimizationEngine.
    """

    def __init__(self, config: ViewProjectionConfig | None = None) -> None:
        self._config = config or ViewProjectionConfig()

    def generate(self, model: CanonicalModel3D) -> ViewProjectionCache:
        mesh = _mesh_data_to_trimesh(model.mesh)
        profile_svg = self._render_projection(mesh, plane="XY")
        top_svg = self._render_projection(mesh, plane="XZ")

        cfg = self._config
        profile_meta = ProjectionMetadata(
            plane="XY",
            camera={"origin": "side", "view_axis": "Z", "up": "Y"},
            scale=self._compute_scale(mesh, plane="XY"),
            width_px=cfg.width_px,
            height_px=cfg.height_px,
        )
        top_meta = ProjectionMetadata(
            plane="XZ",
            camera={"origin": "top", "view_axis": "Y", "up": "Z"},
            scale=self._compute_scale(mesh, plane="XZ"),
            width_px=cfg.width_px,
            height_px=cfg.height_px,
        )

        return ViewProjectionCache(
            model_id=model.id,
            profile_view=ProjectedView(
                plane="XY",
                asset_path=None,
                svg_content=profile_svg,
                metadata=profile_meta,
            ),
            top_view=ProjectedView(
                plane="XZ",
                asset_path=None,
                svg_content=top_svg,
                metadata=top_meta,
            ),
            projection_metadata={
                "visualization_only": True,
                "source_model_id": model.id,
                "note": "Not for use in simulation or optimization",
            },
        )

    def _render_projection(self, mesh: trimesh.Trimesh, plane: str) -> str:
        cfg = self._config
        edges = mesh.edges_unique
        vertices = np.asarray(mesh.vertices, dtype=np.float64)

        if plane == "XZ":
            coords = vertices[:, [0, 2]]
        elif plane == "XY":
            coords = vertices[:, [0, 1]]
        else:
            raise ValueError(f"Unsupported projection plane: {plane}")

        scale, offset = _fit_to_viewport(
            coords,
            width=cfg.width_px,
            height=cfg.height_px,
            padding=cfg.padding_px,
        )

        lines: list[str] = []
        for e0, e1 in edges:
            p0 = coords[e0] * scale + offset
            p1 = coords[e1] * scale + offset
            lines.append(
                f'<line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" '
                f'x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" '
                f'stroke="{cfg.stroke_color}" stroke-width="1"/>'
            )

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{cfg.width_px}" height="{cfg.height_px}" '
            f'viewBox="0 0 {cfg.width_px} {cfg.height_px}">'
            f'<g>{"".join(lines)}</g></svg>'
        )

    def _compute_scale(self, mesh: trimesh.Trimesh, plane: str) -> float:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if plane == "XZ":
            coords = vertices[:, [0, 2]]
        else:
            coords = vertices[:, [0, 1]]
        scale, _ = _fit_to_viewport(
            coords,
            width=self._config.width_px,
            height=self._config.height_px,
            padding=self._config.padding_px,
        )
        return float(scale)


def _mesh_data_to_trimesh(mesh_data: MeshData) -> trimesh.Trimesh:
    vertices = np.array(mesh_data.vertices, dtype=np.float64)
    faces = np.array(mesh_data.faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _fit_to_viewport(
    coords: np.ndarray,
    width: int,
    height: int,
    padding: int,
) -> tuple[float, np.ndarray]:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    extent = hi - lo
    extent = np.where(extent == 0, 1.0, extent)

    draw_w = width - 2 * padding
    draw_h = height - 2 * padding
    scale = min(draw_w / extent[0], draw_h / extent[1])

    scaled_extent = extent * scale
    offset = np.array(
        [
            padding + (draw_w - scaled_extent[0]) / 2 - lo[0] * scale,
            padding + (draw_h - scaled_extent[1]) / 2 - lo[1] * scale,
        ],
        dtype=np.float64,
    )
    return scale, offset
