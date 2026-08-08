"""Visualization-only helpers for demo viewer (no business logic)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

# Viewport defaults — must match demo ViewProjectionConfig
VIEW_WIDTH = 900
VIEW_HEIGHT = 650
VIEW_PADDING = 48

# CAD orthographic conventions (visualization only), Y-up jar frame:
# - profile view: projection on XY, camera looks along ±Z (height vs width)
# - top view: projection on XZ, camera looks along ±Y (footprint)
VIEW_CONVENTIONS: dict[str, dict[str, str]] = {
    "side": {
        "plane": "XY",
        "view_axis": "Z",
        "label_fr": "Vue de profil",
        "label_en": "Profile View",
        "dimension_axes": "X × Y",
    },
    "top": {
        "plane": "XZ",
        "view_axis": "Y",
        "label_fr": "Vue de dessus",
        "label_en": "Top View",
        "dimension_axes": "X × Z",
    },
}


@dataclass(frozen=True)
class ProjectionLayers:
    """SVG fragments derived exclusively from one canonical mesh."""

    contour: str
    wireframe: str
    vertices: str
    bounding_box: str
    principal_axes: str


def fit_to_viewport(
    coords: np.ndarray,
    width: int = VIEW_WIDTH,
    height: int = VIEW_HEIGHT,
    padding: int = VIEW_PADDING,
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


def projected_extent_mm(extents_mm: tuple[float, float, float], plane: str) -> tuple[float, float]:
    """Return the two world-axis spans visible in an orthographic projection."""
    dx, dy, dz = extents_mm
    if plane == "XZ":
        return dx, dz
    if plane == "XY":
        return dx, dy
    raise ValueError(f"Unsupported projection plane: {plane}")


def displayed_view_entry(
    *,
    view_name: str,
    filename: str,
    sha256: str,
    canonical_mesh_sha256: str,
) -> dict[str, str]:
    """Manifest entry for one orthographic view."""
    spec = VIEW_CONVENTIONS[view_name]
    return {
        "view_name": view_name,
        "plane": spec["plane"],
        "view_axis": spec["view_axis"],
        "label_fr": spec["label_fr"],
        "label_en": spec["label_en"],
        "dimension_axes": spec["dimension_axes"],
        "filename": filename,
        "sha256": sha256,
        "canonical_mesh_sha256": canonical_mesh_sha256,
    }


def build_cad_reference_projection_svg(
    geometry: object,
    *,
    plane: str,
    model_id: str,
    canonical_mesh_sha256: str,
    clearance_mm: float = 0.0,
) -> str:
    """Build user-facing reference views from CadReferenceGeometry B-Rep — no mesh."""
    from nutella_scraper.domain.models.cad_reference_geometry import CadReferenceGeometry
    from nutella_scraper.engines.visualization.cad_reference_projector import cad_reference_projection

    if not isinstance(geometry, CadReferenceGeometry):
        raise TypeError("geometry must be CadReferenceGeometry")

    contour = cad_reference_projection(
        geometry,
        plane=plane,
        clearance_mm=clearance_mm,
    )
    layers = ProjectionLayers(
        contour=contour,
        wireframe="",
        vertices="",
        bounding_box="",
        principal_axes="",
    )
    if plane == "TOP_XZ":
        view_axis_label = "Y"
    elif plane == "PROFILE":
        view_axis_label = "Y"
    elif plane == VIEW_CONVENTIONS["top"]["plane"]:
        view_axis_label = VIEW_CONVENTIONS["top"]["view_axis"]
    elif plane == VIEW_CONVENTIONS["side"]["plane"]:
        view_axis_label = VIEW_CONVENTIONS["side"]["view_axis"]
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")

    return _projection_document(
        layers,
        model_id=model_id,
        canonical_mesh_sha256=canonical_mesh_sha256,
        plane=plane,
        view_axis=view_axis_label,
    )


def build_analytical_projection_svg(
    profile: object,
    *,
    plane: str,
    model_id: str,
    canonical_mesh_sha256: str,
    debug_mesh: trimesh.Trimesh | None = None,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    principal_axes: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
) -> str:
    """Build user-facing views from InternalJarProfile — wireframe optional for debug."""
    from nutella_scraper.domain.models.internal_jar_profile import InternalJarProfile
    from nutella_scraper.engines.visualization.profile_projector import profile_contour_svg_fragment

    if not isinstance(profile, InternalJarProfile):
        raise TypeError("profile must be InternalJarProfile")

    contour = profile_contour_svg_fragment(profile, plane=plane)
    wireframe = ""
    vertices = ""
    bounding_box = ""
    principal_axes_svg = ""
    if debug_mesh is not None:
        vertices_np = np.asarray(debug_mesh.vertices, dtype=np.float64)
        coords, component_indices, view_axis_index = _project_vertices(vertices_np, plane)
        scale, offset = fit_to_viewport(coords)
        wireframe_edges = np.asarray(debug_mesh.edges_unique, dtype=np.int64)
        wireframe = _segments_path(
            coords[wireframe_edges],
            scale,
            offset,
            css_class="wireframe",
        )
        vertices = _vertices_path(coords, scale, offset)
        bounding_box = _bounding_box_path(coords, scale, offset)
        principal_axes_svg = _principal_axes_svg(
            center=np.asarray(center, dtype=np.float64),
            axes=np.asarray(principal_axes, dtype=np.float64),
            component_indices=component_indices,
            model_extent=float(np.max(debug_mesh.extents)),
            scale=scale,
            offset=offset,
        )

    layers = ProjectionLayers(
        contour=contour,
        wireframe=wireframe,
        vertices=vertices,
        bounding_box=bounding_box,
        principal_axes=principal_axes_svg,
    )
    view_axis_label = "Y"
    if plane in ("XZ", "TOP_XZ"):
        view_axis_label = "Y"
    elif plane == "XY":
        view_axis_label = "Z"
    elif plane == VIEW_CONVENTIONS["side"]["plane"]:
        view_axis_label = VIEW_CONVENTIONS["side"]["view_axis"]
    elif plane == VIEW_CONVENTIONS["top"]["plane"]:
        view_axis_label = VIEW_CONVENTIONS["top"]["view_axis"]
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")

    return _projection_document(
        layers,
        model_id=model_id,
        canonical_mesh_sha256=canonical_mesh_sha256,
        plane=plane,
        view_axis=view_axis_label,
    )


def build_projection_svg(
    mesh: trimesh.Trimesh,
    *,
    plane: str,
    center: tuple[float, float, float],
    principal_axes: tuple[tuple[float, float, float], ...],
    model_id: str,
    canonical_mesh_sha256: str,
) -> str:
    """Build independently toggleable 2D layers from a canonical 3D mesh."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    coords, component_indices, view_axis_index = _project_vertices(vertices, plane)
    scale, offset = fit_to_viewport(coords)

    contour_edges = _silhouette_edges(mesh, view_axis_index)
    wireframe_edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    layers = ProjectionLayers(
        contour=_segments_path(
            coords[contour_edges],
            scale,
            offset,
            css_class="contour",
        ),
        wireframe=_segments_path(
            coords[wireframe_edges],
            scale,
            offset,
            css_class="wireframe",
        ),
        vertices=_vertices_path(coords, scale, offset),
        bounding_box=_bounding_box_path(coords, scale, offset),
        principal_axes=_principal_axes_svg(
            center=np.asarray(center, dtype=np.float64),
            axes=np.asarray(principal_axes, dtype=np.float64),
            component_indices=component_indices,
            model_extent=float(np.max(mesh.extents)),
            scale=scale,
            offset=offset,
        ),
    )
    view_axis_label = "Y"
    if plane in ("XZ", "TOP_XZ"):
        view_axis_label = "Y"
    elif plane == "XY":
        view_axis_label = "Z"
    elif plane == VIEW_CONVENTIONS["side"]["plane"]:
        view_axis_label = VIEW_CONVENTIONS["side"]["view_axis"]
    elif plane == VIEW_CONVENTIONS["top"]["plane"]:
        view_axis_label = VIEW_CONVENTIONS["top"]["view_axis"]
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")

    return _projection_document(
        layers,
        model_id=model_id,
        canonical_mesh_sha256=canonical_mesh_sha256,
        plane=plane,
        view_axis=view_axis_label,
    )


def _project_vertices(
    vertices: np.ndarray,
    plane: str,
) -> tuple[np.ndarray, tuple[int, int], int]:
    if plane in ("XZ", "TOP_XZ"):
        indices = (0, 2)
        view_axis = 1
    elif plane == "XY":
        indices = (0, 1)
        view_axis = 2
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")
    return vertices[:, indices], indices, view_axis


def _silhouette_edges(mesh: trimesh.Trimesh, view_axis: int) -> np.ndarray:
    """
    Return boundary/frontier edges for an orthographic view.

    An edge belongs to the contour when it is open, or when at least one
    adjacent face points toward the camera and another does not.
    """
    edges_sorted = np.asarray(mesh.edges_sorted, dtype=np.int64)
    unique_edges, inverse = np.unique(edges_sorted, axis=0, return_inverse=True)
    face_ids = np.repeat(np.arange(len(mesh.faces), dtype=np.int64), 3)
    front_facing = np.asarray(mesh.face_normals[:, view_axis] > 1e-9, dtype=np.int8)

    counts = np.bincount(inverse, minlength=len(unique_edges))
    has_front = np.zeros(len(unique_edges), dtype=np.int8)
    has_other = np.zeros(len(unique_edges), dtype=np.int8)
    np.maximum.at(has_front, inverse, front_facing[face_ids])
    np.maximum.at(has_other, inverse, 1 - front_facing[face_ids])

    is_contour = (counts == 1) | ((has_front == 1) & (has_other == 1))
    return unique_edges[is_contour]


def _segments_path(
    segments: np.ndarray,
    scale: float,
    offset: np.ndarray,
    *,
    css_class: str,
) -> str:
    if len(segments) == 0:
        return ""
    projected = segments * scale + offset
    rounded = np.round(projected, 2)
    non_degenerate = np.any(rounded[:, 0] != rounded[:, 1], axis=1)
    rounded = rounded[non_degenerate]
    if len(rounded) == 0:
        return ""

    canonical = np.sort(rounded, axis=1)
    _, unique_indices = np.unique(canonical.reshape(len(canonical), 4), axis=0, return_index=True)
    rounded = rounded[np.sort(unique_indices)]
    commands = " ".join(f"M{p0[0]:.2f},{p0[1]:.2f}L{p1[0]:.2f},{p1[1]:.2f}" for p0, p1 in rounded)
    return f'<path class="{css_class}" d="{commands}"/>'


def _vertices_path(coords: np.ndarray, scale: float, offset: np.ndarray) -> str:
    projected = np.round(coords * scale + offset, 2)
    projected = np.unique(projected, axis=0)
    commands = " ".join(
        f"M{x - 1.25:.2f},{y:.2f}h2.5M{x:.2f},{y - 1.25:.2f}v2.5" for x, y in projected
    )
    return f'<path class="vertices" d="{commands}"/>'


def _bounding_box_path(coords: np.ndarray, scale: float, offset: np.ndarray) -> str:
    lo = coords.min(axis=0) * scale + offset
    hi = coords.max(axis=0) * scale + offset
    width, height = hi - lo
    return (
        f'<rect class="bounding-box" x="{lo[0]:.2f}" y="{lo[1]:.2f}" '
        f'width="{width:.2f}" height="{height:.2f}"/>'
    )


def _principal_axes_svg(
    *,
    center: np.ndarray,
    axes: np.ndarray,
    component_indices: tuple[int, int],
    model_extent: float,
    scale: float,
    offset: np.ndarray,
) -> str:
    colors = ("#ff5252", "#66ff66", "#4d8dff")
    fragments: list[str] = []
    half_length = max(model_extent * 0.3, 1.0)
    for index, (axis, color) in enumerate(zip(axes, colors, strict=False), start=1):
        endpoints = np.vstack([center - axis * half_length, center + axis * half_length])
        projected = endpoints[:, component_indices] * scale + offset
        fragments.append(
            f'<line class="principal-axis" data-axis="{index}" '
            f'x1="{projected[0, 0]:.2f}" y1="{projected[0, 1]:.2f}" '
            f'x2="{projected[1, 0]:.2f}" y2="{projected[1, 1]:.2f}" '
            f'stroke="{color}"/>'
        )
    return "".join(fragments)


def _projection_document(
    layers: ProjectionLayers,
    *,
    model_id: str,
    canonical_mesh_sha256: str,
    plane: str,
    view_axis: str,
) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{VIEW_WIDTH}" height="{VIEW_HEIGHT}" '
        f'viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}" '
        f'data-model-id="{model_id}" '
        f'data-canonical-mesh-sha256="{canonical_mesh_sha256}" '
        f'data-plane="{plane}" '
        f'data-view-axis="{view_axis}" '
        f'preserveAspectRatio="xMidYMid meet">'
        "<style>"
        ".contour{fill:none;stroke:#fff;stroke-width:2;stroke-linejoin:round;"
        "stroke-linecap:round;vector-effect:non-scaling-stroke}"
        ".interior-profile-contour{fill:none;stroke:#a855f7;stroke-width:1.1;"
        "stroke-linejoin:round;stroke-linecap:round;vector-effect:non-scaling-stroke}"
        ".wireframe{fill:none;stroke:#777;stroke-width:.65;opacity:.55;"
        "vector-effect:non-scaling-stroke}"
        ".vertices{fill:none;stroke:#55dfff;stroke-width:1;vector-effect:non-scaling-stroke}"
        ".bounding-box{fill:none;stroke:#ffb74d;stroke-width:1.5;stroke-dasharray:7 5;"
        "vector-effect:non-scaling-stroke}"
        ".principal-axis{stroke-width:2;vector-effect:non-scaling-stroke}"
        "</style>"
        '<rect width="100%" height="100%" fill="#000"/>'
        f'<g data-layer="wireframe" style="display:none">{layers.wireframe}</g>'
        f'<g data-layer="vertices" style="display:none">{layers.vertices}</g>'
        f'<g data-layer="bounding-box" style="display:none">{layers.bounding_box}</g>'
        f'<g data-layer="principal-axes" style="display:none">{layers.principal_axes}</g>'
        f'<g data-layer="contour">{layers.contour}</g>'
        "</svg>"
    )


def load_jar_profile(jar_json: Path) -> dict[str, object]:
    with jar_json.open(encoding="utf-8") as f:
        return json.load(f)


def jar_profile_xz_coords(jar: dict[str, object]) -> np.ndarray:
    """Meridian inner wall in XZ (x = radius, y = z)."""
    profile = jar["meridian_profile"]
    assert isinstance(profile, list)
    points: list[list[float]] = []
    for pt in profile:
        assert isinstance(pt, dict)
        z = float(pt["z_mm"])
        r = float(pt["r_mm"])
        points.append([r, z])
        points.append([-r, z])
    return np.array(points, dtype=np.float64)


def jar_top_xy_coords(jar: dict[str, object]) -> np.ndarray:
    """Rings for top view: outer body + neck opening."""
    profile = jar["meridian_profile"]
    assert isinstance(profile, list)
    max_r = max(float(pt["r_mm"]) for pt in profile if isinstance(pt, dict))
    neck_r = float(jar["neck_inner_diameter_mm"]) / 2.0
    t = np.linspace(0, 2 * np.pi, 64)
    outer = np.column_stack([max_r * np.cos(t), max_r * np.sin(t)])
    inner = np.column_stack([neck_r * np.cos(t), neck_r * np.sin(t)])
    return np.vstack([outer, inner])


def _line_segments(coords: np.ndarray, scale: float, offset: np.ndarray) -> str:
    lines: list[str] = []
    for i in range(len(coords) - 1):
        p0 = coords[i] * scale + offset
        p1 = coords[i + 1] * scale + offset
        lines.append(
            f'<line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" '
            f'x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" '
            f'stroke="#ffffff" stroke-width="1.5" vector-effect="non-scaling-stroke"/>'
        )
    return "".join(lines)


def jar_profile_svg(jar: dict[str, object], combined_coords: np.ndarray) -> str:
    """Jar meridian outline on profile view (visualization only)."""
    profile = jar["meridian_profile"]
    assert isinstance(profile, list)
    scale, offset = fit_to_viewport(combined_coords)

    right = np.array([[float(pt["r_mm"]), float(pt["z_mm"])] for pt in profile])
    left = np.array([[-float(pt["r_mm"]), float(pt["z_mm"])] for pt in reversed(profile)])

    segs = _line_segments(right, scale, offset)
    segs += _line_segments(left, scale, offset)
    # close bottom
    if len(right) > 0:
        p0 = right[0] * scale + offset
        p1 = left[-1] * scale + offset
        segs += (
            f'<line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" '
            f'x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" '
            f'stroke="#ffffff" stroke-width="1.5" vector-effect="non-scaling-stroke"/>'
        )

    return segs


def jar_top_svg(jar: dict[str, object], combined_coords: np.ndarray) -> str:
    """Jar top view circles (visualization only)."""
    scale, offset = fit_to_viewport(combined_coords)
    profile = jar["meridian_profile"]
    assert isinstance(profile, list)
    max_r = max(float(pt["r_mm"]) for pt in profile if isinstance(pt, dict))
    neck_r = float(jar["neck_inner_diameter_mm"]) / 2.0
    cx, cy = offset  # origin at 0,0 in model space
    # Center jar at origin in XY
    center = np.array([0.0, 0.0]) * scale + offset
    outer_r = max_r * scale
    neck_r_scaled = neck_r * scale
    return (
        f'<circle cx="{center[0]:.2f}" cy="{center[1]:.2f}" r="{outer_r:.2f}" '
        f'fill="none" stroke="#ffffff" stroke-width="1.5" vector-effect="non-scaling-stroke"/>'
        f'<circle cx="{center[0]:.2f}" cy="{center[1]:.2f}" r="{neck_r_scaled:.2f}" '
        f'fill="none" stroke="#ffffff" stroke-width="1.5" stroke-dasharray="4 3" '
        f'vector-effect="non-scaling-stroke"/>'
    )


def scraper_lines_from_coords(
    edge_coords: np.ndarray,
    combined_coords: np.ndarray,
) -> str:
    """Draw scraper edges with unified viewport."""
    scale, offset = fit_to_viewport(combined_coords)
    lines: list[str] = []
    for p0, p1 in edge_coords:
        a = p0 * scale + offset
        b = p1 * scale + offset
        lines.append(
            f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" '
            f'x2="{b[0]:.2f}" y2="{b[1]:.2f}" '
            f'stroke="#ffffff" stroke-width="1" opacity="0.9" vector-effect="non-scaling-stroke"/>'
        )
    return "".join(lines)


def build_composite_svg(
    jar_layer: str,
    scraper_layer: str,
    *,
    model_id: str,
    canonical_mesh_sha256: str,
) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{VIEW_WIDTH}" height="{VIEW_HEIGHT}" '
        f'viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}" '
        f'data-model-id="{model_id}" '
        f'data-canonical-mesh-sha256="{canonical_mesh_sha256}" '
        f'preserveAspectRatio="xMidYMid meet">'
        f'<rect width="100%" height="100%" fill="#000000"/>'
        f'<g id="jar-layer">{jar_layer}</g>'
        f'<g id="scraper-layer">{scraper_layer}</g>'
        f"</svg>"
    )


def scraper_edge_coords_xz(
    vertices: np.ndarray, edges: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    coords = vertices[:, [0, 2]]
    return [(coords[e0], coords[e1]) for e0, e1 in edges]


def scraper_edge_coords_xy(
    vertices: np.ndarray, edges: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    coords = vertices[:, [0, 1]]
    return [(coords[e0], coords[e1]) for e0, e1 in edges]
