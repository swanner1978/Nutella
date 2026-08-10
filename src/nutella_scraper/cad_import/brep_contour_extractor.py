"""Extract exact 2D interior contours from inner B-Rep faces (OpenCascade)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nutella_scraper.domain.models.cad_reference_geometry import (
    CadProjectedContour,
    ProjectedPolyline2D,
)

# Same orthographic planes as VIEW_CONVENTIONS / build_projection_svg (Y-up jar).
PLANE_PROFILE = "XY"
PLANE_TOP_XZ = "XZ"

DEFAULT_DEFLECTION_MM = 0.05


@dataclass(frozen=True)
class _ViewSpec:
    plane: str
    view_axis: str


def extract_inner_contours(
    inner_shape: object,
    *,
    bounding_box_center: tuple[float, float, float],
    deflection_mm: float = DEFAULT_DEFLECTION_MM,
) -> tuple[CadProjectedContour, CadProjectedContour]:
    """
    Extract interior frontiers in the same planes as the mesh views.

    - Profile (XY, view along Z): Z = 0 section of the inner faces → (X, Y).
    - Top (XZ, view along Y): free rim wire of the cavity opening → (X, Z).
    """
    del bounding_box_center  # reserved for future HLR; rim/section use world axes
    profile = _section_profile_xy_contour(inner_shape, deflection_mm=deflection_mm)
    top = _rim_top_xz_contour(inner_shape, deflection_mm=deflection_mm)
    return profile, top


def _section_profile_xy_contour(
    shape: object,
    *,
    deflection_mm: float,
) -> CadProjectedContour:
    """Inner cavity wall cut by the mid-plane Z = 0, stored in XY (same as Vue de profil)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

    plane = gp_Pln(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
    section = BRepAlgoAPI_Section(shape, plane)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    section.Build()
    section_shape = section.Shape()
    if section_shape.IsNull():
        return CadProjectedContour(
            plane=PLANE_PROFILE,
            view_axis="Z",
            polylines=(),
            edge_count=0,
            source="opencascade_brep_section_xy",
        )

    segments: list[list[tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    edge_count = 0
    explorer = TopExp_Explorer(section_shape, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        points_3d = _discretize_edge(edge, deflection_mm=deflection_mm)
        segment = [(x_mm, y_mm) for x_mm, y_mm, _z_mm in points_3d]
        if len(segment) >= 2:
            key = tuple(segment)
            if key not in seen:
                seen.add(key)
                segments.append(list(segment))
        edge_count += 1
        explorer.Next()

    chains = _stitch_polyline_segments(segments)
    polylines = [
        ProjectedPolyline2D(points_mm=tuple(chain), is_closed=_chain_is_closed(chain))
        for chain in chains
        if len(chain) >= 2
    ]

    return CadProjectedContour(
        plane=PLANE_PROFILE,
        view_axis="Z",
        polylines=tuple(polylines),
        edge_count=edge_count,
        source="opencascade_brep_section_xy",
    )


def _rim_top_xz_contour(
    shape: object,
    *,
    deflection_mm: float,
) -> CadProjectedContour:
    """
    Cavity opening frontier: the free closed wire of the inner faces with highest Y.

    Projected to XZ — same plane as Vue de dessus.
    """
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    free_bounds = ShapeAnalysis_FreeBounds(shape, True, True)
    closed = free_bounds.GetClosedWires()
    if closed is None or closed.IsNull():
        return CadProjectedContour(
            plane=PLANE_TOP_XZ,
            view_axis="Y",
            polylines=(),
            edge_count=0,
            source="opencascade_brep_rim_xz",
        )

    best_wire: object | None = None
    best_mean_y = float("-inf")
    wire_explorer = TopExp_Explorer(closed, TopAbs_WIRE)
    while wire_explorer.More():
        wire = TopoDS.Wire_s(wire_explorer.Current())
        samples = _sample_wire_points(wire, deflection_mm=deflection_mm)
        if samples:
            mean_y = sum(point[1] for point in samples) / len(samples)
            if mean_y > best_mean_y:
                best_mean_y = mean_y
                best_wire = wire
        wire_explorer.Next()

    if best_wire is None:
        return CadProjectedContour(
            plane=PLANE_TOP_XZ,
            view_axis="Y",
            polylines=(),
            edge_count=0,
            source="opencascade_brep_rim_xz",
        )

    segments: list[list[tuple[float, float]]] = []
    edge_count = 0
    edge_explorer = TopExp_Explorer(best_wire, TopAbs_EDGE)
    while edge_explorer.More():
        edge = TopoDS.Edge_s(edge_explorer.Current())
        points_3d = _discretize_edge(edge, deflection_mm=deflection_mm)
        segment = [(x_mm, z_mm) for x_mm, _y_mm, z_mm in points_3d]
        if len(segment) >= 2:
            segments.append(segment)
        edge_count += 1
        edge_explorer.Next()

    chains = _stitch_polyline_segments(segments)
    polylines = [
        ProjectedPolyline2D(points_mm=tuple(chain), is_closed=True)
        for chain in chains
        if len(chain) >= 3
    ]
    if not polylines and chains:
        polylines = [
            ProjectedPolyline2D(
                points_mm=tuple(chains[0]),
                is_closed=_chain_is_closed(chains[0]),
            )
        ]

    return CadProjectedContour(
        plane=PLANE_TOP_XZ,
        view_axis="Y",
        polylines=tuple(polylines),
        edge_count=edge_count,
        source="opencascade_brep_rim_xz",
    )


def _sample_wire_points(
    wire: object,
    *,
    deflection_mm: float,
) -> list[tuple[float, float, float]]:
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    points: list[tuple[float, float, float]] = []
    explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        points.extend(_discretize_edge(edge, deflection_mm=deflection_mm))
        explorer.Next()
    return points


def _stitch_polyline_segments(
    segments: list[list[tuple[float, float]]],
    *,
    tolerance_mm: float = 0.08,
) -> list[list[tuple[float, float]]]:
    """Connect line segments that share endpoints into maximal chains."""
    if not segments:
        return []

    def close(
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> bool:
        return (
            abs(left[0] - right[0]) <= tolerance_mm
            and abs(left[1] - right[1]) <= tolerance_mm
        )

    unused = [list(segment) for segment in segments if len(segment) >= 2]
    chains: list[list[tuple[float, float]]] = []

    while unused:
        chain = unused.pop(0)
        changed = True
        while changed:
            changed = False
            for index, segment in enumerate(unused):
                if close(chain[-1], segment[0]):
                    chain.extend(segment[1:])
                    unused.pop(index)
                    changed = True
                    break
                if close(chain[-1], segment[-1]):
                    chain.extend(reversed(segment[:-1]))
                    unused.pop(index)
                    changed = True
                    break
                if close(chain[0], segment[-1]):
                    chain = segment[:-1] + chain
                    unused.pop(index)
                    changed = True
                    break
                if close(chain[0], segment[0]):
                    chain = list(reversed(segment[1:])) + chain
                    unused.pop(index)
                    changed = True
                    break
        chains.append(_dedupe_points(chain))

    return chains


def _chain_is_closed(chain: list[tuple[float, float]]) -> bool:
    if len(chain) < 3:
        return False
    return _points_close(chain[0], chain[-1])


def _discretize_edge(edge: object, *, deflection_mm: float) -> list[tuple[float, float, float]]:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection

    curve = BRepAdaptor_Curve(edge)
    disc = GCPnts_QuasiUniformDeflection(curve, deflection_mm)
    if not disc.IsDone() or disc.NbPoints() < 2:
        p0 = curve.Value(curve.FirstParameter())
        p1 = curve.Value(curve.LastParameter())
        return [(p0.X(), p0.Y(), p0.Z()), (p1.X(), p1.Y(), p1.Z())]

    points: list[tuple[float, float, float]] = []
    for index in range(1, disc.NbPoints() + 1):
        point = disc.Value(index)
        points.append((float(point.X()), float(point.Y()), float(point.Z())))
    return points


def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return points
    deduped = [points[0]]
    for point in points[1:]:
        if not _points_close(point, deduped[-1]):
            deduped.append(point)
    return deduped


def _points_close(
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    tolerance_mm: float = 0.02,
) -> bool:
    return abs(left[0] - right[0]) <= tolerance_mm and abs(left[1] - right[1]) <= tolerance_mm


def offset_contour(
    contour: CadProjectedContour,
    offset_mm: float,
) -> CadProjectedContour:
    """Radial inward offset for envelope display — preserves CAD topology."""
    if offset_mm <= 0.0:
        return contour

    offset_polylines: list[ProjectedPolyline2D] = []
    for polyline in contour.polylines:
        if contour.plane == PLANE_TOP_XZ:
            points = _offset_radial_xz_points(polyline.points_mm, offset_mm)
        elif contour.plane == PLANE_PROFILE:
            points = _offset_profile_xy_points(polyline.points_mm, offset_mm)
        else:
            points = list(polyline.points_mm)
        if len(points) >= 2:
            offset_polylines.append(
                ProjectedPolyline2D(
                    points_mm=tuple(points),
                    is_closed=polyline.is_closed,
                )
            )

    return CadProjectedContour(
        plane=contour.plane,
        view_axis=contour.view_axis,
        polylines=tuple(offset_polylines),
        edge_count=contour.edge_count,
        source=f"{contour.source}+offset_{offset_mm:.3f}mm",
    )


def _offset_radial_xz_points(
    points: tuple[tuple[float, float], ...],
    offset_mm: float,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for x_mm, z_mm in points:
        radius = math.hypot(x_mm, z_mm)
        if radius <= 1e-9:
            result.append((0.0, 0.0))
            continue
        scale = max(radius - offset_mm, 0.0) / radius
        result.append((x_mm * scale, z_mm * scale))
    return result


def _offset_profile_xy_points(
    points: tuple[tuple[float, float], ...],
    offset_mm: float,
) -> list[tuple[float, float]]:
    """Move profile wall points toward the Y axis (inward along X)."""
    result: list[tuple[float, float]] = []
    for x_mm, y_mm in points:
        sign = 1.0 if x_mm >= 0.0 else -1.0
        adjusted = sign * max(abs(x_mm) - offset_mm, 0.0)
        result.append((adjusted, y_mm))
    return result
