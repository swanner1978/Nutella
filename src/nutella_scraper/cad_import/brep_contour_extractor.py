"""Extract exact 2D contours from inner B-Rep via OpenCascade HLR."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nutella_scraper.domain.models.cad_reference_geometry import (
    CadProjectedContour,
    ProjectedPolyline2D,
)

PLANE_PROFILE = "PROFILE"
PLANE_TOP_XZ = "TOP_XZ"

DEFAULT_DEFLECTION_MM = 0.05


@dataclass(frozen=True)
class _ViewSpec:
    plane: str
    view_axis: str
    direction: tuple[float, float, float]
    center: tuple[float, float, float]


def extract_inner_contours(
    inner_shape: object,
    *,
    bounding_box_center: tuple[float, float, float],
    deflection_mm: float = DEFAULT_DEFLECTION_MM,
) -> tuple[CadProjectedContour, CadProjectedContour]:
    """Project inner B-Rep face edges to profile (R×Y) and top (X×Z) reference planes."""
    profile_spec = _ViewSpec(
        plane=PLANE_PROFILE,
        view_axis="X",
        direction=(1.0, 0.0, 0.0),
        center=bounding_box_center,
    )
    top_spec = _ViewSpec(
        plane=PLANE_TOP_XZ,
        view_axis="Y",
        direction=(0.0, -1.0, 0.0),
        center=bounding_box_center,
    )
    profile = _section_meridian_contour(inner_shape, profile_spec, deflection_mm=deflection_mm)
    top = _direct_edge_contour(inner_shape, top_spec, deflection_mm=deflection_mm)
    return profile, top


def _section_meridian_contour(
    shape: object,
    spec: _ViewSpec,
    *,
    deflection_mm: float,
) -> CadProjectedContour:
    """
    Extract the inner meridian by sectioning the B-Rep with the YZ plane (X = 0).

    Bodies of revolution expose many meridional edges in a side projection; only the
    true meridian section matches the STEP profile contour.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

    plane = gp_Pln(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
    section = BRepAlgoAPI_Section(shape, plane)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    section.Build()
    section_shape = section.Shape()
    if section_shape.IsNull():
        return CadProjectedContour(
            plane=spec.plane,
            view_axis=spec.view_axis,
            polylines=(),
            edge_count=0,
            source="opencascade_brep_section",
        )

    segments: list[list[tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    edge_count = 0
    explorer = TopExp_Explorer(section_shape, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        points_3d = _discretize_edge(edge, deflection_mm=deflection_mm)
        segment = _project_meridian_section_points(points_3d)
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
    polylines = _mirror_meridian_polylines(polylines)

    return CadProjectedContour(
        plane=spec.plane,
        view_axis=spec.view_axis,
        polylines=tuple(polylines),
        edge_count=edge_count,
        source="opencascade_brep_section",
    )


def _project_meridian_section_points(
    points_3d: list[tuple[float, float, float]],
) -> list[tuple[float, float]]:
    """Project a YZ section polyline to the profile plane (R×Y, R ≥ 0)."""
    projected: list[tuple[float, float]] = []
    for x_mm, y_mm, z_mm in points_3d:
        radius = math.hypot(x_mm, z_mm)
        projected.append((radius, y_mm))
    return projected


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


def _mirror_meridian_polylines(
    polylines: list[ProjectedPolyline2D],
) -> list[ProjectedPolyline2D]:
    """Duplicate the meridian on the negative-R side for symmetric profile display."""
    mirrored: list[ProjectedPolyline2D] = []
    for polyline in polylines:
        right = _dedupe_points(list(polyline.points_mm))
        if len(right) < 2:
            continue
        mirrored.append(
            ProjectedPolyline2D(
                points_mm=tuple(right),
                is_closed=polyline.is_closed,
            )
        )
        left = [(-radius, y_mm) for radius, y_mm in reversed(right)]
        mirrored.append(
            ProjectedPolyline2D(
                points_mm=tuple(left),
                is_closed=polyline.is_closed,
            )
        )
    return mirrored


def _direct_edge_contour(
    shape: object,
    spec: _ViewSpec,
    *,
    deflection_mm: float,
) -> CadProjectedContour:
    """
    Orthographic projection of inner B-Rep edges.

    HLR collapses bodies of revolution when viewed along the axis of symmetry
    (top view becomes a flat line). We therefore project tessellated edge geometry
    directly and keep meridional vs horizontal edges depending on the target plane.
    """
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    polylines: list[ProjectedPolyline2D] = []
    edge_count = 0
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        points_3d = _discretize_edge(edge, deflection_mm=deflection_mm)
        meridional = _edge_is_meridional(points_3d)
        if spec.plane == PLANE_TOP_XZ and meridional:
            explorer.Next()
            continue
        if spec.plane == PLANE_PROFILE and not meridional:
            explorer.Next()
            continue

        projected = _project_points(points_3d, spec.plane)
        if len(projected) >= 2:
            is_closed = _points_close(projected[0], projected[-1])
            polylines.append(
                ProjectedPolyline2D(
                    points_mm=tuple(projected),
                    is_closed=is_closed,
                )
            )
        edge_count += 1
        explorer.Next()

    return CadProjectedContour(
        plane=spec.plane,
        view_axis=spec.view_axis,
        polylines=tuple(polylines),
        edge_count=edge_count,
        source="opencascade_brep_edges",
    )


def _edge_is_meridional(
    points_3d: list[tuple[float, float, float]],
    *,
    tolerance_mm: float = 0.5,
) -> bool:
    """True when an edge lies in a plane containing the Y axis (meridian generator)."""
    z_values = [z for _, _, z in points_3d]
    return max(z_values) - min(z_values) <= tolerance_mm


def _hlr_contour(
    shape: object,
    spec: _ViewSpec,
    *,
    deflection_mm: float,
) -> CadProjectedContour:
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.HLRAlgo import HLRAlgo_Projector
    from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer

    cx, cy, cz = spec.center
    dx, dy, dz = spec.direction
    projector = HLRAlgo_Projector(
        gp_Ax2(gp_Pnt(cx, cy, cz), gp_Dir(dx, dy, dz)),
    )
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(projector)
    algo.Update()
    algo.Hide()
    hlr = HLRBRep_HLRToShape(algo)
    visible = hlr.VCompound()

    polylines: list[ProjectedPolyline2D] = []
    edge_count = 0
    if visible is not None and not visible.IsNull():
        from OCP.TopoDS import TopoDS

        explorer = TopExp_Explorer(visible, TopAbs_EDGE)
        while explorer.More():
            edge = TopoDS.Edge_s(explorer.Current())
            points_3d = _discretize_edge(edge, deflection_mm=deflection_mm)
            projected = _project_points(points_3d, spec.plane)
            if len(projected) >= 2:
                is_closed = _points_close(projected[0], projected[-1])
                polylines.append(
                    ProjectedPolyline2D(
                        points_mm=tuple(projected),
                        is_closed=is_closed,
                    )
                )
            edge_count += 1
            explorer.Next()

    if spec.plane == PLANE_PROFILE:
        polylines = _mirror_profile_polylines(polylines)

    return CadProjectedContour(
        plane=spec.plane,
        view_axis=spec.view_axis,
        polylines=tuple(polylines),
        edge_count=edge_count,
    )


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


def _project_points(
    points_3d: list[tuple[float, float, float]],
    plane: str,
) -> list[tuple[float, float]]:
    projected: list[tuple[float, float]] = []
    for x_mm, y_mm, z_mm in points_3d:
        if plane == PLANE_TOP_XZ:
            projected.append((x_mm, z_mm))
        elif plane == PLANE_PROFILE:
            radius = math.hypot(x_mm, z_mm)
            projected.append((radius, y_mm))
        else:
            raise ValueError(f"Unsupported projection plane: {plane}")
    return projected


def _mirror_profile_polylines(
    polylines: list[ProjectedPolyline2D],
) -> list[ProjectedPolyline2D]:
    """Build a closed meridian: right profile (r≥0) mirrored to −r."""
    if not polylines:
        return polylines

    merged: list[tuple[float, float]] = []
    for polyline in polylines:
        merged.extend(polyline.points_mm)

    if len(merged) < 2:
        return polylines

    positive = [(abs(r), y) for r, y in merged if r >= -1e-6]
    positive.sort(key=lambda item: item[1])
    if len(positive) < 2:
        return polylines

    right = _dedupe_points(positive)
    left = [(-r, y) for r, y in reversed(right)]
    closed = right + left[1:]
    return [
        ProjectedPolyline2D(
            points_mm=tuple(closed),
            is_closed=True,
        )
    ]


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
            points = _offset_top_points(polyline.points_mm, offset_mm)
        elif contour.plane == PLANE_PROFILE:
            points = _offset_profile_points(polyline.points_mm, offset_mm)
        else:
            points = polyline.points_mm
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


def _offset_top_points(
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


def _offset_profile_points(
    points: tuple[tuple[float, float], ...],
    offset_mm: float,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for radius, y_mm in points:
        sign = 1.0 if radius >= 0.0 else -1.0
        adjusted = sign * max(abs(radius) - offset_mm, 0.0)
        result.append((adjusted, y_mm))
    return result
