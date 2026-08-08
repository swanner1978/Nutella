"""Select inner cavity faces from a STEP B-Rep solid."""

from __future__ import annotations

import math

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Vec


def select_inner_outer_faces(shape: object) -> tuple[list[object], list[object]]:
    """
    Classify faces as inner or outer cavity walls using inward-facing normals.

    For a Y-up jar, inner faces have normals pointing toward the vertical axis
    (negative dot product with the outward radial vector in XZ).
    """
    inner_faces: list[object] = []
    outer_faces: list[object] = []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        surf = BRepAdaptor_Surface(face)
        u_mid = 0.5 * (surf.FirstUParameter() + surf.LastUParameter())
        v_mid = 0.5 * (surf.FirstVParameter() + surf.LastVParameter())

        point = gp_Pnt()
        d1u = gp_Vec()
        d1v = gp_Vec()
        surf.D1(u_mid, v_mid, point, d1u, d1v)
        normal = d1u.Crossed(d1v)
        if face.Orientation() == 1:
            normal.Reverse()

        radial_len = math.hypot(point.X(), point.Z())
        if radial_len <= 1e-3:
            explorer.Next()
            continue

        radial = gp_Vec(point.X() / radial_len, 0.0, point.Z() / radial_len)
        alignment = normal.Dot(radial)
        if alignment < -0.25:
            inner_faces.append(face)
        elif alignment > 0.25:
            outer_faces.append(face)

        explorer.Next()

    return inner_faces, outer_faces


def build_compound(faces: list[object]) -> object:
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for face in faces:
        builder.Add(compound, face)
    return compound
