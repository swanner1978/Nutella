"""Diagnostic-only helpers to read STEP face colors via OpenCascade XCAF.

Does not alter the production STEP → B-Rep loader (STEPControl_Reader), which
does not carry appearance/colour data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

TARGET_RGB_255 = (85, 255, 255)
DEFAULT_TOLERANCE_255 = 2
DEFAULT_MESH_LINEAR_DEFLECTION_MM = 0.35


@dataclass(frozen=True)
class FaceColorSample:
    face_id: int
    rgb_255: tuple[int, int, int]
    rgb_normalized: tuple[float, float, float]
    raw_red_green_blue: tuple[float, float, float]
    area_mm2: float
    source: str


@dataclass(frozen=True)
class StepFaceColorDiagnostic:
    step_path: str
    total_faces: int
    faces_with_readable_color: int
    target_rgb_255: tuple[int, int, int]
    matching_faces: tuple[FaceColorSample, ...]
    unique_colors_255: tuple[tuple[int, int, int], ...]
    color_information_available: bool
    loader: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def matching_face_count(self) -> int:
        return len(self.matching_faces)

    @property
    def total_target_area_mm2(self) -> float:
        return float(sum(sample.area_mm2 for sample in self.matching_faces))


@dataclass(frozen=True)
class TargetFaceMesh:
    """Tessellation of faces selected solely by STEP colour match."""

    diagnostic: StepFaceColorDiagnostic
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]


def rgb255_matches(
    rgb_255: tuple[int, int, int],
    target: tuple[int, int, int] = TARGET_RGB_255,
    *,
    tolerance: int = DEFAULT_TOLERANCE_255,
) -> bool:
    """Compare 8-bit RGB channels with an absolute integer tolerance."""
    return all(
        abs(int(value) - int(expected)) <= tolerance
        for value, expected in zip(rgb_255, target, strict=True)
    )


def normalized_srgb_to_rgb255(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(  # type: ignore[return-value]
        int(round(max(0.0, min(1.0, float(channel))) * 255.0)) for channel in rgb
    )


def diagnose_step_face_colors(
    step_path: Path,
    *,
    target_rgb_255: tuple[int, int, int] = TARGET_RGB_255,
    tolerance_255: int = DEFAULT_TOLERANCE_255,
) -> StepFaceColorDiagnostic:
    """
    Load ``step_path`` with STEPCAFControl_Reader (XCAF) and inspect face colours.

    Production ``load_step_shape`` remains unchanged and is not used here.
    """
    return extract_target_face_mesh(
        step_path,
        target_rgb_255=target_rgb_255,
        tolerance_255=tolerance_255,
        collect_mesh=False,
    ).diagnostic


def extract_target_face_mesh(
    step_path: Path,
    *,
    target_rgb_255: tuple[int, int, int] = TARGET_RGB_255,
    tolerance_255: int = DEFAULT_TOLERANCE_255,
    mesh_linear_deflection_mm: float = DEFAULT_MESH_LINEAR_DEFLECTION_MM,
    collect_mesh: bool = True,
) -> TargetFaceMesh:
    """
    Select faces by STEP colour only, optionally tessellating those faces.

    Face IDs match ``diagnose_step_face_colors`` (1-based explorer order).
    """
    path = step_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        from OCP.BRep import BRep_Builder, BRep_Tool
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDF import TDF_LabelSequence
        from OCP.TDocStd import TDocStd_Document
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS, TopoDS_Compound
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "cadquery-ocp is required for STEP face-color diagnostics"
        ) from exc

    application = XCAFApp_Application.GetApplication_s()
    fmt = TCollection_ExtendedString("MDTV-XCAF")
    document = TDocStd_Document(fmt)
    application.NewDocument(fmt, document)

    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEPCAFControl_Reader failed for {path} (status={status})")
    if not reader.Transfer(document):
        raise RuntimeError(f"STEPCAFControl_Reader.Transfer failed for {path}")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())

    free_shapes = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free_shapes)
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for index in range(1, free_shapes.Length() + 1):
        builder.Add(compound, shape_tool.GetShape_s(free_shapes.Value(index)))

    faces: list[object] = []
    explorer = TopExp_Explorer(compound, TopAbs_FACE)
    while explorer.More():
        faces.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()

    color_types = (
        XCAFDoc_ColorType.XCAFDoc_ColorSurf,
        XCAFDoc_ColorType.XCAFDoc_ColorGen,
        XCAFDoc_ColorType.XCAFDoc_ColorCurv,
    )
    matching: list[FaceColorSample] = []
    unique_colors: set[tuple[int, int, int]] = set()
    readable = 0
    sample_notes: list[str] = []
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    for face_id, face in enumerate(faces, start=1):
        color = Quantity_Color()
        source = ""
        for color_type in color_types:
            if color_tool.GetColor(face, color_type, color):
                source = f"XCAFDoc_ColorTool.GetColor/{color_type}"
                break
            if color_tool.GetInstanceColor(face, color_type, color):
                source = f"XCAFDoc_ColorTool.GetInstanceColor/{color_type}"
                break
        if not source:
            continue

        readable += 1
        raw = (float(color.Red()), float(color.Green()), float(color.Blue()))
        srgb = tuple(float(channel) for channel in color.Values(Quantity_TOC_sRGB))
        rgb_255 = normalized_srgb_to_rgb255(srgb)  # type: ignore[arg-type]
        unique_colors.add(rgb_255)

        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area_mm2 = float(props.Mass())

        if len(sample_notes) < 5:
            sample_notes.append(
                f"Face {face_id}: raw RGB methods={raw}, "
                f"sRGB normalized={srgb}, compared_as_rgb255={rgb_255}, source={source}"
            )

        if not rgb255_matches(rgb_255, target_rgb_255, tolerance=tolerance_255):
            continue

        matching.append(
            FaceColorSample(
                face_id=face_id,
                rgb_255=rgb_255,
                rgb_normalized=srgb,  # type: ignore[arg-type]
                raw_red_green_blue=raw,
                area_mm2=area_mm2,
                source=source,
            )
        )

        if not collect_mesh:
            continue

        mesher = BRepMesh_IncrementalMesh(
            face,
            float(mesh_linear_deflection_mm),
            False,
            0.5,
            True,
        )
        mesher.Perform()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            continue
        transform = location.Transformation()
        base_index = len(vertices)
        for node_index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(node_index)
            point.Transform(transform)
            vertices.append((float(point.X()), float(point.Y()), float(point.Z())))
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            triangle = triangulation.Triangle(triangle_index)
            n1, n2, n3 = triangle.Get()
            triangles.append(
                (
                    base_index + int(n1) - 1,
                    base_index + int(n2) - 1,
                    base_index + int(n3) - 1,
                )
            )

    notes = [
        "Production loader uses STEPControl_Reader (geometry only; no colours).",
        "This diagnostic uses STEPCAFControl_Reader + XCAF colour tables.",
        "Comparison uses Quantity_TOC_sRGB -> 8-bit RGB (FreeCAD COLOUR_RGB).",
        *sample_notes,
    ]
    diagnostic = StepFaceColorDiagnostic(
        step_path=str(path),
        total_faces=len(faces),
        faces_with_readable_color=readable,
        target_rgb_255=target_rgb_255,
        matching_faces=tuple(matching),
        unique_colors_255=tuple(sorted(unique_colors)),
        color_information_available=readable > 0,
        loader="STEPCAFControl_Reader + XCAFDoc_ColorTool",
        notes=tuple(notes),
    )
    return TargetFaceMesh(
        diagnostic=diagnostic,
        vertices=np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
        faces=np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
    )


def format_step_face_color_report(diagnostic: StepFaceColorDiagnostic) -> str:
    lines = [
        "STEP color diagnostic",
        f"File: {diagnostic.step_path}",
        f"Loader: {diagnostic.loader}",
        "",
        (
            "COLOR INFORMATION AVAILABLE"
            if diagnostic.color_information_available
            else "COLOR INFORMATION NOT AVAILABLE"
        ),
        "",
        f"Total B-Rep faces: {diagnostic.total_faces}",
        f"Faces with readable color: {diagnostic.faces_with_readable_color}",
        (
            f"Target color RGB{diagnostic.target_rgb_255}: "
            f"{diagnostic.matching_face_count}"
        ),
        f"Unique readable colors (RGB 0-255): {list(diagnostic.unique_colors_255)}",
        "",
        "Detected faces:",
    ]
    if not diagnostic.matching_faces:
        lines.append("- (none)")
    else:
        for sample in diagnostic.matching_faces:
            lines.append(
                f"- Face {sample.face_id} — area {sample.area_mm2:.3f} mm² "
                f"— RGB{sample.rgb_255}"
            )
    lines.extend(
        [
            "",
            f"Total target area: {diagnostic.total_target_area_mm2:.3f} mm²",
            "",
            "Notes:",
            *[f"- {note}" for note in diagnostic.notes],
        ]
    )
    return "\n".join(lines)
