"""File-based persistence for CanonicalModel3D."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh

from nutella_scraper.cad_import.geometry_metadata import mesh_to_mesh_data
from nutella_scraper.domain.models.canonical import (
    BoundingBox,
    CanonicalModel3D,
    GeometricMetadata,
    MeshData,
    RigidTransform,
)
from nutella_scraper.domain.models.common import ModelFormat


from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.domain.models.cad_reference_geometry import (
    CadBoundingBox,
    CadProjectedContour,
    CadReferenceGeometry,
    ProjectedPolyline2D,
)


class CadReferenceNotAvailableError(FileNotFoundError):
    """Raised when CAD B-Rep reference geometry is unavailable (e.g. STL-only import)."""


class ModelStore:
    """File-based store for canonical 3D models, InternalJarSurface, InternalJarProfile, and CadReferenceGeometry."""

    INTERNAL_STL = "internal.stl"
    INTERNAL_META = "internal_surface.json"
    INTERNAL_PROFILE_META = "internal_profile.json"
    CAD_REFERENCE_META = "cad_reference.json"
    REFERENCE_STEP = "reference.step"

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def persist(self, model: CanonicalModel3D) -> str:
        model_dir = self._base_dir / model.id
        model_dir.mkdir(parents=True, exist_ok=True)

        mesh = _mesh_data_to_trimesh(model.mesh)
        mesh.export(str(model_dir / "canonical.stl"))

        internal = _build_internal_surface(model)
        self.persist_internal(model.id, internal)
        profile = _build_internal_profile(internal)
        self.persist_profile(model.id, profile)

        payload = _model_to_dict(model)
        (model_dir / "metadata.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return model.id

    def persist_internal(self, model_id: str, surface: InternalJarSurface) -> None:
        from nutella_scraper.engines.compute.internal_jar_surface_builder import (
            internal_mesh_to_trimesh,
        )

        model_dir = self._base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        internal_mesh = internal_mesh_to_trimesh(surface)
        internal_mesh.export(str(model_dir / self.INTERNAL_STL))
        payload = _internal_surface_to_dict(surface)
        (model_dir / self.INTERNAL_META).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def persist_profile(self, model_id: str, profile: object) -> None:
        from nutella_scraper.domain.models.internal_jar_profile import InternalJarProfile

        assert isinstance(profile, InternalJarProfile)
        model_dir = self._base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = _internal_profile_to_dict(profile)
        (model_dir / self.INTERNAL_PROFILE_META).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def persist_cad_reference(
        self,
        model_id: str,
        geometry: CadReferenceGeometry,
        *,
        step_path: Path | None = None,
    ) -> None:
        """Persist CAD B-Rep reference geometry for 2D reference views."""
        model_dir = self._base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        if step_path is not None and step_path.exists():
            shutil.copy2(step_path, model_dir / self.REFERENCE_STEP)
        payload = _cad_reference_to_dict(geometry)
        (model_dir / self.CAD_REFERENCE_META).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def get_cad_reference(self, model_id: str) -> CadReferenceGeometry:
        """Load CAD B-Rep reference geometry; rebuild from STEP when missing."""
        model_dir = self._base_dir / model_id
        meta_path = model_dir / self.CAD_REFERENCE_META
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return _cad_reference_from_dict(data)

        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "step":
            raise CadReferenceNotAvailableError(
                f"CAD reference geometry requires STEP source for model {model_id} "
                f"(format={metadata.get('format')})"
            )

        step_path = model_dir / self.REFERENCE_STEP
        if not step_path.exists():
            step_path = Path(str(metadata.get("source_path", "")))
        if not step_path.exists():
            raise CadReferenceNotAvailableError(
                f"STEP file missing for CAD reference rebuild: {model_id}"
            )

        from nutella_scraper.cad_import.cad_reference_builder import CadReferenceGeometryBuilder

        geometry = CadReferenceGeometryBuilder().from_step(step_path, model_id=model_id)
        self.persist_cad_reference(model_id, geometry, step_path=step_path)
        return geometry

    def get_profile(self, model_id: str) -> object:
        from nutella_scraper.domain.models.internal_jar_profile import InternalJarProfile
        model_dir = self._base_dir / model_id
        meta_path = model_dir / self.INTERNAL_PROFILE_META
        if not meta_path.exists():
            internal = self.get_internal(model_id)
            profile = _build_internal_profile(internal)
            self.persist_profile(model_id, profile)
            return profile
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return _internal_profile_from_dict(data)

    def get_internal(self, model_id: str) -> InternalJarSurface:
        model_dir = self._base_dir / model_id
        meta_path = model_dir / self.INTERNAL_META
        stl_path = model_dir / self.INTERNAL_STL
        if not meta_path.exists() or not stl_path.exists():
            canonical = self.get(model_id)
            surface = _build_internal_surface(canonical)
            self.persist_internal(model_id, surface)
            return surface
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        loaded = trimesh.load(str(stl_path), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Expected Trimesh when loading internal surface for {model_id}")
        return _internal_surface_from_dict(data, mesh_to_mesh_data(loaded))

    def get(self, model_id: str) -> CanonicalModel3D:
        model_dir = self._base_dir / model_id
        metadata_path = model_dir / "metadata.json"
        stl_path = model_dir / "canonical.stl"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        if not stl_path.exists():
            raise FileNotFoundError(f"Mesh file missing for model: {model_id}")

        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        loaded = trimesh.load(str(stl_path), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Expected Trimesh when loading model {model_id}")

        mesh_data = mesh_to_mesh_data(loaded)
        return _model_from_dict(data, mesh_data)

    def delete(self, model_id: str) -> None:
        model_dir = self._base_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)


def _mesh_data_to_trimesh(mesh_data: MeshData) -> trimesh.Trimesh:
    vertices = np.array(mesh_data.vertices, dtype=np.float64)
    faces = np.array(mesh_data.faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _model_to_dict(model: CanonicalModel3D) -> dict[str, object]:
    g = model.geometry
    return {
        "id": model.id,
        "source_hash": model.source_hash,
        "format": model.format,
        "source_path": str(model.source_path),
        "provenance": model.provenance,
        "bounds": asdict(model.bounds),
        "geometry": {
            "bounding_box": asdict(g.bounding_box),
            "dimensions_mm": list(g.dimensions_mm),
            "center_mm": list(g.center_mm),
            "principal_axes": [list(a) for a in g.principal_axes],
            "volume_mm3": g.volume_mm3,
            "is_watertight": g.is_watertight,
            "vertex_count": g.vertex_count,
            "face_count": g.face_count,
        },
        "frame": [list(row) for row in model.frame.matrix],
    }


def _model_from_dict(data: dict[str, object], mesh: MeshData) -> CanonicalModel3D:
    bounds_data = data["bounds"]
    assert isinstance(bounds_data, dict)
    bounds = BoundingBox(**bounds_data)

    geom_data = data["geometry"]
    assert isinstance(geom_data, dict)
    bbox_data = geom_data["bounding_box"]
    assert isinstance(bbox_data, dict)

    principal_raw = geom_data["principal_axes"]
    assert isinstance(principal_raw, list)

    geometry = GeometricMetadata(
        bounding_box=BoundingBox(**bbox_data),
        dimensions_mm=(
            float(geom_data["dimensions_mm"][0]),  # type: ignore[index]
            float(geom_data["dimensions_mm"][1]),  # type: ignore[index]
            float(geom_data["dimensions_mm"][2]),  # type: ignore[index]
        ),
        center_mm=(
            float(geom_data["center_mm"][0]),  # type: ignore[index]
            float(geom_data["center_mm"][1]),  # type: ignore[index]
            float(geom_data["center_mm"][2]),  # type: ignore[index]
        ),
        principal_axes=tuple(
            (float(axis[0]), float(axis[1]), float(axis[2])) for axis in principal_raw
        ),  # type: ignore[assignment]
        volume_mm3=(
            float(geom_data["volume_mm3"]) if geom_data.get("volume_mm3") is not None else None
        ),
        is_watertight=bool(geom_data["is_watertight"]),
        vertex_count=int(geom_data["vertex_count"]),  # type: ignore[arg-type]
        face_count=int(geom_data["face_count"]),  # type: ignore[arg-type]
    )

    frame_raw = data.get("frame", [])
    assert isinstance(frame_raw, list)
    frame = RigidTransform(
        matrix=tuple(
            (float(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in frame_raw
        )
    )

    source_path = Path(str(data["source_path"]))
    model_format: ModelFormat = "step" if data["format"] == "step" else "stl"

    return CanonicalModel3D(
        id=str(data["id"]),
        source_hash=str(data["source_hash"]),
        format=model_format,
        source_path=source_path,
        mesh=mesh,
        bounds=bounds,
        geometry=geometry,
        frame=frame,
        provenance="canonical_3d",
    )


def _build_internal_surface(model: CanonicalModel3D) -> InternalJarSurface:
    from nutella_scraper.engines.compute.internal_jar_surface_builder import (
        InternalJarSurfaceBuilder,
    )

    return InternalJarSurfaceBuilder().from_canonical(model)


def _build_internal_profile(surface: InternalJarSurface) -> object:
    from nutella_scraper.engines.compute.internal_jar_profile_builder import (
        InternalJarProfileBuilder,
    )

    return InternalJarProfileBuilder().from_internal(surface)


def _internal_profile_to_dict(profile: object) -> dict[str, object]:
    from dataclasses import asdict

    return {
        "jar_id": profile.jar_id,
        "canonical_mesh_sha256": profile.canonical_mesh_sha256,
        "meridian_spline": _bspline1d_to_dict(profile.meridian_spline),
        "top_contour_spline": _bspline2d_to_dict(profile.top_contour_spline),
        "meridian": [
            {"y_mm": point.y_mm, "radius_mm": point.radius_mm}
            for point in profile.meridian
        ],
        "top_contour": [
            {"x_mm": point.x_mm, "z_mm": point.z_mm} for point in profile.top_contour
        ],
        "top_reference_y_mm": profile.top_reference_y_mm,
        "top_inner_radius_mm": profile.top_inner_radius_mm,
        "y_min_mm": profile.y_min_mm,
        "y_max_mm": profile.y_max_mm,
        "reconstruction": asdict(profile.reconstruction),
        "metadata": profile.metadata,
    }


def _bspline1d_to_dict(spline: object) -> dict[str, object]:
    return {
        "degree": spline.degree,  # type: ignore[attr-defined]
        "knots": list(spline.knots),  # type: ignore[attr-defined]
        "coefficients": list(spline.coefficients),  # type: ignore[attr-defined]
    }


def _bspline2d_to_dict(spline: object) -> dict[str, object]:
    return {
        "degree": spline.degree,  # type: ignore[attr-defined]
        "knots": list(spline.knots),  # type: ignore[attr-defined]
        "coefficients_x": list(spline.coefficients_x),  # type: ignore[attr-defined]
        "coefficients_z": list(spline.coefficients_z),  # type: ignore[attr-defined]
    }


def _internal_profile_from_dict(data: dict[str, object]) -> object:
    from nutella_scraper.domain.models.internal_jar_profile import (
        MeridianPoint,
        ProfileReconstructionQuality,
        TopContourSample,
    )
    from nutella_scraper.engines.compute.curve_fitting import (
        BSpline1D,
        BSpline2DLoop,
        fit_meridian_bspline,
        fit_top_contour_bspline,
    )

    meridian_raw = data.get("meridian", [])
    assert isinstance(meridian_raw, list)
    top_raw = data.get("top_contour", [])
    assert isinstance(top_raw, list)
    metadata = data.get("metadata", {})
    assert isinstance(metadata, dict)

    meridian_spline_raw = data.get("meridian_spline")
    if isinstance(meridian_spline_raw, dict):
        meridian_spline = BSpline1D(
            degree=int(meridian_spline_raw["degree"]),  # type: ignore[arg-type]
            knots=tuple(float(value) for value in meridian_spline_raw["knots"]),  # type: ignore[index]
            coefficients=tuple(
                float(value) for value in meridian_spline_raw["coefficients"]
            ),  # type: ignore[index]
        )
    else:
        y_values = np.array(
            [float(entry["y_mm"]) for entry in meridian_raw],  # type: ignore[index]
            dtype=np.float64,
        )
        r_values = np.array(
            [float(entry["radius_mm"]) for entry in meridian_raw],  # type: ignore[index]
            dtype=np.float64,
        )
        meridian_spline, _ = fit_meridian_bspline(y_values, r_values)

    top_spline_raw = data.get("top_contour_spline")
    if isinstance(top_spline_raw, dict):
        top_contour_spline = BSpline2DLoop(
            degree=int(top_spline_raw["degree"]),  # type: ignore[arg-type]
            knots=tuple(float(value) for value in top_spline_raw["knots"]),  # type: ignore[index]
            coefficients_x=tuple(
                float(value) for value in top_spline_raw["coefficients_x"]
            ),  # type: ignore[index]
            coefficients_z=tuple(
                float(value) for value in top_spline_raw["coefficients_z"]
            ),  # type: ignore[index]
        )
    else:
        x_values = np.array(
            [float(entry["x_mm"]) for entry in top_raw],  # type: ignore[index]
            dtype=np.float64,
        )
        z_values = np.array(
            [
                float(entry.get("z_mm", entry.get("y_mm", 0.0)))  # type: ignore[union-attr]
                for entry in top_raw
            ],
            dtype=np.float64,
        )
        top_contour_spline, _ = fit_top_contour_bspline(x_values, z_values)

    reconstruction_raw = data.get("reconstruction", {})
    if isinstance(reconstruction_raw, dict) and reconstruction_raw:
        reconstruction = ProfileReconstructionQuality(
            meridian_max_error_mm=float(reconstruction_raw["meridian_max_error_mm"]),  # type: ignore[index]
            meridian_rms_error_mm=float(reconstruction_raw["meridian_rms_error_mm"]),  # type: ignore[index]
            meridian_hausdorff_mm=float(reconstruction_raw["meridian_hausdorff_mm"]),  # type: ignore[index]
            top_contour_max_error_mm=float(reconstruction_raw["top_contour_max_error_mm"]),  # type: ignore[index]
            top_contour_rms_error_mm=float(reconstruction_raw["top_contour_rms_error_mm"]),  # type: ignore[index]
            top_contour_hausdorff_mm=float(reconstruction_raw["top_contour_hausdorff_mm"]),  # type: ignore[index]
            top_contour_circularity=float(reconstruction_raw["top_contour_circularity"]),  # type: ignore[index]
            top_contour_is_circular=bool(reconstruction_raw["top_contour_is_circular"]),  # type: ignore[index]
        )
    else:
        reconstruction = ProfileReconstructionQuality(
            meridian_max_error_mm=0.0,
            meridian_rms_error_mm=0.0,
            meridian_hausdorff_mm=0.0,
            top_contour_max_error_mm=0.0,
            top_contour_rms_error_mm=0.0,
            top_contour_hausdorff_mm=0.0,
            top_contour_circularity=1.0,
            top_contour_is_circular=True,
        )

    return InternalJarProfile(
        jar_id=str(data["jar_id"]),
        canonical_mesh_sha256=str(data["canonical_mesh_sha256"]),
        meridian_spline=meridian_spline,
        top_contour_spline=top_contour_spline,
        meridian=tuple(
            MeridianPoint(
                y_mm=float(entry["y_mm"]),  # type: ignore[index]
                radius_mm=float(entry["radius_mm"]),  # type: ignore[index]
            )
            for entry in meridian_raw
        ),
        top_contour=tuple(
            TopContourSample(
                x_mm=float(entry["x_mm"]),  # type: ignore[index]
                z_mm=float(entry.get("z_mm", entry.get("y_mm", 0.0))),  # type: ignore[union-attr]
            )
            for entry in top_raw
        ),
        top_reference_y_mm=float(data["top_reference_y_mm"]),  # type: ignore[arg-type]
        top_inner_radius_mm=float(data["top_inner_radius_mm"]),  # type: ignore[arg-type]
        y_min_mm=float(data["y_min_mm"]),  # type: ignore[arg-type]
        y_max_mm=float(data["y_max_mm"]),  # type: ignore[arg-type]
        reconstruction=reconstruction,
        metadata=metadata,  # type: ignore[arg-type]
    )


def _internal_surface_to_dict(surface: InternalJarSurface) -> dict[str, object]:
    return {
        "jar_id": surface.jar_id,
        "canonical_mesh_sha256": surface.canonical_mesh_sha256,
        "y_min_mm": surface.y_min_mm,
        "y_max_mm": surface.y_max_mm,
        "slices": [
            {"y_mm": slice_.y_mm, "inner_radius_mm": slice_.inner_radius_mm}
            for slice_ in surface.slices
        ],
        "sample_points_mm": [list(point) for point in surface.sample_points_mm],
        "sample_areas_mm2": list(surface.sample_areas_mm2),
        "source_face_count": surface.source_face_count,
        "metadata": surface.metadata,
    }


def _internal_surface_from_dict(data: dict[str, object], mesh: MeshData) -> InternalJarSurface:
    from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurfaceSlice

    slices_raw = data.get("slices", [])
    assert isinstance(slices_raw, list)
    samples_raw = data.get("sample_points_mm", [])
    assert isinstance(samples_raw, list)
    areas_raw = data.get("sample_areas_mm2", [])
    assert isinstance(areas_raw, list)
    metadata = data.get("metadata", {})
    assert isinstance(metadata, dict)

    return InternalJarSurface(
        jar_id=str(data["jar_id"]),
        canonical_mesh_sha256=str(data["canonical_mesh_sha256"]),
        mesh=mesh,
        y_min_mm=float(data["y_min_mm"]),  # type: ignore[arg-type]
        y_max_mm=float(data["y_max_mm"]),  # type: ignore[arg-type]
        slices=tuple(
            InternalJarSurfaceSlice(
                y_mm=float(entry["y_mm"]),  # type: ignore[index]
                inner_radius_mm=float(entry["inner_radius_mm"]),  # type: ignore[index]
            )
            for entry in slices_raw
        ),
        sample_points_mm=tuple(
            (float(point[0]), float(point[1]), float(point[2])) for point in samples_raw
        ),
        sample_areas_mm2=tuple(float(value) for value in areas_raw),
        source_face_count=int(data["source_face_count"]),  # type: ignore[arg-type]
        metadata=metadata,  # type: ignore[arg-type]
    )


def _cad_reference_to_dict(geometry: CadReferenceGeometry) -> dict[str, object]:
    return {
        "model_id": geometry.model_id,
        "step_path": geometry.step_path,
        "step_sha256": geometry.step_sha256,
        "bounding_box": asdict(geometry.bounding_box),
        "inner_face_count": geometry.inner_face_count,
        "outer_face_count": geometry.outer_face_count,
        "profile_contour": _cad_contour_to_dict(geometry.profile_contour),
        "top_contour": _cad_contour_to_dict(geometry.top_contour),
        "metadata": geometry.metadata,
    }


def _cad_contour_to_dict(contour: CadProjectedContour | None) -> dict[str, object] | None:
    if contour is None:
        return None
    return {
        "plane": contour.plane,
        "view_axis": contour.view_axis,
        "polylines": [
            {
                "points_mm": [list(point) for point in polyline.points_mm],
                "is_closed": polyline.is_closed,
            }
            for polyline in contour.polylines
        ],
        "edge_count": contour.edge_count,
        "source": contour.source,
    }


def _cad_reference_from_dict(data: dict[str, object]) -> CadReferenceGeometry:
    bbox_raw = data["bounding_box"]
    assert isinstance(bbox_raw, dict)
    metadata = data.get("metadata", {})
    assert isinstance(metadata, dict)
    return CadReferenceGeometry(
        model_id=str(data["model_id"]),
        step_path=str(data["step_path"]),
        step_sha256=str(data["step_sha256"]),
        bounding_box=CadBoundingBox(**bbox_raw),
        inner_face_count=int(data["inner_face_count"]),  # type: ignore[arg-type]
        outer_face_count=int(data["outer_face_count"]),  # type: ignore[arg-type]
        profile_contour=_cad_contour_from_dict(data.get("profile_contour")),
        top_contour=_cad_contour_from_dict(data.get("top_contour")),
        metadata=metadata,  # type: ignore[arg-type]
    )


def _cad_contour_from_dict(raw: object) -> CadProjectedContour | None:
    if not isinstance(raw, dict):
        return None
    polylines_raw = raw.get("polylines", [])
    assert isinstance(polylines_raw, list)
    polylines = tuple(
        ProjectedPolyline2D(
            points_mm=tuple(
                (float(point[0]), float(point[1])) for point in entry["points_mm"]  # type: ignore[index]
            ),
            is_closed=bool(entry.get("is_closed", False)),  # type: ignore[union-attr]
        )
        for entry in polylines_raw
    )
    return CadProjectedContour(
        plane=str(raw["plane"]),
        view_axis=str(raw["view_axis"]),
        polylines=polylines,
        edge_count=int(raw["edge_count"]),  # type: ignore[arg-type]
        source=str(raw.get("source", "opencascade_hlr")),
    )
