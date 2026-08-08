"""InternalJarSurface extraction and projection consistency tests."""

from __future__ import annotations

import numpy as np
import trimesh

from nutella_scraper.cad_import.geometry_metadata import mesh_to_mesh_data
from nutella_scraper.domain.models.canonical import (
    BoundingBox,
    CanonicalModel3D,
    GeometricMetadata,
    RigidTransform,
)
from nutella_scraper.domain.models.contact import ContactSimulationConfig, TrajectoryConfig
from nutella_scraper.engines.compute.contact_simulator import ContactSimulationEngine
from nutella_scraper.engines.compute.internal_jar_surface_builder import (
    InternalJarSurfaceBuilder,
    internal_mesh_to_trimesh,
)
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder
from nutella_scraper.engines.visualization.projection_math import silhouette_edges


def _y_aligned_cylinder(*, radius: float, height: float, sections: int = 32) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    transform = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    mesh.apply_transform(transform)
    return mesh


def _double_wall_canonical(*, inner_r: float = 40.0, outer_r: float = 45.0, height: float = 100.0):
    inner = _y_aligned_cylinder(radius=inner_r, height=height)
    outer = _y_aligned_cylinder(radius=outer_r, height=height)
    combined = trimesh.util.concatenate([inner, outer])
    mesh_data = mesh_to_mesh_data(combined)
    bounds = BoundingBox(
        min_x=float(combined.bounds[0][0]),
        min_y=float(combined.bounds[0][1]),
        min_z=float(combined.bounds[0][2]),
        max_x=float(combined.bounds[1][0]),
        max_y=float(combined.bounds[1][1]),
        max_z=float(combined.bounds[1][2]),
    )
    geometry = GeometricMetadata(
        bounding_box=bounds,
        dimensions_mm=(
            bounds.max_x - bounds.min_x,
            bounds.max_y - bounds.min_y,
            bounds.max_z - bounds.min_z,
        ),
        center_mm=(
            (bounds.min_x + bounds.max_x) / 2.0,
            (bounds.min_y + bounds.max_y) / 2.0,
            (bounds.min_z + bounds.max_z) / 2.0,
        ),
        principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        volume_mm3=None,
        is_watertight=False,
        vertex_count=len(combined.vertices),
        face_count=len(combined.faces),
    )
    return CanonicalModel3D(
        id="double_wall_jar",
        source_hash="test-double-wall",
        format="stl",
        source_path=__file__,
        mesh=mesh_data,
        bounds=bounds,
        geometry=geometry,
        frame=RigidTransform(),
    )


def _side_silhouette_max_x(mesh: trimesh.Trimesh) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    coords = vertices[:, (0, 2)]
    edges = silhouette_edges(mesh, view_axis=1)
    if len(edges) == 0:
        return float(coords[:, 0].max())
    return float(coords[edges].reshape(-1, 2)[:, 0].max())


class TestInternalJarSurfaceBuilder:
    def test_extracts_inner_shell_from_double_wall_mesh(self) -> None:
        jar = _double_wall_canonical(inner_r=40.0, outer_r=45.0)
        raw_mesh = JarMeshBuilder().from_canonical_raw(jar)
        internal = InternalJarSurfaceBuilder().from_canonical(jar)
        inner_mesh = internal_mesh_to_trimesh(internal)

        assert internal.face_count < len(raw_mesh.faces)
        assert internal.metadata["excluded_face_count"] > 0

        raw_extent = _side_silhouette_max_x(raw_mesh)
        inner_extent = _side_silhouette_max_x(inner_mesh)
        assert inner_extent <= 40.5
        assert raw_extent >= 44.5
        assert inner_extent < raw_extent - 2.0

    def test_side_silhouette_is_narrower_than_raw_double_wall(self) -> None:
        jar = _double_wall_canonical()
        internal = InternalJarSurfaceBuilder().from_canonical(jar)
        inner_mesh = internal_mesh_to_trimesh(internal)
        raw_mesh = JarMeshBuilder().from_canonical_raw(jar)

        assert _side_silhouette_max_x(inner_mesh) < _side_silhouette_max_x(raw_mesh) - 2.0

    def test_top_projection_excludes_outer_radius(self) -> None:
        jar = _double_wall_canonical(inner_r=40.0, outer_r=45.0)
        internal = InternalJarSurfaceBuilder().from_canonical(jar)
        inner_mesh = internal_mesh_to_trimesh(internal)
        radial = np.sqrt(inner_mesh.vertices[:, 0] ** 2 + inner_mesh.vertices[:, 2] ** 2)
        assert float(radial.max()) <= 40.5


class TestContactSimulationUsesInternalJarSurface:
    def test_simulation_reports_internal_geometry_source(
        self,
        cylindrical_jar_canonical,
        internal_jar_surface,
        wall_scraper_geometry,
        wall_scraper_pose,
    ) -> None:
        config = ContactSimulationConfig(
            trajectory=TrajectoryConfig(angular_step_deg=90.0, vertical_step_mm=50.0),
            contact_threshold_mm=1.0,
            clearance_mm=0.15,
            mesh_tolerance_mm=0.1,
        )
        result = ContactSimulationEngine().simulate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            wall_scraper_pose,
            config,
            internal=internal_jar_surface,
        )

        geometry_info = result.diagnostics["compute_geometry"]
        assert geometry_info["geometry_source"] == "InternalJarSurface"
        assert geometry_info["jar_mesh_faces"] == internal_jar_surface.face_count
        assert len(result.contact_distance_map) == internal_jar_surface.face_count

    def test_coverage_remains_finite_after_refactor(
        self,
        cylindrical_jar_canonical,
        internal_jar_surface,
        wall_scraper_geometry,
        wall_scraper_pose,
    ) -> None:
        config = ContactSimulationConfig(
            trajectory=TrajectoryConfig(angular_step_deg=90.0, vertical_step_mm=50.0),
            contact_threshold_mm=1.0,
            clearance_mm=0.15,
            mesh_tolerance_mm=0.1,
        )
        result = ContactSimulationEngine().simulate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            wall_scraper_pose,
            config,
            internal=internal_jar_surface,
        )
        assert 0.0 <= result.coverage_score <= 1.0
