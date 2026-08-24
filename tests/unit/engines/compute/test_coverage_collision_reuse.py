"""Coverage collision hot-path reuse — same physics, fewer nearest / copies."""

from __future__ import annotations

import numpy as np
import pytest
from tests.unit.engines.compute.test_coverage_simulator import (
    _a0_parameters,
    _fast_surface,
)

from nutella_scraper.engines.compute import scraper_envelope_collision as col_mod
from nutella_scraper.engines.compute.scraper_envelope_collision import (
    evaluate_envelope_collision,
    rigid_pose_neighborhood,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    apply_rigid_transform,
    build_rigid_scraper_artifact,
    envelope_contact_frame,
    rigid_transform_between_frames,
    transform_points,
)


@pytest.fixture(scope="module")
def a0_collision_bundle():
    surface = _fast_surface()
    params = _a0_parameters(surface)
    artifact = build_rigid_scraper_artifact(surface, params)
    surface_mesh = surface.to_trimesh()
    return {
        "surface": surface,
        "params": params,
        "artifact": artifact,
        "surface_mesh": surface_mesh,
    }


def test_envelope_contact_frame_reuses_mesh_without_mutating_it(a0_collision_bundle) -> None:
    surface = a0_collision_bundle["surface"]
    params = a0_collision_bundle["params"]
    mesh = a0_collision_bundle["surface_mesh"]
    verts_before = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces_before = np.asarray(mesh.faces, dtype=np.int64).copy()
    shared = envelope_contact_frame(
        surface, params, surface_progress_deg=20.0, surface_mesh=mesh
    )
    fresh = envelope_contact_frame(surface, params, surface_progress_deg=20.0)
    assert np.allclose(shared.origin_mm, fresh.origin_mm, atol=1e-9)
    assert np.allclose(shared.rotation, fresh.rotation, atol=1e-9)
    assert np.allclose(np.asarray(mesh.vertices), verts_before, atol=1e-12)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_vertex_array_se3_matches_mesh_copy_collision(a0_collision_bundle) -> None:
    surface = a0_collision_bundle["surface"]
    params = a0_collision_bundle["params"].with_updates(surface_progress_deg=20.0)
    artifact = a0_collision_bundle["artifact"]
    surface_mesh = a0_collision_bundle["surface_mesh"]
    vertices0 = np.asarray(artifact.mesh.vertices, dtype=np.float64)
    faces0 = np.asarray(artifact.mesh.faces, dtype=np.int64)
    edges0 = np.asarray(artifact.mesh.edges_unique, dtype=np.int64)
    nominal = envelope_contact_frame(
        surface, params, surface_progress_deg=20.0, surface_mesh=surface_mesh
    )
    compared = 0
    for frame in rigid_pose_neighborhood(nominal):
        transform = rigid_transform_between_frames(artifact.design_frame, frame)
        copied = apply_rigid_transform(artifact.mesh, transform)
        via_copy = evaluate_envelope_collision(
            copied, surface, params, surface_mesh=surface_mesh
        )
        via_arrays = evaluate_envelope_collision(
            artifact.mesh,
            surface,
            params,
            surface_mesh=surface_mesh,
            vertices=transform_points(vertices0, transform),
            faces=faces0,
            edges_unique=edges0,
        )
        assert via_arrays.admissible == via_copy.admissible
        assert via_arrays.has_collision == via_copy.has_collision
        assert via_arrays.vertex_hit == via_copy.vertex_hit
        assert via_arrays.edge_hit == via_copy.edge_hit
        assert via_arrays.face_hit == via_copy.face_hit
        assert via_arrays.contact_face_ids == via_copy.contact_face_ids
        assert via_arrays.min_signed_interior_mm == pytest.approx(
            via_copy.min_signed_interior_mm, abs=1e-9
        )
        assert via_arrays.min_unsigned_distance_mm == pytest.approx(
            via_copy.min_unsigned_distance_mm, abs=1e-9
        )
        compared += 1
    assert compared == 17


def test_second_proximity_skipped_when_vertices_penetrate(
    a0_collision_bundle, monkeypatch
) -> None:
    surface = a0_collision_bundle["surface"]
    params = a0_collision_bundle["params"]
    artifact = a0_collision_bundle["artifact"]
    surface_mesh = a0_collision_bundle["surface_mesh"]
    orig = col_mod._proximity
    calls = {"n": 0}

    def counted(mesh, points):
        calls["n"] += 1
        return orig(mesh, points)

    monkeypatch.setattr(col_mod, "_proximity", counted)
    shoved = apply_rigid_transform(artifact.mesh, np.eye(4, dtype=np.float64))
    shoved.vertices += np.array([80.0, 0.0, 0.0])
    evaluate_envelope_collision(shoved, surface, params, surface_mesh=surface_mesh)
    assert calls["n"] == 1


def test_interior_pose_still_queries_wall_extras(
    a0_collision_bundle, monkeypatch
) -> None:
    surface = a0_collision_bundle["surface"]
    params = a0_collision_bundle["params"].with_updates(surface_progress_deg=0.0)
    artifact = a0_collision_bundle["artifact"]
    surface_mesh = a0_collision_bundle["surface_mesh"]
    orig = col_mod._proximity
    calls = {"n": 0}

    def counted(mesh, points):
        calls["n"] += 1
        return orig(mesh, points)

    monkeypatch.setattr(col_mod, "_proximity", counted)
    report = evaluate_envelope_collision(
        artifact.mesh, surface, params, surface_mesh=surface_mesh
    )
    assert report.vertex_hit is False
    assert calls["n"] == 2


def test_cached_proximity_matches_trimesh_on_surface(a0_collision_bundle) -> None:
    surface = a0_collision_bundle["surface"]
    params = a0_collision_bundle["params"].with_updates(surface_progress_deg=20.0)
    artifact = a0_collision_bundle["artifact"]
    mesh = a0_collision_bundle["surface_mesh"]
    verts_before = np.asarray(mesh.vertices, dtype=np.float64).copy()
    from nutella_scraper.engines.compute.envelope_surface_proximity import (
        closest_on_envelope_surface,
    )

    nominal = envelope_contact_frame(
        surface, params, surface_progress_deg=20.0, surface_mesh=mesh
    )
    vertices0 = np.asarray(artifact.mesh.vertices, dtype=np.float64)
    faces0 = np.asarray(artifact.mesh.faces, dtype=np.int64)
    edges0 = np.asarray(artifact.mesh.edges_unique, dtype=np.int64)
    for frame in rigid_pose_neighborhood(nominal):
        transform = rigid_transform_between_frames(artifact.design_frame, frame)
        posed = transform_points(vertices0, transform)
        extra = np.vstack(
            [
                0.5 * (posed[edges0[:, 0]] + posed[edges0[:, 1]]),
                posed[faces0].mean(axis=1),
            ]
        )
        for points in (posed, extra):
            ref_close, ref_dist, ref_tid = mesh.nearest.on_surface(points)
            close, dist, tid = closest_on_envelope_surface(mesh, points)
            assert np.allclose(dist, ref_dist, atol=1e-12, rtol=0.0)
            assert np.array_equal(tid, ref_tid)
            assert np.allclose(close, ref_close, atol=1e-12, rtol=0.0)
    assert np.allclose(np.asarray(mesh.vertices), verts_before, atol=1e-12)
