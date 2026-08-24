"""Fast envelope closest-point — identical to Trimesh, fewer rebuilds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.unit.engines.compute.test_coverage_simulator import (
    _a0_parameters,
    _fast_surface,
)

from nutella_scraper.engines.compute.envelope_surface_proximity import (
    bind_envelope_proximity,
    closest_on_envelope_surface,
    closest_on_envelope_surface_fast,
    closest_on_envelope_surface_legacy,
    reset_proximity_stats,
)
from nutella_scraper.engines.compute.scraper_envelope_collision import (
    rigid_pose_neighborhood,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    build_rigid_scraper_artifact,
    envelope_contact_frame,
    rigid_transform_between_frames,
    transform_points,
)

HTML_SRC = Path("scripts/templates/demo_viewer.html")
PROX_SRC = Path("src/nutella_scraper/engines/compute/envelope_surface_proximity.py")

# Explicit identity tolerances vs Trimesh.nearest.on_surface.
_DIST_ATOL = 1e-12
_POINT_ATOL = 1e-12


@pytest.fixture(scope="module")
def a0_mesh_bundle():
    surface = _fast_surface()
    params = _a0_parameters(surface)
    artifact = build_rigid_scraper_artifact(surface, params)
    mesh = surface.to_trimesh()
    bind_envelope_proximity(mesh)
    return {
        "surface": surface,
        "params": params,
        "artifact": artifact,
        "mesh": mesh,
        "vertices": np.asarray(artifact.mesh.vertices, dtype=np.float64),
        "faces": np.asarray(artifact.mesh.faces, dtype=np.int64),
        "edges": np.asarray(artifact.mesh.edges_unique, dtype=np.int64),
    }


def _posed_samples(bundle, *, progress_deg: float) -> list[np.ndarray]:
    surface = bundle["surface"]
    params = bundle["params"].with_updates(surface_progress_deg=float(progress_deg))
    mesh = bundle["mesh"]
    artifact = bundle["artifact"]
    if abs(float(progress_deg)) <= 1e-9:
        nominal = artifact.design_frame
    else:
        nominal = envelope_contact_frame(
            surface, params, surface_progress_deg=float(progress_deg), surface_mesh=mesh
        )
    verts = bundle["vertices"]
    faces = bundle["faces"]
    edges = bundle["edges"]
    samples: list[np.ndarray] = []
    for frame in rigid_pose_neighborhood(nominal):
        transform = rigid_transform_between_frames(artifact.design_frame, frame)
        posed = transform_points(verts, transform)
        extra = np.vstack(
            [
                0.5 * (posed[edges[:, 0]] + posed[edges[:, 1]]),
                posed[faces].mean(axis=1),
            ]
        )
        samples.append(posed)
        samples.append(extra)
    return samples


def _assert_match(
    fast,
    ref,
    *,
    dist_atol: float = _DIST_ATOL,
    point_atol: float = _POINT_ATOL,
) -> None:
    close_f, dist_f, tri_f = fast
    close_r, dist_r, tri_r = ref
    assert dist_f.shape == dist_r.shape
    assert np.allclose(dist_f, dist_r, atol=dist_atol, rtol=0.0)
    assert np.array_equal(tri_f, tri_r)
    assert np.allclose(close_f, close_r, atol=point_atol, rtol=0.0)


def test_fast_helper_does_not_touch_viewer() -> None:
    text = PROX_SRC.read_text(encoding="utf-8")
    assert "closest_on_envelope_surface_fast" in text
    assert "engines.visualization" not in text
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "closest_on_envelope_surface_fast" not in html


def test_fast_matches_trimesh_on_design_and_posed_samples(a0_mesh_bundle) -> None:
    mesh = a0_mesh_bundle["mesh"]
    verts_before = np.asarray(mesh.vertices, dtype=np.float64).copy()
    n_mismatch_tid = 0
    n_points = 0
    max_dist = 0.0
    max_point = 0.0
    for progress in (0.0, 20.0, 44.0):
        for points in _posed_samples(a0_mesh_bundle, progress_deg=progress):
            n_points += len(points)
            ref = mesh.nearest.on_surface(points)
            fast = closest_on_envelope_surface_fast(mesh, points)
            _assert_match(fast, ref)
            max_dist = max(max_dist, float(np.max(np.abs(fast[1] - ref[1]))))
            max_point = max(
                max_point,
                float(np.max(np.linalg.norm(fast[0] - ref[0], axis=1))),
            )
            n_mismatch_tid += int(np.count_nonzero(fast[2] != ref[2]))
    assert n_mismatch_tid == 0
    assert n_points > 50_000
    assert max_dist <= _DIST_ATOL
    assert max_point <= _POINT_ATOL
    assert np.allclose(np.asarray(mesh.vertices), verts_before, atol=1e-12)


def test_fast_matches_legacy_on_posed_samples(a0_mesh_bundle) -> None:
    mesh = a0_mesh_bundle["mesh"]
    for points in _posed_samples(a0_mesh_bundle, progress_deg=20.0):
        fast = closest_on_envelope_surface_fast(mesh, points)
        legacy = closest_on_envelope_surface_legacy(mesh, points)
        _assert_match(fast, legacy)


def test_production_helper_uses_fast_path(a0_mesh_bundle) -> None:
    mesh = a0_mesh_bundle["mesh"]
    points = a0_mesh_bundle["vertices"]
    reset_proximity_stats()
    prod = closest_on_envelope_surface(mesh, points)
    fast = closest_on_envelope_surface_fast(mesh, points)
    _assert_match(prod, fast)


def test_edge_vertex_floor_and_near_tie_samples(a0_mesh_bundle) -> None:
    mesh = a0_mesh_bundle["mesh"]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tris = verts[faces]
    centroids = tris.mean(axis=1)
    edge_mids = 0.5 * (tris[:, 0] + tris[:, 1])
    floor = verts[np.argsort(verts[:, 1])[:48]]
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    along_normal = centroids + 0.4 * normals
    against_normal = centroids - 0.4 * normals
    samples = np.vstack(
        [verts, edge_mids, centroids, floor, along_normal, against_normal]
    )
    _assert_match(
        closest_on_envelope_surface_fast(mesh, samples),
        mesh.nearest.on_surface(samples),
    )
