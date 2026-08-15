"""Visualization-only visual.stl artefact — independent of compute tessellation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from nutella_scraper.cad_import.geometry_normalizer import GeometryNormalizer
from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.cad_import.pipeline import ImportPipeline
from nutella_scraper.cad_import.trimesh_loader import (
    DEFAULT_STEP_TOL_ANGULAR_RAD,
    DEFAULT_STEP_TOL_LINEAR_MM,
    TrimeshLoader,
)
from nutella_scraper.cad_import.visual_stl import (
    DEFAULT_VISUAL_TOL_ANGULAR_RAD,
    DEFAULT_VISUAL_TOL_LINEAR_MM,
    VISUAL_ROLE,
    VISUAL_STL_NAME,
    VisualStlError,
    VisualTessellationConfig,
    compare_visual_frame,
    tessellate_visual_mesh,
    validate_visual_mesh,
)


def test_visual_tessellation_is_independent_of_compute_defaults() -> None:
    assert DEFAULT_STEP_TOL_LINEAR_MM == 0.01
    assert DEFAULT_STEP_TOL_ANGULAR_RAD == 0.1
    assert DEFAULT_VISUAL_TOL_LINEAR_MM != DEFAULT_STEP_TOL_LINEAR_MM
    assert DEFAULT_VISUAL_TOL_ANGULAR_RAD != DEFAULT_STEP_TOL_ANGULAR_RAD
    cfg = VisualTessellationConfig()
    assert cfg.tol_linear_mm == DEFAULT_VISUAL_TOL_LINEAR_MM
    assert cfg.tol_angular_rad == DEFAULT_VISUAL_TOL_ANGULAR_RAD


def test_visual_tessellation_passes_display_tolerances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step_path = tmp_path / "jar.step"
    step_path.write_bytes(b"STEP")
    captured: dict[str, object] = {}

    def fake_load(path: str, **kwargs: object) -> trimesh.Trimesh:
        captured["path"] = path
        captured.update(kwargs)
        mesh = trimesh.creation.box(extents=(80.0, 120.0, 80.0))
        mesh.units = "millimeters"
        return mesh

    monkeypatch.setattr(trimesh, "load", fake_load)
    mesh = tessellate_visual_mesh(step_path)

    assert captured["tol_linear"] == DEFAULT_VISUAL_TOL_LINEAR_MM
    assert captured["tol_angular"] == DEFAULT_VISUAL_TOL_ANGULAR_RAD
    assert captured["tol_relative"] is False
    assert len(mesh.faces) > 0


def test_stl_import_does_not_write_visual_stl(box_stl_path: Path, tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "models")
    pipeline = ImportPipeline(
        normalizer=GeometryNormalizer(),
        model_store=store,
    )
    result = pipeline.import_stl(box_stl_path, generate_views=False)
    model_dir = tmp_path / "models" / result.model_id
    assert (model_dir / "canonical.stl").exists()
    assert (model_dir / "internal.stl").exists()
    assert not (model_dir / "visual.stl").exists()
    assert not (model_dir / "visual.json").exists()


def test_persist_visual_writes_dedicated_file_and_keeps_compute_meshes(
    box_stl_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ModelStore(tmp_path / "models")
    canonical = GeometryNormalizer().normalize_from_stl(box_stl_path)
    model_id = store.persist(canonical)
    step_path = tmp_path / "jar.step"
    step_path.write_bytes(b"STEP")

    def fake_load(path: str, **kwargs: object) -> trimesh.Trimesh:
        if Path(path).suffix.lower() in {".step", ".stp"}:
            mesh = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
            mesh.units = "millimeters"
            return mesh
        return original_load(path, **kwargs)

    original_load = trimesh.load
    monkeypatch.setattr(trimesh, "load", fake_load)
    payload = store.persist_visual(model_id, step_path=step_path, canonical=canonical)

    model_dir = tmp_path / "models" / model_id
    assert (model_dir / "visual.stl").exists()
    assert (model_dir / "canonical.stl").exists()
    assert (model_dir / "internal.stl").exists()
    assert payload["filename"] == VISUAL_STL_NAME
    assert payload["role"] == VISUAL_ROLE
    assert payload["frame_check"]["aligned"] is True
    assert store.get_visual_path(model_id).name == "visual.stl"

    import json

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["visual_stl"]["filename"] == "visual.stl"
    assert metadata["visual_stl"]["vertex_count"] == payload["vertex_count"]

    reloaded_canonical = store.get(model_id)
    assert len(reloaded_canonical.mesh.vertices) == len(canonical.mesh.vertices)


def test_validate_visual_rejects_translated_mesh(box_stl_path: Path) -> None:
    canonical = GeometryNormalizer().normalize_from_stl(box_stl_path)
    shifted = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    shifted.apply_translation([5.0, 0.0, 0.0])
    with pytest.raises(VisualStlError, match="frame does not match"):
        validate_visual_mesh(
            shifted,
            canonical,
            VisualTessellationConfig(),
            source_path="visual.stl",
        )


def test_compare_visual_frame_zero_for_identical_box(box_stl_path: Path) -> None:
    canonical = GeometryNormalizer().normalize_from_stl(box_stl_path)
    visual = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    comparison = compare_visual_frame(visual, canonical, abs_tolerance_mm=0.25)
    assert comparison.frame_aligned
    assert comparison.max_frame_delta_mm == pytest.approx(0.0, abs=1e-9)


def test_import_step_writes_visual_stl(jar_step_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("cascadio", reason="STEP tessellation required")
    store = ModelStore(tmp_path / "models")
    pipeline = ImportPipeline(
        normalizer=GeometryNormalizer(loader=TrimeshLoader()),
        model_store=store,
    )
    result = pipeline.import_step(jar_step_path, generate_views=False)
    visual_path = store.get_visual_path(result.model_id)
    assert visual_path.exists()
    assert visual_path.name == "visual.stl"

    model_dir = tmp_path / "models" / result.model_id
    assert (model_dir / "canonical.stl").exists()
    assert (model_dir / "internal.stl").exists()
    assert (model_dir / "internal.stl").resolve() != visual_path.resolve()

    meta = store.get_visual_metadata(result.model_id)
    assert meta["role"] == VISUAL_ROLE
    assert meta["tessellation"]["tol_linear_mm"] == DEFAULT_VISUAL_TOL_LINEAR_MM
    assert meta["tessellation"]["tol_angular_rad"] == DEFAULT_VISUAL_TOL_ANGULAR_RAD
    assert int(meta["vertex_count"]) > 1000
    assert int(meta["face_count"]) > 1000
    assert int(meta["face_count"]) < 750_000
    assert meta["frame_check"]["aligned"] is True
    assert float(meta["frame_check"]["max_frame_delta_mm"]) <= float(
        meta["frame_check"]["abs_tolerance_mm"]
    )

    loaded = trimesh.load(str(visual_path), force="mesh")
    assert isinstance(loaded, trimesh.Trimesh)
    assert len(loaded.faces) == meta["face_count"]

    vis_center = 0.5 * (loaded.bounds[0] + loaded.bounds[1])
    can = result.canonical
    can_center = np.array(
        [
            0.5 * (can.bounds.min_x + can.bounds.max_x),
            0.5 * (can.bounds.min_y + can.bounds.max_y),
            0.5 * (can.bounds.min_z + can.bounds.max_z),
        ]
    )
    np.testing.assert_allclose(vis_center, can_center, atol=0.3)
    vis_dim = loaded.extents
    can_dim = np.array(can.geometry.dimensions_mm, dtype=np.float64)
    np.testing.assert_allclose(vis_dim, can_dim, atol=0.3)
