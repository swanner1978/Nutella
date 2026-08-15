#!/usr/bin/env python3
"""
Demonstration du CAD Import Pipeline.

Charge un fichier STEP (export SolidWorks), construit CanonicalModel3D,
affiche les metadonnees geometriques et genere les vues SVG profil/dessus.

Usage:
    python scripts/demo_import.py --step path/to/racloir.step
    python scripts/demo_import.py --sample          # mode demo sans fichier SW
    python scripts/demo_import.py --step racloir.step --serve
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

# Allow running without editable install
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.visualization_helpers import (  # noqa: E402
    VIEW_CONVENTIONS,
    build_projection_svg,
    displayed_view_entry,
)

from nutella_scraper.cad_import import (  # noqa: E402
    GeometryNormalizer,
    ImportPipeline,
    ModelStore,
)
from nutella_scraper.cad_import.trimesh_loader import (  # noqa: E402
    CANONICAL_UNITS,
    TrimeshLoader,
)
from nutella_scraper.domain.models.canonical import CanonicalModel3D  # noqa: E402

DEFAULT_OUTPUT = _ROOT / "output" / "views"
DEFAULT_SAMPLE = _ROOT / "data" / "reference" / "demo_racloir.step"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "demo_viewer.html"
_LOG = logging.getLogger("nutella_scraper.demo_import")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_mesh_sha256(model: CanonicalModel3D) -> str:
    """Hash the exact vertices/faces used by the visualization."""
    import numpy as np

    vertices = np.asarray(model.mesh.vertices, dtype="<f8")
    faces = np.asarray(model.mesh.faces, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"CanonicalModel3D.mesh.v1\0")
    digest.update(np.asarray(vertices.shape, dtype="<i8").tobytes())
    digest.update(vertices.tobytes(order="C"))
    digest.update(np.asarray(faces.shape, dtype="<i8").tobytes())
    digest.update(faces.tobytes(order="C"))
    return digest.hexdigest()


def _create_sample_step(path: Path) -> Path:
    """Generate a watertight box mesh exported as STEP (demo substitute)."""
    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit('trimesh requis. Installez : pip install -e ".[cad_import,dev]"') from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=(75.0, 14.0, 105.0))

    try:
        mesh.export(str(path))
        if path.exists() and path.stat().st_size > 0:
            print(f"[sample] Fichier STEP de demo cree : {path}")
            print("         Remplacez par un export SolidWorks reel via --step")
            return path
    except Exception:
        pass

    stl_path = path.with_suffix(".stl")
    mesh.export(str(stl_path))
    print("[sample] Export STEP indisponible (installez cascadio pour STEP).")
    print(f"[sample] Fichier STL de demo cree : {stl_path}")
    return stl_path


def _print_metadata(model: CanonicalModel3D) -> None:
    g = model.geometry
    b = g.bounding_box
    print("\n=== CanonicalModel3D ===")
    print(f"  ID            : {model.id}")
    print(f"  Source        : {model.source_path}")
    print(f"  Hash          : {model.source_hash[:16]}...")
    print(f"  Provenance    : {model.provenance}")
    print("\n=== GeometricMetadata ===")
    print(
        f"  Dimensions    : {g.dimensions_mm[0]:.3f} x "
        f"{g.dimensions_mm[1]:.3f} x {g.dimensions_mm[2]:.3f} mm"
    )
    print(
        f"  Centre        : ({g.center_mm[0]:.3f}, {g.center_mm[1]:.3f}, {g.center_mm[2]:.3f}) mm"
    )
    vol = f"{g.volume_mm3:.2f}" if g.volume_mm3 is not None else "N/A"
    print(f"  Volume        : {vol} mm3")
    print(f"  Watertight    : {g.is_watertight}")
    print(f"  Vertices      : {g.vertex_count}")
    print(f"  Faces         : {g.face_count}")
    print("\n=== Bounding box ===")
    print(f"  X : [{b.min_x:.3f}, {b.max_x:.3f}] mm")
    print(f"  Y : [{b.min_y:.3f}, {b.max_y:.3f}] mm")
    print(f"  Z : [{b.min_z:.3f}, {b.max_z:.3f}] mm")
    print("\n=== Axes principaux ===")
    for i, axis in enumerate(g.principal_axes, start=1):
        print(f"  Axe {i} : ({axis[0]:+.4f}, {axis[1]:+.4f}, {axis[2]:+.4f})")


def _metadata_dict(
    model: CanonicalModel3D,
    *,
    canonical_mesh_hash: str,
) -> dict[str, object]:
    g = model.geometry
    b = g.bounding_box
    data: dict[str, object] = {
        "model_id": model.id,
        "source_path": str(model.source_path),
        "source_hash": model.source_hash,
        "canonical_mesh_sha256": canonical_mesh_hash,
        "provenance": model.provenance,
        "dimensions_mm": [round(v, 4) for v in g.dimensions_mm],
        "center_mm": [round(v, 4) for v in g.center_mm],
        "volume_mm3": round(g.volume_mm3, 4) if g.volume_mm3 is not None else None,
        "is_watertight": g.is_watertight,
        "vertex_count": g.vertex_count,
        "face_count": g.face_count,
        "bounding_box": {
            "min_x": b.min_x,
            "min_y": b.min_y,
            "min_z": b.min_z,
            "max_x": b.max_x,
            "max_y": b.max_y,
            "max_z": b.max_z,
        },
        "principal_axes": [list(axis) for axis in g.principal_axes],
        "visualization_only_note": "SVG views are not used for simulation or optimization",
    }
    return data


def _canonical_trimesh(model: CanonicalModel3D) -> object:
    import numpy as np
    import trimesh

    vertices = np.array(model.mesh.vertices, dtype=np.float64)
    faces = np.array(model.mesh.faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _save_views(
    model: CanonicalModel3D,
    output_dir: Path,
    *,
    canonical_mesh_hash: str,
    tessellation: dict[str, object],
    request_id: str | None = None,
) -> Path:
    """Generate an immutable, import-specific viewer directory."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / model.id
    run_dir.mkdir(exist_ok=False)

    req = request_id or "cli"
    mesh = _canonical_trimesh(model)

    _LOG.info(
        "[upload:%s] [visualization] CanonicalModel3D id=%s | mesh_sha256=%s",
        req,
        model.id,
        canonical_mesh_hash,
    )
    _LOG.info(
        "[visualization] canonical_faces=%d | vertices=%d | output=%s",
        len(mesh.faces),
        len(mesh.vertices),
        run_dir,
    )

    _LOG.info("[visualization] external_overlay=none")
    displayed_views: dict[str, dict[str, str]] = {}
    for view_name in ("side", "top", "left", "right", "bottom"):
        svg = build_projection_svg(
            mesh,
            plane=VIEW_CONVENTIONS[view_name]["plane"],
            center=model.geometry.center_mm,
            principal_axes=model.geometry.principal_axes,
            model_id=model.id,
            canonical_mesh_sha256=canonical_mesh_hash,
        )
        filename = f"{view_name}_composite.svg"
        path = run_dir / filename
        path.write_text(svg, encoding="utf-8")
        file_hash = _file_sha256(path)
        displayed_views[view_name] = displayed_view_entry(
            view_name=view_name,
            filename=filename,
            sha256=file_hash,
            canonical_mesh_sha256=canonical_mesh_hash,
        )
        _LOG.info("[view:%s] file=%s | sha256=%s", view_name, path, file_hash)

    metadata = _metadata_dict(
        model,
        canonical_mesh_hash=canonical_mesh_hash,
    )
    metadata["tessellation"] = tessellation
    metadata["displayed_views"] = displayed_views
    metadata["viewer_directory"] = str(run_dir)

    meta_path = run_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    html_path = run_dir / "index.html"
    if TEMPLATE.exists():
        shutil.copy(TEMPLATE, html_path)
    else:
        html_path.write_text("<!doctype html><html><body>#000</body></html>", encoding="utf-8")

    _LOG.info("[viewer] file=%s | sha256=%s", html_path, _file_sha256(html_path))
    _LOG.info("[manifest] file=%s | sha256=%s", meta_path, _file_sha256(meta_path))
    return run_dir


def refresh_orthographic_views(view_dir: Path, models_root: Path) -> None:
    """Rewrite SVG views when Top/Bottom conventions change (visualization only)."""
    from nutella_scraper.cad_import.model_store import ModelStore
    from scripts.visualization_helpers import VIEW_CONVENTIONS, build_projection_svg

    meta_path = view_dir / "metadata.json"
    if not meta_path.exists():
        return
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    views = metadata.get("displayed_views") or {}
    top = views.get("top") or {}
    if "bottom" in views and top.get("plane") == VIEW_CONVENTIONS["top"]["plane"]:
        return
    model_id = str(metadata["model_id"])
    canonical_hash = str(metadata["canonical_mesh_sha256"])
    model = ModelStore(models_root).get(model_id)
    mesh = _canonical_trimesh(model)
    displayed_views: dict[str, dict[str, str]] = {}
    for view_name in ("side", "top", "left", "right", "bottom"):
        svg = build_projection_svg(
            mesh,
            plane=VIEW_CONVENTIONS[view_name]["plane"],
            center=model.geometry.center_mm,
            principal_axes=model.geometry.principal_axes,
            model_id=model.id,
            canonical_mesh_sha256=canonical_hash,
        )
        filename = f"{view_name}_composite.svg"
        path = view_dir / filename
        path.write_text(svg, encoding="utf-8")
        displayed_views[view_name] = displayed_view_entry(
            view_name=view_name,
            filename=filename,
            sha256=_file_sha256(path),
            canonical_mesh_sha256=canonical_hash,
        )
    metadata["displayed_views"] = displayed_views
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _serve_directory(directory: Path, port: int) -> None:
    from scripts.serve_viewer import serve

    serve(directory, port, open_browser=True)


def import_model_for_viewer(
    step_path: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    request_id: str | None = None,
) -> Path:
    """Run the existing CAD import pipeline and return its new view directory."""
    req = request_id or "cli"
    requested_path = step_path
    step_path = step_path.expanduser().resolve()
    if not step_path.exists():
        raise FileNotFoundError(f"Fichier introuvable — {step_path}")

    source_hash_before = _file_sha256(step_path)
    source_stat = step_path.stat()
    loader = TrimeshLoader()
    pipeline = ImportPipeline(
        normalizer=GeometryNormalizer(loader=loader),
        model_store=ModelStore(_ROOT / "output" / "models"),
    )

    suffix = step_path.suffix.lower()
    _LOG.info("[upload:%s] [source] path=%s", req, requested_path)
    _LOG.info("[upload:%s] [source] resolved_path=%s", req, step_path)
    _LOG.info(
        "[upload:%s] [source] sha256=%s | size=%d bytes | mtime_ns=%d",
        req,
        source_hash_before,
        source_stat.st_size,
        source_stat.st_mtime_ns,
    )
    if suffix in (".step", ".stp"):
        _LOG.info("[upload:%s] [import_pipeline] ImportPipeline.import_step() début", req)
        result = pipeline.import_step(step_path, generate_views=False)
        _LOG.info("[upload:%s] [import_pipeline] ImportPipeline.import_step() fin", req)
    elif suffix == ".stl":
        _LOG.info("[upload:%s] [import_pipeline] ImportPipeline.import_stl() début", req)
        result = pipeline.import_stl(step_path, generate_views=False)
        _LOG.info("[upload:%s] [import_pipeline] ImportPipeline.import_stl() fin", req)
    else:
        raise ValueError(f"Format non supporté : {suffix}")

    _LOG.info(
        "[upload:%s] [canonical_created] id=%s | source=%s",
        req,
        result.canonical.id,
        result.canonical.source_path,
    )

    source_hash_after = _file_sha256(step_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError(f"Le fichier source a changé pendant l'import : {step_path}")
    if result.canonical.source_path != step_path:
        raise RuntimeError(
            "Le CanonicalModel3D ne référence pas le fichier demandé : "
            f"{result.canonical.source_path} != {step_path}"
        )
    if result.canonical.source_hash != source_hash_before:
        raise RuntimeError(
            "Le hash source du CanonicalModel3D ne correspond pas au fichier importé"
        )

    canonical_mesh_hash = _canonical_mesh_sha256(result.canonical)
    _LOG.info(
        "[upload:%s] [canonical_created] mesh_sha256=%s | vertices=%d | faces=%d",
        req,
        canonical_mesh_hash,
        result.canonical.geometry.vertex_count,
        result.canonical.geometry.face_count,
    )
    _print_metadata(result.canonical)
    tessellation = {
        "applies_to": "STEP" if suffix in (".step", ".stp") else "STL (pre-tessellated)",
        "linear_deflection_mm": (
            loader.step_tol_linear_mm if suffix in (".step", ".stp") else None
        ),
        "angular_deflection_rad": (
            loader.step_tol_angular_rad if suffix in (".step", ".stp") else None
        ),
        "relative": loader.step_tol_relative if suffix in (".step", ".stp") else None,
        "canonical_units": CANONICAL_UNITS,
    }
    _LOG.info("[upload:%s] [projections] génération des projections début", req)
    viewer_dir = _save_views(
        result.canonical,
        output_dir,
        canonical_mesh_hash=canonical_mesh_hash,
        tessellation=tessellation,
        request_id=req,
    )
    _LOG.info("[upload:%s] [projections] génération des projections fin | dir=%s", req, viewer_dir)

    _LOG.info(
        "[upload:%s] [canonical-store] %s", req, _ROOT / "output" / "models" / result.model_id
    )
    _LOG.info("[upload:%s] [proof] source, canonical mesh and displayed SVG hashes recorded", req)
    return viewer_dir


def run_demo(step_path: Path, output_dir: Path, *, serve: bool, port: int) -> int:
    try:
        viewer_dir = import_model_for_viewer(step_path, output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    if serve:
        _serve_directory(viewer_dir, port)

    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Demo CAD Import Pipeline — STEP → CanonicalModel3D + vues SVG",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--step",
        type=Path,
        help="Chemin vers un fichier STEP exporte depuis SolidWorks (.step / .stp)",
    )
    source.add_argument(
        "--sample",
        action="store_true",
        help="Generer un STEP de demo (boite) si aucun export SolidWorks disponible",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Dossier de sortie des vues SVG (defaut: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Lancer un serveur HTTP local et ouvrir le viewer dans le navigateur",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port du serveur local (defaut: 8765)",
    )
    args = parser.parse_args()

    step_path = args.step
    if args.sample:
        step_path = _create_sample_step(DEFAULT_SAMPLE)
    assert step_path is not None

    return run_demo(step_path, args.output, serve=args.serve, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
