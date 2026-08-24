"""Viewer HTTP API contract — shared between serve_viewer and the frontend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

API_IMPORT_STEP = "/api/import-step"
API_SIMULATE_CONTACT = "/api/simulate-contact"
API_BUILD_SCRAPER = "/api/build-scraper"
API_SCRAPER_SHAPE_CANDIDATES = "/api/scraper-shape-candidates"
API_COVERAGE_RANK_CATALOG = "/api/coverage-rank-catalog"
API_COVERAGE_ANGLE_AUDIT = "/api/coverage-angle-audit"
API_COVERAGE_TARGET_REGION = "/api/coverage-target-region"
API_INTERIOR_CONTOUR = "/api/interior-contour"
API_DEBUG_STEP_FACE_COLORS = "/api/debug-step-face-colors"
API_SIMULATIONS = "/api/simulations"
API_RUNTIME = "/api/runtime"
API_VIEWER_SCENE = "/api/viewer-scene"

VIEWER_POST_ENDPOINTS: tuple[str, ...] = (
    API_IMPORT_STEP,
    API_SIMULATE_CONTACT,
    API_BUILD_SCRAPER,
    API_SCRAPER_SHAPE_CANDIDATES,
    API_INTERIOR_CONTOUR,
    API_DEBUG_STEP_FACE_COLORS,
)


@dataclass(frozen=True)
class SimulateContactRequest:
    """Optional body for POST /api/simulate-contact."""

    model_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SimulateContactRequest:
        if not payload:
            return cls()
        model_id = payload.get("model_id")
        if model_id is None:
            return cls()
        return cls(model_id=str(model_id))


@dataclass(frozen=True)
class BuildScraperRequest:
    """Optional body for POST /api/build-scraper."""

    model_id: str | None = None
    parameters: dict[str, Any] | None = None
    include_svg_overlays: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> BuildScraperRequest:
        if not payload:
            return cls()
        model_id = payload.get("model_id")
        raw_params = payload.get("parameters")
        parameters = raw_params if isinstance(raw_params, dict) else None
        return cls(
            model_id=None if model_id is None else str(model_id),
            parameters=parameters,
            include_svg_overlays=payload.get("include_svg_overlays") is True,
        )


@dataclass(frozen=True)
class ScraperShapeCandidatesRequest:
    """Optional body for POST /api/scraper-shape-candidates."""

    model_id: str | None = None
    count: int = 100

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ScraperShapeCandidatesRequest:
        if not payload:
            return cls()
        model_id = payload.get("model_id")
        raw_count = payload.get("count", 100)
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 100
        return cls(
            model_id=None if model_id is None else str(model_id),
            count=count,
        )


def normalize_api_path(path: str) -> str:
    """Normalize request paths so /api/foo and /api/foo/ resolve identically."""
    normalized = urlsplit(path).path or "/"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if normalized != "/":
        normalized = normalized.rstrip("/")
    return normalized


def simulation_id_from_path(path: str) -> str | None:
    """Return the job id addressed by /api/simulations/{id}."""
    normalized = normalize_api_path(path)
    prefix = f"{API_SIMULATIONS}/"
    if not normalized.startswith(prefix):
        return None
    simulation_id = normalized[len(prefix) :]
    if not simulation_id or "/" in simulation_id:
        return None
    return simulation_id


def simulation_result_id_from_path(path: str) -> str | None:
    """Return the job id addressed by /api/simulations/{id}/result."""
    normalized = normalize_api_path(path)
    prefix = f"{API_SIMULATIONS}/"
    suffix = "/result"
    if not normalized.startswith(prefix) or not normalized.endswith(suffix):
        return None
    simulation_id = normalized[len(prefix) : -len(suffix)]
    if not simulation_id or "/" in simulation_id:
        return None
    return simulation_id


def simulation_pose_path(path: str) -> tuple[str, int | None] | None:
    """Parse /api/simulations/{id}/poses[/index]."""
    normalized = normalize_api_path(path)
    prefix = f"{API_SIMULATIONS}/"
    if not normalized.startswith(prefix):
        return None
    parts = normalized[len(prefix) :].split("/")
    if len(parts) == 2 and parts[0] and parts[1] == "poses":
        return parts[0], None
    if len(parts) == 3 and parts[0] and parts[1] == "poses":
        try:
            index = int(parts[2])
        except ValueError:
            return None
        return parts[0], index
    return None


def build_not_found_payload(
    *,
    path: str,
    method: str,
    request_id: str,
) -> dict[str, Any]:
    """Structured 404 payload for unknown viewer API routes."""
    return {
        "error": "Endpoint introuvable",
        "message": (
            f"Aucun endpoint {method} n'est enregistré pour « {path} ». "
            f"Endpoints POST disponibles : {', '.join(VIEWER_POST_ENDPOINTS)}"
        ),
        "path": path,
        "method": method,
        "available_endpoints": list(VIEWER_POST_ENDPOINTS),
        "request_id": request_id,
    }


def parse_json_body(raw_body: bytes) -> dict[str, Any]:
    """Parse an optional JSON request body."""
    if not raw_body:
        return {}
    payload = json.loads(raw_body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Le corps JSON doit être un objet")
    return payload


def resolve_view_dir(
    *,
    output_root: Path,
    active_view_dir: Path | None,
    model_id: str | None,
) -> Path:
    """
    Resolve the immutable viewer directory for the active model.

    Prefers an explicit ``model_id`` sent by the frontend, then falls back to
    the server-side active import directory.
    """
    if model_id:
        candidate = output_root / model_id
        metadata_path = candidate / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(
                f"Modèle de visualisation introuvable : {model_id} "
                f"(manifest absent sous {candidate})"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("model_id")) != model_id:
            raise ValueError(
                f"Manifest incohérent pour le modèle {model_id} "
                f"(model_id={metadata.get('model_id')!r})"
            )
        return candidate

    if active_view_dir is not None and active_view_dir.exists():
        return active_view_dir

    raise ValueError(
        "Aucun modèle actif — importez un fichier STEP ou fournissez model_id "
        "dans le corps JSON de POST /api/simulate-contact"
    )


def resolve_pose_view_dir(
    *,
    output_root: Path,
    manifest: dict[str, Any],
    active_view_dir: Path | None,
) -> Path:
    """Resolve the viewer directory for per-pose Scraper3D overlay generation."""
    view_dir_name = manifest.get("view_dir_name")
    if view_dir_name:
        candidate = output_root / str(view_dir_name)
        if (candidate / "metadata.json").exists():
            return candidate

    if active_view_dir is not None and (active_view_dir / "metadata.json").exists():
        return active_view_dir

    legacy = output_root / str(manifest.get("model_id", ""))
    if (legacy / "metadata.json").exists():
        return legacy

    raise FileNotFoundError(
        "Répertoire de visualisation introuvable pour reconstruire le Scraper3D "
        f"(view_dir_name={view_dir_name!r}, model_id={manifest.get('model_id')!r})"
    )


def validate_simulate_contact_response(payload: dict[str, Any]) -> None:
    """Ensure the simulate endpoint returns the frontend contract."""
    required_top_level = ("model_id", "coverage_score_display", "metrics", "overlays")
    for key in required_top_level:
        if key not in payload:
            raise ValueError(f"Réponse simulation incomplète — clé manquante : {key}")

    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("metrics doit être un objet JSON")

    metric_keys = (
        "coverage_score_percent",
        "covered_face_count",
        "total_face_count",
        "contact_point_count",
        "has_collision",
        "max_penetration_depth_mm",
    )
    for key in metric_keys:
        if key not in metrics:
            raise ValueError(f"metrics.{key} manquant dans la réponse simulation")

    overlays = payload["overlays"]
    if not isinstance(overlays, dict):
        raise ValueError("overlays doit être un objet JSON")
    for view_name in ("side", "top", "left", "right"):
        if view_name not in overlays:
            raise ValueError(f"overlays.{view_name} manquant dans la réponse simulation")
