"""Integration tests for viewer HTTP API routes used by the frontend."""

from __future__ import annotations

import json
import threading
import time
from functools import partial
from pathlib import Path

import pytest
from scripts.serve_viewer import ViewerHTTPRequestHandler, ViewerHTTPServer
from scripts.viewer_api import (
    API_IMPORT_STEP,
    API_SIMULATE_CONTACT,
    VIEWER_POST_ENDPOINTS,
    build_not_found_payload,
    normalize_api_path,
    resolve_view_dir,
    resolve_pose_view_dir,
    validate_simulate_contact_response,
)

from nutella_scraper.application.simulation_execution import SimulationExecutionManager
from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.contact_simulator import ContactSimulationEngine


def test_normalize_api_path_strips_trailing_slash() -> None:
    assert normalize_api_path("/api/simulate-contact/") == API_SIMULATE_CONTACT
    assert normalize_api_path("/api/import-step/") == API_IMPORT_STEP


def test_build_not_found_payload_is_explicit() -> None:
    payload = build_not_found_payload(
        path="/api/unknown",
        method="POST",
        request_id="req-test",
    )

    assert payload["error"] == "Endpoint introuvable"
    assert "POST" in payload["message"]
    assert "/api/unknown" in payload["message"]
    assert payload["available_endpoints"] == list(VIEWER_POST_ENDPOINTS)


def test_frontend_post_endpoints_are_registered() -> None:
    from scripts.viewer_api import API_BUILD_SCRAPER, API_INTERIOR_CONTOUR

    assert API_IMPORT_STEP in VIEWER_POST_ENDPOINTS
    assert API_SIMULATE_CONTACT in VIEWER_POST_ENDPOINTS
    assert API_BUILD_SCRAPER in VIEWER_POST_ENDPOINTS
    assert API_INTERIOR_CONTOUR in VIEWER_POST_ENDPOINTS


@pytest.fixture
def viewer_server_setup(
    tmp_path: Path,
    cylindrical_jar_canonical: object,
    wall_scraper_geometry: ScraperGeometry,
    wall_scraper_pose: ScraperPose,
    coarse_simulation_config: ContactSimulationConfig,
) -> tuple[str, Path, str]:
    import numpy as np
    import trimesh
    from nutella_scraper.cad_import.model_store import ModelStore
    from scripts.visualization_helpers import VIEW_CONVENTIONS, build_projection_svg

    from tests.unit.cad_import.conftest import persist_test_cad_reference

    model = cylindrical_jar_canonical
    models_root = tmp_path / "models"
    output_root = tmp_path / "views"
    store = ModelStore(models_root)
    store.persist(model)

    jar_step = Path(__file__).resolve().parents[3] / "Solidworks" / "jar.STEP"
    if jar_step.exists():
        persist_test_cad_reference(store, model.id, jar_step)

    ContactSimulationEngine().simulate(
        model,
        wall_scraper_geometry,
        wall_scraper_pose,
        coarse_simulation_config,
    )

    view_dir = output_root / model.id
    view_dir.mkdir(parents=True)
    vertices = np.array(model.mesh.vertices, dtype=np.float64)
    faces = np.array(model.mesh.faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    side_svg = build_projection_svg(
        mesh,
        plane=VIEW_CONVENTIONS["side"]["plane"],
        center=model.geometry.center_mm,
        principal_axes=model.geometry.principal_axes,
        model_id=model.id,
        canonical_mesh_sha256="test_hash",
    )
    top_svg = build_projection_svg(
        mesh,
        plane=VIEW_CONVENTIONS["top"]["plane"],
        center=model.geometry.center_mm,
        principal_axes=model.geometry.principal_axes,
        model_id=model.id,
        canonical_mesh_sha256="test_hash",
    )
    (view_dir / "side_composite.svg").write_text(side_svg, encoding="utf-8")
    (view_dir / "top_composite.svg").write_text(top_svg, encoding="utf-8")
    (view_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": model.id,
                "displayed_views": {
                    "side": {
                        "filename": "side_composite.svg",
                        "plane": VIEW_CONVENTIONS["side"]["plane"],
                    },
                    "top": {
                        "filename": "top_composite.svg",
                        "plane": VIEW_CONVENTIONS["top"]["plane"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    handler = partial(ViewerHTTPRequestHandler, directory=str(output_root))
    server = ViewerHTTPServer(("127.0.0.1", 0), handler)
    server.output_root = output_root
    server.upload_root = tmp_path / "uploads"
    server.models_root = models_root
    server.active_view_dir = view_dir
    server.import_lock = threading.Lock()
    server.dev_mode = True
    server.simulation_manager = SimulationExecutionManager(tmp_path / "simulations")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    yield base_url, view_dir, model.id

    server.simulation_manager.shutdown()
    server.shutdown()
    server.server_close()


def _post_json(base_url: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    import http.client
    from urllib.parse import urlparse

    parsed = urlparse(f"{base_url}{path}")
    body = json.dumps(payload or {}).encode("utf-8")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    try:
        connection.request(
            "POST",
            parsed.path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read().decode("utf-8")
        return response.status, json.loads(raw) if raw else {}
    finally:
        connection.close()


def _request_json(base_url: str, method: str, path: str) -> tuple[int, dict]:
    import http.client
    from urllib.parse import urlparse

    parsed = urlparse(f"{base_url}{path}")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    try:
        connection.request(method, parsed.path)
        response = connection.getresponse()
        raw = response.read().decode("utf-8")
        return response.status, json.loads(raw) if raw else {}
    finally:
        connection.close()


def _wait_for_simulation(base_url: str, started: dict) -> dict:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        status, payload = _request_json(base_url, "GET", started["status_url"])
        assert status == 200
        if payload["state"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("simulation did not reach a terminal state")


class TestViewerFrontendIntegration:
    def test_post_simulate_contact_endpoint_exists(
        self,
        viewer_server_setup: tuple[str, Path, str],
    ) -> None:
        base_url, _view_dir, model_id = viewer_server_setup
        status, payload = _post_json(
            base_url,
            API_SIMULATE_CONTACT,
            {"model_id": model_id},
        )

        assert status == 202
        completed = _wait_for_simulation(base_url, payload)
        assert completed["state"] == "completed"
        assert "result" not in completed
        assert completed["profile_ms"]["serialization"] >= 0.0
        result_status, result = _request_json(
            base_url,
            "GET",
            completed["result_url"],
        )
        assert result_status == 200
        validate_simulate_contact_response(result)
        assert result["model_id"] == model_id
        assert "interior-envelope" in result["overlays"]["side"]
        assert "contact-covered" not in result["overlays"]["side"]
        assert result["metrics"]["coverage_score_percent"] >= 0.0
        assert result["overlay_profile"]["face_count"] > 0
        manifest_status, manifest = _request_json(
            base_url,
            "GET",
            f"/api/simulations/{completed['simulation_id']}/poses",
        )
        assert manifest_status == 200
        assert manifest["pose_count"] > 0
        pose_status, pose = _request_json(
            base_url,
            "GET",
            manifest["pose_url_template"].replace("{index}", "0"),
        )
        assert pose_status == 200
        assert "contact-covered" in pose["overlays"]["side"]
        assert "scraper-volume" in pose["overlays"]["side"]
        assert "scraper-contour" in pose["overlays"]["top"]
        assert pose["scraper"]["provenance"] == "captured_exact_simulation_mesh"

    def test_post_simulate_contact_accepts_trailing_slash(
        self,
        viewer_server_setup: tuple[str, Path, str],
    ) -> None:
        base_url, _view_dir, model_id = viewer_server_setup
        status, payload = _post_json(
            base_url,
            f"{API_SIMULATE_CONTACT}/",
            {"model_id": model_id},
        )

        assert status == 202
        completed = _wait_for_simulation(base_url, payload)
        assert completed["state"] == "completed"
        result_status, result = _request_json(
            base_url,
            "GET",
            completed["result_url"],
        )
        assert result_status == 200
        validate_simulate_contact_response(result)

    def test_unknown_post_endpoint_returns_explicit_error(
        self,
        viewer_server_setup: tuple[str, Path, str],
    ) -> None:
        base_url, _, _model_id = viewer_server_setup
        status, payload = _post_json(base_url, "/api/does-not-exist")

        assert status == 404
        assert payload["error"] == "Endpoint introuvable"
        assert payload["path"] == "/api/does-not-exist"
        assert API_SIMULATE_CONTACT in payload["available_endpoints"]
        assert "Endpoints POST disponibles" in payload["message"]

    def test_resolve_pose_view_dir_uses_manifest_view_dir_name(
        self,
        viewer_server_setup: tuple[str, Path, str],
    ) -> None:
        _base_url, view_dir, model_id = viewer_server_setup
        resolved = resolve_pose_view_dir(
            output_root=view_dir.parent,
            manifest={"model_id": model_id, "view_dir_name": view_dir.name},
            active_view_dir=None,
        )
        assert resolved == view_dir

    def test_resolve_view_dir_uses_frontend_model_id(
        self,
        viewer_server_setup: tuple[str, Path, str],
    ) -> None:
        base_url, view_dir, model_id = viewer_server_setup
        resolved = resolve_view_dir(
            output_root=view_dir.parent,
            active_view_dir=None,
            model_id=model_id,
        )
        assert resolved == view_dir

        status, _payload = _post_json(
            base_url,
            API_SIMULATE_CONTACT,
            {"model_id": model_id},
        )
        assert status == 202
        cancel_status, cancelled = _request_json(
            base_url,
            "DELETE",
            _payload["cancel_url"],
        )
        assert cancel_status == 200
        assert cancelled["state"] in {"cancelled", "completed"}

    def test_simulate_without_model_returns_actionable_error(
        self,
        tmp_path: Path,
    ) -> None:
        output_root = tmp_path / "views"
        output_root.mkdir()
        handler = partial(ViewerHTTPRequestHandler, directory=str(output_root))
        server = ViewerHTTPServer(("127.0.0.1", 0), handler)
        server.output_root = output_root
        server.upload_root = tmp_path / "uploads"
        server.models_root = tmp_path / "models"
        server.active_view_dir = None
        server.import_lock = threading.Lock()
        server.dev_mode = True
        server.simulation_manager = SimulationExecutionManager(tmp_path / "simulations")

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.05)
        host, port = server.server_address
        base_url = f"http://{host}:{port}"

        try:
            status, payload = _post_json(base_url, API_SIMULATE_CONTACT, {})
        finally:
            server.simulation_manager.shutdown()
            server.shutdown()
            server.server_close()

        assert status == 422
        assert payload["exception_type"] == "ValueError"
        assert "model_id" in payload["message"] or "modèle actif" in payload["message"].lower()
