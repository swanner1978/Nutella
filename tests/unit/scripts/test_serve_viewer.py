"""Tests for atomic viewer projection replacement."""

from __future__ import annotations

import http.client
import threading
import time
import urllib.request
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import pytest
from scripts.serve_viewer import (
    ViewerHTTPRequestHandler,
    ViewerHTTPServer,
    _remove_previous_view,
    _remove_stale_view_directories,
    build_error_payload,
    viewer_document_path,
)

from nutella_scraper.application.simulation_execution import SimulationExecutionManager


def test_viewer_document_path_aliases_index_to_html() -> None:
    assert viewer_document_path("") == "/index.html"
    assert viewer_document_path("/") == "/index.html"
    assert viewer_document_path("/index") == "/index.html"
    assert viewer_document_path("/index.html") == "/index.html"
    assert viewer_document_path("/api/runtime") == "/api/runtime"
    assert viewer_document_path("/favicon.ico") == "/favicon.ico"


def test_index_html_is_served_inline_not_as_attachment(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(
        "<!doctype html><html lang='fr'><head><title>viewer</title></head><body>ok</body></html>",
        encoding="utf-8",
    )
    handler = partial(ViewerHTTPRequestHandler, directory=str(tmp_path))
    server = ViewerHTTPServer(("127.0.0.1", 0), handler)
    server.output_root = tmp_path
    server.upload_root = tmp_path / "uploads"
    server.models_root = tmp_path / "models"
    server.active_view_dir = None
    server.import_lock = threading.Lock()
    server.dev_mode = False
    server.simulation_manager = SimulationExecutionManager(tmp_path / "simulations")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    try:
        host, port = server.server_address
        base = f"http://{host}:{port}"
        with urllib.request.urlopen(f"{base}/index.html", timeout=5) as response:
            content_type = response.headers.get("Content-Type") or response.headers.get(
                "Content-type"
            )
            disposition = response.headers.get("Content-Disposition")
            body = response.read().decode("utf-8")
            assert response.status == 200
            assert content_type is not None
            assert content_type.startswith("text/html")
            assert "attachment" not in (disposition or "").lower()
            assert (disposition or "inline").lower().startswith("inline")
            assert "<!doctype html>" in body.lower()
        parsed = urlparse(base)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            connection.request("GET", "/index")
            redirect = connection.getresponse()
            assert redirect.status == 302
            assert redirect.getheader("Location") == "/index.html"
            assert "attachment" not in (redirect.getheader("Content-Disposition") or "").lower()
        finally:
            connection.close()
    finally:
        server.simulation_manager.shutdown()
        server.shutdown()
        server.server_close()


def test_build_error_payload_includes_traceback_in_dev_mode() -> None:
    request_id = "req-123"
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        payload = build_error_payload(
            exc,
            request_id=request_id,
            stage="import_pipeline",
            dev_mode=True,
        )

    assert payload["exception_type"] == "RuntimeError"
    assert payload["message"] == "boom"
    assert payload["stage"] == "import_pipeline"
    assert payload["request_id"] == request_id
    assert "Traceback" in payload["traceback"]


def test_build_error_payload_hides_traceback_outside_dev_mode() -> None:
    payload = build_error_payload(
        ValueError("invalid"),
        request_id="req-456",
        stage="validate_size",
        dev_mode=False,
    )

    assert payload["exception_type"] == "ValueError"
    assert "traceback" not in payload


def test_remove_previous_view_deletes_only_active_projection(tmp_path: Path) -> None:
    output_root = tmp_path / "views"
    previous = output_root / "old-model"
    current = output_root / "new-model"
    previous.mkdir(parents=True)
    current.mkdir()
    (previous / "profile.svg").write_text("old", encoding="utf-8")
    (current / "profile.svg").write_text("new", encoding="utf-8")

    _remove_previous_view(previous, output_root)

    assert not previous.exists()
    assert current.exists()


def test_remove_previous_view_refuses_path_outside_viewer(tmp_path: Path) -> None:
    output_root = tmp_path / "views"
    outside = tmp_path / "models" / "canonical"
    output_root.mkdir()
    outside.mkdir(parents=True)

    with pytest.raises(ValueError, match="hors du viewer"):
        _remove_previous_view(outside, output_root)

    assert outside.exists()


def test_remove_stale_view_directories_keeps_only_current_import(tmp_path: Path) -> None:
    output_root = tmp_path / "views"
    current = output_root / "current"
    stale = output_root / "stale"
    unrelated = output_root / "assets"
    for directory in (current, stale, unrelated):
        directory.mkdir(parents=True)
    (current / "metadata.json").write_text('{"model_id":"current"}', encoding="utf-8")
    (stale / "metadata.json").write_text('{"model_id":"stale"}', encoding="utf-8")
    (unrelated / "metadata.json").write_text('{"kind":"other"}', encoding="utf-8")

    _remove_stale_view_directories(output_root, current)

    assert current.exists()
    assert not stale.exists()
    assert unrelated.exists()


def test_viewer_template_exposes_progress_cancel_and_diagnostics() -> None:
    template = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "templates"
        / "demo_viewer.html"
    ).read_text(encoding="utf-8")

    assert 'id="cancel-simulation"' not in template
    assert 'id="toolbar-toggle-overlays"' not in template
    assert 'id="toolbar-toggle-envelope"' not in template
    assert 'id="toolbar-toggle-trajectory"' not in template
    assert 'id="toolbar-reset"' not in template
    assert 'id="simulate-contact"' in template
    assert 'id="toggle-envelope"' in template
    assert 'id="simulation-progress"' in template
    assert 'id="simulation-diagnostics"' in template
    assert 'method: "DELETE"' in template
    assert "async function cancelSimulation()" in template
    assert template.count("applyContactOverlays(pose);") == 1
    assert "result_url" in template
    assert "frontend_transfer" in template
    assert 'id="trajectory-pose"' in template
    assert 'data-visibility-toggle="scraper"' in template
    assert 'data-visibility-toggle="pot"' in template
    assert 'id="toolbar-build-scraper"' in template
    assert "/api/build-scraper" in template
    assert 'id="toggle-scene-scraper"' in template
    assert 'id="toggle-coordinate-frame"' in template


def test_build_scraper_request_defaults_skip_svg_overlays() -> None:
    from scripts.viewer_api import BuildScraperRequest

    assert BuildScraperRequest.from_dict(None).include_svg_overlays is False
    assert BuildScraperRequest.from_dict({"model_id": "x"}).include_svg_overlays is False
    assert (
        BuildScraperRequest.from_dict({"include_svg_overlays": True}).include_svg_overlays
        is True
    )
    assert (
        BuildScraperRequest.from_dict({"include_svg_overlays": "true"}).include_svg_overlays
        is False
    )
