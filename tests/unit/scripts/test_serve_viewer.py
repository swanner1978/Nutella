"""Tests for atomic viewer projection replacement."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.serve_viewer import (
    _remove_previous_view,
    _remove_stale_view_directories,
    build_error_payload,
)


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

    assert 'id="cancel-simulation"' in template
    assert 'id="simulation-progress"' in template
    assert 'id="simulation-diagnostics"' in template
    assert 'method: "DELETE"' in template
    assert template.count("applyContactOverlays(pose);") == 1
    assert "result_url" in template
    assert "frontend_transfer" in template
    assert 'id="trajectory-pose"' in template
    assert 'data-visibility-toggle="scraper"' in template
    assert 'data-visibility-toggle="pot"' in template
    assert 'id="toolbar-build-scraper"' in template
    assert "/api/build-scraper" in template
