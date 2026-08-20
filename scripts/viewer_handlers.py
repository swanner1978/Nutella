"""HTTP handlers for viewer API routes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts.viewer_api import (
    BuildScraperRequest,
    ScraperShapeCandidatesRequest,
    SimulateContactRequest,
    validate_simulate_contact_response,
)

from nutella_scraper.engines.visualization.viewer_bridge import (
    build_contact_visualization_response,
)

_LOG = logging.getLogger("nutella_scraper.serve_viewer")


def handle_simulate_contact(
    *,
    view_dir: Path,
    models_root: Path,
    request_id: str,
) -> dict[str, Any]:
    """Run contact simulation and return the viewer overlay payload."""
    payload = build_contact_visualization_response(
        view_dir=view_dir,
        models_root=models_root,
    )
    validate_simulate_contact_response(payload)
    payload["request_id"] = request_id
    _LOG.info(
        "[simulate:%s] model_id=%s | coverage=%.3f",
        request_id,
        payload["model_id"],
        payload["coverage_score_display"],
    )
    return payload


def read_simulate_contact_request(raw_body: bytes) -> SimulateContactRequest:
    """Parse POST /api/simulate-contact JSON body."""
    from scripts.viewer_api import parse_json_body

    if not raw_body:
        return SimulateContactRequest()
    try:
        return SimulateContactRequest.from_dict(parse_json_body(raw_body))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corps JSON invalide pour /api/simulate-contact : {exc}") from exc


def read_viewer_model_request(raw_body: bytes) -> SimulateContactRequest:
    """Parse POST bodies that accept an optional ``model_id``."""
    return read_simulate_contact_request(raw_body)


def read_build_scraper_request(raw_body: bytes) -> BuildScraperRequest:
    """Parse POST /api/build-scraper JSON body (model_id + optional parameters)."""
    from scripts.viewer_api import BuildScraperRequest, parse_json_body

    if not raw_body:
        return BuildScraperRequest()
    try:
        return BuildScraperRequest.from_dict(parse_json_body(raw_body))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corps JSON invalide pour /api/build-scraper : {exc}") from exc


def read_scraper_shape_candidates_request(raw_body: bytes) -> ScraperShapeCandidatesRequest:
    """Parse POST /api/scraper-shape-candidates JSON body."""
    from scripts.viewer_api import ScraperShapeCandidatesRequest, parse_json_body

    if not raw_body:
        return ScraperShapeCandidatesRequest()
    try:
        return ScraperShapeCandidatesRequest.from_dict(parse_json_body(raw_body))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Corps JSON invalide pour /api/scraper-shape-candidates : {exc}"
        ) from exc
