"""Visualization routes — read-only view overlays."""

from __future__ import annotations

from fastapi import APIRouter, Request

from nutella_scraper.application.dto import ViewOverlayResponseDTO

router = APIRouter()


@router.get("/models/{model_id}/overlay", response_model=ViewOverlayResponseDTO)
def get_view_overlay(
    request: Request,
    model_id: str,
    contact_result_id: str,
) -> ViewOverlayResponseDTO:
    orchestrator = request.app.state.container.orchestrator
    return orchestrator.get_view_overlay(model_id, contact_result_id)
