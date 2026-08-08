"""CAD import routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from nutella_scraper.application.dto import ImportRequestDTO, ImportResponseDTO

router = APIRouter()


@router.post("/solidworks", response_model=ImportResponseDTO)
def import_solidworks(request: Request, body: ImportRequestDTO) -> ImportResponseDTO:
    orchestrator = request.app.state.container.orchestrator
    return orchestrator.import_model(body)
