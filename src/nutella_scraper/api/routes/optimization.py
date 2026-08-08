"""Optimization routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from nutella_scraper.application.dto import OptimizationRequestDTO, OptimizationResponseDTO

router = APIRouter()


@router.post("/runs", response_model=OptimizationResponseDTO)
def start_optimization(
    request: Request,
    body: OptimizationRequestDTO,
) -> OptimizationResponseDTO:
    orchestrator = request.app.state.container.orchestrator
    return orchestrator.start_optimization(body)


@router.get("/runs/{run_id}")
def get_optimization_run(request: Request, run_id: str) -> dict[str, str]:
    return {"run_id": run_id, "status": "not_implemented"}
