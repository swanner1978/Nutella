"""Pydantic schemas for HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nutella_scraper.application.dto import (
    ImportRequestDTO,
    ImportResponseDTO,
    OptimizationRequestDTO,
    OptimizationResponseDTO,
    SimulateRequestDTO,
    SimulateResponseDTO,
    ViewOverlayResponseDTO,
)

__all__ = [
    "ImportRequestDTO",
    "ImportResponseDTO",
    "OptimizationRequestDTO",
    "OptimizationResponseDTO",
    "SimulateRequestDTO",
    "SimulateResponseDTO",
    "ViewOverlayResponseDTO",
]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ModelMetadataResponse(BaseModel):
    model_id: str
    source_hash: str
    format: str


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"


class WebSocketSimulateMessage(BaseModel):
    model_id: str
    jar_id: str = "nutella_400g"
    simulation_profile: str = "contact_default"


class WebSocketSimulateResult(BaseModel):
    coverage_score: float = Field(description="From ComputeEngine — not derived from pixels")
    contact_result_id: str | None = None
