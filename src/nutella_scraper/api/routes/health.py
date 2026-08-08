"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter

from nutella_scraper.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse()
