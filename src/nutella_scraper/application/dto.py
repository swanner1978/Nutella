"""Application DTOs for API and CLI boundaries."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImportRequestDTO(BaseModel):
    sldprt_path: str | None = None
    step_path: str | None = None
    stl_path: str | None = None


class ImportResponseDTO(BaseModel):
    model_id: str
    views_id: str | None = None
    source_hash: str


class SimulateRequestDTO(BaseModel):
    model_id: str
    jar_id: str = "nutella_400g"
    simulation_profile: str = "contact_default"


class SimulateResponseDTO(BaseModel):
    coverage_score: float
    contact_result_id: str | None = None
    feasible: bool = True


class OptimizationRequestDTO(BaseModel):
    jar_id: str = "nutella_400g"
    design_space_id: str = "default"
    optimization_profile: str = "fdm_pla"


class OptimizationResponseDTO(BaseModel):
    run_id: str
    status: str


class ViewOverlayResponseDTO(BaseModel):
    model_id: str
    profile_svg: str
    top_svg: str
    coverage_score_display: float = Field(
        description="Read-only copy of ComputeEngine coverage_score"
    )
