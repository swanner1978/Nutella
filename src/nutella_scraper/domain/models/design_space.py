"""Design space and optimization domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParameterType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    INTEGER = "integer"


@dataclass(frozen=True)
class DesignParameter:
    """Single optimizable design parameter."""

    name: str
    type: ParameterType
    lower_bound: float
    upper_bound: float
    default: float
    unit: str = "mm"


@dataclass(frozen=True)
class DesignSpace:
    """
    Parametric design space for scraper geometry.

    Maps to SolidWorks driving dimensions or meta-parameters.
    """

    id: str
    parameters: tuple[DesignParameter, ...]
    version: str = "1"


@dataclass(frozen=True)
class ParameterSample:
    """Single sample drawn from DesignSpace."""

    values: dict[str, float]


@dataclass(frozen=True)
class OptimizationBudget:
    """Resource budget for an optimization run."""

    max_trials: int = 200
    timeout_s: int = 3600
    seed: int = 42


@dataclass(frozen=True)
class ObjectiveSpec:
    """Objective function specification."""

    name: str
    weight: float
    direction: str  # "maximize" | "minimize"


@dataclass(frozen=True)
class DesignCandidate:
    """Single evaluated design candidate."""

    id: str
    parameter_sample: ParameterSample
    metrics: dict[str, float]
    feasible: bool
    model_id: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a completed optimization run."""

    run_id: str
    candidates: tuple[DesignCandidate, ...]
    pareto_front_ids: tuple[str, ...]
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationRun:
    """Persisted optimization run metadata."""

    id: str
    design_space_id: str
    jar_id: str
    status: str
    budget: OptimizationBudget
    seed: int
    created_at: str
    updated_at: str
