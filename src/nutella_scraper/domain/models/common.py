"""Shared domain types and provenance markers."""

from __future__ import annotations

from typing import Literal

Provenance = Literal["canonical_3d", "visualization_projection", "computed_metric"]
ModelFormat = Literal["step", "stl"]
ExportFormat = Literal["stl", "step", "3mf"]
