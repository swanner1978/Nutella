"""Analytical B-spline representation of the jar interior cavity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nutella_scraper.engines.compute.curve_fitting import BSpline1D, BSpline2DLoop


@dataclass(frozen=True)
class MeridianPoint:
    """Sample on the fitted meridian (y, inner radius)."""

    y_mm: float
    radius_mm: float


@dataclass(frozen=True)
class TopContourSample:
    """One point on the horizontal inner contour (XZ plane, Y-up world)."""

    x_mm: float
    z_mm: float


@dataclass(frozen=True)
class ProfileReconstructionQuality:
    meridian_max_error_mm: float
    meridian_rms_error_mm: float
    meridian_hausdorff_mm: float
    top_contour_max_error_mm: float
    top_contour_rms_error_mm: float
    top_contour_hausdorff_mm: float
    top_contour_circularity: float
    top_contour_is_circular: bool


@dataclass(frozen=True)
class InternalJarProfile:
    """
    CAD-like interior profile reconstructed from section data + B-splines.

    Triangle edges are never displayed — only fitted curves derived from
    aggregated inner-wall sections.
    """

    jar_id: str
    canonical_mesh_sha256: str
    meridian_spline: BSpline1D
    top_contour_spline: BSpline2DLoop
    meridian: tuple[MeridianPoint, ...]
    top_contour: tuple[TopContourSample, ...]
    top_reference_y_mm: float
    top_inner_radius_mm: float
    y_min_mm: float
    y_max_mm: float
    reconstruction: ProfileReconstructionQuality
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)

    @property
    def meridian_point_count(self) -> int:
        return len(self.meridian)

    @property
    def top_contour_point_count(self) -> int:
        return len(self.top_contour)

    def meridian_radius_at(self, y_mm: float) -> float:
        from nutella_scraper.engines.compute.curve_fitting import evaluate_meridian_spline

        y_clamped = float(np.clip(y_mm, self.y_min_mm, self.y_max_mm))
        return float(evaluate_meridian_spline(self.meridian_spline, np.array([y_clamped]))[0])

    def with_radial_offset(self, offset_mm: float) -> InternalJarProfile:
        from nutella_scraper.engines.compute.internal_jar_profile_builder import (
            offset_internal_profile,
        )

        return offset_internal_profile(self, offset_mm)
