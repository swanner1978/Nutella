"""Reconstruct InternalJarProfile via section extraction and B-spline fitting."""

from __future__ import annotations

import math

import numpy as np

from nutella_scraper.domain.models.internal_jar_profile import (
    InternalJarProfile,
    MeridianPoint,
    ProfileReconstructionQuality,
    TopContourSample,
)
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.engines.compute.curve_fitting import (
    circularity_score,
    evaluate_meridian_spline,
    evaluate_top_contour_spline,
    fit_meridian_bspline,
    fit_top_contour_bspline,
    hausdorff_mm,
)

DEFAULT_MAX_ERROR_MM = 0.12
SECTION_COUNT = 160
TOP_SECTION_COUNT = 180
CIRCULARITY_THRESHOLD = 0.992


class InternalJarProfileBuilder:
    """Extract sections from InternalJarSurface and fit faithful B-splines."""

    def from_internal(
        self,
        surface: InternalJarSurface,
        *,
        max_error_mm: float = DEFAULT_MAX_ERROR_MM,
        section_count: int = SECTION_COUNT,
        top_section_count: int = TOP_SECTION_COUNT,
    ) -> InternalJarProfile:
        y_samples, r_samples = _meridian_from_surface(
            surface,
            section_count=section_count,
        )
        meridian_spline, meridian_fit = fit_meridian_bspline(
            y_samples,
            r_samples,
            max_error_mm=max_error_mm,
        )

        y_dense = np.linspace(float(y_samples.min()), float(y_samples.max()), 240)
        r_dense = evaluate_meridian_spline(meridian_spline, y_dense)
        meridian_dense = np.column_stack([y_dense, r_dense])
        fitted_at_samples = evaluate_meridian_spline(meridian_spline, y_samples)
        meridian_hausdorff = hausdorff_mm(
            np.column_stack([y_samples, r_samples]),
            np.column_stack([y_samples, fitted_at_samples]),
        )

        reference_radius_mm = float(np.max(r_samples))
        y_ref, x_samples, z_samples = _top_contour_from_surface(
            surface,
            section_count=top_section_count,
            reference_radius_mm=reference_radius_mm,
        )
        circularity = circularity_score(x_samples, z_samples)
        is_circular = circularity >= CIRCULARITY_THRESHOLD

        if is_circular:
            theta = np.linspace(-math.pi, math.pi, top_section_count, endpoint=False)
            x_samples = reference_radius_mm * np.cos(theta)
            z_samples = reference_radius_mm * np.sin(theta)

        extracted = np.column_stack([x_samples, z_samples])

        if is_circular:
            top_spline, top_fit = _fit_exact_circle_spline(
                reference_radius_mm,
                sample_count=top_section_count,
            )
            fitted_circle = evaluate_top_contour_spline(
                top_spline,
                sample_count=max(len(extracted), top_section_count),
            )
            top_hausdorff = hausdorff_mm(extracted, fitted_circle)
        else:
            top_spline, top_fit = fit_top_contour_bspline(
                x_samples,
                z_samples,
                max_error_mm=max_error_mm,
            )
            top_dense_eval = evaluate_top_contour_spline(
                top_spline,
                sample_count=max(360, len(extracted) * 2),
            )
            top_hausdorff = hausdorff_mm(extracted, top_dense_eval)
        top_dense = evaluate_top_contour_spline(top_spline, sample_count=top_section_count)

        meridian = tuple(
            MeridianPoint(y_mm=float(y), radius_mm=float(r))
            for y, r in zip(y_dense[::8], r_dense[::8], strict=False)
        )
        top_contour = tuple(
            TopContourSample(x_mm=float(x), z_mm=float(z))
            for x, z in top_dense[::2]
        )
        mean_top_radius = float(np.mean(np.sqrt(top_dense[:, 0] ** 2 + top_dense[:, 1] ** 2)))

        return InternalJarProfile(
            jar_id=surface.jar_id,
            canonical_mesh_sha256=surface.canonical_mesh_sha256,
            meridian_spline=meridian_spline,
            top_contour_spline=top_spline,
            meridian=meridian,
            top_contour=top_contour,
            top_reference_y_mm=y_ref,
            top_inner_radius_mm=mean_top_radius,
            y_min_mm=float(y_samples.min()),
            y_max_mm=float(y_samples.max()),
            reconstruction=ProfileReconstructionQuality(
                meridian_max_error_mm=meridian_fit.max_error_mm,
                meridian_rms_error_mm=meridian_fit.rms_error_mm,
                meridian_hausdorff_mm=meridian_hausdorff,
                top_contour_max_error_mm=top_fit.max_error_mm,
                top_contour_rms_error_mm=top_fit.rms_error_mm,
                top_contour_hausdorff_mm=top_hausdorff,
                top_contour_circularity=circularity,
                top_contour_is_circular=is_circular,
            ),
            metadata={
                "builder": "InternalJarProfileBuilder",
                "section_count": section_count,
                "top_section_count": top_section_count,
                "max_error_mm": max_error_mm,
                "meridian_control_points": meridian_fit.control_point_count,
                "top_control_points": top_fit.control_point_count,
            },
        )


def offset_internal_profile(profile: InternalJarProfile, offset_mm: float) -> InternalJarProfile:
    y_values = np.array([point.y_mm for point in profile.meridian], dtype=np.float64)
    r_values = np.maximum(
        evaluate_meridian_spline(profile.meridian_spline, y_values) - offset_mm,
        0.0,
    )
    meridian_spline, meridian_fit = fit_meridian_bspline(
        y_values,
        r_values,
        max_error_mm=DEFAULT_MAX_ERROR_MM,
    )
    top_dense = evaluate_top_contour_spline(profile.top_contour_spline)
    scaled = top_dense.copy()
    radial = np.sqrt(scaled[:, 0] ** 2 + scaled[:, 1] ** 2)
    mask = radial > 1e-9
    scale = np.ones_like(radial)
    scale[mask] = np.maximum(radial[mask] - offset_mm, 0.0) / radial[mask]
    scaled[:, 0] *= scale
    scaled[:, 1] *= scale
    top_spline, top_fit = fit_top_contour_bspline(
        scaled[:, 0],
        scaled[:, 1],
        max_error_mm=DEFAULT_MAX_ERROR_MM,
    )
    return InternalJarProfile(
        jar_id=profile.jar_id,
        canonical_mesh_sha256=profile.canonical_mesh_sha256,
        meridian_spline=meridian_spline,
        top_contour_spline=top_spline,
        meridian=tuple(
            MeridianPoint(y_mm=float(y), radius_mm=float(r))
            for y, r in zip(y_values, r_values, strict=True)
        ),
        top_contour=tuple(
            TopContourSample(x_mm=float(x), z_mm=float(z)) for x, z in scaled
        ),
        top_reference_y_mm=profile.top_reference_y_mm,
        top_inner_radius_mm=max(profile.top_inner_radius_mm - offset_mm, 0.0),
        y_min_mm=profile.y_min_mm,
        y_max_mm=profile.y_max_mm,
        reconstruction=ProfileReconstructionQuality(
            meridian_max_error_mm=meridian_fit.max_error_mm,
            meridian_rms_error_mm=meridian_fit.rms_error_mm,
            meridian_hausdorff_mm=profile.reconstruction.meridian_hausdorff_mm,
            top_contour_max_error_mm=top_fit.max_error_mm,
            top_contour_rms_error_mm=top_fit.rms_error_mm,
            top_contour_hausdorff_mm=profile.reconstruction.top_contour_hausdorff_mm,
            top_contour_circularity=profile.reconstruction.top_contour_circularity,
            top_contour_is_circular=profile.reconstruction.top_contour_is_circular,
        ),
        metadata={
            **profile.metadata,
            "radial_offset_mm": offset_mm,
        },
    )


def _meridian_from_surface(
    surface: InternalJarSurface,
    *,
    section_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(surface.sample_points_mm, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("InternalJarSurface has no sample points for profile extraction")

    y_values = points[:, 1]
    radial = np.sqrt(points[:, 0] ** 2 + points[:, 2] ** 2)
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    bins = np.linspace(y_min, y_max, max(section_count, 8))
    half = max((y_max - y_min) / max(section_count * 2, 4), 1e-6)

    y_samples: list[float] = []
    r_samples: list[float] = []
    max_radius = float(np.max(radial))
    threshold = 0.45 * max_radius
    for y_value in bins:
        mask = np.abs(y_values - y_value) <= half
        if not np.any(mask):
            continue
        slice_radii = radial[mask]
        radius = float(np.max(slice_radii))
        if radius < threshold:
            continue
        y_samples.append(float(y_value))
        r_samples.append(radius)

    if len(y_samples) < 4:
        for slice_ in surface.slices:
            if slice_.inner_radius_mm >= threshold:
                y_samples.append(float(slice_.y_mm))
                r_samples.append(float(slice_.inner_radius_mm))

    if len(y_samples) < 4:
        raise ValueError("Unable to extract enough meridian samples from InternalJarSurface")

    order = np.argsort(y_samples)
    y_sorted = np.asarray(y_samples, dtype=np.float64)[order]
    r_sorted = np.asarray(r_samples, dtype=np.float64)[order]
    return y_sorted, r_sorted


def _top_contour_from_surface(
    surface: InternalJarSurface,
    *,
    section_count: int,
    reference_radius_mm: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    points = np.asarray(surface.sample_points_mm, dtype=np.float64)
    y_values = points[:, 1]
    y_ref = float(y_values.max())
    span = max(float(y_values.max() - y_values.min()), 1e-6)
    half = max(span / max(section_count, 8), 1e-6)
    mask = y_values >= y_ref - half
    section = points[mask]
    if len(section) < 8:
        section = points

    x_values = section[:, 0]
    z_values = section[:, 2]
    radial = np.sqrt(x_values**2 + z_values**2)
    max_radius = float(np.max(radial))
    threshold = 0.45 * max_radius
    wall_mask = radial >= threshold
    if np.count_nonzero(wall_mask) >= 8:
        x_values = x_values[wall_mask]
        z_values = z_values[wall_mask]

    theta = np.arctan2(z_values, x_values)
    theta_bins = np.linspace(-math.pi, math.pi, max(section_count, 16), endpoint=False)
    x_samples: list[float] = []
    z_samples: list[float] = []
    for left, right in zip(theta_bins, np.roll(theta_bins, -1), strict=False):
        t_mask = (theta >= left) & (theta < right)
        if not np.any(t_mask):
            continue
        candidates_x = x_values[t_mask]
        candidates_z = z_values[t_mask]
        candidate_radius = np.sqrt(candidates_x**2 + candidates_z**2)
        if reference_radius_mm is not None and reference_radius_mm > 1e-6:
            best = int(np.argmin(np.abs(candidate_radius - reference_radius_mm)))
        else:
            best = int(np.argmax(candidate_radius))
        x_samples.append(float(candidates_x[best]))
        z_samples.append(float(candidates_z[best]))

    if len(x_samples) < 8:
        x_samples = [float(value) for value in x_values]
        z_samples = [float(value) for value in z_values]

    return y_ref, np.asarray(x_samples, dtype=np.float64), np.asarray(z_samples, dtype=np.float64)


def _fit_exact_circle_spline(radius_mm: float, *, sample_count: int):
    from nutella_scraper.engines.compute.curve_fitting import BSpline2DLoop, CurveFitMetrics

    theta = np.linspace(0.0, 2.0 * math.pi, max(sample_count, 16), endpoint=False)
    x = radius_mm * np.cos(theta)
    z = radius_mm * np.sin(theta)
    loop, metrics = fit_top_contour_bspline(x, z, max_error_mm=1e-6)
    return loop, CurveFitMetrics(
        max_error_mm=metrics.max_error_mm,
        rms_error_mm=metrics.rms_error_mm,
        sample_count=len(theta),
        control_point_count=metrics.control_point_count,
    )
