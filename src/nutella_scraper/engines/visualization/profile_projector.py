"""Project InternalJarProfile B-splines to SVG paths — no mesh edges."""

from __future__ import annotations

import numpy as np

from nutella_scraper.domain.models.internal_jar_profile import InternalJarProfile
from nutella_scraper.domain.models.views import SvgLayer
from nutella_scraper.engines.compute.curve_fitting import (
    evaluate_meridian_spline,
    evaluate_top_contour_spline,
)
from nutella_scraper.engines.visualization.projection_math import (
    PLANE_PROFILE,
    PLANE_TOP_XZ,
    fit_to_viewport,
)

LAYER_INTERIOR_PROFILE = "interior-envelope"
PROFILE_STROKE = "#a855f7"
PROFILE_STROKE_WIDTH = 1.1
POT_STROKE = "#ffffff"
POT_STROKE_WIDTH = 2.0

MERIDIAN_SAMPLES = 180
TOP_SAMPLES = 160


def profile_side_path(profile: InternalJarProfile) -> np.ndarray:
    """Closed meridian in profile plane: x = radius, y = height."""
    y_dense = np.linspace(profile.y_min_mm, profile.y_max_mm, MERIDIAN_SAMPLES)
    r_dense = evaluate_meridian_spline(profile.meridian_spline, y_dense)
    right = np.column_stack([r_dense, y_dense])
    left = np.column_stack([-r_dense[::-1], y_dense[::-1]])
    return np.vstack([right, left[1:]])


def profile_top_path(profile: InternalJarProfile) -> np.ndarray:
    """Closed top contour in XZ plane."""
    return evaluate_top_contour_spline(
        profile.top_contour_spline,
        sample_count=TOP_SAMPLES,
    )


def _smooth_path(
    coords: np.ndarray,
    *,
    scale: float,
    offset: np.ndarray,
    css_class: str,
    stroke: str,
    stroke_width: float,
) -> str:
    if len(coords) == 0:
        return ""
    projected = coords * scale + offset
    commands = [
        (
            f"{'M' if index == 0 else 'L'}"
            f"{projected[index, 0]:.2f},{projected[index, 1]:.2f}"
        )
        for index in range(len(projected))
    ]
    commands.append("Z")
    return (
        f'<path class="{css_class}" d="{" ".join(commands)}" fill="none" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" '
        f'stroke-linejoin="round" stroke-linecap="round" '
        f'vector-effect="non-scaling-stroke"/>'
    )


def profile_layers_for_plane(
    profile: InternalJarProfile,
    *,
    plane: str,
    view_key: str,
    css_class: str = "interior-profile-contour",
    stroke: str = PROFILE_STROKE,
    stroke_width: float = PROFILE_STROKE_WIDTH,
    layer_type: str = LAYER_INTERIOR_PROFILE,
) -> tuple[SvgLayer, ...]:
    if plane == PLANE_PROFILE:
        coords = profile_side_path(profile)
    elif plane == PLANE_TOP_XZ:
        coords = profile_top_path(profile)
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")

    if len(coords) == 0:
        return ()

    scale, offset = fit_to_viewport(coords)
    path = _smooth_path(
        coords,
        scale=scale,
        offset=offset,
        css_class=css_class,
        stroke=stroke,
        stroke_width=stroke_width,
    )
    if not path:
        return ()
    return (
        SvgLayer(
            id=f"{view_key}-{layer_type}",
            z_index=15,
            svg_fragment=path,
            layer_type=layer_type,
        ),
    )


def profile_contour_svg_fragment(
    profile: InternalJarProfile,
    *,
    plane: str,
    css_class: str = "contour",
    stroke: str = POT_STROKE,
    stroke_width: float = POT_STROKE_WIDTH,
) -> str:
    if plane == PLANE_PROFILE:
        coords = profile_side_path(profile)
    elif plane == PLANE_TOP_XZ:
        coords = profile_top_path(profile)
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")
    if len(coords) == 0:
        return ""
    scale, offset = fit_to_viewport(coords)
    return _smooth_path(
        coords,
        scale=scale,
        offset=offset,
        css_class=css_class,
        stroke=stroke,
        stroke_width=stroke_width,
    )
