"""Parametric scraper-profile families in one sagittal frame (A0 meridian).

All families are graphs ``r(y)``: parameter ``t`` in [0, 1] runs from the
opening (top) to the floor. ``y(t)`` is strictly decreasing by construction.
This module generates curves only — it does not loft, collide, or score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import (
    ScraperEnvelopePathBuilder,
    scraper_length_span,
)
from nutella_scraper.engines.compute.trajectory_contact_cache import (
    reference_scraper_parameters,
)

PROFILE_SAMPLE_COUNT = 48
# Thin blade: thickness is the only section size. Width is fixed, never a
# shape-search parameter (not 10 mm+).
SCRAPER_THICKNESS_MM = 2.0
SCRAPER_WIDTH_MM = 2.0
BLADE_THICKNESS_MM = SCRAPER_THICKNESS_MM
BLADE_WIDTH_MM = SCRAPER_WIDTH_MM
# Physical blade lengths. Independent of jar height and of trajectory length.
# 40 mm is the A0 historical solid, not a claimed optimum.
DEFAULT_SCRAPER_LENGTH_MM = 40.0
SCRAPER_LENGTHS_MM: tuple[float, ...] = (20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0)


@dataclass(frozen=True)
class SagittalFrame:
    """Shared (y, r) frame of the A0 design meridian. Not a coverage grid."""

    y_top_mm: float
    y_bot_mm: float
    meridian_y_mm: NDArray[np.float64]
    meridian_r_mm: NDArray[np.float64]
    meridian_xyz_mm: NDArray[np.float64]
    r_max_mm: float
    useful_height_mm: float

    def y_at(self, t: NDArray[np.float64] | float) -> NDArray[np.float64]:
        tt = np.asarray(t, dtype=np.float64)
        return float(self.y_top_mm) + tt * (float(self.y_bot_mm) - float(self.y_top_mm))

    def window_for_length(self, length_mm: float) -> SagittalFrame:
        """First ``length_mm`` of the opening-to-floor meridian. Not the full jar."""
        requested = float(length_mm)
        if requested <= 1e-6:
            raise ValueError("length_mm must be positive")
        cap = max(float(self.useful_height_mm), 1.0)
        target = min(requested, cap)
        xyz = np.asarray(self.meridian_xyz_mm, dtype=np.float64)
        if len(xyz) < 2:
            raise ValueError("Meridian is too short to window")
        ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(ds)])
        target = min(target, float(arc[-1]))
        n = max(16, int(np.ceil(target)) + 1)
        s_query = np.linspace(0.0, target, n, dtype=np.float64)
        y = np.interp(s_query, arc, self.meridian_y_mm)
        r = np.interp(s_query, arc, self.meridian_r_mm)
        x = np.interp(s_query, arc, xyz[:, 0])
        z = np.interp(s_query, arc, xyz[:, 2])
        wall = np.column_stack((x, y, z))
        return SagittalFrame(
            y_top_mm=float(y[0]),
            y_bot_mm=float(y[-1]),
            meridian_y_mm=np.asarray(y, dtype=np.float64),
            meridian_r_mm=np.asarray(r, dtype=np.float64),
            meridian_xyz_mm=np.asarray(wall, dtype=np.float64),
            r_max_mm=float(np.max(r)),
            useful_height_mm=float(target),
        )

    def r_wall_at_y(self, y_mm: NDArray[np.float64] | float) -> NDArray[np.float64]:
        y = np.asarray(y_mm, dtype=np.float64)
        order = np.argsort(self.meridian_y_mm)
        return np.interp(
            y,
            self.meridian_y_mm[order],
            self.meridian_r_mm[order],
        )


@dataclass(frozen=True)
class SampledProfile:
    """One candidate centreline in the A0 sagittal plane."""

    family_id: str
    parameters: tuple[float, ...]
    t: NDArray[np.float64]
    y_mm: NDArray[np.float64]
    r_mm: NDArray[np.float64]
    points_mm: NDArray[np.float64]
    length_mm: float = 0.0


def build_sagittal_frame(surface: InteriorSurfaceReference) -> SagittalFrame:
    """A0 meridian of the interior envelope. Not the historical A0 point grid."""
    params = reference_scraper_parameters(surface)
    opening_y, lower_y, max_length = scraper_length_span(surface)
    full = params.with_updates(length_mm=float(max_length))
    wall = np.asarray(
        ScraperEnvelopePathBuilder().build(surface, full).wall_curve_mm,
        dtype=np.float64,
    )
    if len(wall) < 2:
        raise ValueError("Interior meridian is too short for a sagittal frame")
    y = np.asarray(wall[:, 1], dtype=np.float64)
    r = np.hypot(wall[:, 0], wall[:, 2])
    if float(y[0]) < float(y[-1]):
        wall = wall[::-1]
        y = y[::-1]
        r = r[::-1]
    return SagittalFrame(
        y_top_mm=float(opening_y),
        y_bot_mm=float(lower_y),
        meridian_y_mm=y,
        meridian_r_mm=r,
        meridian_xyz_mm=wall,
        r_max_mm=float(np.max(r)),
        useful_height_mm=float(max_length),
    )


def _bezier_1d(ctrl: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    degree = len(ctrl) - 1
    out = np.zeros_like(t, dtype=np.float64)
    for k, pk in enumerate(ctrl):
        out += float(comb(degree, k)) * (t**k) * ((1.0 - t) ** (degree - k)) * float(pk)
    return out


def _linear_r(
    t: NDArray[np.float64],
    r0: float,
    r1: float,
) -> NDArray[np.float64]:
    return (1.0 - t) * float(r0) + t * float(r1)


def _poly_r(
    t: NDArray[np.float64],
    r0: float,
    r1: float,
    extras: NDArray[np.float64],
) -> NDArray[np.float64]:
    base = _linear_r(t, r0, r1)
    if len(extras) == 0:
        return base
    bump = np.zeros_like(t, dtype=np.float64)
    for k, ck in enumerate(extras):
        bump += float(ck) * (t ** (k + 1))
    return base + t * (1.0 - t) * bump


def _circle_from_3(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float] | None:
    d = 2.0 * (
        a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])
    )
    if abs(d) < 1e-9:
        return None
    a2 = float(np.dot(a, a))
    b2 = float(np.dot(b, b))
    c2 = float(np.dot(c, c))
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    center = np.array([ux, uy], dtype=np.float64)
    radius = float(np.linalg.norm(center - a))
    if radius < 1e-6:
        return None
    return center, radius


def _arc_r(
    t: NDArray[np.float64],
    frame: SagittalFrame,
    r0: float,
    r1: float,
    bulge_mm: float,
) -> NDArray[np.float64]:
    if abs(float(bulge_mm)) < 1e-9:
        return _linear_r(t, r0, r1)
    p0 = np.array([float(frame.y_top_mm), float(r0)], dtype=np.float64)
    p1 = np.array([float(frame.y_bot_mm), float(r1)], dtype=np.float64)
    chord = p1 - p0
    nrm = float(np.linalg.norm(chord))
    if nrm < 1e-9:
        return _linear_r(t, r0, r1)
    normal = np.array([-chord[1], chord[0]], dtype=np.float64) / nrm
    mid = 0.5 * (p0 + p1) + float(bulge_mm) * normal
    circle = _circle_from_3(p0, mid, p1)
    if circle is None:
        return _linear_r(t, r0, r1)
    center, radius = circle
    a0 = float(np.arctan2(p0[1] - center[1], p0[0] - center[0]))
    am = float(np.arctan2(mid[1] - center[1], mid[0] - center[0]))
    a1 = float(np.arctan2(p1[1] - center[1], p1[0] - center[0]))

    def _unwrap(start: float, through: float, end: float) -> NDArray[np.float64]:
        path = [start]
        cur = start
        for target in (through, end):
            delta = (target - cur + np.pi) % (2.0 * np.pi) - np.pi
            cur = cur + delta
            path.append(cur)
        s = np.linspace(0.0, 1.0, 64)
        ang01 = path[0] + s * (path[1] - path[0])
        ang12 = path[1] + s * (path[2] - path[1])
        return np.concatenate([ang01, ang12[1:]])

    angles = _unwrap(a0, am, a1)
    pts = np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        )
    )
    y_line = pts[:, 0]
    if float(np.min(np.diff(y_line))) >= -1e-9:
        y_line = y_line[::-1]
        pts = pts[::-1]
    y_query = frame.y_at(t)
    return np.interp(y_query, y_line[::-1], pts[::-1, 1])


def _sigmoid_r(
    t: NDArray[np.float64],
    r0: float,
    r1: float,
    t0: float,
    k: float,
) -> NDArray[np.float64]:
    width = max(float(k), 1e-3)
    s = 1.0 / (1.0 + np.exp(-(t - float(t0)) / width))
    s0 = 1.0 / (1.0 + np.exp(-(0.0 - float(t0)) / width))
    s1 = 1.0 / (1.0 + np.exp(-(1.0 - float(t0)) / width))
    denom = max(float(s1 - s0), 1e-9)
    u = (s - s0) / denom
    return _linear_r(u, r0, r1)


def _fourier_r(
    t: NDArray[np.float64],
    a0: float,
    harmonics: NDArray[np.float64],
) -> NDArray[np.float64]:
    out = np.full_like(t, float(a0), dtype=np.float64)
    n = len(harmonics) // 2
    for k in range(n):
        ak = float(harmonics[2 * k])
        bk = float(harmonics[2 * k + 1])
        ang = 2.0 * np.pi * (k + 1) * t
        out += ak * np.cos(ang) + bk * np.sin(ang)
    return out


RadiusFn = Callable[[NDArray[np.float64], NDArray[np.float64], SagittalFrame], NDArray[np.float64]]


@dataclass(frozen=True)
class ShapeFamily:
    family_id: str
    display_name: str
    n_parameters: int
    _radius: RadiusFn
    _defaults: Callable[[SagittalFrame], NDArray[np.float64]]
    _bounds: Callable[[SagittalFrame], NDArray[np.float64]]

    def default_params(self, frame: SagittalFrame) -> NDArray[np.float64]:
        return np.asarray(self._defaults(frame), dtype=np.float64)

    def bounds(self, frame: SagittalFrame) -> NDArray[np.float64]:
        return np.asarray(self._bounds(frame), dtype=np.float64)

    def radius(
        self,
        t: NDArray[np.float64],
        params: NDArray[np.float64],
        frame: SagittalFrame,
    ) -> NDArray[np.float64]:
        return np.asarray(self._radius(t, params, frame), dtype=np.float64)


def _r_span(frame: SagittalFrame) -> tuple[float, float, float]:
    r0 = float(frame.meridian_r_mm[0])
    r1 = float(frame.meridian_r_mm[-1])
    r_hi = max(float(frame.r_max_mm) * 1.05, 1.0)
    return r0, r1, r_hi


def _r_bounds(n: int, r_hi: float) -> NDArray[np.float64]:
    return np.tile(np.array([[0.0, r_hi]], dtype=np.float64), (n, 1))


def _poly_family(degree: int) -> ShapeFamily:
    extras = max(degree - 1, 0)

    def radius(
        t: NDArray[np.float64],
        params: NDArray[np.float64],
        frame: SagittalFrame,
    ) -> NDArray[np.float64]:
        return _poly_r(t, float(params[0]), float(params[1]), params[2:])

    def defaults(frame: SagittalFrame) -> NDArray[np.float64]:
        r0, r1, _hi = _r_span(frame)
        return np.concatenate(
            [np.array([r0, r1], dtype=np.float64), np.zeros(extras, dtype=np.float64)]
        )

    def bounds(frame: SagittalFrame) -> NDArray[np.float64]:
        _r0, _r1, r_hi = _r_span(frame)
        span = max(r_hi, 1.0)
        lo_hi = _r_bounds(2, r_hi)
        extra = np.tile(np.array([[-span, span]], dtype=np.float64), (extras, 1))
        return np.vstack([lo_hi, extra]) if extras else lo_hi

    return ShapeFamily(
        family_id=f"poly_{degree}",
        display_name=f"Polynôme degré {degree}",
        n_parameters=2 + extras,
        _radius=radius,
        _defaults=defaults,
        _bounds=bounds,
    )


def _bezier_family(n_ctrl: int) -> ShapeFamily:
    def radius(
        t: NDArray[np.float64],
        params: NDArray[np.float64],
        frame: SagittalFrame,
    ) -> NDArray[np.float64]:
        return _bezier_1d(params, t)

    def defaults(frame: SagittalFrame) -> NDArray[np.float64]:
        t_ctrl = np.linspace(0.0, 1.0, n_ctrl)
        y = frame.y_at(t_ctrl)
        return np.asarray(frame.r_wall_at_y(y), dtype=np.float64)

    def bounds(frame: SagittalFrame) -> NDArray[np.float64]:
        return _r_bounds(n_ctrl, _r_span(frame)[2])

    return ShapeFamily(
        family_id=f"bezier_{n_ctrl}",
        display_name=f"Bézier {n_ctrl} points de contrôle",
        n_parameters=n_ctrl,
        _radius=radius,
        _defaults=defaults,
        _bounds=bounds,
    )


def _fourier_family(harmonics: int) -> ShapeFamily:
    n = 1 + 2 * harmonics

    def radius(
        t: NDArray[np.float64],
        params: NDArray[np.float64],
        frame: SagittalFrame,
    ) -> NDArray[np.float64]:
        return _fourier_r(t, float(params[0]), params[1:])

    def defaults(frame: SagittalFrame) -> NDArray[np.float64]:
        mean_r = float(np.mean(frame.meridian_r_mm))
        return np.concatenate(
            [np.array([mean_r], dtype=np.float64), np.zeros(2 * harmonics, dtype=np.float64)]
        )

    def bounds(frame: SagittalFrame) -> NDArray[np.float64]:
        _r0, _r1, r_hi = _r_span(frame)
        span = max(r_hi, 1.0)
        rows = [np.array([0.0, r_hi], dtype=np.float64)]
        for _ in range(2 * harmonics):
            rows.append(np.array([-span, span], dtype=np.float64))
        return np.vstack(rows)

    return ShapeFamily(
        family_id=f"fourier_{harmonics}",
        display_name=f"Fourier tronquée {harmonics} harmoniques",
        n_parameters=n,
        _radius=radius,
        _defaults=defaults,
        _bounds=bounds,
    )


def _straight() -> ShapeFamily:
    def radius(
        t: NDArray[np.float64],
        params: NDArray[np.float64],
        frame: SagittalFrame,
    ) -> NDArray[np.float64]:
        return _linear_r(t, float(params[0]), float(params[1]))

    def defaults(frame: SagittalFrame) -> NDArray[np.float64]:
        r0, r1, _hi = _r_span(frame)
        return np.array([r0, r1], dtype=np.float64)

    def bounds(frame: SagittalFrame) -> NDArray[np.float64]:
        return _r_bounds(2, _r_span(frame)[2])

    return ShapeFamily(
        family_id="straight",
        display_name="Ligne droite",
        n_parameters=2,
        _radius=radius,
        _defaults=defaults,
        _bounds=bounds,
    )


def _arc() -> ShapeFamily:
    def radius(
        t: NDArray[np.float64],
        params: NDArray[np.float64],
        frame: SagittalFrame,
    ) -> NDArray[np.float64]:
        return _arc_r(t, frame, float(params[0]), float(params[1]), float(params[2]))

    def defaults(frame: SagittalFrame) -> NDArray[np.float64]:
        r0, r1, _hi = _r_span(frame)
        y_mid = 0.5 * (frame.y_top_mm + frame.y_bot_mm)
        r_mid = float(frame.r_wall_at_y(y_mid))
        r_chord = 0.5 * (r0 + r1)
        # Full wall sag often leaves the envelope; keep a constructible arc.
        bulge = 0.35 * (r_mid - r_chord)
        return np.array([r0, r1, bulge], dtype=np.float64)

    def bounds(frame: SagittalFrame) -> NDArray[np.float64]:
        _r0, _r1, r_hi = _r_span(frame)
        height = max(float(frame.useful_height_mm), 1.0)
        return np.vstack(
            [
                _r_bounds(2, r_hi),
                np.array([[-0.5 * height, 0.5 * height]], dtype=np.float64),
            ]
        )

    return ShapeFamily(
        family_id="circular_arc",
        display_name="Arc de cercle",
        n_parameters=3,
        _radius=radius,
        _defaults=defaults,
        _bounds=bounds,
    )


def _mild_arc(family_id: str, display_name: str, *, inward: bool) -> ShapeFamily:
    """Small constant-curvature sag. ``inward`` bows toward the jar axis."""

    sign = 1.0 if inward else -1.0

    def radius(
        t: NDArray[np.float64],
        params: NDArray[np.float64],
        frame: SagittalFrame,
    ) -> NDArray[np.float64]:
        return _arc_r(
            t,
            frame,
            float(params[0]),
            float(params[1]),
            sign * abs(float(params[2])),
        )

    def defaults(frame: SagittalFrame) -> NDArray[np.float64]:
        r0, r1, _hi = _r_span(frame)
        y_mid = 0.5 * (frame.y_top_mm + frame.y_bot_mm)
        r_mid = float(frame.r_wall_at_y(y_mid))
        r_chord = 0.5 * (r0 + r1)
        sag = abs(r_mid - r_chord)
        bulge = 0.20 * sag if sag > 0.5 else 0.03 * max(float(frame.useful_height_mm), 1.0)
        return np.array([r0, r1, bulge], dtype=np.float64)

    def bounds(frame: SagittalFrame) -> NDArray[np.float64]:
        _r0, _r1, r_hi = _r_span(frame)
        height = max(float(frame.useful_height_mm), 1.0)
        return np.vstack(
            [
                _r_bounds(2, r_hi),
                np.array([[0.0, 0.12 * height]], dtype=np.float64),
            ]
        )

    return ShapeFamily(
        family_id=family_id,
        display_name=display_name,
        n_parameters=3,
        _radius=radius,
        _defaults=defaults,
        _bounds=bounds,
    )


def _sigmoid() -> ShapeFamily:
    def radius(
        t: NDArray[np.float64],
        params: NDArray[np.float64],
        frame: SagittalFrame,
    ) -> NDArray[np.float64]:
        return _sigmoid_r(
            t,
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
        )

    def defaults(frame: SagittalFrame) -> NDArray[np.float64]:
        r0, r1, _hi = _r_span(frame)
        return np.array([r0, r1, 0.5, 0.12], dtype=np.float64)

    def bounds(frame: SagittalFrame) -> NDArray[np.float64]:
        _r0, _r1, r_hi = _r_span(frame)
        return np.vstack(
            [
                _r_bounds(2, r_hi),
                np.array([[0.05, 0.95], [0.03, 0.45]], dtype=np.float64),
            ]
        )

    return ShapeFamily(
        family_id="sigmoid",
        display_name="S-curve / sigmoïde",
        n_parameters=4,
        _radius=radius,
        _defaults=defaults,
        _bounds=bounds,
    )


SHAPE_FAMILIES: tuple[ShapeFamily, ...] = (
    _straight(),
    _mild_arc("concave", "Légèrement concave", inward=True),
    _mild_arc("convex", "Légèrement convexe", inward=False),
    _arc(),
    _poly_family(2),
    _poly_family(3),
    _poly_family(4),
    _poly_family(5),
    _poly_family(6),
    _bezier_family(4),
    _bezier_family(6),
    _bezier_family(8),
    _bezier_family(10),
    _sigmoid(),
    _fourier_family(3),
    _fourier_family(5),
)

FAMILY_BY_ID: dict[str, ShapeFamily] = {item.family_id: item for item in SHAPE_FAMILIES}
VALIDATION_FAMILY_IDS: tuple[str, ...] = ("straight", "bezier_4")
SEARCH_FAMILY_IDS: tuple[str, ...] = (
    "straight",
    "concave",
    "convex",
    "circular_arc",
    "bezier_4",
)
PRELIMINARY_FAMILY_IDS: tuple[str, ...] = SEARCH_FAMILY_IDS
LENGTH_PRETEST_SPECS: tuple[tuple[str, float], ...] = (
    ("straight", 20.0),
    ("straight", 30.0),
    ("straight", 40.0),
    ("straight", 50.0),
    ("concave", 40.0),
    ("convex", 40.0),
)


def sample_profile(
    family: ShapeFamily,
    params: NDArray[np.float64] | tuple[float, ...] | list[float],
    frame: SagittalFrame,
    *,
    sample_count: int = PROFILE_SAMPLE_COUNT,
    length_mm: float | None = None,
) -> SampledProfile:
    vector = np.asarray(params, dtype=np.float64).reshape(-1)
    if len(vector) != int(family.n_parameters):
        raise ValueError(
            f"{family.family_id} expects {family.n_parameters} parameters, got {len(vector)}"
        )
    requested = (
        float(length_mm)
        if length_mm is not None
        else min(DEFAULT_SCRAPER_LENGTH_MM, float(frame.useful_height_mm))
    )
    window = frame.window_for_length(requested)
    t = np.linspace(0.0, 1.0, max(int(sample_count), 2), dtype=np.float64)
    y = window.y_at(t)
    r = family.radius(t, vector, window)
    points = np.column_stack((r, y, np.zeros_like(y)))
    return SampledProfile(
        family_id=str(family.family_id),
        parameters=tuple(float(v) for v in vector),
        t=t,
        y_mm=y,
        r_mm=np.asarray(r, dtype=np.float64),
        points_mm=np.asarray(points, dtype=np.float64),
        length_mm=float(window.useful_height_mm),
    )


def clip_params_to_bounds(
    family: ShapeFamily,
    params: NDArray[np.float64],
    frame: SagittalFrame,
) -> NDArray[np.float64]:
    bounds = family.bounds(frame)
    vector = np.asarray(params, dtype=np.float64).reshape(-1)
    return np.clip(vector, bounds[:, 0], bounds[:, 1])
