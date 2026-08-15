"""Active scraper edge from interior surface — ordered stations + smooth tip curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.interior_surface_reference import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    InteriorSurfaceReference,
)

# Sparse surface anchors (from mesh section) → dense smooth resampling for loft.
_LENGTH_SAMPLES = 48
_WIDTH_ANCHORS = 11
_WIDTH_SAMPLES = 33
NUMERIC_GAP_MM = 1e-3


@dataclass(frozen=True)
class EnvelopeStation:
    """One loft station along scraper length on the interior product surface."""

    s_mm: float
    y_mm: float
    tip_points_mm: NDArray[np.float64]
    inward_normals: NDArray[np.float64]
    tangent_length: NDArray[np.float64]
    wall_points_mm: NDArray[np.float64]


@dataclass(frozen=True)
class ScraperEnvelopePath:
    """Active-edge trajectory derived from InteriorSurfaceReference mesh only."""

    stations: tuple[EnvelopeStation, ...]
    source: str = SOURCE_INTERIOR_PRODUCT_SURFACE

    @property
    def tip_curve_mm(self) -> NDArray[np.float64]:
        mid = self.stations[0].tip_points_mm.shape[0] // 2
        return np.asarray(
            [station.tip_points_mm[mid] for station in self.stations],
            dtype=np.float64,
        )

    @property
    def wall_curve_mm(self) -> NDArray[np.float64]:
        mid = self.stations[0].wall_points_mm.shape[0] // 2
        return np.asarray(
            [station.wall_points_mm[mid] for station in self.stations],
            dtype=np.float64,
        )


def endpoint_match_costs(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
) -> tuple[float, float]:
    """Return (cost_same_order, cost_flipped_order) for tip/wall endpoint pairing."""
    d_same = float(
        np.linalg.norm(previous[0] - current[0])
        + np.linalg.norm(previous[-1] - current[-1])
    )
    d_flip = float(
        np.linalg.norm(previous[0] - current[-1])
        + np.linalg.norm(previous[-1] - current[0])
    )
    return d_same, d_flip


def prefers_flipped_order(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
) -> bool:
    """True if reversing ``current`` better matches ``previous`` endpoints."""
    d_same, d_flip = endpoint_match_costs(previous, current)
    return d_flip + 1e-9 < d_same


def normalize_row_orders(
    wall_rows: list[NDArray[np.float64]],
    normal_rows: list[NDArray[np.float64]],
) -> None:
    """
    In-place: deterministic first-station winding, then pairwise continuity.

    After this, no consecutive pair should prefer a flipped correspondence.
    """
    if not wall_rows:
        return

    mid = wall_rows[0].shape[0] // 2
    n0 = normal_rows[0][mid]
    chord = wall_rows[0][-1] - wall_rows[0][0]
    # Width direction: n × +Y  (right-handed with upward length).
    width_dir = np.cross(n0, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    w_norm = float(np.linalg.norm(width_dir))
    if w_norm > 1e-9 and float(np.dot(chord, width_dir)) < 0.0:
        wall_rows[0] = wall_rows[0][::-1].copy()
        normal_rows[0] = normal_rows[0][::-1].copy()

    for index in range(1, len(wall_rows)):
        if prefers_flipped_order(wall_rows[index - 1], wall_rows[index]):
            wall_rows[index] = wall_rows[index][::-1].copy()
            normal_rows[index] = normal_rows[index][::-1].copy()


def assert_no_inverted_station_pairs(stations: tuple[EnvelopeStation, ...]) -> None:
    """Raise if any consecutive tip rows still prefer a flipped pairing."""
    for index in range(1, len(stations)):
        prev = stations[index - 1].tip_points_mm
        curr = stations[index].tip_points_mm
        if prefers_flipped_order(prev, curr):
            raise AssertionError(
                f"Stations {index - 1}→{index} still prefer flipped tip correspondence"
            )


class ScraperEnvelopePathBuilder:
    """
    Build the active edge from interior surface anchors.

    Anchors come from mesh ∩ plane(y); the tip curve is a cubic arc-length
    interpolation of those anchors (not the raw section polyline).
    """

    def build(
        self,
        surface: InteriorSurfaceReference,
        parameters: ScraperParameters,
    ) -> ScraperEnvelopePath:
        if surface.vertex_count == 0 or surface.face_count == 0:
            raise ValueError("InteriorSurfaceReference is empty")

        mesh = surface.to_trimesh()
        y_min = float(surface.y_min_mm)
        y_max = float(surface.y_max_mm)
        center_y = float(np.clip(parameters.position_z_mm, y_min, y_max))
        half_length = 0.5 * float(parameters.length_mm)
        half_width = 0.5 * float(parameters.width_mm)
        clearance = float(parameters.clearance_mm) + NUMERIC_GAP_MM

        # Design approach is fixed for the rigid manufacturing solid.
        # Surface progress is applied later as a free SE(3) pose, not a reshape.
        yaw = 0.0
        ux0 = float(np.cos(yaw))
        uz0 = float(-np.sin(yaw))

        s_values = np.linspace(-half_length, half_length, _LENGTH_SAMPLES)
        wall_rows: list[NDArray[np.float64]] = []
        normal_rows: list[NDArray[np.float64]] = []

        for s_mm in s_values:
            y_mm = float(np.clip(center_y + float(s_mm), y_min, y_max))
            contour = self._horizontal_contour(mesh, y_mm)
            anchors = self._sample_contour_by_arc_length(
                contour,
                ux=ux0,
                uz=uz0,
                half_width_mm=half_width,
                sample_count=_WIDTH_ANCHORS,
            )
            # Snap anchors to the surface (reference fidelity), then smooth.
            anchors, _dist, tri_ids = mesh.nearest.on_surface(anchors)
            anchors = np.asarray(anchors, dtype=np.float64)
            smooth = self._interpolate_curve_arc_length(anchors, _WIDTH_SAMPLES)
            # Normals from surface at the smooth samples (positions stay smooth).
            _c, _d, tri_smooth = mesh.nearest.on_surface(smooth)
            normals = np.asarray(
                [
                    self._inward_normal_at(mesh, smooth[i], int(tri_smooth[i]))
                    for i in range(len(smooth))
                ],
                dtype=np.float64,
            )
            wall_rows.append(smooth)
            normal_rows.append(normals)
            del tri_ids

        normalize_row_orders(wall_rows, normal_rows)

        stations: list[EnvelopeStation] = []
        mid = _WIDTH_SAMPLES // 2
        for index, s_mm in enumerate(s_values):
            wall_row = wall_rows[index]
            normal_row = normal_rows[index]
            tip_row = wall_row + normal_row * clearance
            if index + 1 < len(s_values):
                delta = wall_rows[index + 1][mid] - wall_row[mid]
            else:
                delta = wall_row[mid] - wall_rows[index - 1][mid]
            tangent = self._safe_unit(delta, fallback=np.asarray([0.0, 1.0, 0.0]))
            stations.append(
                EnvelopeStation(
                    s_mm=float(s_mm),
                    y_mm=float(wall_row[mid, 1]),
                    tip_points_mm=tip_row,
                    inward_normals=normal_row,
                    tangent_length=tangent,
                    wall_points_mm=wall_row,
                )
            )

        path = ScraperEnvelopePath(stations=tuple(stations), source=surface.source)
        assert_no_inverted_station_pairs(path.stations)
        return path

    def _horizontal_contour(
        self,
        mesh: object,
        y_mm: float,
    ) -> NDArray[np.float64]:
        """Return the longest polyline of mesh ∩ plane y = y_mm (anchors only)."""
        section = mesh.section(  # type: ignore[attr-defined]
            plane_origin=[0.0, float(y_mm), 0.0],
            plane_normal=[0.0, 1.0, 0.0],
        )
        if section is None:
            for delta in (0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0):
                section = mesh.section(  # type: ignore[attr-defined]
                    plane_origin=[0.0, float(y_mm + delta), 0.0],
                    plane_normal=[0.0, 1.0, 0.0],
                )
                if section is not None:
                    break
        if section is None:
            raise ValueError(
                f"No interior-surface section at y={y_mm:.3f} mm "
                "(plane does not intersect the product-surface mesh)"
            )

        discrete = getattr(section, "discrete", None)
        if not discrete:
            vertices = np.asarray(section.vertices, dtype=np.float64)
            if len(vertices) < 2:
                raise ValueError(f"Degenerate interior section at y={y_mm:.3f}")
            return vertices

        poly = max(
            (np.asarray(part, dtype=np.float64) for part in discrete),
            key=lambda arr: float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
            if len(arr) > 1
            else 0.0,
        )
        if len(poly) < 2:
            raise ValueError(f"Interior section polyline too short at y={y_mm:.3f}")
        return poly

    def _sample_contour_by_arc_length(
        self,
        contour: NDArray[np.float64],
        *,
        ux: float,
        uz: float,
        half_width_mm: float,
        sample_count: int,
    ) -> NDArray[np.float64]:
        """Sample sparse surface anchors around the approach point by arc length."""
        pts = np.asarray(contour, dtype=np.float64)
        if len(pts) < 2:
            raise ValueError("Contour needs at least two points")

        if np.linalg.norm(pts[0] - pts[-1]) <= 1e-6:
            pts = pts[:-1].copy()
        if len(pts) < 2:
            raise ValueError("Contour collapsed after removing closing vertex")

        diffs = np.diff(pts, axis=0, append=pts[:1])
        seg_len = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = float(cum[-1])
        if total <= 1e-9:
            raise ValueError("Contour has zero arc length")

        proj = pts[:, 0] * ux + pts[:, 2] * uz
        i0 = int(np.argmax(proj))
        s0 = float(cum[i0])

        offsets = np.linspace(-half_width_mm, half_width_mm, sample_count)
        samples = np.empty((sample_count, 3), dtype=np.float64)
        for index, offset in enumerate(offsets):
            target = (s0 + float(offset)) % total
            samples[index] = self._interpolate_polyline(pts, cum, target)
        return samples

    @staticmethod
    def _interpolate_curve_arc_length(
        anchors: NDArray[np.float64],
        sample_count: int,
    ) -> NDArray[np.float64]:
        """
        Cubic arc-length interpolation through surface anchors.

        Preserves anchor geometry (interpolating, not approximating) so the tip
        is smooth without becoming a generic circle.
        """
        pts = np.asarray(anchors, dtype=np.float64)
        if len(pts) < 2:
            raise ValueError("Need at least two anchors to interpolate a tip curve")
        if len(pts) == 2 or sample_count <= len(pts):
            # Linear fallback / already dense enough.
            if sample_count == len(pts):
                return pts.copy()
            t = np.linspace(0.0, 1.0, sample_count)
            out = np.empty((sample_count, 3), dtype=np.float64)
            for i, ti in enumerate(t):
                out[i] = pts[0] * (1.0 - ti) + pts[-1] * ti
            return out

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total <= 1e-9:
            return np.repeat(pts[:1], sample_count, axis=0)

        # Chord-length parameter in [0, 1].
        u = cum / total
        # Clamp degree to available unique samples.
        unique_u, unique_idx = np.unique(u, return_index=True)
        unique_pts = pts[unique_idx]
        if len(unique_u) < 2:
            return np.repeat(pts[:1], sample_count, axis=0)

        try:
            from scipy.interpolate import make_interp_spline

            degree = min(3, len(unique_u) - 1)
            spline = make_interp_spline(unique_u, unique_pts, k=degree, axis=0)
            u_new = np.linspace(0.0, 1.0, sample_count)
            return np.asarray(spline(u_new), dtype=np.float64)
        except Exception:
            # Piecewise-linear arc-length resample if spline unavailable.
            u_new = np.linspace(0.0, total, sample_count)
            out = np.empty((sample_count, 3), dtype=np.float64)
            for i, s in enumerate(u_new):
                out[i] = ScraperEnvelopePathBuilder._interpolate_polyline(
                    pts, cum, float(s), closed=False
                )
            return out

    @staticmethod
    def _interpolate_polyline(
        pts: NDArray[np.float64],
        cum: NDArray[np.float64],
        s: float,
        *,
        closed: bool = True,
    ) -> NDArray[np.float64]:
        s_clamped = float(np.clip(s, float(cum[0]), float(cum[-1])))
        idx = int(np.searchsorted(cum, s_clamped, side="right") - 1)
        idx = max(0, min(idx, len(pts) - 1))
        s0 = float(cum[idx])
        s1 = float(cum[min(idx + 1, len(cum) - 1)])
        span = max(s1 - s0, 1e-12)
        t = (s_clamped - s0) / span
        p0 = pts[idx]
        if idx + 1 < len(pts):
            p1 = pts[idx + 1]
        elif closed:
            p1 = pts[0]
        else:
            p1 = pts[idx]
        return p0 * (1.0 - t) + p1 * t

    @staticmethod
    def _inward_normal_at(
        mesh: object,
        point: NDArray[np.float64],
        triangle_id: int,
    ) -> NDArray[np.float64]:
        """Face normal oriented toward the interior volume (explicit probe)."""
        normals = np.asarray(mesh.face_normals, dtype=np.float64)  # type: ignore[attr-defined]
        if 0 <= triangle_id < len(normals):
            normal = np.asarray(normals[triangle_id], dtype=np.float64).copy()
        else:
            _, _, tri_ids = mesh.nearest.on_surface(point.reshape(1, 3))  # type: ignore[attr-defined]
            normal = np.asarray(normals[int(tri_ids[0])], dtype=np.float64).copy()

        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            radial = np.asarray([-point[0], 0.0, -point[2]], dtype=np.float64)
            radial_n = float(np.linalg.norm(radial))
            if radial_n <= 1e-12:
                return np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
            return radial / radial_n
        normal /= norm

        eps = 0.25
        plus = point + normal * eps
        minus = point - normal * eps
        r_plus = float(np.hypot(plus[0], plus[2]))
        r_minus = float(np.hypot(minus[0], minus[2]))
        if r_plus <= r_minus:
            return normal
        return -normal

    @staticmethod
    def _safe_unit(
        vector: NDArray[np.float64],
        *,
        fallback: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            return np.asarray(fallback, dtype=np.float64)
        return np.asarray(vector / norm, dtype=np.float64)
