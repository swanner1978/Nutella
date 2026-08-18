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
_MIN_LENGTH_SAMPLES = 8
_MAX_LENGTH_SAMPLES = 48
_TARGET_LENGTH_SPACING_MM = 2.5
_MERIDIAN_SPACING_MM = 1.5
_MIN_STATION_SEPARATION_MM = 1e-3
# Drop trailing meridian samples that run along the rim (almost no +Y).
_MIN_SIDEWALL_SLOPE = 0.35
_WIDTH_ANCHORS = 11
_WIDTH_SAMPLES = 33
NUMERIC_GAP_MM = 1e-3
LENGTH_EXCEEDS_ENVELOPE_MSG = (
    "Requested scraper length exceeds available interior envelope."
)


def jar_longitudinal_limits(
    surface: InteriorSurfaceReference,
) -> tuple[float, float]:
    """Useful scraper corridor along +Y: median plane → opening.

    * Opening = highest interior-mesh Y (``surface.y_max_mm``).
    * Median plane = AABB mid-height of InteriorSurfaceReference, i.e. the
      horizontal plane through the jar-frame origin of the green +Y axis.
    """
    y_min = float(surface.y_min_mm)
    y_max = float(surface.y_max_mm)
    median_y = 0.5 * (y_min + y_max)
    opening_y = y_max
    if opening_y - median_y < 1e-3:
        raise ValueError("Interior envelope has no usable opening-to-median span")
    return median_y, opening_y


# Physical length (``length_mm``) is independent of the collision corridor.
# The blade is always anchored at the opening, never at the viewer red axis.
LONGITUDINAL_ANCHOR = "upper_opening"


def scraper_length_span(
    surface: InteriorSurfaceReference,
) -> tuple[float, float, float]:
    """Opening, interior floor, and max physical length (Y down from opening).

    Lower stop is ``surface.y_min_mm`` — the lowest sectionable interior Y —
    not the AABB mid-height (viewer frame origin / star). Stopping at
    mid-height froze ``length_mm`` once it exceeded ~half the jar.
    """
    opening_y = float(surface.y_max_mm)
    lower_y = float(surface.y_min_mm)
    max_length_mm = opening_y - lower_y
    if max_length_mm < 1e-3:
        raise ValueError("Interior envelope has no usable opening-to-floor span")
    return opening_y, lower_y, float(max_length_mm)


def apply_effective_length(
    parameters: ScraperParameters,
    surface: InteriorSurfaceReference,
) -> tuple[ScraperParameters, dict[str, object]]:
    """Clamp ``length_mm`` to the geometrically available span; keep the top fixed."""
    opening_y, lower_y, max_length_mm = scraper_length_span(surface)
    requested = float(parameters.length_mm)
    effective = min(requested, max_length_mm)
    clamped = requested > max_length_mm + 1e-6
    updated = (
        parameters.with_updates(length_mm=effective) if clamped else parameters
    )
    warning = (
        f"Attention : longueur maximale = {max_length_mm:.0f} mm" if clamped else None
    )
    return updated, {
        "max_length_mm": max_length_mm,
        "effective_length_mm": float(updated.length_mm),
        "requested_length_mm": requested,
        "clamped": clamped,
        "warning": warning,
        "opening_y_mm": opening_y,
        "lower_y_mm": lower_y,
        "anchor": LONGITUDINAL_ANCHOR,
    }


def interior_centroid_mm(surface: InteriorSurfaceReference) -> NDArray[np.float64]:
    """Jar-axis point at mid interior height — interior-side normal probe."""
    return np.asarray(
        [0.0, 0.5 * (float(surface.y_min_mm) + float(surface.y_max_mm)), 0.0],
        dtype=np.float64,
    )


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
    # Width = n × length. Length is +Y on the wall; on the floor n ≈ +Y so
    # fall back to the constructed chord (constant blade width direction).
    width_dir = np.cross(n0, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    w_norm = float(np.linalg.norm(width_dir))
    if w_norm <= 1e-9:
        width_dir = chord
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
    Build the active edge as a 3D strip along the interior meridian.

    The centreline C(s) is the sagittal profile of InteriorSurfaceReference
    (opening → fillet → floor). Each station is a constant-width chord in the
    local surface frame (N, T, W), projected back onto the envelope.
    Horizontal Y-slices are not used for station geometry: they jump azimuth
    on the floor and loft a disk instead of a narrow blade.
    """

    def build(
        self,
        surface: InteriorSurfaceReference,
        parameters: ScraperParameters,
    ) -> ScraperEnvelopePath:
        if surface.vertex_count == 0 or surface.face_count == 0:
            raise ValueError("InteriorSurfaceReference is empty")

        mesh = surface.to_trimesh()
        half_width = 0.5 * float(parameters.width_mm)
        clearance = float(parameters.clearance_mm) + NUMERIC_GAP_MM
        length_mm = float(parameters.length_mm)
        opening_y, lower_y, max_length_mm = scraper_length_span(surface)
        length_mm = min(length_mm, max_length_mm)
        interior_target = interior_centroid_mm(surface)

        # Design approach is fixed for the rigid manufacturing solid.
        # Surface progress is applied later as a free SE(3) pose, not a reshape.
        yaw = 0.0
        ux0 = float(np.cos(yaw))
        uz0 = float(-np.sin(yaw))

        meridian_pts, meridian_y, meridian_s = self._interior_meridian(
            mesh,
            y_min=lower_y,
            y_max=opening_y,
            ux=ux0,
            uz=uz0,
        )
        s_lo, s_hi, s_values, extra_above = self._longitudinal_arc_samples(
            meridian_y=meridian_y,
            meridian_s=meridian_s,
            length_mm=length_mm,
        )
        centre = np.column_stack(
            (
                np.interp(s_values, meridian_s, meridian_pts[:, 0]),
                np.interp(s_values, meridian_s, meridian_pts[:, 1]),
                np.interp(s_values, meridian_s, meridian_pts[:, 2]),
            )
        )
        centre, _dist_c, tri_c = mesh.nearest.on_surface(centre)
        centre = np.asarray(centre, dtype=np.float64)

        wall_rows: list[NDArray[np.float64]] = []
        normal_rows: list[NDArray[np.float64]] = []
        tangents: list[NDArray[np.float64]] = []
        prev_width: NDArray[np.float64] | None = None

        for index, point in enumerate(centre):
            if index + 1 < len(centre):
                raw_t = centre[index + 1] - point
            else:
                raw_t = point - centre[index - 1]
            tangent = self._safe_unit(raw_t, fallback=np.asarray([0.0, 1.0, 0.0]))
            normal = self._inward_normal_at(
                mesh,
                point,
                int(tri_c[index]),
                interior_target=interior_target,
            )
            width_dir, prev_width = self._width_direction(normal, tangent, prev_width)
            anchors = self._constant_width_anchors(
                mesh,
                origin=point,
                width_dir=width_dir,
                half_width_mm=half_width,
                sample_count=_WIDTH_ANCHORS,
            )
            smooth = self._interpolate_curve_arc_length(anchors, _WIDTH_SAMPLES)
            _c, _d, tri_smooth = mesh.nearest.on_surface(smooth)
            smooth = np.asarray(_c, dtype=np.float64)
            normals = np.asarray(
                [
                    self._inward_normal_at(
                        mesh,
                        smooth[i],
                        int(tri_smooth[i]),
                        interior_target=interior_target,
                    )
                    for i in range(len(smooth))
                ],
                dtype=np.float64,
            )
            wall_rows.append(smooth)
            normal_rows.append(normals)
            tangents.append(tangent)

        normalize_row_orders(wall_rows, normal_rows)

        stations: list[EnvelopeStation] = []
        mid = _WIDTH_SAMPLES // 2
        s_mid = 0.5 * (s_lo + s_hi)
        for index, s_abs in enumerate(s_values):
            wall_row = wall_rows[index]
            normal_row = normal_rows[index]
            tip_row = wall_row + normal_row * clearance
            stations.append(
                EnvelopeStation(
                    s_mm=float(s_abs - s_mid),
                    y_mm=float(wall_row[mid, 1]),
                    tip_points_mm=tip_row,
                    inward_normals=normal_row,
                    tangent_length=tangents[index],
                    wall_points_mm=wall_row,
                )
            )

        self._assert_unique_station_progression(stations)
        if extra_above > _MIN_STATION_SEPARATION_MM:
            self._append_overhang_stations(stations, extra_above_mm=extra_above)
            self._assert_unique_station_progression(stations)
        path = ScraperEnvelopePath(stations=tuple(stations), source=surface.source)
        assert_no_inverted_station_pairs(path.stations)
        return path

    @staticmethod
    def length_sample_count(length_mm: float) -> int:
        n = int(round(float(length_mm) / _TARGET_LENGTH_SPACING_MM)) + 1
        return int(np.clip(n, _MIN_LENGTH_SAMPLES, _MAX_LENGTH_SAMPLES))

    def _interior_meridian(
        self,
        mesh: object,
        *,
        y_min: float,
        y_max: float,
        ux: float,
        uz: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """3D centreline C(s) in the approach plane, floor/axis → opening."""
        try:
            pts = self._sagittal_meridian(mesh, ux=ux, uz=uz)
        except ValueError:
            pts = self._stacked_slice_meridian(
                mesh, y_min=y_min, y_max=y_max, ux=ux, uz=uz
            )
        if len(pts) < 2:
            raise ValueError(
                "Interior envelope has no usable longitudinal trajectory"
            )
        pts, _, _tri_ids = mesh.nearest.on_surface(pts)  # type: ignore[attr-defined]
        pts = np.asarray(pts, dtype=np.float64)
        y_arr = np.asarray(pts[:, 1], dtype=np.float64)
        pts, y_arr = self._trim_trailing_rim(pts, y_arr)
        pts, y_arr = self._dedup_polyline(pts, y_arr)
        if len(pts) < 2:
            raise ValueError(
                "Interior envelope has no usable longitudinal trajectory"
            )
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s_arr = np.concatenate([[0.0], np.cumsum(seg)])
        return pts, y_arr, s_arr

    def _sagittal_meridian(
        self,
        mesh: object,
        *,
        ux: float,
        uz: float,
    ) -> NDArray[np.float64]:
        """Profile curve: mesh ∩ plane(Y, approach). One half, axis → opening."""
        plane_normal = np.asarray([-uz, 0.0, ux], dtype=np.float64)
        nrm = float(np.linalg.norm(plane_normal))
        if nrm <= 1e-9:
            plane_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            plane_normal = plane_normal / nrm
        section = mesh.section(  # type: ignore[attr-defined]
            plane_origin=[0.0, 0.0, 0.0],
            plane_normal=plane_normal.tolist(),
        )
        if section is None:
            raise ValueError("No sagittal section of the interior envelope")
        polylines = self._path3d_polylines(section)
        if not polylines:
            raise ValueError("Sagittal section has no polylines")
        raw = max(
            polylines,
            key=lambda arr: float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
            if len(arr) > 1
            else 0.0,
        )
        chain = self._approach_half_chain(raw, ux=ux, uz=uz)
        if len(chain) < 2:
            raise ValueError("Sagittal approach half is too short")
        chain = self._resample_polyline(chain, _MERIDIAN_SPACING_MM)
        return chain

    @staticmethod
    def _path3d_polylines(section: object) -> list[NDArray[np.float64]]:
        discrete = getattr(section, "discrete", None)
        if discrete:
            return [np.asarray(part, dtype=np.float64) for part in discrete if len(part) >= 2]
        polylines: list[NDArray[np.float64]] = []
        verts = np.asarray(section.vertices, dtype=np.float64)
        for entity in getattr(section, "entities", []):
            idx = np.asarray(getattr(entity, "points", []), dtype=np.int64)
            if len(idx) >= 2:
                polylines.append(verts[idx])
        return polylines

    @staticmethod
    def _approach_half_chain(
        pts: NDArray[np.float64],
        *,
        ux: float,
        uz: float,
    ) -> NDArray[np.float64]:
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 2:
            return pts
        proj = pts[:, 0] * ux + pts[:, 2] * uz
        radii = np.hypot(pts[:, 0], pts[:, 2])
        eligible = np.where(proj >= -1.0)[0]
        if len(eligible) < 2:
            eligible = np.arange(len(pts))
        i_open = int(eligible[int(np.argmax(pts[eligible, 1]))])

        def _walk(start: int, step: int) -> list[int]:
            chain = [start]
            index = start
            while True:
                nxt = index + step
                if nxt < 0 or nxt >= len(pts):
                    break
                if float(proj[nxt]) < -1.0 and float(radii[nxt]) > 1.0:
                    break
                chain.append(nxt)
                index = nxt
            return chain

        left = _walk(i_open, -1)
        right = _walk(i_open, 1)
        idxs = list(reversed(left[1:])) + [i_open] + right[1:]
        chain = pts[np.asarray(idxs, dtype=np.int64)]
        if float(chain[0, 1]) > float(chain[-1, 1]):
            chain = chain[::-1].copy()
        return chain

    def _stacked_slice_meridian(
        self,
        mesh: object,
        *,
        y_min: float,
        y_max: float,
        ux: float,
        uz: float,
    ) -> NDArray[np.float64]:
        """Fallback: horizontal slices with azimuth continuity (no floor jumps)."""
        span = max(float(y_max) - float(y_min), 1e-6)
        count = int(np.clip(round(span / _MERIDIAN_SPACING_MM) + 1, 16, 96))
        ys = np.linspace(float(y_min), float(y_max), count)
        inset = min(0.08, 0.002 * span)
        ys[0] += inset
        ys[-1] -= inset
        points: list[NDArray[np.float64]] = []
        previous: NDArray[np.float64] | None = None
        for y_mm in ys:
            try:
                contour = self._horizontal_contour(mesh, float(y_mm), allow_nudge=False)
            except ValueError:
                continue
            if previous is None:
                point = self._approach_point(contour, ux, uz)
            else:
                dists = np.linalg.norm(
                    np.asarray(contour, dtype=np.float64) - previous, axis=1
                )
                point = np.asarray(contour[int(np.argmin(dists))], dtype=np.float64).copy()
            points.append(point)
            previous = point
        if len(points) < 2:
            raise ValueError(
                "Interior envelope has no usable longitudinal trajectory"
            )
        return np.asarray(points, dtype=np.float64)

    @staticmethod
    def _resample_polyline(
        pts: NDArray[np.float64],
        spacing_mm: float,
    ) -> NDArray[np.float64]:
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 2:
            return pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total <= 1e-9:
            return pts[:1].copy()
        count = int(np.clip(round(total / max(spacing_mm, 0.5)) + 1, 8, 256))
        targets = np.linspace(0.0, total, count)
        out = np.empty((count, 3), dtype=np.float64)
        for i, s in enumerate(targets):
            out[i] = ScraperEnvelopePathBuilder._interpolate_polyline(
                pts, cum, float(s), closed=False
            )
        return out

    @staticmethod
    def _dedup_polyline(
        pts: NDArray[np.float64],
        y_arr: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if len(pts) < 2:
            return pts, y_arr
        keep = np.concatenate(
            [[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]
        )
        return pts[keep], y_arr[keep]

    @classmethod
    def _width_direction(
        cls,
        normal: NDArray[np.float64],
        tangent: NDArray[np.float64],
        previous: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        width = np.cross(normal, tangent)
        if float(np.linalg.norm(width)) <= 1e-9:
            width = np.cross(normal, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(width)) <= 1e-9:
            width = np.cross(normal, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
        width = cls._safe_unit(width, fallback=np.asarray([0.0, 0.0, 1.0]))
        if previous is not None and float(np.dot(width, previous)) < 0.0:
            width = -width
        return width, width

    def _constant_width_anchors(
        self,
        mesh: object,
        *,
        origin: NDArray[np.float64],
        width_dir: NDArray[np.float64],
        half_width_mm: float,
        sample_count: int,
    ) -> NDArray[np.float64]:
        """Place a width_mm chord in the local W direction, then snap to the envelope."""
        offsets = np.linspace(-float(half_width_mm), float(half_width_mm), sample_count)
        seeds = np.asarray(origin, dtype=np.float64)[None, :] + np.asarray(
            width_dir, dtype=np.float64
        )[None, :] * offsets[:, None]
        snapped, _dist, _tri = mesh.nearest.on_surface(seeds)  # type: ignore[attr-defined]
        return np.asarray(snapped, dtype=np.float64)

    @staticmethod
    def _trim_trailing_rim(
        pts: NDArray[np.float64],
        y_arr: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Stop envelope-following before a nearly-horizontal rim/flange run."""
        if len(pts) < 3:
            return pts, y_arr
        seg = np.diff(pts, axis=0)
        ds = np.linalg.norm(seg, axis=1)
        dy = np.diff(y_arr)
        slope = np.abs(dy) / np.maximum(ds, 1e-9)
        last = len(pts) - 1
        while last >= 2 and float(slope[last - 1]) < _MIN_SIDEWALL_SLOPE:
            last -= 1
        return pts[: last + 1], y_arr[: last + 1]

    @classmethod
    def _longitudinal_arc_samples(
        cls,
        *,
        meridian_y: NDArray[np.float64],
        meridian_s: NDArray[np.float64],
        length_mm: float,
    ) -> tuple[float, float, NDArray[np.float64], float]:
        """Grow the blade downward from the opening along C(s); top station stays put."""
        y_open = float(meridian_y[-1])
        y_floor = float(np.min(meridian_y))
        max_length = max(y_open - y_floor, _MIN_STATION_SEPARATION_MM)
        effective = min(float(length_mm), max_length)
        y_bottom = y_open - effective
        follow_hi = float(meridian_s[-1])
        # Walk from the opening toward the floor until Y reaches y_bottom.
        # On a flat floor Y is nearly constant, so this includes the inward
        # path to the centre when length is at the geometric maximum.
        idx = len(meridian_y) - 1
        while idx > 0 and float(meridian_y[idx]) > y_bottom + 1e-6:
            idx -= 1
        follow_lo = float(meridian_s[idx])
        extra_above = 0.0
        if follow_hi - follow_lo < _MIN_STATION_SEPARATION_MM:
            raise ValueError("Interior envelope has no usable longitudinal trajectory")
        n_samples = cls.length_sample_count(max(effective, 1.0))
        s_values = np.linspace(follow_lo, follow_hi, n_samples)
        unique = np.concatenate([[True], np.diff(s_values) > 1e-9])
        s_values = s_values[unique]
        if len(s_values) < 2:
            raise ValueError("Interior envelope has no usable longitudinal trajectory")
        return follow_lo, follow_hi, s_values, extra_above

    @classmethod
    def _append_overhang_stations(
        cls,
        stations: list[EnvelopeStation],
        *,
        extra_above_mm: float,
    ) -> None:
        """Rigid continuation above the opening — not an invented envelope surface."""
        last = stations[-1]
        tangent = np.asarray(last.tangent_length, dtype=np.float64)
        if float(tangent[1]) < 0.0:
            tangent = -tangent
        # A rim-tangent is nearly horizontal; extra length must leave the pot
        # toward the opening (+Y), not slide along the flange.
        if float(tangent[1]) < _MIN_SIDEWALL_SLOPE:
            tangent = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            tangent = cls._safe_unit(tangent, fallback=np.asarray([0.0, 1.0, 0.0]))
        n_extra = max(1, cls.length_sample_count(extra_above_mm) - 1)
        for index in range(1, n_extra + 1):
            ds = extra_above_mm * (index / n_extra)
            offset = tangent * ds
            stations.append(
                EnvelopeStation(
                    s_mm=float(last.s_mm + ds),
                    y_mm=float(last.y_mm + offset[1]),
                    tip_points_mm=np.asarray(last.tip_points_mm, dtype=np.float64) + offset,
                    inward_normals=np.asarray(last.inward_normals, dtype=np.float64),
                    tangent_length=tangent,
                    wall_points_mm=np.asarray(last.wall_points_mm, dtype=np.float64) + offset,
                )
            )

    @staticmethod
    def _assert_unique_station_progression(stations: list[EnvelopeStation]) -> None:
        mid = stations[0].wall_points_mm.shape[0] // 2
        for index in range(1, len(stations)):
            delta = stations[index].wall_points_mm[mid] - stations[index - 1].wall_points_mm[mid]
            if float(np.linalg.norm(delta)) < _MIN_STATION_SEPARATION_MM:
                raise ValueError(LENGTH_EXCEEDS_ENVELOPE_MSG)

    @staticmethod
    def _approach_point(
        contour: NDArray[np.float64],
        ux: float,
        uz: float,
    ) -> NDArray[np.float64]:
        pts = np.asarray(contour, dtype=np.float64)
        proj = pts[:, 0] * ux + pts[:, 2] * uz
        return np.asarray(pts[int(np.argmax(proj))], dtype=np.float64).copy()

    def _horizontal_contour(
        self,
        mesh: object,
        y_mm: float,
        *,
        allow_nudge: bool = True,
    ) -> NDArray[np.float64]:
        """Return the longest polyline of mesh ∩ plane y = y_mm (anchors only)."""
        section = mesh.section(  # type: ignore[attr-defined]
            plane_origin=[0.0, float(y_mm), 0.0],
            plane_normal=[0.0, 1.0, 0.0],
        )
        if section is None:
            if allow_nudge:
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

    def _inward_normal_at(
        self,
        mesh: object,
        point: NDArray[np.float64],
        triangle_id: int,
        *,
        interior_target: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Face normal oriented toward the interior volume (centroid probe)."""
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
                return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            return radial / radial_n
        normal /= norm

        if interior_target is None:
            bounds = np.asarray(mesh.bounds, dtype=np.float64)  # type: ignore[attr-defined]
            interior_target = np.asarray(
                [0.0, 0.5 * (float(bounds[0, 1]) + float(bounds[1, 1])), 0.0],
                dtype=np.float64,
            )
        to_inside = np.asarray(interior_target, dtype=np.float64) - np.asarray(
            point, dtype=np.float64
        )
        if float(np.dot(normal, to_inside)) < 0.0:
            normal = -normal
        return normal

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
