"""Cached closest-point on the interior envelope — same results as trimesh.

``Trimesh.nearest.on_surface`` calls ``proximity.closest_point``, which:

1. Builds a **new** ``cKDTree`` of referenced vertices on every call.
2. Queries ``mesh.triangles_tree`` (R-tree of triangle AABBs, cached).
3. Runs point-triangle closest on those AABB candidates.
4. Picks the two nearest candidates in Python and tie-breaks triangle ids
   with face normals.

The R-tree AABB walk dominates A0 (~50 ms / 5544 extras). This helper keeps
the vertex KD-tree, a centroid KD-tree, and the triangle soup on the mesh.

Fast path (``closest_on_surface_fast``):

- Evaluate the ``K`` nearest triangle centroids (fixed-width, fully vectorized).
- The result is exact when ``best_dist + tol.merge <= d_K - max_extent``:
  any unqueried triangle has centroid farther than ``d_K`` and cannot beat
  ``best_dist``.
- Otherwise expand with the conservative centroid ball
  (nearest-vertex distance + max centroid-to-vertex extent).
- When the two best distances are within ``tol.merge``, fall back to
  ``mesh.nearest.on_surface`` so triangle ids match Trimesh's tie-break.

Does not subsample scraper vertices or extras. Does not change SE(3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from trimesh.constants import tol
from trimesh.triangles import closest_point as closest_point_on_triangles
from trimesh.util import diagonal_dot

_CACHE_KEY = "_nutella_envelope_proximity"
_MERGE = float(tol.merge)
# Fixed-width centroid k-NN. Bound check expands if K is not enough.
_KNN_K = 16

# Diagnostic counters for the A0 profiler (not used by the solver).
PROXIMITY_STATS: dict[str, int] = {
    "calls": 0,
    "points": 0,
    "tied_fallback_points": 0,
    "empty_ball_points": 0,
    "triangles_examined": 0,
    "fast_path_points": 0,
    "bound_expand_points": 0,
}


def reset_proximity_stats() -> None:
    for key in PROXIMITY_STATS:
        PROXIMITY_STATS[key] = 0


def _engine_for(mesh: trimesh.Trimesh) -> EnvelopeSurfaceProximity:
    cached = mesh.metadata.get(_CACHE_KEY)
    if isinstance(cached, EnvelopeSurfaceProximity) and cached.mesh is mesh:
        return cached
    return bind_envelope_proximity(mesh)


def _empty_result() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    empty = np.zeros((0, 3), dtype=np.float64)
    return empty, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)


class EnvelopeSurfaceProximity:
    """Closest point / distance / triangle id on a fixed interior mesh."""

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
        if len(vertices) == 0 or len(triangles) == 0:
            raise ValueError("Empty interior mesh")
        centroids = triangles.mean(axis=1)
        extents = np.linalg.norm(triangles - centroids[:, None, :], axis=2)
        self.mesh = mesh
        self.triangles = triangles
        self.face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        self._vertex_tree = mesh.kdtree
        self._centroid_tree = cKDTree(centroids)
        self._max_extent = float(np.max(extents))
        self._knn_k = int(min(_KNN_K, len(triangles)))
        # Touch cached R-tree so a tied fallback does not rebuild it cold.
        _ = mesh.triangles_tree

    def closest_on_surface(
        self,
        points: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Production entry: fast path with exact Trimesh fallback."""
        return self.closest_on_surface_fast(points)

    def closest_on_surface_legacy(
        self,
        points: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Previous engine: centroid ball + Python top-2 + full-tie Trimesh fallback."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (n, 3)")
        n_points = int(len(points))
        PROXIMITY_STATS["calls"] += 1
        PROXIMITY_STATS["points"] += n_points
        if n_points == 0:
            return _empty_result()

        vertex_d = np.asarray(self._vertex_tree.query(points)[0], dtype=np.float64)
        radii = vertex_d + self._max_extent + _MERGE
        balls: list[Any] = self._centroid_tree.query_ball_point(points, radii)
        counts = np.fromiter((len(ball) for ball in balls), dtype=np.int64, count=n_points)
        if np.any(counts == 0):
            PROXIMITY_STATS["empty_ball_points"] += int(np.count_nonzero(counts == 0))
            close, dist, tri = self.mesh.nearest.on_surface(points)
            return (
                np.asarray(close, dtype=np.float64),
                np.asarray(dist, dtype=np.float64),
                np.asarray(tri, dtype=np.int64),
            )

        close, dist, tri, tied, n_tri = self._closest_from_balls(points, balls, counts)
        PROXIMITY_STATS["triangles_examined"] += int(n_tri)
        if np.any(tied):
            c2, d2, t2 = self.mesh.nearest.on_surface(points[tied])
            close[tied] = c2
            dist[tied] = d2
            tri[tied] = t2
            PROXIMITY_STATS["tied_fallback_points"] += int(np.count_nonzero(tied))
            PROXIMITY_STATS["fast_path_points"] += n_points - int(np.count_nonzero(tied))
        else:
            PROXIMITY_STATS["fast_path_points"] += n_points
        return close, dist, tri

    def closest_on_surface_fast(
        self,
        points: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """k-NN centroids + conservative expand + Trimesh tie-break fallback."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (n, 3)")
        n_points = int(len(points))
        PROXIMITY_STATS["calls"] += 1
        PROXIMITY_STATS["points"] += n_points
        if n_points == 0:
            return _empty_result()

        k = self._knn_k
        centroid_d, centroid_idx = self._centroid_tree.query(points, k=k)
        centroid_d = np.asarray(centroid_d, dtype=np.float64)
        centroid_idx = np.asarray(centroid_idx, dtype=np.int64)
        if centroid_d.ndim == 1:
            centroid_d = centroid_d[:, None]
            centroid_idx = centroid_idx[:, None]

        close, dist, tri, two_dists, n_tri = self._closest_from_fixed_k(
            points, centroid_idx
        )
        PROXIMITY_STATS["triangles_examined"] += int(n_tri)

        d_k = centroid_d[:, -1]
        bound_ok = dist + _MERGE <= d_k - self._max_extent
        expand = ~bound_ok
        tied = np.ptp(two_dists, axis=1) < _MERGE
        fallback = tied.copy()

        if np.any(expand):
            n_exp = int(np.count_nonzero(expand))
            PROXIMITY_STATS["bound_expand_points"] += n_exp
            c_exp, d_exp, t_exp, tied_exp, n_ball = self._closest_ball_subset(points[expand])
            PROXIMITY_STATS["triangles_examined"] += int(n_ball)
            close[expand] = c_exp
            dist[expand] = d_exp
            tri[expand] = t_exp
            fallback[expand] = tied_exp

        if np.any(fallback):
            c2, d2, t2 = self.mesh.nearest.on_surface(points[fallback])
            close[fallback] = c2
            dist[fallback] = d2
            tri[fallback] = t2
            n_fb = int(np.count_nonzero(fallback))
            PROXIMITY_STATS["tied_fallback_points"] += n_fb
            PROXIMITY_STATS["fast_path_points"] += n_points - n_fb
        else:
            PROXIMITY_STATS["fast_path_points"] += n_points
        return close, dist, tri

    def _closest_from_fixed_k(
        self,
        points: NDArray[np.float64],
        centroid_idx: NDArray[np.int64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.float64],
        int,
    ]:
        n_points, k = centroid_idx.shape
        tile = np.repeat(np.arange(n_points, dtype=np.int64), k)
        query_close = closest_point_on_triangles(
            self.triangles[centroid_idx.reshape(-1)], points[tile]
        )
        query_distance = diagonal_dot(points[tile] - query_close, points[tile] - query_close)
        dist_sq = query_distance.reshape(n_points, k)
        idx0 = np.argmin(dist_sq, axis=1)
        rows = np.arange(n_points)
        masked = dist_sq.copy()
        masked[rows, idx0] = np.inf
        idx1 = np.argmin(masked, axis=1)
        if k == 1:
            idx1 = idx0
        two_dists = np.column_stack((dist_sq[rows, idx0], dist_sq[rows, idx1]))
        close = np.asarray(query_close.reshape(n_points, k, 3)[rows, idx0], dtype=np.float64)
        dist = np.sqrt(np.maximum(two_dists[:, 0], 0.0))
        tri = np.asarray(centroid_idx[rows, idx0], dtype=np.int64)
        return close, dist, tri, two_dists, int(n_points * k)

    def _closest_ball_subset(
        self,
        points: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.bool_],
        int,
    ]:
        if len(points) == 0:
            empty, dist, tri = _empty_result()
            return empty, dist, tri, np.zeros((0,), dtype=np.bool_), 0
        vertex_d = np.asarray(self._vertex_tree.query(points)[0], dtype=np.float64)
        radii = vertex_d + self._max_extent + _MERGE
        balls: list[Any] = self._centroid_tree.query_ball_point(points, radii)
        counts = np.fromiter((len(ball) for ball in balls), dtype=np.int64, count=len(points))
        if np.any(counts == 0):
            PROXIMITY_STATS["empty_ball_points"] += int(np.count_nonzero(counts == 0))
            close, dist, tri = self.mesh.nearest.on_surface(points)
            n = int(len(points))
            return (
                np.asarray(close, dtype=np.float64),
                np.asarray(dist, dtype=np.float64),
                np.asarray(tri, dtype=np.int64),
                np.zeros((n,), dtype=np.bool_),
                0,
            )
        return self._closest_from_balls(points, balls, counts)

    def _flatten_balls(
        self,
        points: NDArray[np.float64],
        balls: list[Any],
        counts: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        n_points = len(points)
        if counts is None:
            counts = np.fromiter((len(ball) for ball in balls), dtype=np.int64, count=n_points)
        n_cand = int(np.sum(counts))
        all_candidates = np.empty(n_cand, dtype=np.int64)
        offset = 0
        for ball in balls:
            n_hit = len(ball)
            if n_hit:
                all_candidates[offset : offset + n_hit] = ball
                offset += n_hit
        tile_idxs = np.repeat(np.arange(n_points, dtype=np.int64), counts)
        return all_candidates, tile_idxs, counts

    def _closest_from_balls(
        self,
        points: NDArray[np.float64],
        balls: list[Any],
        counts: NDArray[np.int64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.bool_],
        int,
    ]:
        all_candidates, tile_idxs, counts = self._flatten_balls(points, balls, counts)
        query_point = points[tile_idxs]
        query_close = closest_point_on_triangles(self.triangles[all_candidates], query_point)
        query_distance = diagonal_dot(query_point - query_close, query_point - query_close)
        query_group = np.cumsum(counts)[:-1]
        qds = np.array_split(query_distance, query_group)
        idxs = np.int32([qd.argsort()[:2] if len(qd) > 1 else [0, 0] for qd in qds])
        idxs[1:] += query_group.reshape(-1, 1)
        two_dists = query_distance[idxs]
        close = np.asarray(query_close[idxs[:, 0]], dtype=np.float64)
        dist = np.sqrt(np.maximum(two_dists[:, 0], 0.0))
        tri = np.asarray(all_candidates[idxs[:, 0]], dtype=np.int64)
        tied = np.ptp(two_dists, axis=1) < _MERGE
        return close, dist, tri, tied, int(len(all_candidates))


def bind_envelope_proximity(mesh: trimesh.Trimesh) -> EnvelopeSurfaceProximity:
    """Attach a reusable closest-point engine to ``mesh`` (spatial indexes once)."""
    engine = EnvelopeSurfaceProximity(mesh)
    mesh.metadata[_CACHE_KEY] = engine
    return engine


def closest_on_envelope_surface(
    mesh: trimesh.Trimesh,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Same ``(closest, distance, triangle_id)`` as ``mesh.nearest.on_surface``."""
    return _engine_for(mesh).closest_on_surface(points)


def closest_on_envelope_surface_fast(
    mesh: trimesh.Trimesh,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Fast path with exact fallback. Same return contract as the production helper."""
    return _engine_for(mesh).closest_on_surface_fast(points)


def closest_on_envelope_surface_legacy(
    mesh: trimesh.Trimesh,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Previous implementation, kept for A0 before/after benchmarks."""
    return _engine_for(mesh).closest_on_surface_legacy(points)
