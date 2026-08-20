"""Visual-only contact-curve smoothing. Does not move cage control points.

PCHIP (Fritsch–Carlson) on chord-length parameterization: interpolating,
coordinate-wise monotone, no overshoot. Used by tests; the viewer copies
the same scheme in demo_viewer.html.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

SAMPLES_PER_SEGMENT = 8


def smooth_contact_polyline(
    points: NDArray[np.float64] | list[list[float]],
    *,
    samples_per_segment: int = SAMPLES_PER_SEGMENT,
) -> NDArray[np.float64]:
    """Return a denser interpolating polyline. Input coordinates are never written."""
    src = np.asarray(points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if len(src) < 3:
        return src.copy()
    t = np.zeros(len(src), dtype=np.float64)
    for i in range(1, len(src)):
        t[i] = t[i - 1] + max(float(np.linalg.norm(src[i] - src[i - 1])), 1e-9)
    slopes = np.column_stack(
        [_pchip_slopes(t, src[:, axis]) for axis in range(3)]
    )
    out: list[NDArray[np.float64]] = []
    samples = max(2, int(samples_per_segment))
    for i in range(len(src) - 1):
        h = float(t[i + 1] - t[i])
        start_k = 0 if i == 0 else 1
        for k in range(start_k, samples + 1):
            s = k / samples
            out.append(
                np.array(
                    [
                        _hermite(src[i, 0], src[i + 1, 0], slopes[i, 0], slopes[i + 1, 0], h, s),
                        _hermite(src[i, 1], src[i + 1, 1], slopes[i, 1], slopes[i + 1, 1], h, s),
                        _hermite(src[i, 2], src[i + 1, 2], slopes[i, 2], slopes[i + 1, 2], h, s),
                    ],
                    dtype=np.float64,
                )
            )
    return np.asarray(out, dtype=np.float64)


def _pchip_slopes(t: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(values)
    h = np.diff(t)
    delta = np.diff(values) / h
    m = np.zeros(n, dtype=np.float64)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    for i in range(n - 1):
        if delta[i] == 0.0:
            m[i] = 0.0
            m[i + 1] = 0.0
            continue
        a = m[i] / delta[i]
        b = m[i + 1] / delta[i]
        s2 = a * a + b * b
        if s2 > 9.0:
            tau = 3.0 / math.sqrt(s2)
            m[i] = tau * a * delta[i]
            m[i + 1] = tau * b * delta[i]
    return m


def _hermite(y0: float, y1: float, m0: float, m1: float, h: float, s: float) -> float:
    s2 = s * s
    s3 = s2 * s
    return (
        (2.0 * s3 - 3.0 * s2 + 1.0) * y0
        + (s3 - 2.0 * s2 + s) * h * m0
        + (-2.0 * s3 + 3.0 * s2) * y1
        + (s3 - s2) * h * m1
    )
