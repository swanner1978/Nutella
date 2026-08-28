"""TRAJECTORY_MODEL_V2_A0_ONLY — pose graph on A0. No other shapes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from nutella_scraper.engines.compute.coverage_reference_matrix import (  # noqa: E402
    build_coverage_reference_matrix,
)
from nutella_scraper.engines.compute.interior_surface_reference import (  # noqa: E402
    load_interior_surface_reference,
)
from nutella_scraper.engines.compute.pose_contact_cache import (  # noqa: E402
    build_pose_contact_cache,
    cache_entries_payload,
    cache_from_payload,
    mask_indices,
)
from nutella_scraper.engines.compute.pose_space import (  # noqa: E402
    DIAGNOSTIC_LABEL,
    TARGET_MATRIX,
    TRAJECTORY_MODEL,
    PoseSamplingConfig,
    limits_payload,
    motion_limits_from_surface,
    sample_pose_specs,
)
from nutella_scraper.engines.compute.pose_trajectory import (  # noqa: E402
    OPTIMIZATION_LABEL,
    assert_path_physical,
    beam_search_pose_trajectories,
    build_pose_edges,
    opening_pose_ids,
)
from nutella_scraper.engines.compute.shape_materialize import materialize_a0  # noqa: E402
from nutella_scraper.engines.compute.trajectory_contact_cache import (  # noqa: E402
    reference_scraper_parameters,
)

OUT_DIR = _ROOT / "output" / "coverage"
JSON_PATH = OUT_DIR / "trajectory_model_v2_diagnostic.json"
HTML_PATH = OUT_DIR / "trajectory_model_v2_a0_replay.html"
CACHE_PATH = OUT_DIR / "trajectory_model_v2_a0_pose_cache.json"
MODEL_ID = "9a1eee56-d73e-46fb-93d1-25d4ea856874"
OLD_COVERAGE_PERCENT = 0.0


def _load():
    models = _ROOT / "output" / "models"
    surface = load_interior_surface_reference(models_root=models, model_id=MODEL_ID)
    matrix = build_coverage_reference_matrix(surface)
    print(
        f"nuage={matrix.point_count} target={matrix.coverage_target_region} "
        f"label={DIAGNOSTIC_LABEL}",
        flush=True,
    )
    if matrix.point_count != 608:
        raise SystemExit(f"expected 608 targets, got {matrix.point_count}")
    if str(matrix.coverage_target_region) != TARGET_MATRIX:
        raise SystemExit(f"unexpected target {matrix.coverage_target_region}")
    return surface, matrix


def _load_or_build_cache(surface, matrix, specs, artifact, parameters):
    if CACHE_PATH.exists():
        blob = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if (
            blob.get("scraper_fingerprint") == artifact.shape_fingerprint
            and int(blob.get("n_specs", -1)) == len(specs)
        ):
            print(f"  cache disque {CACHE_PATH.name}", flush=True)
            return cache_from_payload(
                blob["entries"],
                n_points=int(blob["n_points"]),
                fingerprint=str(blob["scraper_fingerprint"]),
                angle_window_deg=tuple(blob["angle_window_deg"]),
            )
    print(f"  construction cache poses A0 n={len(specs)} …", flush=True)
    cache = build_pose_contact_cache(
        surface,
        matrix,
        specs,
        artifact=artifact,
        parameters=parameters,
    )
    blob = {
        "family_id": "A0",
        "n_points": int(cache.n_points),
        "n_specs": len(specs),
        "scraper_fingerprint": str(cache.scraper_fingerprint),
        "physics_queries": int(cache.physics_queries),
        "angle_window_deg": list(cache.angle_window_deg),
        "entries": cache_entries_payload(cache),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(blob), encoding="utf-8")
    print(f"  wrote {CACHE_PATH}", flush=True)
    return cache


def _xy(point, axes, lo, scale, size, pad):
    return (
        pad + (float(point[axes[0]]) - float(lo[0])) * scale,
        size - pad - (float(point[axes[1]]) - float(lo[1])) * scale,
    )


def _svg_replay(points, *, axes, title, touched, origins, scraper_lines, size=440.0):
    extras = [list((o[axes[0]], o[axes[1]])) for o in origins]
    for a, b in scraper_lines:
        extras.append([a[axes[0]], a[axes[1]]])
        extras.append([b[axes[0]], b[axes[1]]])
    all_pts = points[:, list(axes)]
    if extras:
        all_pts = np.vstack([all_pts, np.asarray(extras, dtype=np.float64)])
    lo = all_pts.min(axis=0)
    span = np.maximum(all_pts.max(axis=0) - lo, 1e-6)
    pad = 18.0
    scale = (size - 2 * pad) / float(np.max(span))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size:.0f}" height="{size:.0f}" '
        f'viewBox="0 0 {size:.0f} {size:.0f}">',
        '<rect width="100%" height="100%" fill="#111"/>',
        f'<text x="12" y="18" fill="#ddd" font-size="12">{title}</text>',
    ]
    for index, point in enumerate(points):
        px, py = _xy(point, axes, lo, scale, size, pad)
        fill = "#3ddc84" if index in touched else "#5a5a6a"
        rad = 2.6 if index in touched else 1.7
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rad}" fill="{fill}"/>')
    if len(origins) >= 2:
        d = []
        for origin in origins:
            px, py = _xy(origin, axes, lo, scale, size, pad)
            d.append(f"{px:.1f},{py:.1f}")
        parts.append(
            f'<polyline points="{" ".join(d)}" fill="none" stroke="#ffdd55" '
            'stroke-width="2.2"/>'
        )
    for step, origin in enumerate(origins):
        px, py = _xy(origin, axes, lo, scale, size, pad)
        parts.append(
            f'<rect x="{px - 3.5:.1f}" y="{py - 3.5:.1f}" width="7" height="7" '
            'fill="#ff6b6b" stroke="#fff" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{px + 6:.1f}" y="{py - 6:.1f}" fill="#ffdd55" '
            f'font-size="10">{step}</text>'
        )
    for a, b in scraper_lines:
        ax, ay = _xy(a, axes, lo, scale, size, pad)
        bx, by = _xy(b, axes, lo, scale, size, pad)
        parts.append(
            f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
            'stroke="#4da3ff" stroke-width="2.4"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _scraper_segment(entry, length_mm: float):
    origin = np.asarray(entry.origin_mm, dtype=np.float64)
    axis = np.asarray(entry.length_axis, dtype=np.float64)
    nrm = float(np.linalg.norm(axis))
    if nrm <= 1e-9:
        axis = np.array([0.0, -1.0, 0.0])
    else:
        axis = axis / nrm
    tip = origin + axis * float(length_mm)
    return (tuple(float(v) for v in origin), tuple(float(v) for v in tip))


def _html(payload, points, best, length_mm: float) -> str:
    touched = set(best.covered_point_indices)
    origins = [item.origin_mm for item in best.path]
    frames = []
    n = len(best.path)
    picks = [0]
    if n > 1:
        picks.extend(sorted({n // 4, n // 2, (3 * n) // 4, n - 1}))
    for index in picks:
        pose = best.path[index]
        running = 0
        for item in best.path[: index + 1]:
            running |= int(item.covered_mask)
        frame_touched = set(mask_indices(running, best.total_points))
        segment = _scraper_segment(pose, length_mm)
        top = _svg_replay(
            points,
            axes=(0, 2),
            title=f"pose {index} dessus XZ y={pose.y_mm:.1f}",
            touched=frame_touched,
            origins=origins[: index + 1],
            scraper_lines=[segment],
        )
        side = _svg_replay(
            points,
            axes=(0, 1),
            title=f"pose {index} profil XY y={pose.y_mm:.1f}",
            touched=frame_touched,
            origins=origins[: index + 1],
            scraper_lines=[segment],
        )
        frames.append(
            f"<h3>pose {index}/{n - 1}  az={pose.azimuth_deg:.1f}°  "
            f"y={pose.y_mm:.1f} mm</h3>"
            f"<div class='row'>{top}{side}</div>"
        )
    overview_top = _svg_replay(
        points,
        axes=(0, 2),
        title="trajectoire complète dessus",
        touched=touched,
        origins=origins,
        scraper_lines=[],
    )
    overview_side = _svg_replay(
        points,
        axes=(0, 1),
        title="trajectoire complète profil",
        touched=touched,
        origins=origins,
        scraper_lines=[],
    )
    return (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'/>"
        "<title>TRAJECTORY_MODEL_V2_A0_ONLY</title>"
        "<style>body{font-family:sans-serif;background:#1a1a1a;color:#eee;margin:24px}"
        ".row{display:flex;gap:12px;flex-wrap:wrap}"
        "td,th{border:1px solid #444;padding:6px 8px} table{border-collapse:collapse}"
        ".note{color:#bbb;max-width:920px}</style></head><body>"
        "<h1>TRAJECTORY_MODEL_V2_A0_ONLY</h1>"
        "<p class='note'>Replay autonome (pas le viewer principal). "
        "Jaune=chemin de poses, rouge=origine successive, bleu=scraper à la pose, "
        "vert=UNION des cibles touchées, gris=non touché. "
        f"Modèle={payload['trajectory_model']}  label={payload['optimization_label']}.</p>"
        "<table><tr><th>poses candidates</th><th>admissibles</th><th>transitions</th>"
        "<th>départs</th><th>poses trajet</th><th>couverture</th>"
        "<th>profondeur Y mm</th></tr>"
        f"<tr><td>{payload['number_of_candidate_poses']}</td>"
        f"<td>{payload['number_of_admissible_poses']}</td>"
        f"<td>{payload['number_of_edges']}</td>"
        f"<td>{len(payload['start_poses'])}</td>"
        f"<td>{payload['number_of_poses']}</td>"
        f"<td>{payload['coverage_percent']:.3f}%</td>"
        f"<td>{payload['max_depth_reached_mm']:.2f}</td></tr></table>"
        "<h2>Trajectoire complète</h2>"
        f"<div class='row'>{overview_top}{overview_side}</div>"
        "<h2>Succession de poses</h2>"
        + "".join(frames)
        + "</body></html>"
    )


def main() -> int:
    print("STOP after A0 — no other family, no campaign", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    surface, matrix = _load()
    parameters = reference_scraper_parameters(surface)
    artifact = materialize_a0(surface)
    sampling = PoseSamplingConfig()
    specs = sample_pose_specs(surface, sampling)
    limits = motion_limits_from_surface(
        surface,
        scraper_length_mm=float(parameters.length_mm),
        sampling=sampling,
    )
    print(
        f"  candidates={len(specs)}  "
        f"dY<={limits.max_vertical_step_mm:.1f}mm  "
        f"dL<={limits.max_lateral_step_mm:.1f}mm  "
        f"dR<={limits.max_rotation_step_deg:.1f}°  "
        f"opening>={limits.opening_y_mm - limits.opening_band_mm:.1f}",
        flush=True,
    )
    cache = _load_or_build_cache(surface, matrix, specs, artifact, parameters)
    n_adm = sum(1 for item in cache.entries if item.admissible)
    starts = opening_pose_ids(cache, limits)
    edges = build_pose_edges(cache, limits)
    print(
        f"  admissible={n_adm}/{len(cache.entries)}  starts={len(starts)}  "
        f"edges={len(edges)}",
        flush=True,
    )
    ranked = beam_search_pose_trajectories(
        cache, limits, edges=edges, beam_width=48, top_k=5
    )
    if not ranked:
        payload = {
            "label": DIAGNOSTIC_LABEL,
            "trajectory_model": TRAJECTORY_MODEL,
            "target_matrix": TARGET_MATRIX,
            "optimization_label": OPTIMIZATION_LABEL,
            "forme": "A0",
            "campaign_launched": False,
            "number_of_candidate_poses": len(specs),
            "number_of_admissible_poses": n_adm,
            "number_of_edges": len(edges),
            "start_poses": list(starts),
            "max_depth_reached_mm": None,
            "trajectory_length_mm": 0.0,
            "number_of_poses": 0,
            "covered_points": 0,
            "coverage_percent": 0.0,
            "previous_model_coverage_percent": OLD_COVERAGE_PERCENT,
            "trajectory_parameters": limits_payload(limits),
            "error": "aucune trajectoire pose-graphe",
        }
        JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("aucune trajectoire", flush=True)
        return 1
    best = ranked[0]
    assert_path_physical(best.path, limits)
    ys = [item.y_mm for item in best.path]
    if any(ys[i] > ys[i - 1] + 1e-9 for i in range(1, len(ys))):
        raise RuntimeError("trajectory climbs")
    payload = {
        "label": DIAGNOSTIC_LABEL,
        "trajectory_model": TRAJECTORY_MODEL,
        "target_matrix": TARGET_MATRIX,
        "optimization_label": OPTIMIZATION_LABEL,
        "forme": "A0",
        "campaign_launched": False,
        "other_families_evaluated": [],
        "number_of_candidate_poses": len(specs),
        "number_of_admissible_poses": n_adm,
        "number_of_edges": len(edges),
        "start_poses": [
            {
                "pose_id": int(pid),
                "y_mm": float(cache.entry_at(pid).y_mm),
                "azimuth_deg": float(cache.entry_at(pid).azimuth_deg),
            }
            for pid in starts
            if cache.entry_at(pid) is not None
        ],
        "max_depth_reached_mm": best.max_depth_reached_mm,
        "trajectory_length_mm": best.path_length_mm,
        "number_of_poses": best.position_count,
        "covered_points": best.covered_points,
        "coverage_percent": best.coverage_percent,
        "previous_model_coverage_percent": OLD_COVERAGE_PERCENT,
        "downward_moves": best.downward_moves,
        "lateral_moves": best.lateral_moves,
        "rotation_changes": best.rotation_changes,
        "physics_queries": int(cache.physics_queries),
        "trajectory_parameters": limits_payload(limits),
        "sampling": {
            "height_step_mm": sampling.height_step_mm,
            "azimuth_step_deg": sampling.azimuth_step_deg,
            "n_specs": len(specs),
        },
        "best_trajectory": best.to_payload(),
        "disclaimer": (
            "meilleure trajectoire trouvée (HEURISTIC), pas un optimum global"
        ),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    points = np.asarray(matrix.points_mm, dtype=np.float64)
    HTML_PATH.write_text(
        _html(payload, points, best, float(parameters.length_mm)),
        encoding="utf-8",
    )
    print(
        f"A0 UNION={best.covered_points}/{best.total_points} "
        f"{best.coverage_percent:.3f}%  poses={best.position_count}  "
        f"Y {best.start_y_mm:.1f}->{best.max_depth_reached_mm:.1f}  "
        f"old={OLD_COVERAGE_PERCENT}%",
        flush=True,
    )
    print(f"json {JSON_PATH}", flush=True)
    print(f"replay {HTML_PATH}", flush=True)
    print("STOP after A0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
