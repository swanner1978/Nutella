"""TRAJECTORY_MODEL_DIAGNOSTIC — no campaign, no viewer change."""

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
from nutella_scraper.engines.compute.shape_families import (  # noqa: E402
    FAMILY_BY_ID,
    build_sagittal_frame,
    sample_profile,
)
from nutella_scraper.engines.compute.shape_fitter import profile_length_mm  # noqa: E402
from nutella_scraper.engines.compute.shape_materialize import (  # noqa: E402
    materialize_a0,
    materialize_profile,
)
from nutella_scraper.engines.compute.trajectory_contact_cache import (  # noqa: E402
    build_contact_cache,
)
from nutella_scraper.engines.compute.trajectory_model_diagnostic import (  # noqa: E402
    DIAGNOSTIC_LABEL,
    MAPPING_STEPS,
    MAX_DOWNWARD_STEP_MEANS,
    analyze_shape_graph,
    cache_entries_payload,
    cache_from_payload,
    height_monotone_pose_chain,
    pose_contact_spans,
    report_to_payload,
    union_indices,
)
from nutella_scraper.engines.compute.trajectory_optimizer import (  # noqa: E402
    beam_search_trajectories,
)
from nutella_scraper.engines.compute.trajectory_search import (  # noqa: E402
    MAX_DOWNWARD_STEP,
    MAX_LATERAL_STEP,
    index_reference_matrix,
)

OUT_DIR = _ROOT / "output" / "coverage" / "trajectory_model_diagnostic"
MODEL_ID = "9a1eee56-d73e-46fb-93d1-25d4ea856874"
SHAPES = ("A0", "straight", "circular_arc", "bezier_4")


def _load_surface():
    models = _ROOT / "output" / "models"
    surface = load_interior_surface_reference(models_root=models, model_id=MODEL_ID)
    matrix = build_coverage_reference_matrix(surface)
    grid = index_reference_matrix(matrix, surface=surface)
    print(
        f"nuage={matrix.point_count} rows={grid.n_rows} cols={grid.n_cols} "
        f"target={matrix.coverage_target_region}",
        flush=True,
    )
    return surface, matrix, grid


def _artifact_for(family_id: str, surface, frame):
    if family_id == "A0":
        return materialize_a0(surface)
    family = FAMILY_BY_ID[family_id]
    profile = sample_profile(
        family, family.default_params(frame), frame, sample_count=32
    )
    return materialize_profile(
        profile, surface, length_mm=max(profile_length_mm(profile), 1.0)
    )


def _cache_path(family_id: str) -> Path:
    return OUT_DIR / f"cache_{family_id}.json"


def _load_or_build_cache(family_id: str, surface, matrix, grid, frame):
    path = _cache_path(family_id)
    if path.exists():
        print(f"  cache disque {path.name}", flush=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cache_from_payload(
            grid,
            payload["entries"],
            n_points=int(payload["n_points"]),
            fingerprint=str(payload["scraper_fingerprint"]),
        )
    print(f"  construction cache {family_id} …", flush=True)
    artifact = _artifact_for(family_id, surface, frame)
    cache = build_contact_cache(
        surface, matrix, grid, artifact=artifact, parameters=None
    )
    blob = {
        "family_id": family_id,
        "n_points": int(cache.n_points),
        "scraper_fingerprint": str(cache.scraper_fingerprint),
        "physics_queries": int(cache.physics_queries),
        "entries": cache_entries_payload(grid, cache),
    }
    path.write_text(json.dumps(blob), encoding="utf-8")
    print(f"  wrote {path}", flush=True)
    return cache


def _xy(point, axes, lo, scale, size, pad):
    return (
        pad + (float(point[axes[0]]) - float(lo[0])) * scale,
        size - pad - (float(point[axes[1]]) - float(lo[1])) * scale,
    )


def _svg_cloud(points, *, axes, title, touched, highlight, origins, rings, size=420.0):
    extras = [list((o[axes[0]], o[axes[1]])) for o in origins + rings]
    all_pts = points[:, list(axes)]
    if extras:
        all_pts = np.vstack([all_pts, np.asarray(extras, dtype=np.float64)])
    lo = all_pts.min(axis=0)
    span = np.maximum(all_pts.max(axis=0) - lo, 1e-6)
    pad = 16.0
    scale = (size - 2 * pad) / float(np.max(span))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size:.0f}" height="{size:.0f}" '
        f'viewBox="0 0 {size:.0f} {size:.0f}">',
        '<rect width="100%" height="100%" fill="#111"/>',
        f'<text x="12" y="18" fill="#ddd" font-size="12">{title}</text>',
    ]
    for index, point in enumerate(points):
        px, py = _xy(point, axes, lo, scale, size, pad)
        if index in highlight:
            fill, rad = "#ffdd55", 3.4
        elif index in touched:
            fill, rad = "#3ddc84", 2.6
        else:
            fill, rad = "#5a5a6a", 1.7
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rad}" fill="{fill}"/>')
    for item in rings:
        px, py = _xy(item, axes, lo, scale, size, pad)
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="none" '
            'stroke="#4da3ff" stroke-width="1.2"/>'
        )
    for origin in origins:
        px, py = _xy(origin, axes, lo, scale, size, pad)
        parts.append(
            f'<rect x="{px - 4:.1f}" y="{py - 4:.1f}" width="8" height="8" '
            'fill="#ff6b6b" stroke="#fff" stroke-width="0.6"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _view_block(points, view_id, view):
    touched = set(view["touched"])
    highlight = set(view.get("highlight", []))
    origins = [tuple(view["origin"])]
    rings = [tuple(p) for p in view.get("admissible_points", [])]
    top = _svg_cloud(
        points, axes=(0, 2), title=f"{view_id} dessus XZ",
        touched=touched, highlight=highlight, origins=origins, rings=rings,
    )
    side = _svg_cloud(
        points, axes=(0, 1), title=f"{view_id} profil XY",
        touched=touched, highlight=highlight, origins=origins, rings=rings,
    )
    return f"<h3>{view_id}</h3><p>{view['caption']}</p><div class='row'>{top}{side}</div>"


def _html_report(payload, points, a0_views):
    sections = [_view_block(points, vid, view) for vid, view in a0_views.items()]
    rows = []
    for shape in payload["formes"]:
        cut = shape["extinction"]
        rows.append(
            "<tr>"
            f"<td>{shape['forme']}</td><td>{shape['cellules_admissibles']}</td>"
            f"<td>{shape['cellules_avec_contact']}</td>"
            f"<td>{shape['premiere_rangee_sans_admissible']}</td>"
            f"<td>{shape['transitions_rangees_successives']}</td>"
            f"<td>{shape['chemins_partiels_atteignables_cellules']}</td>"
            f"<td>{shape['profondeur_max_row']}</td>"
            f"<td>{cut['kind']} @ {cut['row']}</td></tr>"
        )
    mapping = "</li><li>".join(MAPPING_STEPS)
    return (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'/>"
        f"<title>{DIAGNOSTIC_LABEL}</title><style>"
        "body{font-family:sans-serif;background:#1a1a1a;color:#eee;margin:24px}"
        ".row{display:flex;gap:12px;flex-wrap:wrap}"
        "td,th{border:1px solid #444;padding:6px 8px} table{border-collapse:collapse}"
        ".note{color:#bbb;max-width:920px}</style></head><body>"
        f"<h1>{DIAGNOSTIC_LABEL}</h1>"
        "<p class='note'>Pas le viewer principal. Gris=cible, vert=touché, "
        "jaune=pose affichée, carré rouge=origine pose, anneau bleu=cellules de rangée.</p>"
        f"<h2>Mapping actuel</h2><ol><li>{mapping}</li></ol>"
        f"<p>MAX_DOWNWARD_STEP={MAX_DOWNWARD_STEP} : {MAX_DOWNWARD_STEP_MEANS}</p>"
        f"<p>MAX_LATERAL_STEP={MAX_LATERAL_STEP} (index de colonne, pas des mm).</p>"
        "<h2>Coupe du graphe</h2><table><tr><th>forme</th><th>adm</th><th>contact</th>"
        "<th>1re row vide</th><th>trans +1</th><th>atteignables</th><th>depth</th>"
        f"<th>extinction</th></tr>{''.join(rows)}</table>"
        f"<h2>A0 — poses</h2>{''.join(sections)}</body></html>"
    )


def _span_view(span, caption, extra):
    if span is None:
        return {
            "caption": caption,
            "origin": (0.0, 0.0, 0.0),
            "touched": [],
            "highlight": [],
            "admissible_points": extra,
        }
    return {
        "caption": caption,
        "origin": span.origin_mm,
        "touched": list(span.covered_indices),
        "highlight": list(span.covered_indices),
        "admissible_points": extra,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    surface, matrix, grid = _load_surface()
    frame = build_sagittal_frame(surface)
    points = np.asarray(matrix.points_mm, dtype=np.float64)
    shape_payloads = []
    a0_cache = None
    for family_id in SHAPES:
        print(f"[{family_id}]", flush=True)
        cache = _load_or_build_cache(family_id, surface, matrix, grid, frame)
        ranked = beam_search_trajectories(grid, cache, beam_width=48, top_k=1)
        report = analyze_shape_graph(
            family_id, grid, cache, beam_complete_paths=len(ranked)
        )
        shape_payloads.append(report_to_payload(report))
        print(
            f"  adm={report.n_admissible} contact={report.n_with_contact} "
            f"empty_row={report.first_row_without_admissible} "
            f"reach={report.n_reachable_from_opening} "
            f"depth={report.max_reachable_row} cut={report.extinction.kind}"
            f"@{report.extinction.row}",
            flush=True,
        )
        if family_id == "A0":
            a0_cache = cache
    if a0_cache is None:
        raise SystemExit("cache A0 manquant")
    spans = pose_contact_spans(grid, a0_cache)
    chain = height_monotone_pose_chain(spans, max_poses=24, min_drop_mm=2.0)
    chain_union = union_indices(chain)
    top_span = spans[0] if spans else None
    opening_span = max(spans, key=lambda item: item.origin_mm[1]) if spans else None
    graph = next(item for item in shape_payloads if item["forme"] == "A0")
    last_zone_row = graph["profondeur_max_row"]
    last_span = None
    if last_zone_row is not None:
        cand = [item for item in spans if item.row == int(last_zone_row)]
        last_span = max(cand, key=lambda item: item.covered_count) if cand else None
    first_dead_row = graph["extinction"]["row"]
    opening_adm = [
        (float(c.x_mm), float(c.y_mm), float(c.z_mm)) for c in grid.cells if c.is_top
    ]
    last_adm = [
        (float(c.x_mm), float(c.y_mm), float(c.z_mm))
        for c in grid.cells
        if last_zone_row is not None and c.row == int(last_zone_row)
    ]
    dead_pts = [
        (float(c.x_mm), float(c.y_mm), float(c.z_mm))
        for c in grid.cells
        if c.row == int(first_dead_row)
    ]
    cap_max = (
        f"pose ({top_span.row},{top_span.col}) contact={top_span.covered_count} "
        f"rows={top_span.n_rows_covered} "
        f"Y[{top_span.touched_y_min_mm:.1f},{top_span.touched_y_max_mm:.1f}]"
        if top_span
        else "aucune pose avec contact"
    )
    a0_views = {
        "pose_max_contact": _span_view(top_span, cap_max, []),
        "pose_ouverture": _span_view(
            opening_span, "pose admissible la plus haute (ouverture)", opening_adm
        ),
        "derniere_zone_possible": _span_view(
            last_span,
            f"dernière rangée atteignable depuis l'ouverture (row={last_zone_row})",
            last_adm,
        ),
        "premiere_zone_impossible": _span_view(
            last_span,
            f"première rangée où le graphe s'éteint (row={first_dead_row})",
            dead_pts,
        ),
    }
    a0_span_stats = {
        "n_poses_avec_contact": len(spans),
        "max_points_par_pose": int(top_span.covered_count) if top_span else 0,
        "max_rangees_par_pose": int(top_span.n_rows_covered) if top_span else 0,
        "y_min_points_touches_pose_max": (
            float(top_span.touched_y_min_mm) if top_span else None
        ),
        "y_max_points_touches_pose_max": (
            float(top_span.touched_y_max_mm) if top_span else None
        ),
        "exemples_max_contact": [
            {
                "row": item.row,
                "col": item.col,
                "covered_count": item.covered_count,
                "n_rows_covered": item.n_rows_covered,
                "touched_y_min_mm": item.touched_y_min_mm,
                "touched_y_max_mm": item.touched_y_max_mm,
                "origin_mm": list(item.origin_mm),
            }
            for item in spans[:8]
        ],
        "hypothese_chaine_hauteur": {
            "n_poses": len(chain),
            "union_points": len(chain_union),
            "union_percent": (
                100.0 * len(chain_union) / float(matrix.point_count)
                if matrix.point_count
                else 0.0
            ),
            "poses": [
                {
                    "row": item.row,
                    "col": item.col,
                    "y": item.origin_mm[1],
                    "covered": item.covered_count,
                }
                for item in chain
            ],
        },
    }
    payload = {
        "label": DIAGNOSTIC_LABEL,
        "campaign_launched": False,
        "model_id": MODEL_ID,
        "mapping_steps": list(MAPPING_STEPS),
        "max_downward_step_means": MAX_DOWNWARD_STEP_MEANS,
        "max_downward_step": MAX_DOWNWARD_STEP,
        "max_lateral_step": MAX_LATERAL_STEP,
        "cellule_egale_pose": True,
        "trois_notions": {
            "A_point_cible": "point du nuage interior_matrix_a0_0_90",
            "B_pose_scraper": "SE(3) yaw=azimuth cellule, Y=y cellule + voisinage",
            "C_trajectoire": "succession de cellules-waypoints admissibles",
        },
        "formes": shape_payloads,
        "A0_contact_span": a0_span_stats,
    }
    json_path = OUT_DIR / "trajectory_model_diagnostic.json"
    html_path = OUT_DIR / "trajectory_model_diagnostic.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(_html_report(payload, points, a0_views), encoding="utf-8")
    print(
        json.dumps(
            {
                "label": DIAGNOSTIC_LABEL,
                "formes": [
                    {
                        "forme": item["forme"],
                        "adm": item["cellules_admissibles"],
                        "cut": item["extinction"]["kind"],
                        "cut_row": item["extinction"]["row"],
                        "depth": item["profondeur_max_row"],
                    }
                    for item in shape_payloads
                ],
                "A0_union_chaine_hauteur": a0_span_stats["hypothese_chaine_hauteur"][
                    "union_percent"
                ],
                "json": str(json_path),
                "html": str(html_path),
            },
            indent=2,
        )
    )
    print("TRAJECTORY_MODEL_DIAGNOSTIC terminé. Campagne NON lancée.")


if __name__ == "__main__":
    main()
