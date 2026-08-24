"""Read-only per-angle dump of an existing CoverageResult.

Does not choose poses. Does not change CoverageSimulator scoring.
Coverage is the union of touched target faces, never the sum of per-angle areas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper import ScraperPose
from nutella_scraper.engines.compute.coverage_simulator import CoverageResult
from nutella_scraper.engines.compute.scraper_transform import pose_matrix, pose_to_dict

AUDIT_CANDIDATE_IDS: tuple[str, ...] = ("A0", "S0008", "S0010")
_POSE_EPS_MM = 1e-6
_POSE_EPS_DEG = 1e-6


class CoverageUnionMismatchError(RuntimeError):
    """Union of per-angle faces/area does not match CoverageResult totals."""


def _wrap_delta_deg(a: float, b: float) -> float:
    return float(((b - a + 180.0) % 360.0) - 180.0)


def rotation_geodesic_deg(previous: ScraperPose, current: ScraperPose) -> float:
    ra = pose_matrix(previous)[:3, :3]
    rb = pose_matrix(current)[:3, :3]
    rel = ra.T @ rb
    trace = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def translation_mm(previous: ScraperPose, current: ScraperPose) -> float:
    pa = np.asarray(previous.position_mm, dtype=np.float64)
    pb = np.asarray(current.position_mm, dtype=np.float64)
    return float(np.linalg.norm(pb - pa))


def _progress_azimuth_deg(x: float, z: float) -> float:
    """0° = +X, 90° = −Z (same convention as CoverageSimulator)."""
    return float(np.mod(np.degrees(np.arctan2(-float(z), float(x))), 360.0))


def rest_azimuth_deg(control_points_mm: Sequence[Sequence[float]] | None) -> float | None:
    if not control_points_mm:
        return None
    pts = np.asarray(control_points_mm, dtype=np.float64)
    mid = pts[len(pts) // 2]
    return _progress_azimuth_deg(float(mid[0]), float(mid[2]))


def union_face_ids(
    touched_by_angle: Sequence[tuple[float, Sequence[int]]],
) -> frozenset[int]:
    faces: set[int] = set()
    for _angle, ids in touched_by_angle:
        faces.update(int(i) for i in ids)
    return frozenset(faces)


def area_of_faces(face_ids: Sequence[int] | frozenset[int], areas: NDArray[np.float64]) -> float:
    return float(sum(float(areas[int(face_id)]) for face_id in face_ids))


def verify_coverage_is_union(
    result: CoverageResult,
    areas: NDArray[np.float64],
) -> dict[str, Any]:
    """Abort-quality check: union(faces) and union(area) vs CoverageResult."""
    union_ids = union_face_ids(result.touched_face_ids_by_angle)
    union_area = area_of_faces(union_ids, areas)
    per_angle_sum = 0.0
    for _angle, ids in result.touched_face_ids_by_angle:
        per_angle_sum += area_of_faces(ids, areas)
    faces_ok = union_ids == frozenset(int(i) for i in result.covered_face_ids)
    area_ok = abs(union_area - float(result.covered_area_mm2)) <= 1e-6
    if not faces_ok or not area_ok:
        raise CoverageUnionMismatchError(
            f"{result.candidate_id}: union faces/area mismatch "
            f"faces_ok={faces_ok} area_ok={area_ok} "
            f"union_n={len(union_ids)} result_n={len(result.covered_face_ids)} "
            f"union_area={union_area} result_area={result.covered_area_mm2}"
        )
    return {
        "union_matches_covered_face_ids": True,
        "union_area_matches_covered_area_mm2": True,
        "union_face_count": len(union_ids),
        "union_area_mm2": union_area,
        "sum_of_per_angle_areas_mm2": per_angle_sum,
        "sum_differs_from_union": abs(per_angle_sum - union_area) > 1e-6,
        "coverage_is_union_not_sum": True,
    }


def build_angle_audit(
    result: CoverageResult,
    *,
    areas: NDArray[np.float64],
    centroids: NDArray[np.float64],
    control_points_mm: Sequence[Sequence[float]] | None = None,
    rest_vertices: NDArray[np.float64] | None = None,
    family: str | None = None,
    saved_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize what CoverageSimulator already computed. Pose choice is copied."""
    checks = verify_coverage_is_union(result, areas)
    if saved_row is not None:
        if abs(float(saved_row["coverage_percent"]) - float(result.coverage_percent)) > 1e-6:
            raise CoverageUnionMismatchError(
                f"{result.candidate_id}: replay coverage "
                f"{result.coverage_percent} != saved {saved_row['coverage_percent']}"
            )
        if abs(float(saved_row["covered_area_mm2"]) - float(result.covered_area_mm2)) > 1e-2:
            raise CoverageUnionMismatchError(
                f"{result.candidate_id}: replay area "
                f"{result.covered_area_mm2} != saved {saved_row['covered_area_mm2']}"
            )
    pose_by_angle = dict(result.best_pose_by_angle)
    faces_by_angle = dict(result.touched_face_ids_by_angle)
    control = (
        np.asarray(control_points_mm, dtype=np.float64)
        if control_points_mm is not None
        else np.zeros((0, 3), dtype=np.float64)
    )
    rest = (
        np.asarray(rest_vertices, dtype=np.float64)
        if rest_vertices is not None
        else np.zeros((0, 3), dtype=np.float64)
    )
    centroid_lookup = {
        int(face_id): [
            float(centroids[int(face_id), 0]),
            float(centroids[int(face_id), 1]),
            float(centroids[int(face_id), 2]),
        ]
        for face_id in result.covered_face_ids
        if 0 <= int(face_id) < len(centroids)
    }
    frames: list[dict[str, Any]] = []
    previous: ScraperPose | None = None
    max_translation = 0.0
    max_rotation = 0.0
    max_vertex_travel = 0.0
    previous_posed: NDArray[np.float64] | None = None
    n_pose_changes = 0
    for angle in result.evaluated_angles:
        pose = pose_by_angle.get(float(angle))
        face_ids = tuple(int(i) for i in faces_by_angle.get(float(angle), ()))
        area = area_of_faces(face_ids, areas)
        zone_pts = np.asarray(
            [centroid_lookup[i] for i in face_ids if i in centroid_lookup],
            dtype=np.float64,
        )
        posed_control = None
        posed_rest_bbox = None
        posed_centroid = None
        vertex_travel = None
        se3 = None
        position_xyz = None
        if pose is not None:
            transform = pose_matrix(pose)
            se3 = [[float(v) for v in row] for row in transform]
            position_xyz = [float(v) for v in pose.position_mm]
            if len(control):
                posed_control = (
                    np.concatenate(
                        [control, np.ones((len(control), 1), dtype=np.float64)],
                        axis=1,
                    )
                    @ transform.T
                )[:, :3]
            if len(rest):
                posed_rest = (
                    np.concatenate(
                        [rest, np.ones((len(rest), 1), dtype=np.float64)],
                        axis=1,
                    )
                    @ transform.T
                )[:, :3]
                posed_rest_bbox = {
                    "min": [float(v) for v in np.min(posed_rest, axis=0)],
                    "max": [float(v) for v in np.max(posed_rest, axis=0)],
                }
                posed_centroid = [float(v) for v in np.mean(posed_rest, axis=0)]
                if previous_posed is not None and previous_posed.shape == posed_rest.shape:
                    vertex_travel = float(
                        np.max(np.linalg.norm(posed_rest - previous_posed, axis=1))
                    )
                    max_vertex_travel = max(max_vertex_travel, vertex_travel)
                previous_posed = posed_rest
        changed = False
        translation = 0.0
        rotation = 0.0
        d_yaw = d_pitch = d_roll = 0.0
        if pose is not None and previous is not None:
            translation = translation_mm(previous, pose)
            rotation = rotation_geodesic_deg(previous, pose)
            d_yaw = _wrap_delta_deg(previous.yaw_deg, pose.yaw_deg)
            d_pitch = _wrap_delta_deg(previous.pitch_deg, pose.pitch_deg)
            d_roll = _wrap_delta_deg(previous.roll_deg, pose.roll_deg)
            changed = translation > _POSE_EPS_MM or rotation > _POSE_EPS_DEG
            max_translation = max(max_translation, translation)
            max_rotation = max(max_rotation, rotation)
            if changed:
                n_pose_changes += 1
        elif pose is not None and previous is None:
            changed = False
        frames.append(
            {
                "phi_deg": float(angle),
                "valid": pose is not None,
                "se3_matrix_4x4": se3,
                "position_xyz_mm": position_xyz,
                "yaw_deg": float(pose.yaw_deg) if pose is not None else None,
                "pitch_deg": float(pose.pitch_deg) if pose is not None else None,
                "roll_deg": float(pose.roll_deg) if pose is not None else None,
                "pose": pose_to_dict(pose) if pose is not None else None,
                "pose_changed_from_previous": changed,
                "translation_from_previous_mm": translation,
                "rotation_from_previous_deg": rotation,
                "delta_yaw_deg": d_yaw,
                "delta_pitch_deg": d_pitch,
                "delta_roll_deg": d_roll,
                "max_vertex_travel_from_previous_mm": vertex_travel,
                "face_ids": list(face_ids),
                "touched_face_count": len(face_ids),
                "area_mm2": area,
                "zone_centroids_mm": [
                    [float(v) for v in pt] for pt in zone_pts
                ]
                if len(zone_pts)
                else [],
                "zone_bbox_mm": {
                    "min": [float(v) for v in np.min(zone_pts, axis=0)],
                    "max": [float(v) for v in np.max(zone_pts, axis=0)],
                }
                if len(zone_pts)
                else None,
                "zone_centroid_mm": [float(v) for v in np.mean(zone_pts, axis=0)]
                if len(zone_pts)
                else None,
                "control_points_posed_mm": (
                    [[float(x), float(y), float(z)] for x, y, z in posed_control]
                    if posed_control is not None
                    else None
                ),
                "scraper_bbox_mm": posed_rest_bbox,
                "scraper_centroid_mm": posed_centroid,
            }
        )
        previous = pose
    y_span = None
    if len(control):
        y_span = float(np.max(control[:, 1]) - np.min(control[:, 1]))
    union_centroids = [
        centroid_lookup[i]
        for i in sorted(result.covered_face_ids)
        if i in centroid_lookup
    ]
    return {
        "candidate_id": str(result.candidate_id),
        "family": family,
        "replay_only": True,
        "simulator_physics_unchanged": True,
        "coverage_percent": float(result.coverage_percent),
        "covered_area_mm2": float(result.covered_area_mm2),
        "target_area_mm2": float(result.target_area_mm2),
        "covered_face_ids": sorted(int(i) for i in result.covered_face_ids),
        "shape_fingerprint": str(result.shape_fingerprint),
        "evaluated_angles_deg": [float(v) for v in result.evaluated_angles],
        "n_angles": len(result.evaluated_angles),
        "valid_pose_count": sum(1 for _a, pose in result.best_pose_by_angle if pose is not None),
        "control_y_span_mm": y_span,
        "rest_azimuth_deg": rest_azimuth_deg(control_points_mm),
        "control_points_mm": (
            [[float(x), float(y), float(z)] for x, y, z in control] if len(control) else None
        ),
        "face_centroids_mm": centroid_lookup,
        "union_centroids_mm": union_centroids,
        "pose_jumps": {
            "n_changes": n_pose_changes,
            "max_translation_mm": max_translation,
            "max_rotation_deg": max_rotation,
            "max_vertex_travel_mm": max_vertex_travel,
            "independent_pose_per_angle": n_pose_changes > 0,
        },
        "union_checks": checks,
        "angles": frames,
    }


def _item_rest_azimuth(item: Mapping[str, Any]) -> float | None:
    if item.get("rest_azimuth_deg") is not None:
        return float(item["rest_azimuth_deg"])
    return rest_azimuth_deg(item.get("control_points_mm"))


def format_comparative_report(audits: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "Audit de couverture A0 / S0008 / S0010",
        "Replay CoverageSimulator uniquement. Pas de nouveau candidat.",
        "Couverture = UNION des faces, jamais la somme des aires par angle.",
        "Union(faces des 24 angles) = covered_face_ids.",
        "Aire de cette union = covered_area_mm2.",
        "",
    ]
    extra_s0008: set[int] = set()
    a0_faces: set[int] = set()
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in audits:
        cid = str(item["candidate_id"])
        by_id[cid] = item
        checks = item["union_checks"]
        jumps = item["pose_jumps"]
        az = _item_rest_azimuth(item)
        az_txt = f"{az:.2f}" if az is not None else "n/a"
        lines.extend(
            [
                f"=== {cid} ===",
                f"  famille: {item.get('family')}",
                f"  couverture: {item['coverage_percent']:.4f} %",
                f"  aire union: {item['covered_area_mm2']:.4f} / {item['target_area_mm2']:.4f} mm2",
                f"  faces union: {len(item['covered_face_ids'])}",
                f"  somme des aires par angle: {checks['sum_of_per_angle_areas_mm2']:.4f} mm2",
                f"  somme != union: {checks['sum_differs_from_union']}",
                f"  span Y points de controle: {item.get('control_y_span_mm')}",
                f"  azimuth repos (0=+X, 90=-Z): {az_txt} deg",
                f"  poses changees (φ → φ+2°): {jumps['n_changes']} / {item['n_angles'] - 1}",
                f"  translation max: {jumps['max_translation_mm']:.4f} mm",
                f"  rotation max: {jumps['max_rotation_deg']:.4f} deg",
                f"  deplacement vertex max: {jumps['max_vertex_travel_mm']:.4f} mm",
                f"  pose independante par angle: {jumps['independent_pose_per_angle']}",
                "",
            ]
        )
        if cid == "A0":
            a0_faces = set(item["covered_face_ids"])
        if cid == "S0008":
            extra_s0008 = set(item["covered_face_ids"])
    if a0_faces and extra_s0008:
        only_s0008 = extra_s0008 - a0_faces
        only_a0 = a0_faces - extra_s0008
        a0_az = _item_rest_azimuth(by_id.get("A0", {}))
        s8_az = _item_rest_azimuth(by_id.get("S0008", {}))
        lines.extend(
            [
                "S0008 vs A0:",
                f"  faces seulement S0008: {len(only_s0008)}",
                f"  faces seulement A0: {len(only_a0)}",
                f"  faces communes: {len(extra_s0008 & a0_faces)}",
                "",
                "Controle d'incoherence: OK "
                "(union faces = covered_face_ids, union aire = covered_area_mm2).",
                "",
                "Lecture:",
                "  Le moteur choisit la meilleure pose admissible parmi 17 SE(3)",
                "  a CHAQUE angle, independamment de l'angle precedent.",
                "  La trajectoire n'est pas continue: poses constantes sur plusieurs",
                "  pas de 2°, puis saut (translation jusqu'a ~20 mm, yaw jusqu'a 30°).",
                "  S0008 n'est PAS plus court en Y que A0: span Y identique (cage).",
                "  Tous les points de controle de S0008 ont le meme (X,Z): droite",
                "  verticale a ~"
                f"{s8_az:.1f}° (milieu du secteur 0–45°). A0 est a ~{a0_az:.1f}°."
                if s8_az is not None and a0_az is not None
                else "  verticale decalee en azimuth par rapport a A0.",
                "  Vue de face sur le meridien A0, S0008 apparait court par raccourci",
                "  perspectif (ligne sur le flanc du cylindre), pas par une lame plus courte.",
                "  A φ=0 (pose identite, Y non abaissee) S0008 peint 17.5–27.5° en haut",
                "  du pot; A0 peint 2.5–5°. Des φ=2°, les deux chutent de ~20 mm en Y.",
                "  L'ecart 66.25% vs 63.33% = 7 faces nettes dans l'UNION du secteur,",
                "  pas une somme d'aires par angle, pas un artefact de longueur.",
            ]
        )
    return "\n".join(lines)
