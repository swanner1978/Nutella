"""Persistence for visualization-only view projection caches."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

from nutella_scraper.domain.models.views import (
    ProjectedView,
    ProjectionMetadata,
    ViewProjectionCache,
)


class ViewCacheStore:
    """
    Stores ViewProjectionCache artifacts.

    @visualization_only — must never be read by ComputeEngine or OptimizationEngine.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, cache: ViewProjectionCache) -> str:
        views_id = str(uuid.uuid4())
        views_dir = self._base_dir / views_id
        views_dir.mkdir(parents=True, exist_ok=True)

        if cache.profile_view.svg_content:
            (views_dir / "profile.svg").write_text(cache.profile_view.svg_content, encoding="utf-8")
        if cache.top_view.svg_content:
            (views_dir / "top.svg").write_text(cache.top_view.svg_content, encoding="utf-8")

        manifest = {
            "views_id": views_id,
            "model_id": cache.model_id,
            "provenance": cache.provenance,
            "visualization_only": True,
            "side": _view_to_dict(cache.profile_view),
            "profile": _view_to_dict(cache.profile_view),
            "top": _view_to_dict(cache.top_view),
            "projection_metadata": cache.projection_metadata,
        }
        (views_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return views_id

    def get(self, views_id: str) -> ViewProjectionCache | None:
        views_dir = self._base_dir / views_id
        manifest_path = views_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        profile_svg = _read_optional(views_dir / "profile.svg")
        top_svg = _read_optional(views_dir / "top.svg")

        profile = _view_from_dict(
            data.get("side", data["profile"]),
            profile_svg,
        )
        top = _view_from_dict(data["top"], top_svg)

        return ViewProjectionCache(
            model_id=str(data["model_id"]),
            profile_view=profile,
            top_view=top,
            projection_metadata=data.get("projection_metadata", {}),
            provenance="visualization_projection",
        )


def _view_to_dict(view: ProjectedView) -> dict[str, object]:
    return {
        "plane": view.plane,
        "asset_path": str(view.asset_path) if view.asset_path else None,
        "metadata": asdict(view.metadata),
    }


def _view_from_dict(data: dict[str, object], svg_content: str | None) -> ProjectedView:
    meta = data["metadata"]
    assert isinstance(meta, dict)
    metadata = ProjectionMetadata(
        plane=str(meta["plane"]),
        camera=dict(meta.get("camera", {})),
        scale=float(meta["scale"]),
        width_px=int(meta["width_px"]),
        height_px=int(meta["height_px"]),
    )
    asset = data.get("asset_path")
    return ProjectedView(
        plane=str(data["plane"]),
        asset_path=Path(str(asset)) if asset else None,
        svg_content=svg_content,
        metadata=metadata,
    )


def _read_optional(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None
