"""Parametric scraper control parameters — UI / generator / placement SoT."""

from __future__ import annotations

from dataclasses import asdict, dataclass


# Manufacturing (shape) fields — changing these rebuilds the rigid solid.
MANUFACTURING_PARAM_KEYS: tuple[str, ...] = (
    "width_mm",
    "length_mm",
    "thickness_mm",
    "position_z_mm",
    "bevel_angle_deg",
    "relief_angle_deg",
    "helix_rate_deg_per_mm",
    "clearance_mm",
)

# Pose / path-progress fields — never reshape the solid.
POSE_PARAM_KEYS: tuple[str, ...] = (
    "surface_progress_deg",
)


@dataclass(frozen=True)
class ScraperParameters:
    """
    Manual parametric scraper V1 controls.

    Strict separation:
      MANUFACTURING → lofted once into a rigid FDM solid
      POSE / PROGRESS → SE(3) placement along the interior envelope

    ``surface_progress_deg`` is path progress on the envelope (where to place
    the tip). It is NOT a rebuild trigger and NOT a forced spin about the jar
    axis. ``rotation_angle_deg`` is kept as a UI/API alias of that progress.
    """

    width_mm: float = 15.0
    length_mm: float = 60.0
    thickness_mm: float = 4.0
    position_z_mm: float = 50.0
    surface_progress_deg: float = 0.0
    bevel_angle_deg: float = 30.0
    relief_angle_deg: float = 10.0
    helix_rate_deg_per_mm: float = 0.0
    clearance_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.length_mm <= 0 or self.thickness_mm <= 0:
            raise ValueError("width_mm, length_mm and thickness_mm must be positive")
        if self.bevel_angle_deg < 0 or self.bevel_angle_deg >= 90:
            raise ValueError("bevel_angle_deg must be in [0, 90)")
        if self.relief_angle_deg < 0 or self.relief_angle_deg >= 90:
            raise ValueError("relief_angle_deg must be in [0, 90)")
        if self.clearance_mm < 0:
            raise ValueError("clearance_mm must be non-negative")

    @property
    def rotation_angle_deg(self) -> float:
        """UI/API alias: envelope path progress (not a geometry rebuild key)."""
        return float(self.surface_progress_deg)

    def with_updates(self, **changes: float) -> ScraperParameters:
        payload = asdict(self)
        if "rotation_deg" in changes:
            if (
                "surface_progress_deg" not in changes
                and "rotation_angle_deg" not in changes
            ):
                changes["surface_progress_deg"] = changes["rotation_deg"]
            del changes["rotation_deg"]
        if "rotation_angle_deg" in changes:
            if "surface_progress_deg" not in changes:
                changes["surface_progress_deg"] = changes["rotation_angle_deg"]
            del changes["rotation_angle_deg"]
        payload.update(changes)
        return ScraperParameters(**payload)

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        # Keep legacy UI field name in sync with surface progress.
        payload["rotation_angle_deg"] = float(self.surface_progress_deg)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> ScraperParameters:
        if not payload:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, float] = {}
        for key, value in payload.items():
            if key in {"rotation_deg", "rotation_angle_deg"}:
                if "surface_progress_deg" not in payload:
                    kwargs["surface_progress_deg"] = float(value)  # type: ignore[arg-type]
                continue
            if key not in known:
                continue
            kwargs[key] = float(value)  # type: ignore[arg-type]
        return cls(**kwargs)

    @classmethod
    def default(cls) -> ScraperParameters:
        return cls()
