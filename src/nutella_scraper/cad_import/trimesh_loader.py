"""Trimesh-based mesh loader for STEP and STL files."""

from __future__ import annotations

import logging
from pathlib import Path

import trimesh

from nutella_scraper.cad_import.exceptions import StepReadError, UnsupportedFormatError

_SUPPORTED = (".step", ".stp", ".stl")
_LOG = logging.getLogger(__name__)

DEFAULT_STEP_TOL_LINEAR_MM = 0.01
DEFAULT_STEP_TOL_ANGULAR_RAD = 0.1
DEFAULT_STEP_TOL_RELATIVE = False
CANONICAL_UNITS = "millimeters"


class TrimeshLoader:
    """Loads STEP/STL files via trimesh (+ cascadio for STEP)."""

    def __init__(
        self,
        *,
        step_tol_linear_mm: float = DEFAULT_STEP_TOL_LINEAR_MM,
        step_tol_angular_rad: float = DEFAULT_STEP_TOL_ANGULAR_RAD,
        step_tol_relative: bool = DEFAULT_STEP_TOL_RELATIVE,
    ) -> None:
        if step_tol_linear_mm <= 0:
            raise ValueError("step_tol_linear_mm must be positive")
        if step_tol_angular_rad <= 0:
            raise ValueError("step_tol_angular_rad must be positive")
        self.step_tol_linear_mm = float(step_tol_linear_mm)
        self.step_tol_angular_rad = float(step_tol_angular_rad)
        self.step_tol_relative = bool(step_tol_relative)

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return _SUPPORTED

    def load(self, path: Path) -> trimesh.Trimesh:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED:
            raise UnsupportedFormatError(str(path), _SUPPORTED)

        load_options: dict[str, object] = {"force": "mesh"}
        if suffix in (".step", ".stp"):
            load_options.update(
                {
                    "tol_linear": self.step_tol_linear_mm,
                    "tol_angular": self.step_tol_angular_rad,
                    "tol_relative": self.step_tol_relative,
                }
            )
            _LOG.info(
                "[tessellation] STEP=%s | tol_linear=%.6f mm | tol_angular=%.6f rad | relative=%s",
                path,
                self.step_tol_linear_mm,
                self.step_tol_angular_rad,
                self.step_tol_relative,
            )

        if suffix in (".step", ".stp"):
            _LOG.info("[step_read] début lecture STEP: %s", path)
        try:
            loaded = trimesh.load(str(path), **load_options)
        except Exception as exc:
            if suffix in (".step", ".stp"):
                raise StepReadError(
                    str(path),
                    f"{exc}. Install cad_import extras: pip install 'nutella-scraper[cad_import]'",
                ) from exc
            raise StepReadError(str(path), str(exc)) from exc

        if suffix in (".step", ".stp"):
            _LOG.info("[step_read] fin lecture STEP: %s", path)

        mesh = _to_single_mesh(loaded, path)
        source_units = mesh.units
        if suffix in (".step", ".stp"):
            if source_units is None:
                raise StepReadError(
                    str(path),
                    "STEP tessellation has no unit metadata; refusing an ambiguous scale",
                )
            if source_units != CANONICAL_UNITS:
                mesh.convert_units(CANONICAL_UNITS)
                _LOG.info(
                    "[units] STEP mesh converted from %s to %s",
                    source_units,
                    CANONICAL_UNITS,
                )

        _LOG.info(
            "[mesh] source=%s | units=%s | vertices=%d | faces=%d | extents=%s",
            path,
            mesh.units,
            len(mesh.vertices),
            len(mesh.faces),
            [float(value) for value in mesh.extents],
        )
        return mesh


def _to_single_mesh(loaded: trimesh.Trimesh | trimesh.Scene, path: Path) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise StepReadError(str(path), "Scene contains no geometry")
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise StepReadError(str(path), "Scene contains no triangular meshes")
        return trimesh.util.concatenate(meshes)

    raise StepReadError(str(path), f"Unexpected load result type: {type(loaded).__name__}")
