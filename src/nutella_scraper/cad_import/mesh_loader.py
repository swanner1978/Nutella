"""Mesh loader protocol — dependency interface for external geometry libraries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class LoadedMesh(Protocol):
    """Minimal mesh surface required by the CAD import pipeline."""

    @property
    def vertices(self) -> Any: ...

    @property
    def faces(self) -> Any: ...

    def bounding_box(self) -> Any: ...

    @property
    def is_watertight(self) -> bool: ...

    @property
    def volume(self) -> float: ...

    def copy(self) -> LoadedMesh: ...


class IMeshLoader(Protocol):
    """Loads a file path into a mesh representation."""

    def load(self, path: Path) -> LoadedMesh:
        """Load mesh from disk. Raises CadImportError subclasses on failure."""
        ...

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """File extensions this loader handles (lowercase, with dot)."""
        ...
