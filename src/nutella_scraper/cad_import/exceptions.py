"""CAD import pipeline errors."""

from __future__ import annotations


class CadImportError(Exception):
    """Base error for CAD import pipeline failures."""


class UnsupportedFormatError(CadImportError):
    """Raised when the file format is not supported."""

    def __init__(self, path: str, supported: tuple[str, ...]) -> None:
        self.path = path
        self.supported = supported
        super().__init__(
            f"Unsupported format for '{path}'. Supported extensions: {', '.join(supported)}"
        )


class StepReadError(CadImportError):
    """Raised when a STEP file cannot be parsed."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to read STEP file '{path}': {reason}")


class InvalidGeometryError(CadImportError):
    """Raised when mesh geometry fails validation."""

    def __init__(self, path: str, violations: tuple[str, ...]) -> None:
        self.path = path
        self.violations = violations
        super().__init__(
            f"Invalid geometry in '{path}': " + "; ".join(violations)
        )
