"""Load STEP files into OpenCascade B-Rep shapes — no tessellation."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from nutella_scraper.cad_import.exceptions import StepReadError
from nutella_scraper.domain.models.cad_reference_geometry import CadBoundingBox

_LOG = logging.getLogger(__name__)


def step_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_step_shape(path: Path) -> object:
    """
    Read a STEP file into a TopoDS_Shape via OpenCascade.

    Raises StepReadError when cadquery-ocp is missing or the file is invalid.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
    except ImportError as exc:
        raise StepReadError(
            str(path),
            "cadquery-ocp is required for B-Rep STEP loading. "
            "Install: pip install 'nutella-scraper[cad_import]'",
        ) from exc

    _LOG.info("[brep] reading STEP B-Rep: %s", path)
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise StepReadError(str(path), f"STEPControl_Reader status={status}")

    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise StepReadError(str(path), "STEP transfer produced a null shape")

    _LOG.info("[brep] STEP B-Rep loaded: %s", path)
    return shape


def shape_bounding_box(shape: object) -> CadBoundingBox:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return CadBoundingBox(
        min_x_mm=float(xmin),
        min_y_mm=float(ymin),
        min_z_mm=float(zmin),
        max_x_mm=float(xmax),
        max_y_mm=float(ymax),
        max_z_mm=float(zmax),
    )
