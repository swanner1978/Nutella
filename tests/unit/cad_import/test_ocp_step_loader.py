"""Verify OCP (cadquery-ocp) is available and can load STEP B-Rep."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("OCP", reason="cadquery-ocp must be installed (pip install -e \".[dev]\")")

from nutella_scraper.cad_import.cad_reference_builder import CadReferenceGeometryBuilder
from nutella_scraper.cad_import.step_brep_loader import load_step_shape


def test_ocp_import_available() -> None:
    import OCP

    assert OCP.__file__ is not None
    assert "site-packages" in OCP.__file__.replace("\\", "/")


def test_step_brep_loader_opens_real_step(jar_step_path: Path) -> None:
    shape = load_step_shape(jar_step_path)
    assert shape is not None
    assert not shape.IsNull()


def test_step_brep_pipeline_produces_cad_reference(jar_step_path: Path) -> None:
    geometry = CadReferenceGeometryBuilder().from_step(jar_step_path, model_id="ocp_test")
    assert geometry.metadata.get("source") == "opencascade_brep"
    assert geometry.profile_contour is not None
    assert geometry.top_contour is not None
    assert geometry.profile_contour.source == "opencascade_brep_section"
    assert geometry.inner_face_count > 0
