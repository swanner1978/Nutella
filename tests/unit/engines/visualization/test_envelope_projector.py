"""Envelope projection tests."""

from __future__ import annotations

from nutella_scraper.domain.models.envelope import EnvelopeSlice, InteriorEnvelope
from nutella_scraper.engines.visualization.envelope_projector import (
    LAYER_ENVELOPE,
    EnvelopeProjector,
)


def test_envelope_projector_emits_layers(cad_reference_geometry) -> None:
    envelope = InteriorEnvelope(
        jar_id=cad_reference_geometry.model_id,
        y_min_mm=0.0,
        y_max_mm=100.0,
        neck_radius_mm=30.0,
        clearance_mm=0.15,
        slices=(
            EnvelopeSlice(y_mm=0.0, max_radial_mm=49.0),
            EnvelopeSlice(y_mm=50.0, max_radial_mm=49.0),
            EnvelopeSlice(y_mm=100.0, max_radial_mm=29.0),
        ),
    )
    projection = EnvelopeProjector().project(envelope, cad_reference_geometry)

    assert projection.profile_layers
    assert projection.top_layers
    assert projection.profile_layers[0].layer_type == LAYER_ENVELOPE
    assert "path" in projection.profile_layers[0].svg_fragment
    assert "#a855f7" in projection.profile_layers[0].svg_fragment
    assert 'fill="none"' in projection.profile_layers[0].svg_fragment
    assert "interior-profile-contour" in projection.profile_layers[0].svg_fragment
    assert projection.profile_layers[0].svg_fragment.count("L") >= 4
