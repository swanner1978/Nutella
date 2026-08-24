"""Shared A0 + catalog setup for coverage-batch tests and the ranking script."""

from __future__ import annotations

from tests.unit.engines.compute.test_scraper_parametric_v1 import (
    _profile_a,
    _reference_from_profile,
)

from nutella_scraper.engines.compute.scraper_envelope_path import scraper_length_span
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    build_rigid_scraper_artifact,
)
from nutella_scraper.engines.visualization.scraper_control_cage import (
    build_control_cage_overlay,
)
from nutella_scraper.engines.visualization.scraper_shape_space import (
    generate_candidate_shapes,
    lattice_from_cage,
)


def synthetic_coverage_surface():
    return _reference_from_profile(
        radius_at_y=lambda _y: 50.0,
        y_min=0.0,
        y_max=80.0,
        y_count=21,
        angular_count=48,
    )


def a0_coverage_parameters(surface):
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    return _profile_a(
        width_mm=2.5,
        thickness_mm=2.5,
        length_mm=min(40.0, max_length),
        clearance_mm=0.0,
        position_z_mm=float(0.5 * (surface.y_min_mm + surface.y_max_mm)),
    )


def load_generated_catalog(*, count: int = 1000):
    """A0 manufacturing solid + existing generate_candidate_shapes catalog."""
    surface = synthetic_coverage_surface()
    parameters = a0_coverage_parameters(surface)
    reference = build_rigid_scraper_artifact(surface, parameters)
    cage = build_control_cage_overlay(reference.design_path, surface)
    lattice = lattice_from_cage(cage, surface)
    catalog = generate_candidate_shapes(lattice, count=count)
    return surface, parameters, reference, catalog
