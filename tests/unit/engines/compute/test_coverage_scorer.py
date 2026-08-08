"""Coverage scorer tests."""

from __future__ import annotations

import trimesh

from nutella_scraper.engines.compute.coverage_scorer import CoverageScorer


class TestCoverageScorer:
    def test_empty_face_sets_return_zero(self) -> None:
        mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        score = CoverageScorer().score(frozenset(), frozenset(), mesh)
        assert score == 0.0

    def test_full_coverage_returns_one(self) -> None:
        mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        all_faces = frozenset(range(len(mesh.faces)))
        score = CoverageScorer().score(all_faces, frozenset(), mesh)
        assert score == 1.0
