"""API integration tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nutella_scraper.api.app import create_app
from nutella_scraper.application.container import build_container


class TestHealthEndpoint:
    def test_health_returns_ok(self) -> None:
        app = create_app(container=build_container())
        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
