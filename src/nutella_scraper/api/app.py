"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nutella_scraper.api.routes import health, import_route, optimization, simulation, visualization
from nutella_scraper.application.container import Container, build_container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.container = build_container()
    yield


def create_app(container: Container | None = None) -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Nutella Scraper API",
        version="0.1.0",
        description="API for scraper geometry optimization and contact simulation",
        lifespan=lifespan if container is None else None,
    )
    if container is not None:
        app.state.container = container

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/v1", tags=["health"])
    app.include_router(import_route.router, prefix="/v1/import", tags=["import"])
    app.include_router(simulation.router, prefix="/v1/simulate", tags=["simulation"])
    app.include_router(visualization.router, prefix="/v1/visualization", tags=["visualization"])
    app.include_router(optimization.router, prefix="/v1/optimization", tags=["optimization"])

    return app
