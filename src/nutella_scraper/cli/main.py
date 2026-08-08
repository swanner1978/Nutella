"""CLI entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from nutella_scraper.application.container import build_container
from nutella_scraper.application.dto import OptimizationRequestDTO, SimulateRequestDTO

app = typer.Typer(
    name="nutella-scraper",
    help="Optimisation geometrique de racloir Nutella (FDM)",
    no_args_is_help=True,
)
console = Console()


@app.command("version")
def version() -> None:
    """Show application version."""
    from nutella_scraper import __version__

    console.print(f"nutella-scraper v{__version__}")


@app.command("import")
def import_model(
    sldprt: Path | None = typer.Option(None, help="Path to SLDPRT file"),
    stl: Path | None = typer.Option(None, help="Path to pre-exported STL"),
    step: Path | None = typer.Option(None, help="Path to pre-exported STEP"),
) -> None:
    """Import a scraper model from SolidWorks export."""
    container = build_container()
    if sldprt is not None:
        result = container.orchestrator.import_sldprt(sldprt)
    elif stl is not None:
        from nutella_scraper.application.dto import ImportRequestDTO

        result = container.orchestrator.import_model(
            ImportRequestDTO(stl_path=str(stl), step_path=str(step) if step else None)
        )
    else:
        raise typer.BadParameter("Provide --sldprt or --stl")
    console.print(f"Imported model: {result.model_id}")


@app.command("simulate")
def simulate(
    model_id: str = typer.Argument(..., help="Canonical model ID"),
    jar_id: str = typer.Option("nutella_400g", help="Jar profile ID"),
) -> None:
    """Run contact simulation for a model."""
    container = build_container()
    result = container.orchestrator.simulate(
        SimulateRequestDTO(model_id=model_id, jar_id=jar_id)
    )
    console.print(f"Coverage score: {result.coverage_score:.2%}")


@app.command("optimize")
def optimize(
    jar_id: str = typer.Option("nutella_400g", help="Jar profile ID"),
    profile: str = typer.Option("fdm_pla", help="Optimization profile"),
) -> None:
    """Start optimization run."""
    container = build_container()
    result = container.orchestrator.start_optimization(
        OptimizationRequestDTO(jar_id=jar_id, optimization_profile=profile)
    )
    console.print(f"Optimization run started: {result.run_id}")


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Start FastAPI server."""
    import uvicorn

    uvicorn.run("nutella_scraper.api.app:create_app", host=host, port=port, factory=True)


if __name__ == "__main__":
    app()
