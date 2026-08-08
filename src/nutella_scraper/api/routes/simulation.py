"""Contact simulation routes."""

from __future__ import annotations

from fastapi import APIRouter, Request, WebSocket

from nutella_scraper.application.dto import SimulateRequestDTO, SimulateResponseDTO

router = APIRouter()


@router.post("/contact", response_model=SimulateResponseDTO)
def simulate_contact(request: Request, body: SimulateRequestDTO) -> SimulateResponseDTO:
    orchestrator = request.app.state.container.orchestrator
    return orchestrator.simulate(body)


@router.websocket("/ws")
async def simulate_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "WebSocket skeleton ready"})
    await websocket.close()
