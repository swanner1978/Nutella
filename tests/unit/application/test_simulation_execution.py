"""Tests for isolated simulation execution and lifecycle management."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from nutella_scraper.application.simulation_execution import SimulationExecutionManager


def _completed_worker(
    simulation_id: str,
    view_dir: str,
    models_root: str,
    result_path: str,
    events: Any,
) -> None:
    del simulation_id, view_dir, models_root
    profile = {
        "model_loading": 2.0,
        "scraper_generation": 3.0,
        "contact_calculation": 5.0,
        "distance_calculation": 7.0,
        "metrics_calculation": 1.0,
        "overlay_generation": 4.0,
        "ui_refresh": 0.0,
        "total": 22.0,
    }
    events.put(
        {
            "type": "progress",
            "phase": "distance_calculation",
            "detail": "pose 1/2",
            "progress_percent": 50.0,
            "profile_ms": profile,
        }
    )
    Path(result_path).write_text(json.dumps({"ok": True}), encoding="utf-8")
    events.put({"type": "completed", "profile_ms": profile})


def _long_worker(
    simulation_id: str,
    view_dir: str,
    models_root: str,
    result_path: str,
    events: Any,
) -> None:
    del simulation_id, view_dir, models_root, result_path
    while True:
        events.put({"type": "heartbeat"})
        time.sleep(0.02)


def _blocked_worker(
    simulation_id: str,
    view_dir: str,
    models_root: str,
    result_path: str,
    events: Any,
) -> None:
    del simulation_id, view_dir, models_root, result_path, events
    time.sleep(5.0)


def _wait_terminal(
    manager: SimulationExecutionManager,
    simulation_id: str,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = manager.status(simulation_id)
        if state["state"] in {"completed", "failed", "cancelled"}:
            return state
        time.sleep(0.02)
    raise AssertionError("simulation did not terminate")


def test_manager_returns_result_and_performance_profile(tmp_path: Path) -> None:
    manager = SimulationExecutionManager(
        tmp_path / "results",
        worker_target=_completed_worker,
    )
    try:
        started = manager.start(
            view_dir=tmp_path,
            models_root=tmp_path,
            request_id="request-1",
        )
        completed = _wait_terminal(manager, started["simulation_id"])
    finally:
        manager.shutdown()

    assert completed["state"] == "completed"
    assert completed["result"] == {"ok": True}
    assert completed["profile_ms"]["distance_calculation"] == 7.0
    assert completed["progress_percent"] == 100.0


def test_cancel_reaps_worker_process(tmp_path: Path) -> None:
    manager = SimulationExecutionManager(
        tmp_path / "results",
        worker_target=_long_worker,
        cancel_timeout_s=1.0,
    )
    started = manager.start(
        view_dir=tmp_path,
        models_root=tmp_path,
        request_id="request-2",
    )

    cancelled = manager.cancel(started["simulation_id"])

    assert cancelled["state"] == "cancelled"
    assert manager.active_worker_pids == ()
    manager.shutdown()


def test_watchdog_detects_worker_without_activity(tmp_path: Path) -> None:
    manager = SimulationExecutionManager(
        tmp_path / "results",
        worker_target=_blocked_worker,
        heartbeat_timeout_s=0.05,
    )
    try:
        started = manager.start(
            view_dir=tmp_path,
            models_root=tmp_path,
            request_id="request-3",
        )
        time.sleep(0.12)
        blocked = manager.status(started["simulation_id"])
        assert blocked["state"] == "blocked"
        assert "Aucune progression" in blocked["detail"]
    finally:
        manager.shutdown()

    assert manager.active_worker_pids == ()


def test_shutdown_cancels_all_running_workers(tmp_path: Path) -> None:
    manager = SimulationExecutionManager(
        tmp_path / "results",
        worker_target=_long_worker,
    )
    manager.start(view_dir=tmp_path, models_root=tmp_path, request_id="request-4")
    manager.start(view_dir=tmp_path, models_root=tmp_path, request_id="request-5")

    manager.shutdown()

    assert manager.active_worker_pids == ()
