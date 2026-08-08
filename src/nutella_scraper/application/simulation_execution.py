"""Background execution and lifecycle management for viewer simulations."""

from __future__ import annotations

import json
import logging
import multiprocessing
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("nutella_scraper.simulation_execution")

PHASE_LABELS: dict[str, str] = {
    "preparation": "Préparation",
    "model_loading": "Chargement du modèle",
    "interior_surface_calculation": "Surface intérieure",
    "envelope_calculation": "Calcul de l'enveloppe",
    "contact_calculation": "Calcul des contacts",
    "distance_calculation": "Calcul des distances",
    "metrics_calculation": "Calcul des métriques",
    "pose_capture": "Capture des poses 3D",
    "overlay_generation": "Génération des overlays",
    "serialization": "Sérialisation du résultat",
    "ui_refresh": "Rafraîchissement de l'interface",
}

WorkerTarget = Callable[[str, str, str, str, Any], None]


@dataclass
class _SimulationJob:
    simulation_id: str
    request_id: str
    process: multiprocessing.Process
    events: Any
    result_path: Path
    view_dir: Path
    started_at: float
    state: str = "running"
    phase: str = "preparation"
    detail: str = ""
    progress_percent: float | None = 0.0
    last_activity_at: float = field(default_factory=time.monotonic)
    last_progress_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    profile_ms: dict[str, float] = field(default_factory=dict)
    error: dict[str, Any] | None = None


class SimulationExecutionManager:
    """Own simulation subprocesses and expose polling/cancellation state."""

    def __init__(
        self,
        result_root: Path,
        *,
        heartbeat_timeout_s: float = 15.0,
        cancel_timeout_s: float = 2.0,
        worker_target: WorkerTarget | None = None,
        multiprocessing_context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        self._result_root = result_root.resolve()
        self._result_root.mkdir(parents=True, exist_ok=True)
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._cancel_timeout_s = cancel_timeout_s
        self._worker_target = worker_target or _run_simulation_process
        self._context = multiprocessing_context or multiprocessing.get_context("spawn")
        self._jobs: dict[str, _SimulationJob] = {}
        self._lock = threading.RLock()
        self._closed = False

    def start(self, *, view_dir: Path, models_root: Path, request_id: str) -> dict[str, Any]:
        """Start one isolated simulation process and return its initial state."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Le gestionnaire de simulations est arrêté")
            simulation_id = str(uuid.uuid4())
            result_path = self._result_root / f"{simulation_id}.json"
            events = self._context.Queue()
            process = self._context.Process(
                target=self._worker_target,
                args=(
                    simulation_id,
                    str(view_dir.resolve()),
                    str(models_root.resolve()),
                    str(result_path),
                    events,
                ),
                name=f"nutella-simulation-{simulation_id[:8]}",
            )
            process.start()
            job = _SimulationJob(
                simulation_id=simulation_id,
                request_id=request_id,
                process=process,
                events=events,
                result_path=result_path,
                view_dir=view_dir.resolve(),
                started_at=time.monotonic(),
            )
            self._jobs[simulation_id] = job
            _LOG.info(
                "[simulation:%s] started | pid=%s | model=%s",
                simulation_id,
                process.pid,
                view_dir.name,
            )
            return self._snapshot(job, include_result=False)

    def status(self, simulation_id: str, *, include_result: bool = True) -> dict[str, Any]:
        """Refresh and return the current job state."""
        with self._lock:
            job = self._get_job(simulation_id)
            self._refresh(job)
            return self._snapshot(job, include_result=include_result)

    def result_path(self, simulation_id: str) -> Path:
        """Return the immutable serialized result for a completed job."""
        with self._lock:
            job = self._get_job(simulation_id)
            self._refresh(job)
            if job.state != "completed" or not job.result_path.exists():
                raise RuntimeError(
                    f"Résultat indisponible pour la simulation {simulation_id} "
                    f"(état : {job.state})"
                )
            return job.result_path

    def pose_snapshot_dir(self, simulation_id: str) -> Path:
        """Return the exact per-pose snapshot directory for a completed job."""
        with self._lock:
            job = self._get_job(simulation_id)
            self._refresh(job)
            directory = job.result_path.with_suffix(".poses")
            if job.state != "completed" or not (directory / "manifest.json").exists():
                raise RuntimeError(
                    f"Poses indisponibles pour la simulation {simulation_id} "
                    f"(état : {job.state})"
                )
            return directory

    def view_dir(self, simulation_id: str) -> Path:
        """Return the viewer directory used when the simulation was started."""
        with self._lock:
            job = self._get_job(simulation_id)
            return job.view_dir

    def cancel(self, simulation_id: str) -> dict[str, Any]:
        """Stop a running subprocess and synchronously reap it."""
        with self._lock:
            job = self._get_job(simulation_id)
            self._refresh(job)
            if job.state in {"completed", "failed", "cancelled"}:
                return self._snapshot(job, include_result=False)

            if job.process.is_alive():
                job.process.terminate()
                job.process.join(self._cancel_timeout_s)
                if job.process.is_alive():
                    job.process.kill()
                    job.process.join(self._cancel_timeout_s)
            job.state = "cancelled"
            job.detail = "Simulation annulée"
            job.finished_at = time.monotonic()
            job.progress_percent = None
            self._close_queue(job)
            _LOG.info(
                "[simulation:%s] cancelled | pid=%s | elapsed_ms=%.1f",
                simulation_id,
                job.process.pid,
                self._elapsed_ms(job),
            )
            return self._snapshot(job, include_result=False)

    def shutdown(self) -> None:
        """Cancel and reap every active simulation."""
        with self._lock:
            if self._closed:
                return
            for simulation_id in list(self._jobs):
                job = self._jobs[simulation_id]
                self._refresh(job)
                if job.process.is_alive() and job.state in {"running", "blocked"}:
                    self.cancel(simulation_id)
                else:
                    job.process.join(timeout=self._cancel_timeout_s)
                    if job.process.is_alive():
                        job.process.terminate()
                        job.process.join(timeout=self._cancel_timeout_s)
                    self._close_queue(job)
            self._closed = True

    @property
    def active_worker_pids(self) -> tuple[int, ...]:
        """Return live worker PIDs, primarily for lifecycle diagnostics."""
        with self._lock:
            for job in self._jobs.values():
                self._refresh(job)
            return tuple(
                job.process.pid
                for job in self._jobs.values()
                if job.process.pid is not None and job.process.is_alive()
            )

    def _get_job(self, simulation_id: str) -> _SimulationJob:
        try:
            return self._jobs[simulation_id]
        except KeyError as exc:
            raise KeyError(f"Simulation introuvable : {simulation_id}") from exc

    def _refresh(self, job: _SimulationJob) -> None:
        if job.state in {"completed", "failed", "cancelled"}:
            if not job.process.is_alive():
                job.process.join(timeout=0.2)
                self._close_queue(job)
            return

        while True:
            try:
                event = job.events.get_nowait()
            except queue.Empty:
                break
            self._apply_event(job, event)

        if job.state in {"completed", "failed"}:
            job.process.join(timeout=0.2)
            return

        if not job.process.is_alive():
            job.process.join(timeout=0.2)
            # A completed worker can exit just before its final queue event is visible.
            if job.result_path.exists() and job.process.exitcode == 0:
                job.state = "completed"
                job.detail = "Simulation terminée"
            else:
                job.state = "failed"
                job.error = {
                    "exception_type": "SimulationProcessError",
                    "message": (
                        "Le processus de simulation s'est arrêté sans résultat "
                        f"(code {job.process.exitcode})"
                    ),
                }
            job.finished_at = time.monotonic()
            return

        inactive_s = time.monotonic() - job.last_progress_at
        if inactive_s > self._heartbeat_timeout_s:
            if job.state != "blocked":
                _LOG.error(
                    "[simulation:%s] blocked watchdog | phase=%s | inactive_s=%.1f",
                    job.simulation_id,
                    job.phase,
                    inactive_s,
                )
            job.state = "blocked"
            job.detail = (
                f"Aucune progression détectée depuis {inactive_s:.1f} s "
                f"(phase : {PHASE_LABELS.get(job.phase, job.phase)})"
            )

    def _apply_event(self, job: _SimulationJob, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        job.last_activity_at = time.monotonic()

        if event_type == "heartbeat":
            return
        if job.state == "blocked":
            job.state = "running"
        if event_type == "progress":
            job.last_progress_at = job.last_activity_at
            job.phase = str(event.get("phase", job.phase))
            job.detail = str(event.get("detail", ""))
            progress = event.get("progress_percent")
            job.progress_percent = float(progress) if progress is not None else None
            timings = event.get("profile_ms")
            if isinstance(timings, dict):
                job.profile_ms = {str(key): float(value) for key, value in timings.items()}
            return
        if event_type == "completed":
            job.state = "completed"
            job.phase = "ui_refresh"
            job.detail = "Calcul terminé, rafraîchissement de l'interface"
            job.progress_percent = 100.0
            job.profile_ms = {
                str(key): float(value)
                for key, value in dict(event.get("profile_ms") or {}).items()
            }
            job.finished_at = time.monotonic()
            self._log_profile(job)
            return
        if event_type == "failed":
            job.state = "failed"
            job.detail = "La simulation a échoué"
            job.error = dict(event.get("error") or {})
            job.profile_ms = {
                str(key): float(value)
                for key, value in dict(event.get("profile_ms") or {}).items()
            }
            job.finished_at = time.monotonic()
            _LOG.error(
                "[simulation:%s] failed | %s: %s\n%s",
                job.simulation_id,
                job.error.get("exception_type", "Exception"),
                job.error.get("message", ""),
                job.error.get("traceback", ""),
            )

    def _snapshot(self, job: _SimulationJob, *, include_result: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "simulation_id": job.simulation_id,
            "request_id": job.request_id,
            "state": job.state,
            "phase": job.phase,
            "phase_label": PHASE_LABELS.get(job.phase, job.phase),
            "detail": job.detail,
            "progress_percent": job.progress_percent,
            "elapsed_ms": round(self._elapsed_ms(job), 1),
            "worker_pid": job.process.pid,
            "profile_ms": {key: round(value, 1) for key, value in job.profile_ms.items()},
        }
        if job.error is not None:
            payload["error"] = dict(job.error)
        if include_result and job.state == "completed":
            payload["result"] = json.loads(job.result_path.read_text(encoding="utf-8"))
        return payload

    @staticmethod
    def _elapsed_ms(job: _SimulationJob) -> float:
        end = job.finished_at if job.finished_at is not None else time.monotonic()
        return (end - job.started_at) * 1000.0

    @staticmethod
    def _close_queue(job: _SimulationJob) -> None:
        try:
            job.events.close()
            job.events.join_thread()
        except (OSError, ValueError):
            pass

    @staticmethod
    def _log_profile(job: _SimulationJob) -> None:
        details = " | ".join(
            f"{PHASE_LABELS.get(phase, phase)}={duration:.1f}ms"
            for phase, duration in job.profile_ms.items()
        )
        _LOG.info("[simulation:%s] profile | %s", job.simulation_id, details)


def _run_simulation_process(
    simulation_id: str,
    view_dir: str,
    models_root: str,
    result_path: str,
    events: Any,
) -> None:
    """Subprocess entry point. Kept top-level for Windows spawn."""
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(1.0):
            events.put({"type": "heartbeat"})

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"simulation-heartbeat-{simulation_id[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()
    profile_ms: dict[str, float] = {}

    def report(
        phase: str,
        detail: str,
        progress_percent: float | None,
        timings: dict[str, float],
    ) -> None:
        profile_ms.clear()
        profile_ms.update(timings)
        events.put(
            {
                "type": "progress",
                "phase": phase,
                "detail": detail,
                "progress_percent": progress_percent,
                "profile_ms": timings,
            }
        )

    try:
        from nutella_scraper.engines.visualization.viewer_bridge import (
            build_contact_visualization_response,
        )

        payload = build_contact_visualization_response(
            view_dir=Path(view_dir),
            models_root=Path(models_root),
            progress_callback=report,
            pose_snapshot_dir=Path(result_path).with_suffix(".poses"),
        )
        destination = Path(result_path)
        temporary = destination.with_suffix(".tmp")
        events.put(
            {
                "type": "progress",
                "phase": "overlay_generation",
                "detail": "Sérialisation JSON des overlays",
                "progress_percent": 97.0,
                "profile_ms": profile_ms,
            }
        )
        serialization_started = time.perf_counter()
        serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        profile_ms["serialization"] = (
            time.perf_counter() - serialization_started
        ) * 1000.0
        temporary.write_bytes(serialized)
        temporary.replace(destination)
        events.put({"type": "completed", "profile_ms": profile_ms})
    except BaseException as exc:
        events.put(
            {
                "type": "failed",
                "profile_ms": profile_ms,
                "error": {
                    "exception_type": exc.__class__.__name__,
                    "message": str(exc) or exc.__class__.__name__,
                    "traceback": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                },
            }
        )
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.5)
