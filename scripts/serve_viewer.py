#!/usr/bin/env python3
"""Local HTTP server for the visualization viewer — opens index.html automatically."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
import uuid
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from urllib.parse import unquote, urlencode, urlsplit

MAX_STEP_UPLOAD_BYTES = 250 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
LEGACY_ROOT_ASSETS = (
    "side_composite.svg",
    "profile_composite.svg",
    "top_composite.svg",
    "left_composite.svg",
    "right_composite.svg",
    "profile.svg",
    "top.svg",
    "left.svg",
    "right.svg",
    "metadata.json",
)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.viewer_api import (  # noqa: E402
    API_BUILD_SCRAPER,
    API_DEBUG_STEP_FACE_COLORS,
    API_IMPORT_STEP,
    API_INTERIOR_CONTOUR,
    API_RUNTIME,
    API_SIMULATE_CONTACT,
    API_SIMULATIONS,
    build_not_found_payload,
    normalize_api_path,
    resolve_view_dir,
    resolve_pose_view_dir,
    simulation_id_from_path,
    simulation_pose_path,
    simulation_result_id_from_path,
)
from scripts.viewer_handlers import (  # noqa: E402
    read_simulate_contact_request,
    read_viewer_model_request,
)

from nutella_scraper.application.simulation_execution import (  # noqa: E402
    SimulationExecutionManager,
)
from nutella_scraper.engines.visualization.pose_snapshot_store import (  # noqa: E402
    PoseSnapshotStore,
)
from nutella_scraper.engines.visualization.pose_visualization import (  # noqa: E402
    build_pose_visualization_response,
)
from nutella_scraper.engines.visualization.viewer_bridge import (  # noqa: E402
    build_debug_step_face_colors_response,
    build_interior_contour_response,
    build_scraper_visualization_response,
)

_LOG = logging.getLogger("nutella_scraper.serve_viewer")


class ViewerHTTPServer(ThreadingHTTPServer):
    """Local-only server state for atomic model replacement."""

    daemon_threads = True
    block_on_close = True

    output_root: Path
    upload_root: Path
    models_root: Path
    active_view_dir: Path | None
    import_lock: Lock
    dev_mode: bool
    simulation_manager: SimulationExecutionManager


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_error_payload(
    exc: BaseException,
    *,
    request_id: str,
    stage: str,
    dev_mode: bool,
) -> dict[str, Any]:
    """Structured JSON error body for the frontend."""
    payload: dict[str, Any] = {
        "error": str(exc) or exc.__class__.__name__,
        "message": str(exc),
        "exception_type": exc.__class__.__name__,
        "stage": stage,
        "request_id": request_id,
    }
    if dev_mode:
        payload["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    return payload


def _log_step(request_id: str, stage: str, message: str, **fields: object) -> None:
    suffix = " | ".join(f"{key}={value}" for key, value in fields.items())
    detail = f" | {suffix}" if suffix else ""
    _LOG.info("[upload:%s] [%s] %s%s", request_id, stage, message, detail)


class ViewerHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve static files; redirect directory root to index.html."""

    server: ViewerHTTPServer

    def __init__(self, *args: object, directory: str | None = None, **kwargs: object) -> None:
        super().__init__(*args, directory=directory, **kwargs)  # type: ignore[arg-type]

    def do_GET(self) -> None:
        request_path = urlsplit(self.path).path
        if normalize_api_path(request_path) == API_RUNTIME:
            self._send_json(
                200,
                {
                    "dev_mode": self.server.dev_mode,
                    "simulation_api": API_SIMULATE_CONTACT,
                    "status_api": f"{API_SIMULATIONS}/{{simulation_id}}",
                },
            )
            return
        pose_path = simulation_pose_path(request_path)
        if pose_path is not None:
            self._handle_simulation_pose(*pose_path)
            return
        result_simulation_id = simulation_result_id_from_path(request_path)
        if result_simulation_id is not None:
            self._handle_simulation_result(result_simulation_id)
            return
        simulation_id = simulation_id_from_path(request_path)
        if simulation_id is not None:
            self._handle_simulation_status(simulation_id)
            return
        if request_path in ("", "/"):
            self.path = "/index.html"
            request_path = self.path

        translated = Path(self.translate_path(request_path))
        if translated.is_file():
            _LOG.info(
                "[http] GET %s -> %s | sha256=%s",
                request_path,
                translated.resolve(),
                _file_sha256(translated),
            )
        else:
            _LOG.info("[http] GET %s -> NOT FOUND", request_path)
        return super().do_GET()

    def do_DELETE(self) -> None:
        simulation_id = simulation_id_from_path(self.path)
        if simulation_id is None:
            self._send_json(
                404,
                {
                    "error": "Endpoint introuvable",
                    "message": (
                        f"Utilisez DELETE {API_SIMULATIONS}/{{simulation_id}} "
                        "pour annuler une simulation"
                    ),
                },
            )
            return
        try:
            payload = self.server.simulation_manager.cancel(simulation_id)
            self._send_json(200, payload)
        except KeyError as exc:
            self._send_error_json(
                404,
                exc,
                request_id=simulation_id,
                stage="cancel_simulation",
            )

    def do_POST(self) -> None:
        request_id = str(uuid.uuid4())
        upload_dir: Path | None = None
        stage = "http_request_received"

        try:
            path = normalize_api_path(self.path)
            if path == API_SIMULATE_CONTACT:
                self._handle_simulate_contact(request_id)
                return
            if path == API_BUILD_SCRAPER:
                self._handle_build_scraper(request_id)
                return
            if path == API_INTERIOR_CONTOUR:
                self._handle_interior_contour(request_id)
                return
            if path == API_DEBUG_STEP_FACE_COLORS:
                self._handle_debug_step_face_colors(request_id)
                return
            if path != API_IMPORT_STEP:
                self._send_json(
                    404,
                    build_not_found_payload(
                        path=path,
                        method="POST",
                        request_id=request_id,
                    ),
                )
                return

            client = self.client_address[0]
            filename = Path(unquote(self.headers.get("X-Filename", ""))).name
            content_length = self._parse_content_length()
            _log_step(
                request_id,
                stage,
                "POST /api/import-step",
                client=client,
                filename=filename,
                content_length=content_length,
            )

            if Path(filename).suffix.lower() not in (".step", ".stp"):
                self._send_error_json(
                    400,
                    ValueError("Sélectionnez un fichier .step ou .stp"),
                    request_id=request_id,
                    stage="validate_filename",
                )
                return
            if content_length <= 0:
                self._send_error_json(
                    400,
                    ValueError("Fichier vide"),
                    request_id=request_id,
                    stage="validate_size",
                )
                return
            if content_length > MAX_STEP_UPLOAD_BYTES:
                self._send_error_json(
                    413,
                    ValueError("Fichier STEP trop volumineux (maximum 250 Mio)"),
                    request_id=request_id,
                    stage="validate_size",
                )
                return

            stage = "temp_save_start"
            upload_dir = self.server.upload_root / request_id
            upload_dir.mkdir(parents=True, exist_ok=False)
            upload_path = upload_dir / filename
            _log_step(
                request_id,
                stage,
                "sauvegarde temporaire",
                path=str(upload_path),
            )

            stage = "body_read"
            received = self._stream_body_to_file(upload_path, content_length, request_id)
            _log_step(
                request_id,
                "temp_saved",
                "fichier temporaire écrit",
                path=str(upload_path),
                bytes=received,
                sha256=_file_sha256(upload_path),
            )

            stage = "import_pipeline"
            _log_step(request_id, stage, "appel ImportPipeline.import_step()")
            with self.server.import_lock:
                from scripts.demo_import import import_model_for_viewer

                previous_view_dir = self.server.active_view_dir
                new_view_dir = import_model_for_viewer(
                    upload_path,
                    self.server.output_root,
                    request_id=request_id,
                )
                metadata_path = new_view_dir / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.server.active_view_dir = new_view_dir
                _remove_previous_view(previous_view_dir, self.server.output_root)

            stage = "http_response"
            response_payload = {
                "model_id": metadata["model_id"],
                "canonical_mesh_sha256": metadata["canonical_mesh_sha256"],
                "metadata_url": f"/{new_view_dir.name}/metadata.json",
                "base_url": f"/{new_view_dir.name}/",
                "request_id": request_id,
            }
            _log_step(
                request_id,
                stage,
                "envoi réponse HTTP 200",
                model_id=metadata["model_id"],
            )
            self._send_json(200, response_payload)
        except Exception as exc:
            _LOG.exception(
                "[upload:%s] [%s] échec: %s: %s",
                request_id,
                stage,
                exc.__class__.__name__,
                exc,
            )
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            status = 422 if isinstance(exc, ValueError) else 500
            self._send_error_json(status, exc, request_id=request_id, stage=stage)

    def _handle_simulate_contact(self, request_id: str) -> None:
        stage = "start_simulation"
        try:
            content_length = self._parse_content_length()
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            request = read_simulate_contact_request(raw_body)
            view_dir = resolve_view_dir(
                output_root=self.server.output_root,
                active_view_dir=self.server.active_view_dir,
                model_id=request.model_id,
            )
            self.server.active_view_dir = view_dir

            _log_step(
                request_id,
                stage,
                "POST /api/simulate-contact",
                view_dir=str(view_dir),
                model_id=request.model_id or view_dir.name,
            )
            payload = self.server.simulation_manager.start(
                view_dir=view_dir,
                models_root=self.server.models_root,
                request_id=request_id,
            )
            simulation_id = payload["simulation_id"]
            payload["status_url"] = f"{API_SIMULATIONS}/{simulation_id}"
            payload["cancel_url"] = payload["status_url"]
            self._send_json(202, payload)
        except Exception as exc:
            _LOG.exception(
                "[simulate:%s] [%s] échec: %s: %s",
                request_id,
                stage,
                exc.__class__.__name__,
                exc,
            )
            status = 404 if isinstance(exc, FileNotFoundError) else 500
            if isinstance(exc, ValueError):
                status = 422
            self._send_error_json(status, exc, request_id=request_id, stage=stage)

    def _handle_build_scraper(self, request_id: str) -> None:
        stage = "build_scraper"
        try:
            content_length = self._parse_content_length()
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            request = read_viewer_model_request(raw_body)
            view_dir = resolve_view_dir(
                output_root=self.server.output_root,
                active_view_dir=self.server.active_view_dir,
                model_id=request.model_id,
            )
            _log_step(
                request_id,
                stage,
                "POST /api/build-scraper",
                view_dir=str(view_dir),
                model_id=request.model_id or view_dir.name,
            )
            payload = build_scraper_visualization_response(
                view_dir=view_dir,
                models_root=self.server.models_root,
            )
            self._send_json(200, payload)
        except Exception as exc:
            _LOG.exception(
                "[build-scraper:%s] [%s] échec: %s: %s",
                request_id,
                stage,
                exc.__class__.__name__,
                exc,
            )
            status = 404 if isinstance(exc, FileNotFoundError) else 500
            if isinstance(exc, ValueError):
                status = 422
            self._send_error_json(status, exc, request_id=request_id, stage=stage)

    def _handle_interior_contour(self, request_id: str) -> None:
        stage = "interior_contour"
        try:
            content_length = self._parse_content_length()
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            request = read_viewer_model_request(raw_body)
            view_dir = resolve_view_dir(
                output_root=self.server.output_root,
                active_view_dir=self.server.active_view_dir,
                model_id=request.model_id,
            )
            _log_step(
                request_id,
                stage,
                "POST /api/interior-contour",
                view_dir=str(view_dir),
                model_id=request.model_id or view_dir.name,
            )
            payload = build_interior_contour_response(
                view_dir=view_dir,
                models_root=self.server.models_root,
            )
            self._send_json(200, payload)
        except Exception as exc:
            _LOG.exception(
                "[interior-contour:%s] [%s] échec: %s: %s",
                request_id,
                stage,
                exc.__class__.__name__,
                exc,
            )
            status = 404 if isinstance(exc, FileNotFoundError) else 500
            if isinstance(exc, ValueError):
                status = 422
            self._send_error_json(status, exc, request_id=request_id, stage=stage)

    def _handle_debug_step_face_colors(self, request_id: str) -> None:
        stage = "debug_step_face_colors"
        try:
            content_length = self._parse_content_length()
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            request = read_viewer_model_request(raw_body)
            view_dir = resolve_view_dir(
                output_root=self.server.output_root,
                active_view_dir=self.server.active_view_dir,
                model_id=request.model_id,
            )
            _log_step(
                request_id,
                stage,
                "POST /api/debug-step-face-colors",
                view_dir=str(view_dir),
                model_id=request.model_id or view_dir.name,
            )
            payload = build_debug_step_face_colors_response(
                view_dir=view_dir,
                models_root=self.server.models_root,
            )
            self._send_json(200, payload)
        except Exception as exc:
            _LOG.exception(
                "[debug-step-face-colors:%s] [%s] échec: %s: %s",
                request_id,
                stage,
                exc.__class__.__name__,
                exc,
            )
            status = 404 if isinstance(exc, FileNotFoundError) else 500
            if isinstance(exc, ValueError):
                status = 422
            self._send_error_json(status, exc, request_id=request_id, stage=stage)

    def _handle_simulation_status(self, simulation_id: str) -> None:
        try:
            payload = self.server.simulation_manager.status(
                simulation_id,
                include_result=False,
            )
            if payload["state"] == "completed":
                payload["result_url"] = (
                    f"{API_SIMULATIONS}/{simulation_id}/result"
                )
            error = payload.get("error")
            if error and not self.server.dev_mode:
                error.pop("traceback", None)
            self._send_json(200, payload)
        except KeyError as exc:
            self._send_error_json(
                404,
                exc,
                request_id=simulation_id,
                stage="simulation_status",
            )

    def _handle_simulation_result(self, simulation_id: str) -> None:
        try:
            result_path = self.server.simulation_manager.result_path(simulation_id)
            self._send_json_file(result_path, simulation_id=simulation_id)
        except KeyError as exc:
            self._send_error_json(
                404,
                exc,
                request_id=simulation_id,
                stage="simulation_result",
            )
        except RuntimeError as exc:
            self._send_error_json(
                409,
                exc,
                request_id=simulation_id,
                stage="simulation_result",
            )

    def _handle_simulation_pose(
        self,
        simulation_id: str,
        pose_index: int | None,
    ) -> None:
        try:
            snapshot_dir = self.server.simulation_manager.pose_snapshot_dir(simulation_id)
            store = PoseSnapshotStore(snapshot_dir)
            manifest = store.manifest()
            if pose_index is None:
                manifest["pose_url_template"] = (
                    f"{API_SIMULATIONS}/{simulation_id}/poses/{{index}}"
                )
                self._send_json(200, manifest)
                return
            if pose_index < 0 or pose_index >= int(manifest["pose_count"]):
                raise ValueError(
                    f"Pose {pose_index} hors limites "
                    f"(0..{int(manifest['pose_count']) - 1})"
                )
            cached = snapshot_dir / f"pose-{pose_index:04d}.json"
            if not cached.exists():
                started = time.perf_counter()
                view_dir = resolve_pose_view_dir(
                    output_root=self.server.output_root,
                    manifest=manifest,
                    active_view_dir=self.server.active_view_dir,
                )
                payload = build_pose_visualization_response(
                    pose_snapshot_dir=snapshot_dir,
                    view_dir=view_dir,
                    models_root=self.server.models_root,
                    pose_index=pose_index,
                )
                temporary = cached.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
                temporary.replace(cached)
                _LOG.info(
                    "[simulation:%s] pose overlay generated | pose=%s | "
                    "duration_ms=%.1f | bytes=%s",
                    simulation_id,
                    pose_index,
                    (time.perf_counter() - started) * 1000.0,
                    cached.stat().st_size,
                )
            self._send_json_file(cached, simulation_id=simulation_id)
        except KeyError as exc:
            self._send_error_json(
                404,
                exc,
                request_id=simulation_id,
                stage="simulation_pose",
            )
        except FileNotFoundError as exc:
            self._send_error_json(
                404,
                exc,
                request_id=simulation_id,
                stage="simulation_pose",
            )
        except (RuntimeError, ValueError) as exc:
            self._send_error_json(
                409,
                exc,
                request_id=simulation_id,
                stage="simulation_pose",
            )

    def _parse_content_length(self) -> int:
        try:
            return int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return 0

    def _stream_body_to_file(
        self,
        destination: Path,
        content_length: int,
        request_id: str,
    ) -> int:
        received = 0
        remaining = content_length
        with destination.open("wb") as handle:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, UPLOAD_CHUNK_BYTES))
                if not chunk:
                    raise ValueError(
                        f"Upload STEP incomplet ({received}/{content_length} bytes reçus)"
                    )
                handle.write(chunk)
                received += len(chunk)
                remaining -= len(chunk)
                if received == content_length or received % (5 * UPLOAD_CHUNK_BYTES) == 0:
                    _log_step(
                        request_id,
                        "body_read",
                        "réception en cours",
                        received=received,
                        expected=content_length,
                    )
        return received

    def _send_error_json(
        self,
        status: int,
        exc: BaseException,
        *,
        request_id: str,
        stage: str,
    ) -> None:
        payload = build_error_payload(
            exc,
            request_id=request_id,
            stage=stage,
            dev_mode=self.server.dev_mode,
        )
        self._send_json(status, payload)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        serialization_started = time.perf_counter()
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        serialization_ms = (time.perf_counter() - serialization_started) * 1000.0
        send_started = time.perf_counter()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(content)
            self.wfile.flush()
            _LOG.info(
                "[http] JSON response | status=%s | bytes=%s | "
                "serialization_ms=%.1f | send_ms=%.1f",
                status,
                len(content),
                serialization_ms,
                (time.perf_counter() - send_started) * 1000.0,
            )
        except (BrokenPipeError, ConnectionResetError) as exc:
            _LOG.warning(
                "[http] client disconnected before JSON response (%s): %s",
                exc.__class__.__name__,
                payload.get("request_id", "?"),
            )

    def _send_json_file(self, path: Path, *, simulation_id: str) -> None:
        size = path.stat().st_size
        send_started = time.perf_counter()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(size))
            self.send_header("Connection", "close")
            self.end_headers()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)
            self.wfile.flush()
            _LOG.info(
                "[simulation:%s] overlay transfer | bytes=%s | send_ms=%.1f",
                simulation_id,
                size,
                (time.perf_counter() - send_started) * 1000.0,
            )
        except (BrokenPipeError, ConnectionResetError) as exc:
            _LOG.warning(
                "[simulation:%s] overlay transfer interrupted | %s",
                simulation_id,
                exc.__class__.__name__,
            )

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def _remove_previous_view(previous: Path | None, output_root: Path) -> None:
    if previous is None or not previous.exists():
        return
    previous = previous.resolve()
    if previous.parent != output_root.resolve():
        raise ValueError(f"Refus de supprimer un dossier hors du viewer : {previous}")
    shutil.rmtree(previous)
    _LOG.info("[viewer] previous projection removed: %s", previous)


def _remove_stale_view_directories(output_root: Path, keep: Path | None) -> None:
    keep_resolved = keep.resolve() if keep is not None else None
    for candidate in output_root.iterdir():
        if not candidate.is_dir() or candidate.resolve() == keep_resolved:
            continue
        metadata_path = candidate / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if str(metadata.get("model_id")) != candidate.name:
            continue
        shutil.rmtree(candidate)
        _LOG.info("[viewer] removed stale projection: %s", candidate)


def _prepare_viewer_root(directory: Path) -> tuple[Path, Path | None]:
    directory = directory.resolve()
    metadata_path = directory / "metadata.json"
    is_import_directory = False
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            is_import_directory = str(metadata.get("model_id")) == directory.name
        except (json.JSONDecodeError, OSError):
            is_import_directory = False

    if is_import_directory:
        output_root = directory.parent
        initial_view_dir: Path | None = directory
    else:
        output_root = directory
        initial_view_dir = None

    output_root.mkdir(parents=True, exist_ok=True)
    for filename in LEGACY_ROOT_ASSETS:
        legacy_path = output_root / filename
        if legacy_path.exists():
            legacy_path.unlink()
            _LOG.info("[viewer] removed legacy asset: %s", legacy_path)
    _remove_stale_view_directories(output_root, initial_view_dir)

    template = Path(__file__).resolve().parent / "templates" / "demo_viewer.html"
    if not template.exists():
        raise FileNotFoundError(f"Template viewer introuvable : {template}")
    shutil.copy(template, output_root / "index.html")
    return output_root, initial_view_dir


def serve(
    directory: Path,
    port: int,
    *,
    open_browser: bool = True,
    dev_mode: bool | None = None,
) -> None:
    if dev_mode is None:
        dev_mode = os.environ.get("NUTELLA_VIEWER_DEV", "").lower() in ("1", "true", "yes")

    output_root, initial_view_dir = _prepare_viewer_root(directory)
    upload_root = output_root.parent / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)

    handler = partial(ViewerHTTPRequestHandler, directory=str(output_root))
    server = ViewerHTTPServer(("127.0.0.1", port), handler)
    server.output_root = output_root
    server.upload_root = upload_root
    server.models_root = _ROOT / "output" / "models"
    server.active_view_dir = initial_view_dir
    server.import_lock = Lock()
    server.dev_mode = dev_mode
    server.simulation_manager = SimulationExecutionManager(
        output_root.parent / "simulations"
    )

    query_values: dict[str, str] = {}
    if initial_view_dir is not None:
        metadata_path = initial_view_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        query_values = {
            "model_id": str(metadata["model_id"]),
            "mesh_sha256": str(metadata["canonical_mesh_sha256"]),
        }
        _LOG.info("[viewer] metadata=%s | sha256=%s", metadata_path, _file_sha256(metadata_path))
        for plane, view in metadata["displayed_views"].items():
            view_path = initial_view_dir / view["filename"]
            _LOG.info(
                "[viewer:%s] file=%s | expected_sha256=%s | actual_sha256=%s",
                plane,
                view_path,
                view["sha256"],
                _file_sha256(view_path),
            )

    query = f"?{urlencode(query_values)}" if query_values else ""
    url = f"http://127.0.0.1:{port}/index.html{query}"
    _LOG.info("[viewer] root=%s", output_root)
    _LOG.info("[viewer] uploads=%s", upload_root)
    _LOG.info("[viewer] dev_mode=%s", dev_mode)
    _LOG.info("[viewer] url=%s", url)

    def _wait_for_stop() -> None:
        try:
            input("Viewer actif — Entree pour arreter...\n")
        except (EOFError, KeyboardInterrupt):
            return
        server.shutdown()

    Thread(target=_wait_for_stop, daemon=True).start()

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.simulation_manager.shutdown()
        server.shutdown()
        server.server_close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Serveur local viewer CAD")
    parser.add_argument(
        "--dir",
        type=Path,
        default=_ROOT / "output" / "views",
        help="Racine des vues ou dossier immuable d'un import",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Inclure la stack trace Python dans les réponses d'erreur JSON",
    )
    args = parser.parse_args()
    serve(args.dir, args.port, open_browser=not args.no_browser, dev_mode=args.dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
