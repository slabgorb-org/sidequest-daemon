"""sidequest-renderer daemon — persistent Z-Image renderer on Unix domain socket.

Hosts the Z-Image worker in a single process with the model pre-loaded.
Serves render requests over a Unix domain socket, routing by tier.
Stays warm between sessions.

This module is socket lifecycle + routing only (story 101-7). The worker
pool, embed worker, per-queue heartbeats, and the image render pipeline
live in sibling modules:

    sidequest_daemon.media.tiers          — tier routing constants
    sidequest_daemon.media.embed_worker   — EmbedWorker
    sidequest_daemon.media.worker_pool    — WorkerPool, WorkerState, heartbeats
    sidequest_daemon.media.render_service — RenderService (image compose+render)

Those names are re-exported here for back-compat: the server's
``DaemonStateMirror`` and the daemon's tests import ``WorkerPool``,
``EmbedWorker``, ``WorkerState``, ``dispatch_request``, ``IMAGE_TIERS``,
etc. from ``sidequest_daemon.media.daemon``.

Usage:
    sidequest-renderer                          # start daemon (loads Z-Image)
    sidequest-renderer --warmup=image           # start + load Z-Image only
    sidequest-renderer --no-warmup              # start without loading models (testing)
    sidequest-renderer --shutdown               # send shutdown to running daemon
    sidequest-renderer --status                 # check daemon status
    sidequest-renderer --genre-packs /path      # set genre packs directory
    sidequest-renderer --output-dir /path       # set output directory
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sidequest_daemon.media.music_pipeline import MusicPipeline

from opentelemetry import trace

# --- Re-exported symbols (story 101-7 extraction) -------------------------
# Existing consumers (server DaemonStateMirror, ~12 daemon test modules)
# import these from this module path. Keep them importable here.
from sidequest_daemon.media.embed_worker import EmbedWorker
from sidequest_daemon.media.render_service import RenderError, RenderService
from sidequest_daemon.media.tiers import (
    EMBED_TIERS,
    IMAGE_TIERS,
    MUSIC_TIERS,
    WARMUP_TARGETS,
    _validate_warmup_target,
)
from sidequest_daemon.media.worker_pool import (  # noqa: F401  (back-compat re-exports)
    _IN_FLIGHT_COUNTS,
    WorkerPool,
    WorkerState,
    _make_heartbeat,
    _write_heartbeat,
)

__all__ = [
    "EmbedWorker",
    "RenderError",
    "RenderService",
    "WorkerPool",
    "WorkerState",
    "EMBED_TIERS",
    "IMAGE_TIERS",
    "MUSIC_TIERS",
    "WARMUP_TARGETS",
    "dispatch_request",
    "send_shutdown",
    "send_status",
    "main",
    # Back-compat re-exports — test_78_3 guards _make_heartbeat against
    # over-deletion; the server's mirror + daemon tests import these names
    # from this module path.
    "_make_heartbeat",
    "_write_heartbeat",
    "_IN_FLIGHT_COUNTS",
]

# Socket / PID paths default to the well-known /tmp locations. They are
# env-overridable (SIDEQUEST_RENDERER_SOCK / SIDEQUEST_RENDERER_PID) so a
# test — or a second daemon instance — can bind an isolated path instead of
# colliding with the running production daemon on the shared socket.
SOCKET_PATH = Path(os.environ.get("SIDEQUEST_RENDERER_SOCK", "/tmp/sidequest-renderer.sock"))
PID_PATH = Path(os.environ.get("SIDEQUEST_RENDERER_PID", "/tmp/sidequest-renderer.pid"))

# Socket ownership guard — set to True only inside ``_run_daemon`` after
# ``asyncio.start_unix_server`` returns successfully. Cleanup paths (the
# shutdown ``finally`` block, the ``send_shutdown`` "stale socket" branch)
# MUST check this flag before calling ``SOCKET_PATH.unlink()``. Without the
# guard, any process that imports this module — a warmup helper, a
# misrouted ``--shutdown`` racing the listening daemon's startup, or a
# future tool that calls into ``daemon.py`` — can unlink the path that the
# real listening daemon has already bound to. The kernel keeps the bound
# socket fd valid (lsof still reports the process holding it), but new
# clients cannot ``connect()`` because the directory entry is gone, and the
# server logs ``render.skipped reason=daemon_unavailable`` for every render.
# Playtest 2026-04-26 [P1] root cause.
_owns_socket: bool = False


def _live_daemon_pid() -> int | None:
    """Return the PID of a running daemon if PID_PATH points to one, else None.

    Used to gate destructive socket cleanup. Reads PID_PATH and probes the
    process with ``os.kill(pid, 0)`` (signal 0 = liveness check, never
    actually delivered). Any failure reading or probing returns None — the
    caller treats that as "no live daemon, safe to clean up."
    """
    if not PID_PATH.exists():
        return None
    try:
        pid = int(PID_PATH.read_text().strip())
    except (OSError, ValueError):
        return None
    if pid <= 0 or pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return None
    return pid


# Story 37-23: OTEL tracer for dispatch-level instrumentation. The GM panel
# consumes these spans via the ADR-058 Claude-subprocess OTEL passthrough to
# verify that the lock split is actually delivering concurrent render+embed
# at runtime — per the CLAUDE.md OTEL obligation (subsystem fixes must be
# GM-panel-visible: "The GM panel is the lie detector").
tracer = trace.get_tracer("sidequest_daemon.media.daemon")

log = logging.getLogger(__name__)


async def dispatch_request(
    request: dict,
    *,
    music_pipeline: "MusicPipeline | None" = None,
    render_service: "RenderService | None" = None,
) -> dict:
    """Route a JSON-RPC render request to the right handler based on tier.

    This is the single render-tier dispatch path (story 101-7). ``tier=music``
    routes to ``music_pipeline.generate``; image tiers route to
    ``render_service.render``; any other tier raises ``ValueError`` loudly
    (No Silent Fallbacks). Image renders are invoked while ``_handle_client``
    holds ``render_lock`` — the lock, dispatch span, and heartbeats stay at
    the socket-dispatch site (story 37-23 / 45-31).
    """
    method = request.get("method")
    if method != "render":
        raise NotImplementedError(
            f"dispatch_request only handles 'render', got {method!r}"
        )

    params = request.get("params", {})
    tier = params.get("tier", "")

    if tier in MUSIC_TIERS:
        if music_pipeline is None:
            raise RuntimeError("MusicPipeline not initialized")
        result = await music_pipeline.generate(Path(params["json_params_path"]))
        return {
            "id": request.get("id"),
            "result": {
                "r2_key": result.r2_key,
                "duration_ms": result.duration_ms,
                "seed": result.seed,
                "elapsed_ms": result.elapsed_ms,
            },
        }

    # Image tiers route to the render service. An empty/unset tier also routes
    # here: a narration-only request relies on SceneInterpreter (inside
    # RenderService) to classify the tier — rejecting it as "unknown" would
    # break the server-doesn't-classify fallback path. A *non-empty* tier that
    # is neither music nor image is a genuine unknown and fails loud below.
    if tier in IMAGE_TIERS or not tier:
        if render_service is None:
            raise RuntimeError("RenderService not initialized")
        result = await render_service.render(params)
        return {"id": request.get("id"), "result": result}

    raise ValueError(f"Unknown tier: {tier!r}")


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    pool: WorkerPool,
    render_lock: asyncio.Lock,
    embed_lock: asyncio.Lock,
) -> None:
    """Handle a single client connection — read JSON lines, dispatch, respond."""
    peer = writer.get_extra_info("peername") or "unix-client"
    log.info("Client connected: %s", peer)

    # Story 45-31: heartbeat on connection accept. Tells the server-side
    # mirror "the daemon is reachable" before the first request lands.
    # Per the no-silent-fallbacks rule: never emit a generic "I'm here"
    # event without per-queue state — the mirror keys on queue.
    _write_heartbeat(writer, "image", WorkerState.READY.value)
    _write_heartbeat(writer, "embed", WorkerState.READY.value)
    try:
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            line_str = line.decode().strip()
            if not line_str:
                continue

            try:
                req = json.loads(line_str)
                req_id = req.get("id", "unknown")
                method = req.get("method")
            except json.JSONDecodeError as e:
                _write(
                    writer, "unknown", error={"code": "PARSE_ERROR", "message": str(e)}
                )
                continue

            if not method:
                _write(
                    writer,
                    req_id,
                    error={"code": "INVALID_REQUEST", "message": "Missing 'method'"},
                )
                continue

            params = req.get("params", {})

            if method == "ping":
                _write(writer, req_id, result={"status": "ok"})
            elif method == "status":
                _write(writer, req_id, result=pool.status())
            elif method == "shutdown":
                _write(writer, req_id, result={"status": "ok"})
                log.info("Shutdown requested by client")
                asyncio.get_event_loop().call_soon(
                    lambda: os.kill(os.getpid(), signal.SIGTERM)
                )
            elif method == "warm_up":
                try:
                    target = params.get("worker", "all")
                    if target not in WARMUP_TARGETS:
                        # Fail loud: an unknown worker (e.g. the retired "flux")
                        # must not return a success response with nothing warmed.
                        _write(
                            writer,
                            req_id,
                            error={
                                "code": "UNKNOWN_WORKER",
                                "message": (
                                    f"Unknown warmup target {target!r}; "
                                    f"valid: {sorted(WARMUP_TARGETS)}"
                                ),
                            },
                        )
                        continue
                    results = {}
                    if target in ("all", "image"):
                        results["image"] = await asyncio.to_thread(pool.warm_up_image)
                    if target in ("all", "embed"):
                        results["embed"] = await asyncio.to_thread(pool.warm_up_embed)
                    _write(
                        writer, req_id, result={"status": "warm", "workers": results}
                    )
                except Exception as e:
                    _write(
                        writer,
                        req_id,
                        error={"code": "WARMUP_FAILED", "message": str(e)},
                    )
            elif method == "render":
                tier = params.get("tier")
                # Music-tier short-circuit. Music render requests have a
                # totally different shape from image-tier requests
                # (json_params_path, no narration / no game_state) AND the
                # music pipeline holds render_lock internally — they must NOT
                # be wrapped in render_lock here (that self-deadlocks). Route
                # them straight through the unified dispatcher.
                if tier in MUSIC_TIERS:
                    factory = getattr(pool, "pipeline_factory", None)
                    music_pipeline = (
                        factory.music_pipeline if factory is not None else None
                    )
                    try:
                        reply = await dispatch_request(
                            req,
                            music_pipeline=music_pipeline,
                        )
                    except Exception as exc:
                        log.exception(
                            "music.dispatch_failed tier=%s exc=%s",
                            tier,
                            exc.__class__.__name__,
                        )
                        _write(
                            writer,
                            req_id,
                            error={
                                "code": "MUSIC_RENDER_FAILED",
                                "message": str(exc),
                            },
                        )
                        continue
                    _write(writer, req_id, result=reply["result"])
                    continue

                # Image tiers: the unified dispatcher routes to RenderService
                # (story 101-7). The render_lock, the daemon.dispatch.render
                # span, and the per-queue heartbeats stay here at the
                # socket-dispatch site — story 37-23 keeps the lock + span +
                # lock_name attribute observable in this module, and the
                # heartbeats are per-connection (story 45-31).
                render_service = RenderService(pool)
                reply: dict | None = None
                with tracer.start_as_current_span("daemon.dispatch.render") as span:
                    span.set_attribute("lock_name", "render_lock")
                    span.set_attribute("tier", tier or "")
                    async with render_lock:
                        # Story 45-31: per-queue heartbeat on render-lock
                        # acquire/release. Increments the in-flight count so
                        # the heartbeat's queue_depth reflects real load.
                        _IN_FLIGHT_COUNTS["image"] += 1
                        _write_heartbeat(writer, "image", WorkerState.BUSY.value)
                        try:
                            await writer.drain()
                        except (ConnectionResetError, BrokenPipeError):
                            pass
                        try:
                            reply = await dispatch_request(
                                req, render_service=render_service
                            )
                        except RenderError as e:
                            # Compose/extraction/generation failure — fail loud
                            # with the structured frame the server expects.
                            err = {"code": e.code, "message": e.message}
                            if e.error_type is not None:
                                err["error_type"] = e.error_type
                            if e.tier is not None:
                                err["tier"] = e.tier
                            span.set_attribute("error", True)
                            span.set_attribute("error_type", e.error_type or e.code)
                            _write(writer, req_id, error=err)
                        except asyncio.CancelledError:
                            # Client disconnect is the most common failure mode;
                            # mark the span so cancellations are distinguishable
                            # from successful renders in the GM panel.
                            span.set_attribute("error", True)
                            span.set_attribute("error_type", "CancelledError")
                            raise
                        except ValueError as e:
                            # Unknown tier — fail loud (No Silent Fallbacks).
                            span.set_attribute("error", True)
                            span.set_attribute("error_type", "ValueError")
                            log.warning("render.unknown_tier — %s", e)
                            _write(
                                writer,
                                req_id,
                                error={"code": "UNKNOWN_TIER", "message": str(e)},
                            )
                        except Exception as e:
                            span.set_attribute("error", True)
                            span.set_attribute("error_type", type(e).__name__)
                            log.exception(
                                "render.dispatch_failed — tier=%s", tier
                            )
                            _write(
                                writer,
                                req_id,
                                error={
                                    "code": "GENERATION_FAILED",
                                    # Truncate — an unexpected exception can carry
                                    # local paths; don't forward verbatim (CWE-209).
                                    "message": str(e)[:512],
                                },
                            )
                        finally:
                            # Story 45-31: heartbeat on render-lock release.
                            # Fires unconditionally (success, error, cancel)
                            # so the mirror always sees the queue return to
                            # READY when the lock is freed.
                            _IN_FLIGHT_COUNTS["image"] = max(
                                0, _IN_FLIGHT_COUNTS["image"] - 1
                            )
                            _write_heartbeat(writer, "image", WorkerState.READY.value)
                            try:
                                await writer.drain()
                            except (ConnectionResetError, BrokenPipeError):
                                pass
                # Success (or beat-filter skip) → write the result frame. Error
                # frames were already written inside the lock; ``reply`` is None.
                if reply is not None:
                    _write(writer, req_id, result=reply["result"])
            elif method == "embed":
                # Story 15-7: Generate sentence embeddings for lore fragments.
                #
                # Architecture (post-37-23):
                # - Route through the singleton ``pool.embed`` — NEVER
                #   construct ``EmbedWorker()`` per request (that was the
                #   2026-04-10 playtest deadlock root cause).
                # - Run on a worker thread via ``asyncio.to_thread`` to
                #   keep the event loop unblocked during inference.
                # - Acquire ``embed_lock`` (NOT ``render_lock``). Embed
                #   runs on CPU and Z-Image runs on MPS — independent devices,
                #   independent locks. Under the old shared-lock design,
                #   10ms embeds serialized behind 5–60s image renders; now
                #   they run in parallel.
                text = params.get("text", "")
                if not text or not text.strip():
                    _write(
                        writer,
                        req_id,
                        error={
                            "code": "INVALID_REQUEST",
                            "message": "embed requires non-empty 'text' field",
                        },
                    )
                    continue
                # Story 37-23: wrap dispatch in OTEL span. The lock_name
                # attribute is the lie detector — if a future regression
                # re-shares the locks, this attribute makes the mistake
                # observable in the GM panel rather than silent.
                with tracer.start_as_current_span("daemon.dispatch.embed") as span:
                    span.set_attribute("lock_name", "embed_lock")
                    span.set_attribute("text_len", len(text))
                    async with embed_lock:
                        # Story 45-31: per-queue heartbeat on embed-lock
                        # acquire/release. Independent of image queue —
                        # busy embed must NEVER show as a busy image.
                        _IN_FLIGHT_COUNTS["embed"] += 1
                        _write_heartbeat(writer, "embed", WorkerState.BUSY.value)
                        try:
                            await writer.drain()
                        except (ConnectionResetError, BrokenPipeError):
                            pass
                        try:
                            import time

                            start = time.monotonic()
                            embedding = await asyncio.to_thread(pool.embed, text)
                            latency_ms = int((time.monotonic() - start) * 1000)
                            span.set_attribute("work_ms", latency_ms)
                            log.info(
                                "embed.generated — model=%s text_len=%d latency_ms=%d",
                                "all-MiniLM-L6-v2",
                                len(text),
                                latency_ms,
                            )
                            _write(
                                writer,
                                req_id,
                                result={
                                    "embedding": embedding,
                                    "model": "all-MiniLM-L6-v2",
                                    "latency_ms": latency_ms,
                                },
                            )
                        except asyncio.CancelledError:
                            # Client disconnect — mark span and propagate so the
                            # event loop can unwind cleanly. CancelledError is a
                            # BaseException and would otherwise bypass the
                            # Exception handler below, leaving the span
                            # attributes unset.
                            span.set_attribute("error", True)
                            span.set_attribute("error_type", "CancelledError")
                            raise
                        except Exception as e:
                            # No silent fallback — fail loud with structured error.
                            # Guard against empty str(exception) — some exceptions
                            # (e.g. RuntimeError("")) produce empty strings, which
                            # surface as "Unknown error" on the Rust/GM panel side.
                            error_msg = str(e) or f"{type(e).__name__} (no message)"
                            span.set_attribute("error", True)
                            span.set_attribute("error_type", type(e).__name__)
                            log.exception("embed.failed — text_len=%d", len(text))
                            _write(
                                writer,
                                req_id,
                                error={"code": "EMBED_FAILED", "message": error_msg},
                            )
                        finally:
                            # Story 45-31: heartbeat on embed-lock release.
                            _IN_FLIGHT_COUNTS["embed"] = max(
                                0, _IN_FLIGHT_COUNTS["embed"] - 1
                            )
                            _write_heartbeat(writer, "embed", WorkerState.READY.value)
                            try:
                                await writer.drain()
                            except (ConnectionResetError, BrokenPipeError):
                                pass
            else:
                _write(
                    writer,
                    req_id,
                    error={"code": "UNKNOWN_METHOD", "message": f"Unknown: {method}"},
                )
    except (ConnectionResetError, BrokenPipeError):
        log.info("Client disconnected: %s", peer)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except BrokenPipeError:
            log.debug("Client already disconnected before wait_closed: %s", peer)
        except Exception:
            log.exception("Failed to close client writer")


def _write(
    writer: asyncio.StreamWriter,
    req_id: str,
    *,
    result: dict | None = None,
    error: dict | None = None,
) -> None:
    """Write a JSON response line to the client."""
    resp: dict = {"id": req_id}
    if result is not None:
        resp["result"] = result
    if error is not None:
        resp["error"] = error
    writer.write((json.dumps(resp) + "\n").encode())


async def _run_daemon(
    *,
    warmup: str | bool = False,
    output_dir: Path | None = None,
    genre_packs: Path | None = None,
) -> None:
    """Start the daemon server.

    warmup can be: False, True/"all", "image", "embed"
    """
    if output_dir is None:
        env_dir = os.environ.get("SIDEQUEST_OUTPUT_DIR")
        output_dir = (
            Path(env_dir) if env_dir else Path(tempfile.mkdtemp(prefix="sq-daemon-"))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Publish the actually-used output_dir to a known handshake location so
    # the server can discover it without needing SIDEQUEST_OUTPUT_DIR set in
    # its own environment. Without this, the dev-default flow (no env var,
    # daemon picks `tempfile.mkdtemp(prefix="sq-daemon-")`) hands the server
    # a tmpdir it has no way of knowing about — every render lands in the
    # daemon's tmpdir but the server's `_render_url_from_path` falls through
    # to the verbatim path, the UI 404s, and the player sees no images.
    # Playtest 2026-04-25 [P1].
    try:
        handshake_dir = Path.home() / ".sidequest"
        handshake_dir.mkdir(parents=True, exist_ok=True)
        handshake_file = handshake_dir / "daemon-output-dir"
        handshake_file.write_text(f"{output_dir.resolve()}\n")
    except OSError:
        # Non-fatal: server falls back to the env var path. Logged so the
        # GM panel / dev shell can spot the discovery hole.
        logging.getLogger(__name__).warning(
            "daemon.handshake_write_failed dir=%s",
            handshake_dir,
        )

    if genre_packs is not None:
        os.environ["SIDEQUEST_GENRE_PACKS"] = str(genre_packs)

    # Validate daemon configuration at startup — fail loud on invalid recipes/cameras
    _daemon_root = Path(__file__).resolve().parents[2]  # sidequest-daemon/
    validate_startup_config(
        recipes_path=_daemon_root / "recipes.yaml",
        cameras_path=_daemon_root / "cameras.yaml",
    )
    pool = WorkerPool(output_dir)
    render_lock = asyncio.Lock()
    # Story 37-23: embed gets its own lock. Z-Image runs on MPS (render_lock);
    # embed runs on CPU (embed_lock). Independent devices, independent locks —
    # a long image render no longer blocks a ~30ms embed request.
    embed_lock = asyncio.Lock()

    # Initialize music pipeline via factory.
    # Per the daemon between-session music generation plan (2026-05-10):
    # the factory now constructs only the music pipeline; audio playback
    # was retired. Image pipelines are still constructed inline via WorkerPool.
    from sidequest_daemon.media.pipeline_factory import MediaPipelineFactory

    pipeline_factory = MediaPipelineFactory()
    pipeline_factory.init_music(render_lock=render_lock)
    pool.pipeline_factory = pipeline_factory
    log.info("MediaPipelineFactory initialized (music pipeline ready)")

    if warmup:
        target = warmup if isinstance(warmup, str) else "all"
        # Fail loud at startup rather than logging "warm and ready" while
        # serving cold. Catches the retired --warmup=flux and any typo.
        _validate_warmup_target(target)
        if target in ("all", "image"):
            log.info("Pre-loading Z-Image model...")
            await asyncio.to_thread(pool.warm_up_image)
        if target in ("all", "embed"):
            log.info("Pre-loading SentenceTransformer embed model...")
            await asyncio.to_thread(pool.warm_up_embed)
        log.info("Models warm and ready")

    # Clean up stale socket — but ONLY if no live process is bound to it.
    # If a PID file exists and points to a running process, refuse to unlink:
    # that would yank the directory entry out from under a live daemon and
    # leave clients unable to connect (Playtest 2026-04-26 [P1]). Fail loud
    # per the no-silent-fallbacks rule.
    if SOCKET_PATH.exists():
        if _live_daemon_pid() is not None:
            raise RuntimeError(
                f"refusing to unlink {SOCKET_PATH}: another daemon "
                f"(pid {_live_daemon_pid()}) is already bound to it. "
                f"Run `just daemon-stop` or check for orphaned processes."
            )
        log.debug("daemon.cleanup_stale_socket path=%s", SOCKET_PATH)
        SOCKET_PATH.unlink()

    server = await asyncio.start_unix_server(
        lambda r, w: _handle_client(r, w, pool, render_lock, embed_lock),
        path=str(SOCKET_PATH),
    )

    # Mark this process as the socket owner. Only the owner may unlink on
    # shutdown — see ``_owns_socket`` docstring at module top.
    global _owns_socket
    _owns_socket = True

    # Write PID file
    PID_PATH.write_text(str(os.getpid()))
    log.info("Daemon listening on %s (pid %d)", SOCKET_PATH, os.getpid())
    log.info("Workers: %s", pool.status())

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _shutdown_signal() -> None:
        log.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown_signal)

    try:
        await stop_event.wait()
    finally:
        log.info("Shutting down daemon...")
        server.close()
        await server.wait_closed()
        pool.cleanup()
        # Only the bind-owner may unlink. Without this guard, a process that
        # entered ``_run_daemon`` and then bailed before bind (or was started
        # in some warmup-only mode in the future) would still hit this
        # ``finally`` block and unlink the live daemon's socket.
        if _owns_socket and SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        elif SOCKET_PATH.exists():
            log.debug(
                "daemon.skip_unlink path=%s reason=not_owner pid=%d",
                SOCKET_PATH,
                os.getpid(),
            )
        if PID_PATH.exists():
            # PID file is keyed to this process — only delete if we wrote it,
            # which only happens after a successful bind (i.e. _owns_socket).
            if _owns_socket:
                PID_PATH.unlink()
        log.info("Daemon stopped")


async def send_shutdown() -> None:
    """Send shutdown command to a running daemon."""
    if not SOCKET_PATH.exists():
        print("No daemon running (socket not found)")
        sys.exit(1)

    try:
        reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))
        req = json.dumps({"id": "shutdown", "method": "shutdown", "params": {}})
        writer.write((req + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        resp = json.loads(line.decode())
        if resp.get("result", {}).get("status") == "ok":
            print("Daemon shutdown requested")
        else:
            print(f"Unexpected response: {resp}")
        writer.close()
    except (ConnectionRefusedError, FileNotFoundError):
        # Distinguish a *truly* stale socket (no live process) from a daemon
        # that's mid-startup (warming models before ``start_unix_server``
        # binds). In the second case the path may not yet exist or the
        # connect raced bind — unlinking would corrupt the live daemon.
        live_pid = _live_daemon_pid()
        if live_pid is not None:
            print(
                f"Daemon not responding but pid {live_pid} is alive "
                "(likely warming up) — refusing to unlink socket"
            )
            return
        print("Daemon not responding — cleaning up stale socket")
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if PID_PATH.exists():
            PID_PATH.unlink()


async def send_status() -> None:
    """Query daemon status."""
    if not SOCKET_PATH.exists():
        print("No daemon running (socket not found)")
        sys.exit(1)

    try:
        reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))
        req = json.dumps({"id": "status", "method": "status", "params": {}})
        writer.write((req + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        resp = json.loads(line.decode())
        if "result" in resp:
            status = resp["result"]
            print(f"Z-Image: {status.get('image', 'unknown')}")
            tiers = status.get("supported_tiers", {})
            print(f"Z-Image tiers: {', '.join(tiers.get('image', []))}")
        else:
            print(f"Error: {resp.get('error', resp)}")
        writer.close()
    except (ConnectionRefusedError, FileNotFoundError):
        print("Daemon not responding")


def _parse_arg(name: str) -> str | None:
    """Extract --name VALUE from sys.argv, return value or None."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return None


def main() -> None:
    """CLI entry point for sidequest-renderer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if "--shutdown" in sys.argv:
        asyncio.run(send_shutdown())
    elif "--status" in sys.argv:
        asyncio.run(send_status())
    else:
        # Warmup is the default — use --no-warmup to skip (e.g. for testing)
        warmup: str | bool = "all"
        for arg in sys.argv[1:]:
            if arg == "--no-warmup":
                warmup = False
            elif arg.startswith("--warmup="):
                warmup = arg.split("=", 1)[1]

        # Parse optional paths
        genre_packs_str = _parse_arg("--genre-packs")
        output_dir_str = _parse_arg("--output-dir")
        genre_packs = Path(genre_packs_str) if genre_packs_str else None
        output_dir = Path(output_dir_str) if output_dir_str else None

        asyncio.run(
            _run_daemon(
                warmup=warmup,
                output_dir=output_dir,
                genre_packs=genre_packs,
            )
        )


def validate_startup_config(*, recipes_path: Path, cameras_path: Path) -> None:
    """Fail-loud validation of recipe + camera YAML at daemon boot."""
    from sidequest_daemon.media.camera_specs import CameraLoader
    from sidequest_daemon.media.recipe_loader import RecipeLoader

    CameraLoader.from_file(cameras_path)
    RecipeLoader.from_file(recipes_path)


if __name__ == "__main__":
    main()
