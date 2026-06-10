"""sidequest-renderer daemon — persistent Z-Image renderer on Unix domain socket.

Hosts the Z-Image worker in a single process with the model pre-loaded.
Serves render requests over a Unix domain socket, routing by tier.
Stays warm between sessions.

Usage:
    sidequest-renderer                          # start daemon (loads Z-Image)
    sidequest-renderer --warmup=flux            # start + load Z-Image only
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
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sidequest_daemon.media.music_pipeline import MusicPipeline

from opentelemetry import trace

from sidequest_daemon.media.recipes import (
    BudgetError,
    CatalogMissError,
    RenderConfigError,
    StyleMissError,
)
from sidequest_daemon.telemetry import emit_watcher_event as _emit_watcher_event

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

# Story 45-31 — daemon worker heartbeat.
#
# The daemon emits ``{"event":"heartbeat", "queue": ..., "state": ...,
# "queue_depth": ..., "ts_monotonic": ...}`` lines on every connection
# state transition (accept, render-lock acquire/release, embed-lock
# acquire/release) and on a periodic timer when idle. The server-side
# ``DaemonStateMirror`` consumes these to track liveness without
# polling — replacing the binary socket-on-disk check that swallowed
# the Felix 13-minute silence (playtest 2026-04-19).
class WorkerState(StrEnum):
    """Heartbeat state values. Independent per queue (image vs. embed)
    so a busy embed does not flag the image queue as busy."""

    READY = "ready"   # warm, idle
    BUSY = "busy"     # render_lock or embed_lock acquired
    PAUSED = "paused"  # GPU coordinator gated the queue (ADR-046)
    COLD = "cold"     # not warmed yet


# Backpressure-counters owned by the daemon. The image / embed in-flight
# counts feed every heartbeat's ``queue_depth`` field so the server-side
# mirror sees per-queue concurrent load even when no requests are in
# flight on the connection that's polling.
_IN_FLIGHT_COUNTS: dict[str, int] = {"image": 0, "embed": 0}


def _make_heartbeat(queue: str, state: str, queue_depth: int) -> dict:
    """Build a heartbeat event payload with ``ts_monotonic`` stamped at
    emit time. Centralized so the schema does not drift across the
    six per-connection emission sites (accept × 2 queues, render-lock
    acquire/release, embed-lock acquire/release) plus the periodic
    emitter."""
    return {
        "event": "heartbeat",
        "queue": queue,
        "state": state,
        "queue_depth": int(queue_depth),
        "ts_monotonic": time.monotonic(),
    }


def _write_heartbeat(writer: asyncio.StreamWriter, queue: str, state: str) -> None:
    """Emit a heartbeat line on a per-connection writer. Called inline
    on the event loop — the writer is already protected by the
    per-connection serialization in ``_handle_client``."""
    payload = _make_heartbeat(queue, state, _IN_FLIGHT_COUNTS.get(queue, 0))
    try:
        writer.write((json.dumps(payload) + "\n").encode())
    except (ConnectionResetError, BrokenPipeError):
        # Client went away before we could flush. Not fatal — the next
        # heartbeat target may still be alive.
        pass


# Tier → worker routing.
IMAGE_TIERS = frozenset(
    {
        "scene_illustration",
        "portrait",
        "portrait_square",
        "landscape",
        "text_overlay",
        "fog_of_war",
    }
)
EMBED_TIERS = frozenset({"embed"})
MUSIC_TIERS = frozenset({"music"})


async def dispatch_request(
    request: dict,
    *,
    music_pipeline: "MusicPipeline | None" = None,
) -> dict:
    """Route a JSON-RPC render request to the right handler based on tier.

    Routes `tier=music` to `music_pipeline.generate()`. Image tiers are
    dispatched inline by `_handle_client` and never reach this function;
    any other tier raises `ValueError` loudly (no silent fallback).
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

    raise ValueError(f"Unknown tier: {tier!r}")


class EmbedWorker:
    """Generates sentence embeddings via sentence-transformers (story 15-7).

    Uses all-MiniLM-L6-v2 for 384-dimensional embeddings — fast and
    good enough for lore fragment similarity search.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_name = "all-MiniLM-L6-v2"

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Story 37-23: pin to CPU. MPS is reserved for Flux renders —
            # running embed on CPU gives it an independent device so the
            # embed path never contends with in-flight image generation
            # and can never re-trigger the 2026-04-10 concurrent-MPS-session
            # deadlock that story 37-5 originally fixed by sharing a lock.
            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def generate_embedding(self, text: str) -> list[float]:
        """Generate a sentence embedding for the given text.

        Raises ValueError if text is empty — no silent fallbacks.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return [float(v) for v in embedding]


class WorkerPool:
    """Manages the Z-Image worker with lazy or eager loading."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._image = None
        self._image_loaded = False
        # Embed worker — singleton, owned by the pool. Constructed eagerly
        # at warmup, never per-request. Per-request construction was the
        # 2026-04-10 playtest deadlock root cause: a fresh SentenceTransformer
        # download/MPS placement on every embed call, racing with Flux on
        # the same MPS device.
        self._embed: EmbedWorker | None = None
        self._embed_loaded = False
        self._embed_warmup_ms = 0
        self.pipeline_factory = None  # Set by _run_daemon after init

    def warm_up_image(self) -> dict:
        """Load and warm up the Z-Image image renderer."""
        if self._image_loaded:
            return {"worker": "image", "status": "already_warm", "warmup_ms": 0}
        from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker

        self._image = ZImageMLXWorker(self.output_dir / "zimage")
        log.info("Loading Z-Image...")
        self._image.load_model()
        result = self._image.warm_up()
        self._image_loaded = True
        log.info("Z-Image warm (%.1fs)", result.get("warmup_ms", 0) / 1000)
        return {"worker": "image", "status": "warm", **result}

    def warm_up_flux(self) -> dict:
        """Deprecated back-compat alias for warm_up_image().

        Retained so the ``--warmup=flux`` CLI flag and existing RPC callers
        that dispatch on ``worker="flux"`` keep working without refactoring.
        """
        return self.warm_up_image()

    def _ensure_image(self) -> None:
        if not self._image_loaded:
            self.warm_up_image()

    def warm_up_embed(self) -> dict:
        """Eagerly construct EmbedWorker and load its SentenceTransformer model.

        Called once at daemon startup (when ``--warmup`` or ``--warmup=all``
        is passed) and never again. The same instance is reused for every
        subsequent embed request via ``pool.embed``.
        """
        if self._embed_loaded:
            return {
                "worker": "embed",
                "status": "already_warm",
                "warmup_ms": 0,
                "model": "all-MiniLM-L6-v2",
            }
        start = time.monotonic()
        log.info("Loading SentenceTransformer all-MiniLM-L6-v2 on CPU...")
        self._embed = EmbedWorker()
        self._embed._load_model()
        self._embed_warmup_ms = int((time.monotonic() - start) * 1000)
        self._embed_loaded = True
        log.info("Embed worker warm (%.1fs)", self._embed_warmup_ms / 1000)
        return {
            "worker": "embed",
            "status": "warm",
            "warmup_ms": self._embed_warmup_ms,
            "model": "all-MiniLM-L6-v2",
        }

    def _ensure_embed(self) -> None:
        if not self._embed_loaded:
            self.warm_up_embed()

    def embed(self, text: str) -> list[float]:
        """Generate a sentence embedding via the singleton EmbedWorker.

        Synchronous — call from ``asyncio.to_thread``. The caller must hold
        ``embed_lock`` before invoking (see ``_handle_client`` dispatch);
        this method itself does not take a lock. Embed runs on CPU (see
        ``EmbedWorker._load_model``) so it has an independent device from
        Flux/MPS and cannot contend with in-flight image generation
        (story 37-23).
        """
        self._ensure_embed()
        assert self._embed is not None  # _ensure_embed populates it
        return self._embed.generate_embedding(text)

    def render(self, params: dict) -> dict:
        """Route render request to the appropriate worker by tier."""
        tier = params.get("tier", "")
        if tier in IMAGE_TIERS:
            self._ensure_image()
            return self._image.render(params)
        else:
            raise ValueError(f"Unknown tier: {tier!r}")

    def status(self) -> dict:
        """Return current worker status.

        Story 45-31: ``queue_states`` is provided for diagnostic
        consumers (GM panel, ``/health``-style introspection); the
        server-side ``DaemonStateMirror`` is populated from
        per-connection heartbeat events, NOT from this field. Distinct
        from the legacy ``image``/``embed`` keys which report
        model-load status ("warm" vs. "cold").
        """
        image_loaded = self._image_loaded
        embed_loaded = self._embed_loaded
        return {
            "image": "warm" if image_loaded else "cold",
            "embed": "warm" if embed_loaded else "cold",
            "queue_states": {
                "image": (WorkerState.READY.value if image_loaded else WorkerState.COLD.value),
                "embed": (WorkerState.READY.value if embed_loaded else WorkerState.COLD.value),
            },
            "queue_depths": dict(_IN_FLIGHT_COUNTS),
            "supported_tiers": {
                "image": sorted(IMAGE_TIERS),
                "embed": sorted(EMBED_TIERS),
            },
        }

    def cleanup(self) -> None:
        """Release all models and clear GPU cache."""
        if self._image is not None:
            self._image.cleanup()
            self._image = None
            self._image_loaded = False
        if self._embed is not None:
            # SentenceTransformer has no explicit close — drop the reference
            # so GC + MPS cache release happens.
            self._embed = None
            self._embed_loaded = False


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
                    results = {}
                    if target in ("all", "flux", "image"):
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
                # Music-tier short-circuit. Music render requests have a
                # totally different shape from image-tier requests
                # (json_params_path, no narration / no game_state) — route
                # them through dispatch_request before the image-tier
                # logic touches params it doesn't understand.
                if params.get("tier") in MUSIC_TIERS:
                    music_pipeline = (
                        pool.pipeline_factory.music_pipeline
                        if pool.pipeline_factory is not None
                        else None
                    )
                    try:
                        reply = await dispatch_request(
                            req,
                            music_pipeline=music_pipeline,
                        )
                    except Exception as exc:
                        log.exception(
                            "music.dispatch_failed tier=%s exc=%s",
                            params.get("tier"),
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

                # Beat filter: skip non-visual beats before expensive GPU work
                if params.get("narration") and params.get("game_state"):
                    from sidequest_daemon.renderer.beat_filter import should_generate
                    from sidequest_daemon.types import (
                        GameState,
                        CombatState,
                        ChaseState,
                        Character,
                    )

                    gs_raw = params["game_state"]
                    game_state = GameState(
                        location=gs_raw.get("location", ""),
                        time_of_day=gs_raw.get("time_of_day", ""),
                        characters=[
                            Character(name=c.get("name", ""))
                            for c in gs_raw.get("characters", [])
                        ],
                        combat=CombatState(
                            in_combat=gs_raw.get("combat", {}).get("in_combat", False)
                        ),
                        chase=ChaseState(
                            in_chase=gs_raw.get("chase", {}).get("in_chase", False)
                        ),
                    )
                    previous_location = params.get("previous_location")
                    if not should_generate(
                        params["narration"], game_state, previous_location
                    ):
                        log.info("beat_filter: skipping non-visual beat")
                        _write(
                            writer,
                            req_id,
                            result={"status": "skipped", "reason": "beat_filter"},
                        )
                        continue

                # If narration is provided, use SceneInterpreter for fast rule-based
                # StageCue extraction, then fall back to LLM subject extraction.
                if params.get("narration") and not params.get("positive_prompt"):
                    from sidequest_daemon.scene_interpreter import SceneInterpreter
                    from sidequest_daemon.types import GameState, Character

                    narrator_text = params["narration"]

                    # Extract documents and strip markers before visual processing
                    scene_interp = SceneInterpreter()
                    genre = params.get("genre", "unknown")
                    doc_events = scene_interp.extract_documents(
                        narrator_text, genre=genre
                    )
                    if doc_events:
                        log.info(
                            "scene_interpreter: extracted %d document(s)",
                            len(doc_events),
                        )
                        params.setdefault("document_events", [])
                        for doc in doc_events:
                            params["document_events"].append(doc.model_dump())
                    narrator_text = scene_interp.strip_document_markers(narrator_text)
                    params["narration"] = narrator_text

                    # Try rule-based StageCue extraction (fast, no LLM)
                    gs_raw = params.get("game_state", {})
                    interp_state = GameState(
                        location=gs_raw.get("location", ""),
                        time_of_day=gs_raw.get("time_of_day", ""),
                        characters=[
                            Character(name=c.get("name", ""))
                            for c in gs_raw.get("characters", [])
                        ],
                    )
                    # Only run rule-based interpretation when the caller did
                    # NOT already supply a structured visual block. The
                    # server-side narrator agent emits {tier, subject, mood,
                    # tags} as structured output (see
                    # sidequest-server/sidequest/agents/narrator.py — the
                    # visual JSON block) and the dispatcher forwards those
                    # fields verbatim. Overriding them here silently
                    # second-guesses the agent's classification, which is
                    # how a narrator-classified `landscape` was being
                    # rewritten to `scene_illustration` by an atmosphere
                    # rule match — then validated as `kind=illustration`
                    # without the PC `participants` the server only
                    # populates on the `tier=scene_illustration` branch.
                    # That mis-routed shape was the playtest 2026-04-30
                    # COMPOSE_FAILED signature.
                    server_supplied_tier = (
                        params.get("tier") in IMAGE_TIERS
                        and bool(params.get("subject"))
                    )
                    if not server_supplied_tier:
                        cues = scene_interp.interpret(narrator_text, interp_state)
                        if cues:
                            top_cue = cues[0]
                            params["subject"] = top_cue.subject
                            params["mood"] = top_cue.mood
                            params["tags"] = top_cue.tags
                            params["tier"] = top_cue.tier.value
                            with tracer.start_as_current_span(
                                "scene_interpreter.classified"
                            ) as cls_span:
                                cls_span.set_attribute("tier", top_cue.tier.value)
                                cls_span.set_attribute("subject", top_cue.subject[:120])
                                cls_span.set_attribute("source", "rule_match")
                            log.info(
                                "scene_interpreter — tier=%s subject=%s",
                                top_cue.tier.value,
                                top_cue.subject[:80],
                            )
                    else:
                        with tracer.start_as_current_span(
                            "scene_interpreter.skipped"
                        ) as skip_span:
                            skip_span.set_attribute(
                                "reason", "server_supplied_visual_block"
                            )
                            skip_span.set_attribute("tier", str(params.get("tier", "")))
                        log.info(
                            "scene_interpreter — skipped (server tier=%s subject=%s)",
                            params.get("tier"),
                            str(params.get("subject", ""))[:80],
                        )

                    # Fall back to LLM subject extraction if SceneInterpreter
                    # didn't produce a subject (or for refinement)
                    if not params.get("subject"):
                        from sidequest_daemon.media.subject_extractor import (
                            SubjectExtractor,
                        )

                        extractor = SubjectExtractor()
                        extracted = await extractor.extract(params["narration"])
                        if not extracted or not extracted.get("subject"):
                            _write(
                                writer,
                                req_id,
                                error={
                                    "code": "EXTRACTION_FAILED",
                                    "message": "SubjectExtractor returned no visual subject from narration. No fallback — refusing to render narrative prose directly.",
                                },
                            )
                            continue
                        # Build StageCue-compatible params from extraction
                        params["subject"] = extracted["subject"]
                        params["mood"] = extracted.get("mood", "")
                        params["tags"] = extracted.get("tags", [])
                        # Override tier if extractor found a better one
                        extracted_tier = extracted.get("tier", "")
                        if extracted_tier:
                            tier_lower = extracted_tier.lower()
                            if tier_lower in IMAGE_TIERS:
                                params["tier"] = tier_lower
                        log.info(
                            "narration_extracted — subject=%s, mood=%s, tier=%s",
                            extracted["subject"][:80],
                            extracted.get("mood"),
                            params.get("tier"),
                        )

                composed = None
                if not params.get("positive_prompt"):
                    missing = [
                        k for k in ("subject", "world", "genre") if not params.get(k)
                    ]
                    try:
                        if missing:
                            with tracer.start_as_current_span(
                                "compose.gate_short_circuit"
                            ) as gate_span:
                                gate_span.set_attribute(
                                    "missing_fields", ",".join(missing)
                                )
                                gate_span.set_attribute(
                                    "tier", params.get("tier", "")
                                )
                            raise RenderConfigError(
                                f"render request missing required field(s): {missing}"
                            )

                        from sidequest_daemon.media.workers.zimage_mlx_worker import (
                            build_cue_from_params,
                            compose_prompt_for,
                        )

                        cue = build_cue_from_params(params)
                        composed = compose_prompt_for(cue)
                        params["positive_prompt"] = composed.positive_prompt
                        params["clip_prompt"] = composed.clip_prompt
                        params["negative_prompt"] = composed.negative_prompt
                        params["seed"] = composed.seed
                        # Story 78-1: forward the resolved camera's post
                        # directive (crop/rotate) so the worker applies it
                        # after generation. JSON-dict form keeps params
                        # socket-serializable; None when the camera sets none.
                        params["post"] = (
                            composed.post.model_dump()
                            if composed.post is not None
                            else None
                        )
                        log.info(
                            "prompt_composed — positive=%s",
                            composed.positive_prompt[:150],
                        )
                    except (
                        RenderConfigError,
                        StyleMissError,
                        CatalogMissError,
                        BudgetError,
                        ValueError,
                        # Pingpong 2026-04-30: ``IndexError`` from
                        # ``_character_lod_plan`` (landscape tier with
                        # empty participants) leaked past this handler,
                        # closing the socket mid-request — server logged
                        # ``render.reply_unavailable error=daemon closed
                        # socket before sending a reply``, the real
                        # IndexError was hidden in the daemon log, and
                        # 2 of 2 landscape dispatches silently disappeared
                        # from the Scrapbook. Adding ``IndexError`` /
                        # ``KeyError`` / ``AttributeError`` / ``TypeError``
                        # (the typical "data shape unexpected" failures)
                        # so future tier-wide compose failures emit the
                        # ``compose.failed`` span + structured COMPOSE_FAILED
                        # error frame instead of breaking the JSON-RPC
                        # transport. Matches the user's pingpong note
                        # request: "Add a daemon.compose_failed watcher
                        # span at the daemon `_handle_client` exception
                        # handler so future tier-wide failures show up in
                        # the dashboard instead of /tmp/sidequest-daemon.log."
                        # The narrower fix in prompt_composer.py prevents
                        # this specific IndexError from firing again, but
                        # this defense-in-depth guards the next data-shape
                        # bug — and converts an opaque socket-close into a
                        # GM-panel-visible event.
                        IndexError,
                        KeyError,
                        AttributeError,
                        TypeError,
                    ) as e:
                        # JSON-RPC contract: a render request must always get
                        # either a result or an error frame. Compose-time
                        # exceptions used to bubble out of `_handle_client`
                        # (only `ConnectionResetError`/`BrokenPipeError` are
                        # caught at the outer scope), which closed the socket
                        # mid-request. The server then reported
                        # `daemon.outcome=eof_before_reply` and
                        # `daemon_unavailable` — masking the real failure
                        # (e.g. `PlaceCatalog` rejecting a non-`where:` ref)
                        # behind a transport error.
                        #
                        # Per CLAUDE.md "OTEL Observability Principle": fail
                        # LOUD to the client, not silently to the socket.
                        with tracer.start_as_current_span(
                            "compose.failed"
                        ) as fail_span:
                            fail_span.set_attribute("tier", params.get("tier", ""))
                            fail_span.set_attribute("error_type", type(e).__name__)
                            fail_span.set_attribute("error_message", str(e)[:512])
                            fail_span.set_attribute(
                                "world", params.get("world", "")
                            )
                            fail_span.set_attribute(
                                "genre", params.get("genre", "")
                            )
                        # Watcher event for the GM panel (pingpong
                        # 2026-04-30 daemon-tier-failure ask). The
                        # ``compose.failed`` OTEL span above is for tracer
                        # consumers (Jaeger / OTLP); the watcher event is
                        # the path the dashboard's Console / Subsystems
                        # tabs read. Both fire so the failure is visible
                        # at every observation tier.
                        # sync — see sidequest_daemon/telemetry/watcher_bridge.py docstring for trade-off rationale
                        _emit_watcher_event(
                            "daemon_compose_failed",
                            {
                                "tier": params.get("tier", ""),
                                "error_type": type(e).__name__,
                                "error_message": str(e)[:512],
                                "world": params.get("world", ""),
                                "genre": params.get("genre", ""),
                                "render_id": params.get("render_id", ""),
                            },
                        )
                        log.warning(
                            "render.compose_failed — tier=%s err_type=%s err=%s",
                            params.get("tier", ""),
                            type(e).__name__,
                            e,
                        )
                        _write(
                            writer,
                            req_id,
                            error={
                                "code": "COMPOSE_FAILED",
                                "message": f"{type(e).__name__}: {e}",
                                "error_type": type(e).__name__,
                                "tier": params.get("tier", ""),
                            },
                        )
                        continue

                # Serialize renders — only one GPU operation at a time.
                # Story 37-23: wrap dispatch in OTEL span so the GM panel can
                # verify render acquired render_lock (not embed_lock).
                with tracer.start_as_current_span("daemon.dispatch.render") as span:
                    span.set_attribute("lock_name", "render_lock")
                    span.set_attribute("tier", params.get("tier", ""))
                    async with render_lock:
                        # Story 45-31: per-queue heartbeat on render-lock
                        # acquire/release. Increments the in-flight count
                        # so the heartbeat's queue_depth reflects real
                        # concurrent load.
                        _IN_FLIGHT_COUNTS["image"] += 1
                        _write_heartbeat(writer, "image", WorkerState.BUSY.value)
                        try:
                            await writer.drain()
                        except (ConnectionResetError, BrokenPipeError):
                            pass
                        try:
                            result = await asyncio.to_thread(pool.render, params)
                            with tracer.start_as_current_span(
                                "render.completed"
                            ) as completed:
                                final_prompt = params.get("positive_prompt", "")
                                completed.set_attribute(
                                    "genre", params.get("genre", "")
                                )
                                completed.set_attribute(
                                    "world", params.get("world", "")
                                )
                                # R2 migration: surface session_id and the
                                # uploaded r2_key on render.completed so the
                                # GM panel can verify the artifact landed.
                                # Empty session_id at this point means the
                                # server didn't thread it through — the
                                # worker fell back to "unknown" and the GM
                                # panel can flag it.
                                completed.set_attribute(
                                    "session_id", params.get("session_id", "")
                                )
                                completed.set_attribute(
                                    "r2_key", str(result.get("r2_key") or "")
                                )
                                completed.set_attribute(
                                    "tier", params.get("tier", "")
                                )
                                completed.set_attribute(
                                    "prompt_length", len(final_prompt)
                                )
                                genre_applied = False
                                world_applied = False
                                if composed is not None:
                                    for layer in composed.layers:
                                        tokens = layer.tokens.strip()
                                        if not tokens:
                                            continue
                                        if (
                                            layer.slot == "ART_SENSIBILITY.GENRE"
                                            and tokens in final_prompt
                                        ):
                                            genre_applied = True
                                        elif (
                                            layer.slot == "ART_SENSIBILITY.WORLD"
                                            and tokens in final_prompt
                                        ):
                                            world_applied = True
                                completed.set_attribute(
                                    "genre_style_applied", genre_applied
                                )
                                completed.set_attribute(
                                    "world_style_applied", world_applied
                                )
                            _write(writer, req_id, result=result)
                        except asyncio.CancelledError:
                            # Client disconnect is the most common failure mode;
                            # mark the span so cancellations are distinguishable
                            # from successful renders in the GM panel.
                            span.set_attribute("error", True)
                            span.set_attribute("error_type", "CancelledError")
                            raise
                        except Exception as e:
                            span.set_attribute("error", True)
                            span.set_attribute("error_type", type(e).__name__)
                            log.exception(
                                "render.failed — tier=%s", params.get("tier", "")
                            )
                            _write(
                                writer,
                                req_id,
                                error={"code": "GENERATION_FAILED", "message": str(e)},
                            )
                        finally:
                            # Story 45-31: heartbeat on render-lock release.
                            # Fires unconditionally (success, error, cancel)
                            # so the mirror always sees the queue return to
                            # READY when the lock is freed.
                            _IN_FLIGHT_COUNTS["image"] = max(
                                0, _IN_FLIGHT_COUNTS["image"] - 1
                            )
                            _write_heartbeat(
                                writer, "image", WorkerState.READY.value
                            )
                            try:
                                await writer.drain()
                            except (ConnectionResetError, BrokenPipeError):
                                pass
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
                #   runs on CPU and Flux runs on MPS — independent devices,
                #   independent locks. Under the old shared-lock design,
                #   10ms embeds serialized behind 5–60s Flux renders; now
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
                            _write_heartbeat(
                                writer, "embed", WorkerState.READY.value
                            )
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

    warmup can be: False, True/"all", "flux"
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
        import logging as _logging

        _logging.getLogger(__name__).warning(
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
    # Story 37-23: embed gets its own lock. Flux runs on MPS (render_lock);
    # embed runs on CPU (embed_lock). Independent devices, independent locks —
    # a long Flux render no longer blocks a ~30ms embed request.
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
        if target in ("all", "flux"):
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
