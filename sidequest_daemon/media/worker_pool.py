"""Worker pool + per-queue heartbeat state for the renderer daemon.

Extracted from ``daemon.py`` (story 101-7) so ``daemon.py`` is socket
lifecycle + routing only. ``daemon.py`` re-exports ``WorkerPool``,
``WorkerState``, and the heartbeat helpers for back-compat — the server's
``DaemonStateMirror`` (story 45-31) and the daemon's own tests import these
from ``sidequest_daemon.media.daemon``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from enum import StrEnum
from pathlib import Path

from sidequest_daemon.media.embed_worker import EmbedWorker
from sidequest_daemon.media.tiers import EMBED_TIERS, IMAGE_TIERS

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


class WorkerPool:
    """Manages the Z-Image worker with lazy or eager loading."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._image = None
        self._image_loaded = False
        # Embed worker — singleton, owned by the pool. Constructed eagerly
        # at warmup, never per-request. Per-request construction was the
        # 2026-04-10 playtest deadlock root cause: a fresh SentenceTransformer
        # download/MPS placement on every embed call, racing with Z-Image
        # on the same MPS device.
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
        Z-Image/MPS and cannot contend with in-flight image generation
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
