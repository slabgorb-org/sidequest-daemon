"""RED tests — Story 45-31 — daemon-side heartbeat emission (AC1, AC2, AC6).

The daemon emits a per-queue heartbeat event line on every state
transition so the server-side ``DaemonStateMirror`` can track liveness
without polling. Heartbeat shape::

    {"event": "heartbeat",
     "queue": "image" | "embed",
     "state": "ready" | "busy" | "paused" | "cold",
     "queue_depth": <int>,
     "ts_monotonic": <float>}

Distinguishable from request replies by the presence of an ``event``
key (replies carry ``id`` + ``result`` / ``error``).

Scope (per the story context):
  - AC1: heartbeats emit on connection accept (ready), render-lock
    acquire (busy), render-lock release (ready).
  - AC1 negative: an embed call MUST NOT emit a ``queue="image"`` busy
    heartbeat.
  - AC2: idle daemon emits a periodic ready heartbeat at the
    configured interval (default 30s).
  - AC6: heartbeat lines coexist with the per-request result line and
    do not break existing clients — a client that ignores ``event``
    lines still parses replies cleanly.

Testing strategy: spin up the real ``_handle_client`` against an
in-process Unix socket with a stub ``WorkerPool`` whose ``render``
blocks until the test releases. Read all lines off the wire and
classify by JSON shape.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def short_sock() -> Path:
    p = Path(f"/tmp/sq-hb-test-{uuid.uuid4().hex[:8]}.sock")
    yield p
    if p.exists():
        p.unlink()


def test_worker_state_enum_has_required_values() -> None:
    """``WorkerState`` enumerates the four states the heartbeat payload
    can report. RED until the enum lands in
    ``sidequest_daemon.media.daemon``."""
    from sidequest_daemon.media.daemon import WorkerState

    assert WorkerState.READY.value == "ready"
    assert WorkerState.BUSY.value == "busy"
    assert WorkerState.PAUSED.value == "paused"
    assert WorkerState.COLD.value == "cold"


def test_worker_pool_status_carries_per_queue_state(tmp_path: Path) -> None:
    """``WorkerPool.status()`` must surface a per-queue state field
    that the heartbeat emit logic consults. The exact key is
    ``queue_states`` (a dict of ``queue → state-name``).

    Pre-fix, ``status()`` returned ``{"image": "warm"|"cold", "embed":
    ...}`` — that value space conflates "model loaded" with "queue
    busy." The heartbeat needs the second axis."""
    from sidequest_daemon.media.daemon import WorkerPool

    pool = WorkerPool(tmp_path)
    status = pool.status()
    assert "queue_states" in status, (
        f"WorkerPool.status() must include 'queue_states' for the "
        f"heartbeat emit logic; got keys={sorted(status.keys())}"
    )
    queue_states = status["queue_states"]
    assert "image" in queue_states
    assert "embed" in queue_states
    # A pool with no work in flight reports ``cold`` (model not warmed)
    # or ``ready`` (model warm, idle). Both are valid for a fresh
    # WorkerPool — what's NOT valid is a missing key or "warm" leaking
    # through.
    assert queue_states["image"] in {"cold", "ready"}, (
        f"queue_states['image'] = {queue_states['image']!r} — "
        "expected 'cold' or 'ready' on a fresh pool"
    )


# ---------------------------------------------------------------------------
# In-process daemon harness
# ---------------------------------------------------------------------------


class _StubWorkerPool:
    """Minimal stand-in for ``WorkerPool`` so tests don't need real
    Z-Image weights. Reports ``warm`` for image so the daemon emits
    ``ready`` instead of ``cold`` on connect.

    ``render_release`` and ``render_started`` are ``threading.Event``
    instances (not asyncio.Event) because ``_handle_client`` runs the
    pool's ``render`` via ``asyncio.to_thread`` — the worker is in a
    different OS thread from the test's event loop, so the events
    must be thread-safe. Tests interact with them via plain ``set()``
    from the loop thread; ``threading.Event.set`` is itself
    thread-safe."""

    def __init__(self) -> None:
        import threading

        self.render_release = threading.Event()
        self.render_started = threading.Event()
        self.render_calls: list[dict[str, Any]] = []

    def status(self) -> dict[str, Any]:
        return {
            "image": "warm",
            "embed": "warm",
            "queue_states": {"image": "ready", "embed": "ready"},
            "supported_tiers": {
                "image": ["scene_illustration"],
                "embed": ["embed"],
            },
        }

    def render(self, params: dict[str, Any]) -> dict[str, Any]:
        """Block synchronously until released. ``_handle_client`` runs
        this on ``asyncio.to_thread`` so blocking here keeps the
        render-lock held — exactly what we need to observe a busy
        heartbeat in flight."""
        self.render_calls.append(params)
        self.render_started.set()
        if not self.render_release.wait(timeout=5.0):
            raise TimeoutError("render_release was never set")
        return {
            "image_url": "/tmp/x.png",
            "width": 64,
            "height": 64,
            "elapsed_ms": 1,
        }

    def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]


async def _serve(pool: _StubWorkerPool, sock: Path) -> asyncio.AbstractServer:
    """Start the real ``_handle_client`` against the stub pool."""
    from sidequest_daemon.media.daemon import _handle_client

    render_lock = asyncio.Lock()
    embed_lock = asyncio.Lock()

    async def handler(reader, writer):  # noqa: ANN001
        await _handle_client(reader, writer, pool, render_lock, embed_lock)

    return await asyncio.start_unix_server(handler, path=str(sock))


async def _read_all_lines(reader: asyncio.StreamReader, timeout: float = 1.0) -> list[dict]:
    """Consume every line the daemon writes to the connection until
    EOF or timeout. Each line is parsed as JSON."""
    out: list[dict] = []
    while True:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
        except TimeoutError:
            break
        if not raw:
            break
        line = raw.decode().strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _is_heartbeat(line: dict) -> bool:
    return line.get("event") == "heartbeat"


def _is_reply(line: dict) -> bool:
    return "id" in line and ("result" in line or "error" in line)


# ---------------------------------------------------------------------------
# AC1: heartbeat sequence around render lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_emits_ready_busy_ready_heartbeats(short_sock: Path) -> None:
    """AC1: a render call MUST emit at least one ``state="busy"``
    heartbeat between the request and the result line, and a
    ``state="ready"`` heartbeat after the lock releases. ``queue`` is
    ``"image"`` on every heartbeat in this sequence."""
    pool = _StubWorkerPool()
    server = await _serve(pool, short_sock)
    try:
        reader, writer = await asyncio.open_unix_connection(str(short_sock))
        try:
            req = {
                "id": "r1",
                "method": "render",
                "params": {
                    "tier": "scene_illustration",
                    "subject": "x",
                    "world": "w",
                    "genre": "g",
                    "positive_prompt": "x",
                    "seed": 1,
                },
            }
            writer.write((json.dumps(req) + "\n").encode())
            await writer.drain()

            # Wait for the worker thread to enter render(), then release.
            # ``render_started`` is a threading.Event — poll on the loop
            # so we don't block the event loop while the worker thread
            # acquires the GIL.
            for _ in range(200):
                if pool.render_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert pool.render_started.is_set(), "worker never entered render()"
            # Brief pause so the daemon has a chance to flush the
            # busy heartbeat onto the wire before we release.
            await asyncio.sleep(0.05)
            pool.render_release.set()

            lines = await _read_all_lines(reader, timeout=1.0)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    heartbeats = [ln for ln in lines if _is_heartbeat(ln)]
    replies = [ln for ln in lines if _is_reply(ln)]

    # AC1: at least one busy heartbeat for queue=image.
    busy_image = [
        h
        for h in heartbeats
        if h.get("queue") == "image" and h.get("state") == "busy"
    ]
    assert busy_image, (
        f"AC1: expected at least one queue=image, state=busy heartbeat "
        f"during a render, got heartbeats={heartbeats}"
    )

    # AC1: at least one ready heartbeat for queue=image after release.
    ready_image = [
        h
        for h in heartbeats
        if h.get("queue") == "image" and h.get("state") == "ready"
    ]
    assert ready_image, (
        f"AC1: expected at least one queue=image, state=ready heartbeat "
        f"after render-lock release, got heartbeats={heartbeats}"
    )

    # AC1: heartbeat payload schema — every heartbeat MUST carry
    # queue, state, queue_depth, ts_monotonic.
    for h in heartbeats:
        for required in ("queue", "state", "queue_depth", "ts_monotonic"):
            assert required in h, (
                f"heartbeat missing field {required!r}: {h}"
            )
        assert isinstance(h["queue_depth"], int), h
        assert isinstance(h["ts_monotonic"], (int, float)), h

    # AC6: result line still arrives — heartbeats coexist with replies.
    assert replies, (
        "AC6: heartbeat lines MUST not displace reply lines; expected "
        "the render reply on the same connection"
    )
    assert any(r.get("id") == "r1" for r in replies)


# ---------------------------------------------------------------------------
# AC1 negative: embed lock does not flip the image queue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_does_not_emit_image_busy_heartbeat(short_sock: Path) -> None:
    """AC1 negative: an embed call MUST NOT emit a ``queue="image"``
    busy heartbeat. The image and embed locks are independent (per
    ADR-035 + 37-23) — the heartbeat must reflect the same separation
    or it lies about which queue is in flight."""
    pool = _StubWorkerPool()
    server = await _serve(pool, short_sock)
    try:
        reader, writer = await asyncio.open_unix_connection(str(short_sock))
        try:
            req = {
                "id": "e1",
                "method": "embed",
                "params": {"text": "hello"},
            }
            writer.write((json.dumps(req) + "\n").encode())
            await writer.drain()
            lines = await _read_all_lines(reader, timeout=1.0)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    heartbeats = [ln for ln in lines if _is_heartbeat(ln)]

    # Positive precondition: an embed call DOES emit at least one
    # heartbeat (ready on accept, plus busy/ready around the embed
    # lock). Without this anchor, the negative assertion below is
    # vacuously true when no heartbeats are emitted at all.
    assert heartbeats, (
        "expected at least one heartbeat on a connection that handled "
        "an embed call (accept-time ready emit at minimum); got none. "
        "The negative assertion below would otherwise pass vacuously."
    )

    bad = [
        h
        for h in heartbeats
        if h.get("queue") == "image" and h.get("state") == "busy"
    ]
    assert not bad, (
        f"AC1 negative: embed call emitted queue=image, state=busy "
        f"heartbeat — image and embed queues must remain independent. "
        f"Got: {bad}"
    )

    # AC1 positive complement: the embed lock SHOULD have produced at
    # least one queue=embed heartbeat on the same wire. Without this
    # the negative result above could be a false reassurance — it
    # might mean the embed path emits nothing at all rather than that
    # it emits the right thing.
    embed_busy = [
        h for h in heartbeats
        if h.get("queue") == "embed" and h.get("state") == "busy"
    ]
    assert embed_busy, (
        "AC1: an embed call must emit at least one queue=embed, "
        "state=busy heartbeat so the mirror can track embed-queue "
        "depth independently of the image queue. "
        f"Got heartbeats={heartbeats}"
    )


# ---------------------------------------------------------------------------
# AC6: heartbeat lines do not corrupt per-request reply parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_lines_do_not_corrupt_reply_parsing(short_sock: Path) -> None:
    """AC6 regression: a client that filters lines by the canonical
    "result/error => reply, event => not reply" rule MUST be able to
    walk a heartbeat-mixed stream without mis-classifying a heartbeat
    as a reply (or vice versa). Catches a regression where the
    heartbeat shape collides with reply shape."""
    pool = _StubWorkerPool()
    pool.render_release.set()  # don't block — we want fast turn-around
    server = await _serve(pool, short_sock)
    try:
        reader, writer = await asyncio.open_unix_connection(str(short_sock))
        try:
            for n in range(5):
                req = {
                    "id": f"r{n}",
                    "method": "render",
                    "params": {
                        "tier": "scene_illustration",
                        "subject": str(n),
                        "world": "w",
                        "genre": "g",
                        "positive_prompt": str(n),
                        "seed": n,
                    },
                }
                writer.write((json.dumps(req) + "\n").encode())
                await writer.drain()
            # Drain whatever the daemon sends.
            lines = await _read_all_lines(reader, timeout=1.5)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    replies = [ln for ln in lines if _is_reply(ln)]
    heartbeats = [ln for ln in lines if _is_heartbeat(ln)]

    # Positive precondition: heartbeats must actually appear on the
    # wire mixed with replies — otherwise the "shapes are mutually
    # exclusive" check below holds vacuously when zero heartbeats are
    # emitted. The whole point of AC6 is regression-guarding the
    # mixed stream.
    assert heartbeats, (
        "AC6 regression: expected heartbeats interleaved with replies "
        "on the same connection. Got zero heartbeat lines — the "
        "mutual-exclusion check below would pass vacuously."
    )

    # AC6: every reply id from the 5 requests must be present, and no
    # reply must have been mis-shaped as a heartbeat.
    reply_ids = {r["id"] for r in replies}
    expected_ids = {f"r{n}" for n in range(5)}
    assert expected_ids <= reply_ids, (
        f"AC6: heartbeat lines stole reply slots. Expected reply ids "
        f"{expected_ids}, got {reply_ids}"
    )

    # AC6: the two shapes are mutually exclusive — a line cannot be
    # both. Catches a hybrid that breaks naive parsers. Each side must
    # be checked independently: a heartbeat carrying *either* `id` or
    # `result` is a hybrid that fails the mutual-exclusion contract,
    # not just one carrying both. (Review M3 — `or` here was a vacuous
    # logic bug; replaced with two independent asserts.)
    for h in heartbeats:
        assert "id" not in h, (
            f"AC6: heartbeat line carries reply field 'id': {h}"
        )
        assert "result" not in h, (
            f"AC6: heartbeat line carries reply field 'result': {h}"
        )
    for r in replies:
        assert "event" not in r, (
            f"AC6: reply line contains heartbeat-shaped fields: {r}"
        )
