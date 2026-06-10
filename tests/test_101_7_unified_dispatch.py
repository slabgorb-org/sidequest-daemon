"""Story 101-7 — unified render dispatch + method wiring (RED).

Today there are TWO dispatch mechanisms for one ``render`` method:

  * ``dispatch_request()`` handles ONLY ``tier=music``. Its own docstring
    admits "Image tiers are dispatched inline by ``_handle_client`` and never
    reach this function."
  * Image tiers are routed by a parallel inline if/elif chain inside
    ``_handle_client``.

Two dispatch paths for one method is the embryo of the next split-brain.
This story unifies them so EVERY render tier flows through the single
``dispatch_request`` path, with a loud ``ValueError`` on an unknown tier.

  AC1 — single dispatch path for all render tiers; unknown tier fails loud.
  AC4 — every advertised RPC method reaches the unified dispatcher.

TEA design decision (logged as a deviation in the session file): the story
names ``dispatch_request()`` as the existing dispatcher that image tiers
"never reach". The contract these tests fix is therefore "``dispatch_request``
is the single render-tier dispatch path; image tiers route to an injected
render service through it." The heavy compose/lock/heartbeat machinery is
extracted into RenderService (see test_101_7_dispatch_extraction) — these
tests do not pin its internals, only that the dispatcher delegates to it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from sidequest_daemon.media.daemon import dispatch_request


# --------------------------------------------------------------------------
# AC1 — single dispatch path for all render tiers
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_request_routes_image_tier_to_render_service():
    """RED: an image tier must reach the injected render service THROUGH
    ``dispatch_request`` — not raise "Unknown tier".

    Currently ``dispatch_request`` only knows ``tier=music`` and raises
    ``ValueError("Unknown tier: 'portrait'")`` for every image tier, because
    image dispatch lives in a parallel inline path. After unification the
    image tier must be routed to the render service and its result returned
    under the JSON-RPC ``result`` envelope.
    """
    fake_service = MagicMock()
    fake_service.render = AsyncMock(
        return_value={"r2_key": "renders/p.png", "seed": 7}
    )

    request = {
        "id": "img-1",
        "method": "render",
        "params": {
            "tier": "portrait",
            "positive_prompt": "a weathered station marshal",
            "world": "coyote_star",
            "genre": "space_opera",
            "subject": "marshal",
        },
    }

    reply = await dispatch_request(request, render_service=fake_service)

    fake_service.render.assert_awaited_once()
    # The params dict (not the whole request) is what the render service
    # consumes — mirror the music branch, which passes the inner payload.
    (called_params,), _ = fake_service.render.await_args
    assert called_params["tier"] == "portrait"
    assert reply["id"] == "img-1"
    assert reply["result"] == {"r2_key": "renders/p.png", "seed": 7}


@pytest.mark.asyncio
async def test_dispatch_request_unknown_tier_fails_loud():
    """AC1: an unrecognized tier must raise ``ValueError`` — no silent
    fallback, even once image tiers are wired in. (Guard: passes today,
    must keep passing.)"""
    request = {"id": "x", "method": "render", "params": {"tier": "hologram"}}
    with pytest.raises(ValueError, match="[Uu]nknown tier"):
        await dispatch_request(request, music_pipeline=None)


@pytest.mark.asyncio
async def test_dispatch_request_music_tier_still_routes(tmp_path):
    """AC1 regression: unifying the image path must not disturb the existing
    music route. (Guard — mirrors test_music_dispatch.)"""
    json_path = tmp_path / "combat_input_params.json"
    json_path.write_text(json.dumps({"task": "text2music", "actual_seeds": [42]}))

    from sidequest_daemon.media.music_pipeline import MusicPipeline, MusicResult

    fake_pipeline = MagicMock(spec=MusicPipeline)
    fake_pipeline.generate = AsyncMock(
        return_value=MusicResult(
            r2_key="genre_packs/cav/audio/music/combat.ogg",
            duration_ms=60_000,
            seed=42,
            elapsed_ms=67_000,
        )
    )

    request = {
        "id": "music-1",
        "method": "render",
        "params": {"tier": "music", "json_params_path": str(json_path)},
    }
    reply = await dispatch_request(request, music_pipeline=fake_pipeline)
    fake_pipeline.generate.assert_awaited_once_with(Path(json_path))
    assert reply["result"]["seed"] == 42


# --------------------------------------------------------------------------
# AC4 — every advertised RPC method reaches the unified dispatcher
# --------------------------------------------------------------------------


class _RecordingWriter:
    """In-memory stand-in for ``asyncio.StreamWriter`` (mirrors the harness
    in test_compose_error_replies.py). Records bytes, filters heartbeats."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        return None

    def get_extra_info(self, key: str) -> str:
        return "test-peer"

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None

    @property
    def replies(self) -> list[dict]:
        joined = b"".join(self.chunks).decode()
        out: list[dict] = []
        for line in joined.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("event") == "heartbeat":
                continue
            out.append(obj)
        return out


class _FakePool:
    """Minimal WorkerPool stand-in: enough surface for the routing tests."""

    pipeline_factory = None

    def status(self) -> dict:
        return {"image": "cold", "embed": "cold"}

    def render(self, params: dict) -> dict:
        return {"r2_key": "renders/fake.png"}


async def _drive(reader_lines: list[dict], pool=None):
    """Feed JSON-RPC request lines through ``_handle_client`` over an
    in-memory reader/writer and return the reply frames."""
    from sidequest_daemon.media.daemon import _handle_client

    reader = asyncio.StreamReader()
    for req in reader_lines:
        reader.feed_data((json.dumps(req) + "\n").encode())
    reader.feed_eof()
    writer = _RecordingWriter()
    await _handle_client(
        reader,
        writer,  # type: ignore[arg-type]
        pool or _FakePool(),  # type: ignore[arg-type]
        asyncio.Lock(),
        asyncio.Lock(),
    )
    return writer.replies


@pytest.mark.asyncio
async def test_image_render_routes_through_unified_dispatcher(monkeypatch):
    """RED: an image-tier ``render`` must be routed through
    ``dispatch_request`` — the same single path music already uses.

    We install a spy in place of ``dispatch_request`` and drive an image
    render through ``_handle_client``. Today the spy is never called for an
    image tier (image dispatch is the parallel inline path), so this fails.
    After unification the spy fires for the image tier exactly as it does for
    music.
    """
    import sidequest_daemon.media.daemon as daemon_mod

    calls: list[dict] = []

    async def _spy(request, **kwargs):
        calls.append(request)
        return {"result": {"status": "ok"}}

    monkeypatch.setattr(daemon_mod, "dispatch_request", _spy)

    replies = await _drive(
        [
            {
                "id": "img-route",
                "method": "render",
                "params": {
                    "tier": "portrait",
                    "positive_prompt": "a knight at dusk",
                    "world": "w",
                    "genre": "g",
                    "subject": "knight",
                },
            }
        ]
    )

    assert calls, (
        "image-tier render did not reach dispatch_request — it is still "
        "handled by the parallel inline path inside _handle_client. "
        "Unify image dispatch through dispatch_request (story 101-7 AC1)."
    )
    assert calls[0]["params"]["tier"] == "portrait"
    assert replies and replies[0]["id"] == "img-route"


@pytest.mark.asyncio
async def test_all_advertised_methods_reachable():
    """AC4 wiring test: every advertised method (minus shutdown, which
    SIGTERMs the process and is covered over the real socket by
    test_daemon_socket_lifecycle) must produce a well-formed JSON-RPC reply
    frame — proving each one is reachable through the single router.

    This is the regression guard for "existing socket integration tests
    must pass unchanged": it pins the wire-protocol reply shape for each
    method. It must stay green before AND after the extraction.
    """
    requests = [
        {"id": "m-ping", "method": "ping", "params": {}},
        {"id": "m-status", "method": "status", "params": {}},
        {"id": "m-embed", "method": "embed", "params": {"text": ""}},
        {"id": "m-warmup", "method": "warm_up", "params": {"worker": "bogus"}},
        {"id": "m-render", "method": "render", "params": {"tier": "bogus"}},
        {"id": "m-unknown", "method": "frobnicate", "params": {}},
    ]
    replies = await _drive(requests)

    by_id = {r["id"]: r for r in replies}
    # Every method got a routed reply — none silently dropped.
    for req in requests:
        assert req["id"] in by_id, (
            f"method {req['method']!r} produced no reply frame — not routed"
        )

    # Spot-check the contract per method (wire protocol unchanged).
    assert by_id["m-ping"]["result"] == {"status": "ok"}
    assert "image" in by_id["m-status"]["result"]
    # Empty embed text is a client-input error → structured error frame.
    assert by_id["m-embed"]["error"]["code"] == "INVALID_REQUEST"
    # Unknown warmup worker fails loud (no silent no-op).
    assert by_id["m-warmup"]["error"]["code"] == "UNKNOWN_WORKER"
    # Unknown render tier fails loud (does not hang, does not EOF).
    assert "error" in by_id["m-render"]
    # Unknown method is rejected explicitly.
    assert by_id["m-unknown"]["error"]["code"] == "UNKNOWN_METHOD"
