"""Story 101-5 — retiring the daemon 'flux' alias must fail LOUD, not silent.

When the ``warm_up_flux()`` back-compat alias and the ``'flux'`` dispatch arm
were removed, an unknown warmup target (the retired ``flux``, or any typo) fell
through both dispatch arms and was treated as success: the ``warm_up`` RPC
returned ``{"status": "warm", "workers": {}}`` and the ``--warmup=`` CLI path
logged "Models warm and ready" against an unloaded worker. That is the exact
silent-no-op the No-Silent-Fallbacks rule forbids.

These tests pin the loud behavior at both dispatch sites:
- RPC ``warm_up`` with an unknown ``worker`` → ``UNKNOWN_WORKER`` JSON-RPC error,
  and the pool's warmup methods are never invoked.
- CLI ``--warmup`` validation (``_validate_warmup_target``) → ``ValueError``,
  and the guard is wired into ``_run_daemon``.
A valid target still warms, so the guard does not over-reject.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from sidequest_daemon.media.daemon import (
    WARMUP_TARGETS,
    _handle_client,
    _run_daemon,
    _validate_warmup_target,
)


class _RecordingWriter:
    """Minimal stand-in for ``asyncio.StreamWriter`` — records sent bytes."""

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


class _SpyPool:
    """Pool whose warmup methods record invocation. Image/embed warmup must
    NOT be called when the requested worker is unknown."""

    def __init__(self) -> None:
        self.image_called = False
        self.embed_called = False

    def warm_up_image(self) -> dict:
        self.image_called = True
        return {"worker": "image", "status": "warm", "warmup_ms": 0}

    def warm_up_embed(self) -> dict:
        self.embed_called = True
        return {"worker": "embed", "status": "warm", "warmup_ms": 0}

    def status(self) -> dict:
        return {}


async def _run_one_request(request: dict, pool: _SpyPool) -> list[dict]:
    reader = asyncio.StreamReader()
    reader.feed_data((json.dumps(request) + "\n").encode())
    reader.feed_eof()
    writer = _RecordingWriter()
    await _handle_client(
        reader,
        writer,  # type: ignore[arg-type]
        pool,  # type: ignore[arg-type]
        asyncio.Lock(),
        asyncio.Lock(),
    )
    return writer.replies


@pytest.mark.asyncio
async def test_warm_up_rpc_rejects_unknown_worker_loudly():
    """worker='flux' (retired) must return UNKNOWN_WORKER, not silent success."""
    pool = _SpyPool()
    replies = await _run_one_request(
        {"id": "rpc-flux", "method": "warm_up", "params": {"worker": "flux"}},
        pool,
    )
    assert len(replies) == 1, f"expected exactly one reply, got {replies}"
    reply = replies[0]
    assert reply["id"] == "rpc-flux"
    assert "error" in reply, f"expected an error frame, got {reply}"
    assert reply["error"]["code"] == "UNKNOWN_WORKER"
    assert "flux" in reply["error"]["message"]
    # Critical: nothing was warmed — no silent partial success.
    assert pool.image_called is False
    assert pool.embed_called is False


@pytest.mark.asyncio
async def test_warm_up_rpc_accepts_known_worker():
    """worker='image' still warms — the guard must not over-reject."""
    pool = _SpyPool()
    replies = await _run_one_request(
        {"id": "rpc-image", "method": "warm_up", "params": {"worker": "image"}},
        pool,
    )
    assert len(replies) == 1
    reply = replies[0]
    assert "error" not in reply, f"valid worker should succeed, got {reply}"
    assert reply["result"]["status"] == "warm"
    assert "image" in reply["result"]["workers"]
    assert pool.image_called is True


def test_validate_warmup_target_rejects_flux():
    """The retired 'flux' value (and typos) raise ValueError at the CLI guard."""
    with pytest.raises(ValueError, match="flux"):
        _validate_warmup_target("flux")
    with pytest.raises(ValueError):
        _validate_warmup_target("imag")  # typo


@pytest.mark.parametrize("target", sorted(WARMUP_TARGETS))
def test_validate_warmup_target_accepts_known(target: str):
    """Every recognized target passes the guard without raising."""
    _validate_warmup_target(target)  # must not raise


def test_run_daemon_wires_the_warmup_guard():
    """Wiring: _run_daemon actually calls the loud guard (not dead code)."""
    src = inspect.getsource(_run_daemon)
    assert "_validate_warmup_target" in src, (
        "_run_daemon must call _validate_warmup_target so an unknown "
        "--warmup value fails loud instead of starting cold"
    )


def test_flux_is_not_a_valid_warmup_target():
    """ADR-070 / story 101-5: the 'flux' alias is fully retired."""
    assert "flux" not in WARMUP_TARGETS
    assert WARMUP_TARGETS == frozenset({"all", "image", "embed"})
