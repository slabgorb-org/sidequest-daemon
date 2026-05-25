"""Smoke test — verify the daemon starts and responds to ping."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


async def _read_rpc_response(reader, expected_id: str, timeout: float = 5.0) -> dict:
    """Read lines until the JSON-RPC response for ``expected_id`` arrives.

    The daemon multiplexes server-push events (e.g. ``{"event": "heartbeat",
    ...}``, which carry no ``id``) onto the same connection as RPC replies.
    A correct client skips push frames and matches its own request ``id``.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise AssertionError(f"no RPC response for id={expected_id!r} within {timeout}s")
        line = await asyncio.wait_for(reader.readline(), timeout=remaining)
        if not line:
            raise AssertionError("connection closed before RPC response arrived")
        msg = json.loads(line)
        if msg.get("id") == expected_id:
            return msg
        # Otherwise a push event (heartbeat etc.) — keep reading.


@pytest.fixture()
def daemon_socket(tmp_path):
    """Start the daemon in a subprocess on an ISOLATED socket, yield the
    socket path, then shut it down.

    The socket/PID paths are pinned under ``tmp_path`` via the
    SIDEQUEST_RENDERER_SOCK / SIDEQUEST_RENDERER_PID overrides so the test
    binds and talks to its OWN daemon instead of colliding with whatever
    daemon happens to be bound at the shared /tmp path (e.g. a running
    ``just up`` dev daemon). ``--no-warmup`` keeps boot fast — ping does not
    need models loaded.
    """
    # The socket must live on a SHORT path — the AF_UNIX sun_path limit
    # (~104 chars on macOS) is shorter than a nested pytest tmp_path. A small
    # mkdtemp dir keeps it well under the limit while staying isolated.
    sockdir = Path(tempfile.mkdtemp(prefix="sqd_"))
    sock = sockdir / "r.sock"
    env = {
        **os.environ,
        "SIDEQUEST_GENRE_PACKS": str(tmp_path),
        "SIDEQUEST_RENDERER_SOCK": str(sock),
        "SIDEQUEST_RENDERER_PID": str(sockdir / "r.pid"),
    }
    proc = subprocess.Popen(
        ["sidequest-renderer", "--no-warmup", "--output-dir", str(tmp_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait for the isolated socket to appear.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if sock.exists():
                break
            time.sleep(0.1)
        else:
            proc.kill()
            out, err = proc.communicate(timeout=5)
            pytest.fail(
                f"daemon never created {sock}\nstdout: {out.decode()}\nstderr: {err.decode()}"
            )
        yield sock
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        shutil.rmtree(sockdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_daemon_ping(daemon_socket):
    """Send a ping over the Unix socket and verify the response."""
    reader, writer = await asyncio.open_unix_connection(str(daemon_socket))
    try:
        request = json.dumps({"id": "smoke", "method": "ping", "params": {}})
        writer.write((request + "\n").encode())
        await writer.drain()

        response = await _read_rpc_response(reader, "smoke")
        assert response["result"]["status"] == "ok"
    finally:
        writer.close()
        await writer.wait_closed()
