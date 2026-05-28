"""Regression tests for the socket-lifecycle race fixed 2026-04-26 [P1].

Bug summary: the daemon would log "Daemon listening on
/tmp/sidequest-renderer.sock", ``lsof`` would confirm the process held a
unix socket bound at that path, but ``ls`` showed no file on disk. The
inode had been unlinked. Server clients then logged
``render.skipped reason=daemon_unavailable`` because ``connect()`` failed.

Root cause: cleanup paths in ``daemon.py`` (the shutdown ``finally`` block
and the ``send_shutdown`` "stale socket" branch) called
``SOCKET_PATH.unlink()`` unconditionally. Any process running through
``_run_daemon`` that exited before binding — or any ``--shutdown`` invoked
while the listening daemon was still loading models — would unlink the
path that the live daemon had bound to. The kernel kept the bound socket
fd alive so ``lsof`` was happy, but new ``connect()`` calls failed because
the directory entry was gone.

Fix: a module-level ``_owns_socket`` flag set only after a successful
``start_unix_server`` bind, plus a ``_live_daemon_pid()`` liveness probe
on the PID file. Cleanup only proceeds when the current process owns the
bind, or when no other live daemon process is running.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections import namedtuple
from pathlib import Path

import pytest

from sidequest_daemon.media import daemon as daemon_mod

# Socket/PID paths are isolated per-test via the daemon_process fixture (see
# SIDEQUEST_RENDERER_SOCK / SIDEQUEST_RENDERER_PID overrides), so these tests
# never collide with a daemon bound at the shared /tmp path.
_DaemonHandle = namedtuple("_DaemonHandle", ["proc", "sock", "pid"])


async def _read_rpc_response(reader, expected_id: str, timeout: float = 5.0) -> dict:
    """Read lines until the JSON-RPC response for ``expected_id`` arrives,
    skipping daemon server-push frames (e.g. ``{"event": "heartbeat", ...}``,
    which carry no matching ``id``) multiplexed onto the same connection."""
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


def test_owns_socket_flag_starts_false():
    """The bind-ownership flag must start False — a fresh import must not
    grant cleanup rights to the importer."""
    # Re-import in a subprocess to get a clean module state.
    out = subprocess.check_output(
        [
            "python",
            "-c",
            "from sidequest_daemon.media import daemon; "
            "print(daemon._owns_socket)",
        ],
        text=True,
    ).strip()
    assert out == "False"


def test_live_daemon_pid_returns_none_when_no_pid_file(tmp_path, monkeypatch):
    """Liveness probe must return None when PID_PATH does not exist."""
    monkeypatch.setattr(daemon_mod, "PID_PATH", tmp_path / "no-such.pid")
    assert daemon_mod._live_daemon_pid() is None


def test_live_daemon_pid_returns_none_for_dead_pid(tmp_path, monkeypatch):
    """Liveness probe must return None when PID_PATH points to a dead PID."""
    pid_file = tmp_path / "dead.pid"
    # Spawn and reap a short-lived process to get a guaranteed-dead PID.
    proc = subprocess.Popen(["true"])
    proc.wait()
    pid_file.write_text(str(proc.pid))
    monkeypatch.setattr(daemon_mod, "PID_PATH", pid_file)
    assert daemon_mod._live_daemon_pid() is None


def test_live_daemon_pid_returns_none_for_self(tmp_path, monkeypatch):
    """Liveness probe must return None if the PID file points at us — we
    are not 'another live daemon'."""
    pid_file = tmp_path / "self.pid"
    pid_file.write_text(str(os.getpid()))
    monkeypatch.setattr(daemon_mod, "PID_PATH", pid_file)
    assert daemon_mod._live_daemon_pid() is None


def test_live_daemon_pid_detects_running_process(tmp_path, monkeypatch):
    """Liveness probe must return the PID when it points at a live process."""
    pid_file = tmp_path / "alive.pid"
    proc = subprocess.Popen(["sleep", "10"])
    try:
        pid_file.write_text(str(proc.pid))
        monkeypatch.setattr(daemon_mod, "PID_PATH", pid_file)
        assert daemon_mod._live_daemon_pid() == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_live_daemon_pid_handles_garbage(tmp_path, monkeypatch):
    """Liveness probe must return None on unparseable PID file content."""
    pid_file = tmp_path / "garbage.pid"
    pid_file.write_text("not-a-number\n")
    monkeypatch.setattr(daemon_mod, "PID_PATH", pid_file)
    assert daemon_mod._live_daemon_pid() is None


@pytest.fixture()
def daemon_process(tmp_path, monkeypatch):
    """Boot the real daemon in a subprocess (no warmup → fast) on an
    ISOLATED socket/PID path. Tear it down after the test.

    Socket isolation: SIDEQUEST_RENDERER_SOCK / SIDEQUEST_RENDERER_PID pin
    the daemon's paths under ``tmp_path`` so the test never collides with a
    daemon bound at the shared ``/tmp`` location (e.g. a running ``just up``
    dev daemon). The in-process ``daemon_mod`` globals are monkeypatched to
    the same paths so in-process helpers (``send_shutdown``) and the test
    assertions agree with the subprocess.

    Critical: ``HOME`` is overridden to ``tmp_path`` so the daemon's
    handshake write (``~/.sidequest/daemon-output-dir``) lands in test
    scope. Without this, every test run leaves a stale handshake
    pointing at a long-cleaned-up pytest tmpdir in the user's real
    ``~/.sidequest/``, which then steers the running dev server's
    ``/renders`` mount onto a non-existent directory the next time
    ``create_app`` is reloaded — exactly the failure that hid behind
    the playtest 2026-04-26 "scrapbook images stopped" report.
    """
    # The socket must live on a SHORT path — the AF_UNIX sun_path limit
    # (~104 chars on macOS) is shorter than a nested pytest tmp_path. A small
    # mkdtemp dir keeps it well under the limit while staying isolated.
    sockdir = Path(tempfile.mkdtemp(prefix="sqd_"))
    sock = sockdir / "r.sock"
    pid = sockdir / "r.pid"
    monkeypatch.setattr(daemon_mod, "SOCKET_PATH", sock)
    monkeypatch.setattr(daemon_mod, "PID_PATH", pid)
    env = {
        **os.environ,
        "SIDEQUEST_GENRE_PACKS": str(tmp_path),
        "HOME": str(tmp_path),
        "SIDEQUEST_RENDERER_SOCK": str(sock),
        "SIDEQUEST_RENDERER_PID": str(pid),
    }
    # --no-warmup keeps this test under ~3s. The race we are guarding
    # against is socket-lifecycle, not model-loading.
    proc = subprocess.Popen(
        [
            "sidequest-renderer",
            "--no-warmup",
            "--output-dir",
            str(tmp_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for the socket file to appear.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if sock.exists():
            break
        time.sleep(0.1)
    else:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"daemon never created {sock}\n"
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )
    yield _DaemonHandle(proc=proc, sock=sock, pid=pid)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    shutil.rmtree(sockdir, ignore_errors=True)


def test_socket_file_present_after_bind(daemon_process):
    """The core fail-mode regression: after the daemon logs that it is
    listening, the socket file must actually exist on disk and be a
    socket node — not unlinked out from under the bound fd."""
    sock = daemon_process.sock
    assert sock.exists(), (
        f"daemon bound the socket but the file is missing from {sock} — "
        "this is the exact failure mode of the 2026-04-26 P1 bug"
    )
    # And it must be a socket, not a regular file.
    assert sock.is_socket(), (
        f"{sock} exists but is not a socket node (mode={sock.stat().st_mode:o})"
    )


@pytest.mark.asyncio
async def test_socket_survives_warmup_helper_invocation(daemon_process):
    """If a second invocation of the renderer entry point — for instance a
    warmup helper, a stray ``--shutdown`` racing the live daemon, or any
    future tool that imports ``daemon.py`` — runs while a real daemon is
    bound, the live daemon's socket file MUST remain on disk and clients
    MUST still be able to ``connect()`` to it.

    Without the ``_owns_socket`` guard + ``_live_daemon_pid()`` probe, the
    second invocation's cleanup paths would unlink the live socket and
    leave the system in the exact state described in the bug report:
    process holds bound fd, file gone from disk, clients fail to connect.
    """
    sock = daemon_process.sock
    # Simulate the racing helper: invoke ``--shutdown`` against the live
    # daemon, but kill it immediately so it can never actually shutdown
    # cleanly. This exercises the ``send_shutdown`` cleanup branch. The
    # helper must target the SAME isolated socket the daemon bound, so it
    # inherits the SIDEQUEST_RENDERER_SOCK/PID overrides via env.
    helper = subprocess.Popen(
        ["sidequest-renderer", "--status"],
        env={
            **os.environ,
            "SIDEQUEST_RENDERER_SOCK": str(sock),
            "SIDEQUEST_RENDERER_PID": str(daemon_process.pid),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    helper.wait(timeout=10)

    # The live daemon's socket must still be on disk and connectable.
    assert sock.exists(), (
        "racing helper unlinked the live daemon's socket — _owns_socket "
        "guard or _live_daemon_pid() probe failed"
    )
    reader, writer = await asyncio.open_unix_connection(str(sock))
    try:
        request = json.dumps({"id": "lifecycle", "method": "ping", "params": {}})
        writer.write((request + "\n").encode())
        await writer.drain()
        response = await _read_rpc_response(reader, "lifecycle")
        assert response["result"]["status"] == "ok"
    finally:
        writer.close()
        await writer.wait_closed()


def test_send_shutdown_refuses_to_unlink_live_daemons_socket(
    daemon_process,
):
    """``send_shutdown`` must NOT unlink the socket file when the PID file
    points at a live daemon, even if the connect attempt fails. Before the
    fix, any ``ConnectionRefusedError`` (e.g. mid-startup race) would
    unlink the path the live daemon had bound to."""
    # Sanity: the daemon is up and the PID file points at it.
    assert daemon_process.pid.exists()
    pid = int(daemon_process.pid.read_text().strip())
    os.kill(pid, 0)  # raises if dead

    # Force ``send_shutdown`` into the cleanup branch by monkey-patching
    # ``open_unix_connection`` to raise ``ConnectionRefusedError`` even
    # though the daemon is alive. This simulates the race in the bug
    # report where the helper saw a transient connect failure.
    import sidequest_daemon.media.daemon as d

    async def _raise_refused(_path):
        raise ConnectionRefusedError("simulated mid-startup race")

    original = asyncio.open_unix_connection
    asyncio.open_unix_connection = _raise_refused  # type: ignore[assignment]
    try:
        asyncio.run(d.send_shutdown())
    finally:
        asyncio.open_unix_connection = original  # type: ignore[assignment]

    # The socket file MUST still be on disk — the live daemon owns it.
    assert daemon_process.sock.exists(), (
        "send_shutdown unlinked the live daemon's socket despite the "
        "PID file pointing at a running process — the _live_daemon_pid() "
        "guard failed"
    )
    assert daemon_process.pid.exists(), (
        "send_shutdown unlinked the live daemon's PID file"
    )
