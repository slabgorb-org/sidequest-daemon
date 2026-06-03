"""Story 78-3 — Daemon deferred observability: wire-or-cut guards.

Both deferred-feature exports audited by `sq-wire-it` (2026-06-02) were
decided **CUT** (Operator call, 2026-06-03), each with an ADR deferral note:

- ``start_periodic_heartbeat`` (ADR-131, ``live``) — honest wiring needed a
  brand-new active-writer registry + broadcast emit + task lifecycle, busting
  the 2-pt budget; and idle liveness is *not* currently broken (the server's
  ``DaemonClient.heartbeat_listener`` reconnects ~every 15s, 4x margin under the
  60s unresponsive threshold). Cut, with an ADR-131 Consequences deferral note.
- ``detect_gpu`` / ``GpuInfo`` (ADR-046, ``retired``) — the ``ModelMemoryManager``
  coordinator it was designed to feed was deleted in ``5118d6c`` (2026-05-10).
  No live consumer. Cut, with a one-line ADR-046 note.

These are **removal-regression guards**: they FAIL now (the orphaned code still
exists — RED) and pass once Dev deletes the exports + their orphaned isolation
tests (GREEN). They also guard against *over*-deletion: the per-connection
heartbeat path (the liveness that actually works today) must survive.

AC3 (live-path span assertion) is vacuously satisfied — nothing is wired.

The ADR-note obligation of AC1/AC2 is verified by code review, NOT here: the
ADRs live in the orchestrator repo (``docs/adr/``), and a daemon-repo test
reaching ``../docs/adr`` is fragile and wrong-repo. See the TEA deviation log.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent


def _assert_sibling_test_lacks(filename: str, *symbols: str, hint: str) -> None:
    """Assert a sibling test file no longer references any of ``symbols``.

    Enforces "Delete Dead Code in the Same PR" — a test pointed at a deleted
    export must be removed alongside it. Reports *which* symbol survived so a
    failure says exactly what is still orphaned.
    """
    source = (_TESTS_DIR / filename).read_text()
    survivors = [s for s in symbols if s in source]
    assert not survivors, (
        f"orphaned reference(s) {survivors} still in {filename} — {hint}"
    )


# ---------------------------------------------------------------------------
# AC1 — start_periodic_heartbeat is CUT (ADR-131 deferral note)
# ---------------------------------------------------------------------------


def test_periodic_heartbeat_export_removed() -> None:
    """AC1 (cut path): the unwired periodic emitter must be deleted.

    A self-documented ``NOT YET WIRED`` coroutine is the epic's most
    expensive defect class — it makes the daemon *look* more observable
    than it is. Cutting it removes the trap.
    """
    daemon_mod = importlib.import_module("sidequest_daemon.media.daemon")
    assert not hasattr(daemon_mod, "start_periodic_heartbeat"), (
        "AC1: start_periodic_heartbeat was decided CUT (ADR-131 deferral). "
        "The unwired periodic emitter must be deleted, not left as a "
        "'NOT YET WIRED' stub."
    )


def test_default_heartbeat_interval_constant_removed() -> None:
    """AC1 (cut path): the interval constant is orphaned by the cut.

    ``DEFAULT_HEARTBEAT_INTERVAL_SECONDS`` is referenced *only* as the
    default value of ``start_periodic_heartbeat``'s ``interval_seconds``
    parameter. Once the coroutine is gone, the constant has zero
    consumers and must go too ("Delete Dead Code in the Same PR").
    """
    daemon_mod = importlib.import_module("sidequest_daemon.media.daemon")
    assert not hasattr(daemon_mod, "DEFAULT_HEARTBEAT_INTERVAL_SECONDS"), (
        "AC1: DEFAULT_HEARTBEAT_INTERVAL_SECONDS is orphaned once "
        "start_periodic_heartbeat is cut — delete it in the same PR."
    )


def test_orphaned_periodic_heartbeat_isolation_test_removed() -> None:
    """AC1 (cut path): the orphaned AC2 isolation test must be deleted.

    ``tests/test_heartbeat_emit.py`` currently drives the periodic emitter
    via ``start_periodic_heartbeat`` (the AC2 isolation test). A test
    pointed at deleted code is forbidden — "Do NOT preserve a test pointed
    at deleted code." This guards that the same PR removes it.

    The *other* heartbeat tests in that file (per-connection accept/render
    heartbeats) must remain — see the preservation guard below.
    """
    _assert_sibling_test_lacks(
        "test_heartbeat_emit.py",
        "start_periodic_heartbeat",
        hint="delete test_idle_daemon_emits_periodic_ready_heartbeat in the same PR (AC1).",
    )


# ---------------------------------------------------------------------------
# AC2 — detect_gpu / GpuInfo is CUT (ADR-046 deferral note)
# ---------------------------------------------------------------------------


def test_gpu_detect_module_removed() -> None:
    """AC2 (cut path): the orphaned GPU-detection module must be deleted.

    ADR-046 is ``retired`` and the ``ModelMemoryManager`` coordinator that
    ``detect_gpu`` was meant to feed was deleted (5118d6c, 2026-05-10).
    With no live consumer, the standalone ``gpu.detect`` diagnostic span
    follows the coordinator out. The whole ``gpu_detect.py`` module
    (``detect_gpu`` + ``GpuInfo`` + ``GpuBackend``) is removed.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sidequest_daemon.media.gpu_detect")


def test_orphaned_gpu_detect_span_test_removed() -> None:
    """AC2 (cut path): the orphaned GPU span test must be deleted.

    ``tests/test_otel_spans.py`` currently imports ``detect_gpu`` inside
    ``TestGpuDetectSpan``. Once the module is gone those tests can only
    error on import — delete the class (and its section header) in the
    same PR. The Z-Image worker span tests in that file stay.
    """
    _assert_sibling_test_lacks(
        "test_otel_spans.py",
        "gpu_detect",
        "detect_gpu",
        hint="remove TestGpuDetectSpan and its section header in the same PR (AC2).",
    )


# ---------------------------------------------------------------------------
# Over-deletion guard — the per-connection heartbeat path is NOT in scope
# ---------------------------------------------------------------------------


def test_per_connection_heartbeat_path_preserved() -> None:
    """Collateral guard: the working liveness path must survive the cut.

    Cutting the *periodic* emitter must not touch the *per-connection*
    heartbeat machinery that keeps liveness fresh today:
    ``_write_heartbeat`` (emitted on connection-accept and render/embed
    lock transitions) and its ``_make_heartbeat`` payload builder. This
    guard is GREEN now and must stay GREEN — it fails only if Dev deletes
    too much.
    """
    daemon_mod = importlib.import_module("sidequest_daemon.media.daemon")
    assert hasattr(daemon_mod, "_write_heartbeat"), (
        "Over-deletion: _write_heartbeat (per-connection liveness) must "
        "survive — only the periodic emitter is cut."
    )
    assert hasattr(daemon_mod, "_make_heartbeat"), (
        "Over-deletion: _make_heartbeat (heartbeat payload builder, used "
        "by the per-connection path) must survive."
    )
    assert hasattr(daemon_mod, "_handle_client"), (
        "Over-deletion: _handle_client must survive — the cut is exports "
        "only, not the connection handler."
    )
