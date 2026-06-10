"""Story 101-7 — daemon dispatch extraction (RED).

daemon.py is a ~1,300-LOC god module hosting the socket server, heartbeat
writer, CLI parsing, EmbedWorker, RenderService-shaped render logic,
warmup/shutdown/status, AND dispatch. This story pulls EmbedWorker and the
image-render orchestration (RenderService) into their own modules, leaving
daemon.py as socket lifecycle + routing only.

These tests pin the STRUCTURAL acceptance criteria:
  AC2 — EmbedWorker + RenderService live outside daemon.py; daemon.py is
        socket lifecycle + routing only (LOC budget).
  AC3 — wire protocol / public import surface unchanged: the symbols that
        existing tests and the server consume must remain importable from
        ``sidequest_daemon.media.daemon`` (via re-export after extraction).

They are RED until Dev performs the extraction.

NOTE (TEA design decision, logged as a deviation in the session file):
the story says "media/embed_worker.py, media/render_service.py *or similar*".
These tests fix the canonical module names the story names. If Dev chooses
different filenames, update the two module-path assertions and log the
rename — the *contract under test* is "extracted to its own module", not the
exact basename.
"""

from __future__ import annotations

from pathlib import Path

import sidequest_daemon.media.daemon as daemon_mod

# The maximum line count for daemon.py once it is socket-lifecycle + routing
# only. The story sets the target at "under ~500 LOC"; 550 gives a little
# slack around the "~" without letting the god module survive (it is 1,319
# lines today).
_DAEMON_LOC_BUDGET = 550

_DAEMON_MODULE = "sidequest_daemon.media.daemon"


def test_daemon_module_under_loc_budget():
    """AC2: daemon.py must shrink to socket lifecycle + routing only.

    RED now — the module is ~1,319 lines because it still defines
    EmbedWorker, WorkerPool render orchestration, and the inline image
    dispatch chain.
    """
    source = Path(daemon_mod.__file__)
    loc = sum(1 for _ in source.open(encoding="utf-8"))
    assert loc <= _DAEMON_LOC_BUDGET, (
        f"daemon.py is {loc} LOC; story 101-7 requires it under "
        f"~{_DAEMON_LOC_BUDGET} (socket lifecycle + routing only). "
        f"EmbedWorker and the render-service orchestration must move out."
    )


def test_embed_worker_extracted_to_own_module():
    """AC2: EmbedWorker must no longer be *defined* in daemon.py.

    It stays importable from daemon (AC3 back-compat, see the dedicated
    re-export test) but its ``__module__`` must point at the extracted
    module, not the daemon god module.
    """
    from sidequest_daemon.media.daemon import EmbedWorker

    assert EmbedWorker.__module__ != _DAEMON_MODULE, (
        "EmbedWorker is still defined inside daemon.py "
        f"(__module__={EmbedWorker.__module__!r}); it must be extracted to "
        "its own module."
    )
    assert EmbedWorker.__module__.endswith("embed_worker"), (
        "EmbedWorker should live in an embed_worker module; found "
        f"{EmbedWorker.__module__!r}."
    )


def test_embed_worker_importable_from_dedicated_module():
    """AC2: the extracted module must be the canonical home and expose the
    same class object that daemon re-exports."""
    from sidequest_daemon.media.daemon import EmbedWorker as ReExported
    from sidequest_daemon.media.embed_worker import EmbedWorker as Direct

    assert Direct is ReExported, (
        "daemon.EmbedWorker must be a re-export of "
        "embed_worker.EmbedWorker (same object), not a second definition."
    )


def test_render_service_extracted_to_own_module():
    """AC2: the image-render orchestration must become a RenderService in
    its own module (currently it is the inline if/elif block inside
    ``_handle_client``).

    RED now — neither the class nor the module exists.
    """
    from sidequest_daemon.media.render_service import RenderService

    assert RenderService.__module__ != _DAEMON_MODULE, (
        "RenderService must not be defined inside daemon.py "
        f"(__module__={RenderService.__module__!r})."
    )
    assert RenderService.__module__.endswith("render_service"), (
        "RenderService should live in a render_service module; found "
        f"{RenderService.__module__!r}."
    )


def test_backcompat_public_symbols_still_importable_from_daemon():
    """AC3: extraction must not break the public import surface.

    Existing daemon tests and the server's daemon_client import these names
    from ``sidequest_daemon.media.daemon``. After extraction they must
    remain importable from there (re-exported), or every consumer breaks.

    This is the regression guard for "existing socket integration tests
    must pass unchanged."
    """
    # Symbols that existing test modules import directly from daemon.
    from sidequest_daemon.media.daemon import (  # noqa: F401
        EMBED_TIERS,
        IMAGE_TIERS,
        MUSIC_TIERS,
        EmbedWorker,
        WorkerPool,
        WorkerState,
        _handle_client,
        _run_daemon,
        dispatch_request,
    )

    # Tier frozensets must keep their values — the server routes on them.
    assert "embed" in EMBED_TIERS
    assert "music" in MUSIC_TIERS
    assert "portrait" in IMAGE_TIERS
    assert "landscape" in IMAGE_TIERS


def test_handle_client_signature_unchanged():
    """AC3: ``_handle_client`` is called positionally by the existing
    in-process harness tests as
    ``_handle_client(reader, writer, pool, render_lock, embed_lock)``.
    The extraction must not change that signature.
    """
    import inspect

    params = list(inspect.signature(daemon_mod._handle_client).parameters)
    assert params[:5] == [
        "reader",
        "writer",
        "pool",
        "render_lock",
        "embed_lock",
    ], (
        "_handle_client signature changed — existing harness tests "
        f"(test_compose_error_replies, test_span_scope_per_call) will break: {params}"
    )
