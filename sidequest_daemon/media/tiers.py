"""Tier routing constants for the renderer daemon.

Leaf module (no intra-package imports) so ``daemon.py``, ``worker_pool.py``,
and ``render_service.py`` can all import the tier frozensets without forming an
import cycle. ``daemon.py`` re-exports these names for back-compat — existing
consumers import ``IMAGE_TIERS`` etc. from ``sidequest_daemon.media.daemon``.
"""

from __future__ import annotations

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

# Valid warmup targets for the `warm_up` RPC (`worker=`) and the `--warmup=` CLI
# flag. "all" warms every worker; "image"/"embed" warm one. Any other value is
# rejected loudly — a silent no-op here would let the daemon report "warm" while
# serving cold (No Silent Fallbacks). The retired "flux" alias is deliberately
# absent (ADR-070; story 101-5).
WARMUP_TARGETS = frozenset({"all", "image", "embed"})


def _validate_warmup_target(target: str) -> None:
    """Raise ``ValueError`` if ``target`` is not a recognized warmup worker.

    Fail-loud guard for the ``--warmup`` CLI flag. A bad value (a typo, or the
    retired ``flux`` alias) must crash startup rather than let the daemon log
    "Models warm and ready" while serving cold (No Silent Fallbacks).
    """
    if target not in WARMUP_TARGETS:
        raise ValueError(
            f"Unknown warmup target {target!r}; valid: {sorted(WARMUP_TARGETS)}"
        )
