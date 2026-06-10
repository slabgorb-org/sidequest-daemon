"""Story 101-1 — Align daemon renderer models with the server contract.

Epic 101 (Split-Brain Remediation) confirmed ONE live drift at the
deliberately-duplicated daemon<->server renderer-model seam:

1. Daemon ``RenderTier`` still carries ``CARTOGRAPHY`` (removed server-side
   2026-04-28) plus its two zimage tier configs (turbo + high-fidelity tables).
2. Daemon ``StageCue.camera`` is typed ``CameraPreset`` enum; the server types
   it plain ``str``. An arbitrary camera string the server is free to send is
   rejected by the stricter daemon model.
3. No contract test holds the seam, so the drift recurred silently.

These tests are the RED phase for that alignment. They fail against the current
(drifted) daemon models and pass once Dev:
  - removes ``RenderTier.CARTOGRAPHY`` + the two ``ZIMAGE_*_TIER_CONFIGS``
    cartography entries (and the now-dead "cartography" routing/R2 strings),
  - retypes ``StageCue.camera`` to ``str | None`` matching the server.

The cross-repo contract test (``test_render_tier_members_match_server`` and
``test_stage_cue_fields_match_server``) loads the *real* server model module
from the co-located ``sidequest-server`` repo and asserts the shared subset is
identical — so future drift on EITHER side fails loudly here. Per the project's
No Silent Fallbacks rule, a missing server repo is a hard failure, not a skip:
a skipped contract test is a silent fallback that gives false confidence the
seam is held.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import get_args

from sidequest_daemon.renderer.models import RenderTier, StageCue

# ---------------------------------------------------------------------------
# AC1 — Daemon RenderTier no longer carries CARTOGRAPHY
# ---------------------------------------------------------------------------


def test_render_tier_has_no_cartography_member() -> None:
    """The CARTOGRAPHY enum member is gone (removed server-side 2026-04-28)."""
    assert not hasattr(RenderTier, "CARTOGRAPHY")


def test_render_tier_has_no_cartography_value() -> None:
    """No surviving tier carries the ``"cartography"`` wire value."""
    values = {tier.value for tier in RenderTier}
    assert "cartography" not in values


def test_render_tier_retains_live_tiers() -> None:
    """Over-deletion guard: the five surviving image tiers must remain.

    Removing CARTOGRAPHY must not collaterally drop a live tier. These five
    are shared with the server contract (FOG_OF_WAR included).
    """
    values = {tier.value for tier in RenderTier}
    for live in (
        "scene_illustration",
        "portrait",
        "portrait_square",
        "landscape",
        "text_overlay",
        "fog_of_war",
    ):
        assert live in values, f"live tier {live!r} was lost"


# ---------------------------------------------------------------------------
# AC1 — The two zimage tier configs for cartography are removed
# ---------------------------------------------------------------------------


def test_turbo_zimage_config_has_no_cartography() -> None:
    """The turbo ``ZIMAGE_TIER_CONFIGS`` table no longer maps cartography."""
    from sidequest_daemon.media.zimage_config import ZIMAGE_TIER_CONFIGS

    config_values = {tier.value for tier in ZIMAGE_TIER_CONFIGS}
    assert "cartography" not in config_values


def test_high_fidelity_zimage_config_has_no_cartography() -> None:
    """The high-fidelity ``ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS`` table too."""
    from sidequest_daemon.media.zimage_config import (
        ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS,
    )

    config_values = {tier.value for tier in ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS}
    assert "cartography" not in config_values


# ---------------------------------------------------------------------------
# AC1 — Daemon-internal consistency: no dead "cartography" routing strings
# ---------------------------------------------------------------------------


def test_image_tier_routing_has_no_dead_tiers() -> None:
    """Every ``IMAGE_TIERS`` routing string must be a real RenderTier value.

    Catches the leftover ``"cartography"`` string in ``daemon.IMAGE_TIERS``
    after the enum member is removed — a tier routed to the image worker that
    no longer exists is a half-removed feature (No Silent Fallbacks).
    """
    from sidequest_daemon.media.daemon import IMAGE_TIERS

    valid = {tier.value for tier in RenderTier}
    orphans = {tier for tier in IMAGE_TIERS if tier not in valid}
    assert not orphans, f"IMAGE_TIERS routes non-existent tiers: {orphans}"


def test_r2_kind_mapping_has_no_dead_tiers() -> None:
    """Every ``_TIER_TO_R2_KIND`` key must be a real RenderTier value.

    Catches the leftover ``"cartography"`` key in the worker's R2-kind table.
    """
    from sidequest_daemon.media.workers.zimage_mlx_worker import _TIER_TO_R2_KIND

    valid = {tier.value for tier in RenderTier}
    orphans = {tier for tier in _TIER_TO_R2_KIND if tier not in valid}
    assert not orphans, f"_TIER_TO_R2_KIND maps non-existent tiers: {orphans}"


# ---------------------------------------------------------------------------
# AC2 — StageCue.camera is plain str, matching the server
# ---------------------------------------------------------------------------


def test_stage_cue_camera_annotation_is_optional_str() -> None:
    """``StageCue.camera`` must be typed ``str | None``, not a CameraPreset enum.

    The server types it ``str | None``; the daemon's CameraPreset typing is the
    drift. Inspect the resolved pydantic field annotation.
    """
    annotation = StageCue.model_fields["camera"].annotation
    # Optional[str] resolves to ``str | None`` -> args (str, NoneType)
    assert str in get_args(annotation), (
        f"StageCue.camera annotation is {annotation!r}, expected str | None"
    )
    assert annotation is not None


def test_stage_cue_accepts_arbitrary_camera_string() -> None:
    """An arbitrary camera string the server may send must be accepted.

    With CameraPreset enum typing, a non-member string raises ValidationError.
    With plain ``str`` typing (the server contract) it is accepted verbatim.
    This is the behavioral heart of the camera alignment.
    """
    cue = StageCue(
        tier=RenderTier.SCENE_ILLUSTRATION,
        subject="a camera angle the daemon enum never enumerated",
        camera="orbital_dutch_tilt_270",
    )
    assert cue.camera == "orbital_dutch_tilt_270"
    assert isinstance(cue.camera, str)


def test_stage_cue_camera_remains_optional() -> None:
    """camera stays optional (defaults to None) after the retype."""
    cue = StageCue(tier=RenderTier.PORTRAIT, subject="rux")
    assert cue.camera is None


# ---------------------------------------------------------------------------
# AC3 — Cross-repo contract: daemon subset == server contract
# ---------------------------------------------------------------------------


def _load_server_renderer_models() -> ModuleType:
    """Load the *real* server ``renderer.models`` from the co-located repo.

    The daemon and server are sibling subrepos under the orchestrator checkout.
    The module depends only on pydantic + stdlib, so it loads in isolation in
    the daemon test env without importing the whole server package.

    Resolution order: ``SIDEQUEST_SERVER_ROOT`` env override, else the sibling
    ``../sidequest-server`` next to this daemon repo. A missing server repo is
    a HARD FAILURE (No Silent Fallbacks) — a skipped contract test silently
    stops holding the seam.
    """
    override = os.environ.get("SIDEQUEST_SERVER_ROOT")
    if override:
        server_root = Path(override).resolve()
    else:
        # tests/ -> sidequest-daemon -> <orchestrator root> -> sidequest-server
        server_root = (
            Path(__file__).resolve().parents[2] / "sidequest-server"
        )

    models_path = server_root / "sidequest" / "renderer" / "models.py"
    if not models_path.is_file():
        raise FileNotFoundError(
            "Cannot hold the daemon<->server renderer contract: server models "
            f"not found at {models_path}. Set SIDEQUEST_SERVER_ROOT to the "
            "sidequest-server repo root, or run from the orchestrator checkout "
            "where both repos are co-located."
        )

    spec = importlib.util.spec_from_file_location(
        "sidequest_server_renderer_models_contract", models_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {models_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_tier_members_match_server() -> None:
    """The daemon RenderTier value set must equal the server's exactly.

    This is the drift guard: if either repo adds/removes a tier without the
    other, this fails loudly.
    """
    server = _load_server_renderer_models()
    daemon_values = {tier.value for tier in RenderTier}
    server_values = {tier.value for tier in server.RenderTier}
    assert daemon_values == server_values, (
        "RenderTier drift between repos — "
        f"daemon-only={daemon_values - server_values}, "
        f"server-only={server_values - daemon_values}"
    )


def test_stage_cue_fields_match_server() -> None:
    """The daemon StageCue field NAME set must equal the server's exactly.

    Field order may differ (the wire is keyed by name), but the set of fields
    in the shared contract must be identical.
    """
    server = _load_server_renderer_models()
    daemon_fields = set(StageCue.model_fields)
    server_fields = set(server.StageCue.model_fields)
    assert daemon_fields == server_fields, (
        "StageCue field drift between repos — "
        f"daemon-only={daemon_fields - server_fields}, "
        f"server-only={server_fields - daemon_fields}"
    )


def test_stage_cue_camera_type_matches_server() -> None:
    """The daemon StageCue.camera annotation must match the server's str typing."""
    server = _load_server_renderer_models()
    daemon_camera = StageCue.model_fields["camera"].annotation
    server_camera = server.StageCue.model_fields["camera"].annotation
    assert daemon_camera == server_camera, (
        f"StageCue.camera type drift — daemon={daemon_camera!r}, "
        f"server={server_camera!r}"
    )
