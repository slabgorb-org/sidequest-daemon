"""Z-Image tier configuration — maps RenderTier to generation parameters.

Importable by the main package for tests and wiring. The worker subprocess
duplicates this table (it cannot import from sidequest_daemon).

Migrated to Z-Image Turbo (Tongyi-MAI/Z-Image-Turbo) 2026-04-26 per S4-PERF
investigation. Turbo is LCM-distilled — `supports_guidance=False` in mflux's
ModelConfig, so guidance is fixed at 0.0 (CFG is a no-op for the distilled
model). Step count drops from 20 → 8, yielding the targeted ~4× speedup
(~108s → ~30s) for the chargen render path.

Story 45-38 (2026-04-30) added a parallel **high-fidelity** table for
genre-pack pre-gen using base Z-Image 1.0 at 20 steps + CFG 4.0. The two
tables coexist; ``get_zimage_config(tier, fidelity)`` is the single lookup
point for callers (composer, worker, OTEL spans).

Default flipped to **high_fidelity** 2026-05-02 — painterly base Z-Image is
the new floor; Turbo is now opt-in via explicit ``fidelity="turbo"`` or
``SIDEQUEST_DAEMON_FIDELITY=turbo``. Trades ~30s → ~108s per render for
quality. The streaming-narration "narrator considers" interstitial covers
the longer wall-clock during in-session renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sidequest_daemon.renderer.models import RenderTier

# Model identifier passed to mflux's ModelConfig.from_name.
# `z-image-turbo` is an mflux alias for `Tongyi-MAI/Z-Image-Turbo`.
ZIMAGE_MODEL_VARIANT: str = "z-image-turbo"

# Base Z-Image 1.0 alias — non-distilled, supports CFG, slower but more
# painterly at 20 steps. Used for genre-pack pre-gen (picker portraits,
# POI landscapes) where wall-clock is not the constraint.
ZIMAGE_BASE_MODEL_VARIANT: str = "z-image"

# Quantization level for ZImage construction. 8-bit is the README-default
# Turbo preset and brings VRAM/cache footprint down without measurable
# quality loss on Apple Silicon.
ZIMAGE_QUANTIZE: int = 8

# Turbo is distilled — CFG is disabled on the model side. Encoded here as
# 0.0 so the value is recorded in OTEL spans and tier configs without
# claiming a guidance scale that the model does not honor.
_TURBO_GUIDANCE: float = 0.0

# 8-step preset across all tiers. Turbo's quality cliff is roughly at
# steps < 6; 8 is a safe production setting per mflux's own README example
# (`--steps 9 --quantize 8`).
_TURBO_STEPS: int = 8

# Base Z-Image 1.0 settings for the high-fidelity tier. Story 45-38: the
# Draw Things reference output (~/Desktop/0_painted_sci_fi_concept_art_*)
# was generated at exactly these values. Locked here so the Coyote Star
# picker portrait regen is reproducible.
_HIGH_FIDELITY_STEPS: int = 20
_HIGH_FIDELITY_GUIDANCE: float = 4.0


# Allowed fidelity values. ``Fidelity`` is a Literal alias rather than an
# Enum so it serializes naturally over JSON-RPC (the daemon's request
# format) without an extra encode/decode step. ``VALID_FIDELITIES`` is the
# runtime tuple — derived from the Literal so the type annotation and the
# runtime allowlist cannot drift apart (Story 45-39).
Fidelity = Literal["turbo", "high_fidelity"]
VALID_FIDELITIES: tuple[Fidelity, ...] = ("turbo", "high_fidelity")


@dataclass(frozen=True)
class ZImageTierConfig:
    """Z-Image generation parameters for a specific render tier.

    ``supersample_factor`` controls optional supersampling before encode:
    the generator renders at ``width * factor`` × ``height * factor`` and
    the worker Lanczos-downscales the result back to ``(width, height)``
    before saving/uploading.  Factor 1 is a true no-op (no resize step is
    applied).  Any other positive integer multiplies both axes.  A value
    ≤ 0 or non-integer is rejected loudly at render time (No Silent
    Fallbacks).
    """

    steps: int
    guidance: float
    width: int
    height: int
    model_variant: str
    supersample_factor: int = 1


ZIMAGE_TIER_CONFIGS: dict[RenderTier, ZImageTierConfig] = {
    RenderTier.SCENE_ILLUSTRATION: ZImageTierConfig(
        steps=_TURBO_STEPS,
        guidance=_TURBO_GUIDANCE,
        width=1024,
        height=768,
        model_variant=ZIMAGE_MODEL_VARIANT,
    ),
    RenderTier.PORTRAIT: ZImageTierConfig(
        steps=_TURBO_STEPS,
        guidance=_TURBO_GUIDANCE,
        width=768,
        height=1024,
        model_variant=ZIMAGE_MODEL_VARIANT,
    ),
    RenderTier.PORTRAIT_SQUARE: ZImageTierConfig(
        steps=_TURBO_STEPS,
        guidance=_TURBO_GUIDANCE,
        width=1024,
        height=1024,
        model_variant=ZIMAGE_MODEL_VARIANT,
    ),
    RenderTier.LANDSCAPE: ZImageTierConfig(
        steps=_TURBO_STEPS,
        guidance=_TURBO_GUIDANCE,
        width=1024,
        height=768,
        model_variant=ZIMAGE_MODEL_VARIANT,
    ),
    RenderTier.TEXT_OVERLAY: ZImageTierConfig(
        steps=_TURBO_STEPS,
        guidance=_TURBO_GUIDANCE,
        width=768,
        height=512,
        model_variant=ZIMAGE_MODEL_VARIANT,
    ),
    RenderTier.FOG_OF_WAR: ZImageTierConfig(
        steps=_TURBO_STEPS,
        guidance=_TURBO_GUIDANCE,
        width=1024,
        height=1024,
        model_variant=ZIMAGE_MODEL_VARIANT,
    ),
}

# Story 45-38: parallel high-fidelity table. Resolution per tier follows
# the turbo aspect ratios EXCEPT portrait, which AC1 explicitly pins at
# 1024x1024 (square — the Draw Things reference dimension that exposed
# the bug). Steps + guidance are model-driven (base 1.0 supports CFG),
# so they're identical across every entry.
ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS: dict[RenderTier, ZImageTierConfig] = {
    RenderTier.SCENE_ILLUSTRATION: ZImageTierConfig(
        steps=_HIGH_FIDELITY_STEPS,
        guidance=_HIGH_FIDELITY_GUIDANCE,
        width=1024,
        height=768,
        model_variant=ZIMAGE_BASE_MODEL_VARIANT,
    ),
    RenderTier.PORTRAIT: ZImageTierConfig(
        steps=_HIGH_FIDELITY_STEPS,
        guidance=_HIGH_FIDELITY_GUIDANCE,
        width=1024,
        height=1024,
        model_variant=ZIMAGE_BASE_MODEL_VARIANT,
    ),
    RenderTier.PORTRAIT_SQUARE: ZImageTierConfig(
        steps=_HIGH_FIDELITY_STEPS,
        guidance=_HIGH_FIDELITY_GUIDANCE,
        width=1024,
        height=1024,
        model_variant=ZIMAGE_BASE_MODEL_VARIANT,
    ),
    RenderTier.LANDSCAPE: ZImageTierConfig(
        steps=_HIGH_FIDELITY_STEPS,
        guidance=_HIGH_FIDELITY_GUIDANCE,
        width=1024,
        height=768,
        model_variant=ZIMAGE_BASE_MODEL_VARIANT,
    ),
    RenderTier.TEXT_OVERLAY: ZImageTierConfig(
        steps=_HIGH_FIDELITY_STEPS,
        guidance=_HIGH_FIDELITY_GUIDANCE,
        width=768,
        height=512,
        model_variant=ZIMAGE_BASE_MODEL_VARIANT,
    ),
    RenderTier.FOG_OF_WAR: ZImageTierConfig(
        steps=_HIGH_FIDELITY_STEPS,
        guidance=_HIGH_FIDELITY_GUIDANCE,
        width=1024,
        height=1024,
        model_variant=ZIMAGE_BASE_MODEL_VARIANT,
    ),
}

ZIMAGE_SUPPORTED_TIERS: frozenset[RenderTier] = frozenset(ZIMAGE_TIER_CONFIGS)


def get_zimage_config(
    tier: RenderTier,
    fidelity: Fidelity = "high_fidelity",
) -> ZImageTierConfig:
    """Return the Z-Image config for ``(tier, fidelity)``.

    Single lookup point so callers (composer, worker, OTEL spans, scripts)
    don't reach into the raw module-level dicts. Unknown ``fidelity``
    values raise ``ValueError`` per CLAUDE.md "No Silent Fallbacks" — the
    only valid strings are ``"turbo"`` and ``"high_fidelity"``. Default is
    ``"high_fidelity"`` (base Z-Image 1.0); pass ``"turbo"`` explicitly to
    use the LCM-distilled variant for latency-sensitive paths.
    """
    if fidelity == "turbo":
        return ZIMAGE_TIER_CONFIGS[tier]
    if fidelity == "high_fidelity":
        return ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS[tier]
    raise ValueError(
        f"unknown fidelity {fidelity!r}; expected 'turbo' or 'high_fidelity'"
    )
