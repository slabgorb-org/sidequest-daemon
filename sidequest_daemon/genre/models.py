"""Genre pack models — daemon-relevant subset.

Copied from sq-2/sidequest/genre/models.py. Only the models needed by the
daemon's media and audio pipelines are included here. (Voice pipeline was
removed in 2026-04 along with Kokoro TTS.)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Pack metadata
# ---------------------------------------------------------------------------


class PackMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    version: str
    description: str
    min_sidequest_version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Visual style (used by prompt_composer.py)
# ---------------------------------------------------------------------------


class VisualStyle(BaseModel):
    model_config = ConfigDict(extra="allow")

    positive_suffix: str = ""
    preferred_model: str = "flux"
    base_seed: int = 0
    visual_tag_overrides: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Audio config (used by interpreter.py, library_backend.py, mixer.py)
# ---------------------------------------------------------------------------


class MoodTrack(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    title: str = ""
    bpm: int = 0


class MixerSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    music_volume: float = 0.4
    sfx_volume: float = 0.7
    duck_amount_db: float = -12.0
    crossfade_default_ms: int = 3000
    loudnorm_target_lufs: float = -16.0


class AIGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    model: str = "musicgen_small"
    max_generation_time_s: int = 15
    cache_generated: bool = True


class Variation(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    path: str


class ThemeFamily(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    base_prompt: str
    mood: str
    variations: list[Variation] = []


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mood_tracks: dict[str, list[MoodTrack]] = {}
    mood_keywords: dict[str, list[str]] = {}
    sfx_library: dict[str, list[str]] = {}
    mixer: MixerSettings = MixerSettings()
    ai_generation: AIGenerationConfig = AIGenerationConfig()
    themes: list[ThemeFamily] = []
    variation_types: list[str] = []


# ---------------------------------------------------------------------------
# GenrePack — lightweight stub with only the fields the daemon accesses
# ---------------------------------------------------------------------------


class GenrePack(BaseModel):
    """Minimal GenrePack for the daemon.

    The full GenrePack in sq-2 has ~30 fields; the daemon only reads these.
    """

    model_config = ConfigDict(extra="allow")

    meta: PackMeta = PackMeta(name="unknown", version="0.0.0", description="")
    audio: AudioConfig = AudioConfig()
    visual_style: VisualStyle = VisualStyle()

    @property
    def name(self) -> str:
        return self.meta.name
