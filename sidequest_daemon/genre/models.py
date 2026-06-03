"""Genre pack model — daemon-relevant subset.

Only ``GenrePack`` remains. It is referenced solely as a ``TYPE_CHECKING``
annotation in ``scene_interpreter.py``; it is never instantiated and none of
its fields are accessed in the daemon. The former media/audio sub-models
(``VisualStyle``, ``AudioConfig``, ``MixerSettings``, ``AIGenerationConfig``,
``ThemeFamily``, ``MoodTrack``, ``Variation``, ``PackMeta``) were removed in
story 78-2 as dead exports — the daemon reads pack data via ``StyleCatalog`` /
YAML directly, not these pydantic models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GenrePack(BaseModel):
    """Minimal GenrePack stub for daemon type annotations.

    Referenced only under ``TYPE_CHECKING`` (``scene_interpreter.py``); never
    instantiated and no fields are read in the daemon. ``extra="allow"`` keeps
    it forward-compatible if a caller ever passes a populated pack.
    """

    model_config = ConfigDict(extra="allow")
