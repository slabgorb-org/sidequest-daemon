"""Camera preset specifications loaded from cameras.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from sidequest_daemon.media.recipes import CameraPreset, PostDirective

# PostDirective is defined in ``recipes`` (so ``ComposedPrompt`` can carry it
# without a circular import) and re-exported here for backward compatibility
# with existing importers (post_processor, tests).
__all__ = ["PostDirective", "CameraSpec", "CameraLoader"]


class CameraSpec(BaseModel):
    prompt: str
    post: PostDirective | None = None


class CameraLoader:
    """Loads and validates cameras.yaml — fails loud on missing or unknown."""

    def __init__(self, specs: dict[CameraPreset, CameraSpec]) -> None:
        self.specs = specs

    @classmethod
    def from_file(cls, path: Path) -> "CameraLoader":
        data = yaml.safe_load(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraLoader":
        known = {p.value for p in CameraPreset}
        provided = set(data.keys())

        missing = known - provided
        if missing:
            raise ValueError(f"cameras.yaml missing presets: {sorted(missing)}")

        unknown = provided - known
        if unknown:
            raise ValueError(
                f"cameras.yaml contains unknown presets: {sorted(unknown)}",
            )

        specs: dict[CameraPreset, CameraSpec] = {}
        for name, spec_data in data.items():
            specs[CameraPreset(name)] = CameraSpec.model_validate(spec_data)
        return cls(specs)

    def get(self, preset: CameraPreset) -> CameraSpec:
        return self.specs[preset]
