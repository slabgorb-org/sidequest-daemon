"""Thin wrapper over the ACE-Step package.

Isolates the daemon's only contact with the `acestep` API so the rest
of the codebase doesn't depend on it directly. Two responsibilities:

1. Sanitize JSON params (strip output fields, override audio_path,
   require a pinned seed) — see `prepare_inference_params`.
2. Run inference and return the WAV path + seed used — see `AceStepAdapter.run`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Fields ACE-Step writes back into the JSON after a run; never used as input.
_OUTPUT_ONLY_FIELDS = frozenset({"timecosts", "retake_seeds"})


def prepare_inference_params(
    json_path: Path, output_wav: Path, ref_audio_override: str | None = None
) -> dict[str, Any]:
    """Read JSON params, strip output fields, force wav, rename to ACE-Step kwargs.

    The JSON params files use ACE-Step's *output* field names (what the
    library writes back after a run): `actual_seeds`, `audio_path`. The
    library's `__call__` signature uses *input* names: `manual_seeds`,
    `save_path`. Rename on the way in.

    `ref_audio_override`, when given, replaces `ref_audio_input` — the caller
    has fetched the base-theme OGG (named by its R2 key in the JSON) to a local
    path, and ACE-Step must read that local file, not the unresolvable R2 key.

    Raises ValueError if `actual_seeds[0]` is missing or non-integer
    (no implicit randomness — see spec §4.2 seed contract).
    """
    raw = json.loads(json_path.read_text())

    cleaned = {k: v for k, v in raw.items() if k not in _OUTPUT_ONLY_FIELDS}

    cleaned["format"] = "wav"

    if ref_audio_override is not None:
        cleaned["ref_audio_input"] = ref_audio_override

    seeds = cleaned.pop("actual_seeds", None)
    if not isinstance(seeds, list) or not seeds or not isinstance(seeds[0], int):
        raise ValueError(
            f"MISSING_SEED: {json_path} must have actual_seeds[0] as an integer "
            f"(got {seeds!r})"
        )
    cleaned["manual_seeds"] = [seeds[0]]

    cleaned.pop("audio_path", None)
    cleaned["save_path"] = str(output_wav)

    return cleaned


@dataclass
class InferenceResult:
    wav_path: Path
    seed: int


class AceStepAdapter:
    """Lazy-loaded wrapper over `acestep.pipeline_ace_step.ACEStepPipeline`.

    Inject `_pipeline` in tests to avoid loading the real model. In prod,
    `_pipeline` is None on construction and lazy-loads on first `run()`.
    """

    def __init__(self, *, _pipeline: Any | None = None) -> None:
        self._pipeline = _pipeline

    def _ensure_loaded(self) -> Any:
        if self._pipeline is None:
            from acestep.pipeline_ace_step import ACEStepPipeline

            self._pipeline = ACEStepPipeline()
            log.info("ACE-Step pipeline loaded (cold start)")
        return self._pipeline

    def run(
        self, json_path: Path, output_wav: Path, ref_audio_override: str | None = None
    ) -> InferenceResult:
        params = prepare_inference_params(json_path, output_wav, ref_audio_override)
        pipeline = self._ensure_loaded()
        pipeline(**params)
        return InferenceResult(wav_path=output_wav, seed=params["manual_seeds"][0])
