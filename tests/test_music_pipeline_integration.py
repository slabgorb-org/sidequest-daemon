"""Integration test — exercises real FFmpeg subprocess but mocks
ACE-Step (no GPU) and R2 (no network). Catches plumbing issues that
unit tests miss: FFmpeg not installed, wrong codec, file handles, etc.
"""
import asyncio
import json
import math
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sidequest_daemon.media.ace_step_adapter import InferenceResult
from sidequest_daemon.media.music_pipeline import MusicPipeline


def _write_sine_wav(path: Path, duration_s: int = 1, freq: float = 440.0) -> None:
    """Write a 1-second 440Hz sine wave at 44.1kHz, 16-bit mono."""
    sample_rate = 44100
    n_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767 * 0.3 * math.sin(2 * math.pi * freq * i / sample_rate))
            w.writeframes(struct.pack("<h", value))


@pytest.mark.asyncio
async def test_full_pipeline_with_real_ffmpeg(tmp_path):
    pack_dir = tmp_path / "genre_packs/cav/audio/music"
    json_path = pack_dir / "combat_input_params.json"
    pack_dir.mkdir(parents=True)
    json_path.write_text(json.dumps({
        "task": "text2music", "prompt": "x", "audio_duration": 1,
        "actual_seeds": [42],
    }))

    # Adapter writes a real sine WAV to the requested output path
    def fake_adapter_run(jp, output_wav, ref_audio_override=None):
        _write_sine_wav(output_wav, duration_s=1)
        return InferenceResult(wav_path=output_wav, seed=42)
    adapter = MagicMock()
    adapter.run.side_effect = fake_adapter_run

    # R2 uploader records the bytes it would upload
    uploaded_bytes = []
    def capture_upload(content_bytes, r2_key, content_type):
        uploaded_bytes.append((r2_key, content_type, len(content_bytes)))
        return r2_key

    pipeline = MusicPipeline(
        adapter=adapter,
        r2_uploader=capture_upload, r2_downloader=MagicMock(),
        watcher=MagicMock(),
        render_lock=asyncio.Lock(),
    )

    result = await pipeline.generate(json_path)

    assert result.r2_key == "genre_packs/cav/audio/music/combat.ogg"
    assert len(uploaded_bytes) == 1
    r2_key, content_type, byte_count = uploaded_bytes[0]
    assert r2_key == "genre_packs/cav/audio/music/combat.ogg"
    assert content_type == "audio/ogg"
    assert byte_count > 100  # OGG of 1s sine should be at least a few hundred bytes
