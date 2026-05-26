"""Ref-audio fetch for audio2audio leitmotif variations (road_warrior et al).

When a params file declares `audio2audio_enable: true` with an `ref_audio_input`
that is an R2 key (`genre_packs/...`), the pipeline must download that base OGG
into the render tempdir and hand the adapter a *local* path — ACE-Step needs a
real local file, and the OGG-only pipeline persists nothing locally on its own.
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sidequest_daemon.media.ace_step_adapter import InferenceResult
from sidequest_daemon.media.music_pipeline import MusicPipeline


def _write_a2a_json(path: Path, ref: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "task": "audio2audio" if ref else "text2music",
        "prompt": "war drums",
        "audio_duration": 60,
        "actual_seeds": [42],
        "audio2audio_enable": bool(ref),
        "ref_audio_strength": 0.4,
        "ref_audio_input": ref,
    }))


def _pipeline(adapter, downloader, watcher):
    return MusicPipeline(
        adapter=adapter, r2_uploader=MagicMock(return_value="k"),
        r2_downloader=downloader, watcher=watcher, render_lock=asyncio.Lock(),
    )


def _ffmpeg_stub():
    def fake(wav, ogg):
        ogg.write_bytes(b"ogg")
    return fake


def test_audio2audio_downloads_ref_and_passes_local_override(tmp_path):
    jp = tmp_path / "genre_packs/rw/audio/music/convoy_war_drums_input_params.json"
    _write_a2a_json(jp, "genre_packs/rw/audio/music/convoy.ogg")

    seen = {}

    def fake_run(json_path, output_wav, ref_audio_override=None):
        seen["override"] = ref_audio_override
        # The override file lives in the render tempdir, deleted on block exit —
        # read it here, while the adapter (which would feed ACE-Step) still can.
        seen["override_bytes"] = (
            Path(ref_audio_override).read_bytes() if ref_audio_override else None
        )
        output_wav.write_bytes(b"wav")
        return InferenceResult(wav_path=output_wav, seed=42)

    adapter = MagicMock()
    adapter.run.side_effect = fake_run
    downloader = MagicMock(return_value=b"BASE_OGG_BYTES")
    watcher = MagicMock()

    with patch("sidequest_daemon.media.music_pipeline._run_ffmpeg",
               side_effect=_ffmpeg_stub()):
        asyncio.run(_pipeline(adapter, downloader, watcher).generate(jp))

    # downloaded the right base key
    downloader.assert_called_once_with("genre_packs/rw/audio/music/convoy.ogg")
    # adapter got a LOCAL path (not the R2 key), and the bytes are on disk there
    override = seen["override"]
    assert override is not None
    assert override != "genre_packs/rw/audio/music/convoy.ogg"
    assert seen["override_bytes"] == b"BASE_OGG_BYTES"
    # watcher announced the fetch
    events = [c.args[0] for c in watcher.call_args_list]
    assert "music.ref_audio.fetch" in events


def test_text2music_does_not_download_or_override(tmp_path):
    jp = tmp_path / "genre_packs/rw/audio/music/convoy_input_params.json"
    _write_a2a_json(jp, None)

    seen = {}

    def fake_run(json_path, output_wav, ref_audio_override=None):
        seen["override"] = ref_audio_override
        output_wav.write_bytes(b"wav")
        return InferenceResult(wav_path=output_wav, seed=42)

    adapter = MagicMock()
    adapter.run.side_effect = fake_run
    downloader = MagicMock()

    with patch("sidequest_daemon.media.music_pipeline._run_ffmpeg",
               side_effect=_ffmpeg_stub()):
        asyncio.run(_pipeline(adapter, downloader, MagicMock()).generate(jp))

    downloader.assert_not_called()
    assert seen["override"] is None


def test_ref_fetch_failure_emits_failed_event_stage_ref_fetch(tmp_path):
    jp = tmp_path / "genre_packs/rw/audio/music/convoy_war_drums_input_params.json"
    _write_a2a_json(jp, "genre_packs/rw/audio/music/convoy.ogg")

    adapter = MagicMock()
    downloader = MagicMock(side_effect=RuntimeError("NoSuchKey: convoy.ogg"))
    watcher = MagicMock()

    with pytest.raises(Exception):
        asyncio.run(_pipeline(adapter, downloader, watcher).generate(jp))

    failed = [c for c in watcher.call_args_list
              if c.args[0] == "music.generation.failed"]
    assert len(failed) == 1
    assert failed[0].args[1]["stage"] == "ref_fetch"
    # the adapter never ran — we failed before inference
    adapter.run.assert_not_called()
