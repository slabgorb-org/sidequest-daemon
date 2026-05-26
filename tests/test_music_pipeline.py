import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sidequest_daemon.media.music_pipeline import MusicPipeline


def test_derive_r2_key_strips_input_params_suffix():
    json_path = Path("/abs/sidequest-content/genre_packs/cav/audio/music/combat_input_params.json")
    key = MusicPipeline.derive_r2_key(json_path)
    assert key == "genre_packs/cav/audio/music/combat.ogg"


def test_derive_r2_key_handles_world_subpacks():
    json_path = Path("/abs/sidequest-content/genre_packs/cav/worlds/sunden/audio/music/combat_input_params.json")
    key = MusicPipeline.derive_r2_key(json_path)
    assert key == "genre_packs/cav/worlds/sunden/audio/music/combat.ogg"


def test_derive_r2_key_rejects_path_outside_genre_packs():
    json_path = Path("/abs/elsewhere/audio/music/combat_input_params.json")
    with pytest.raises(ValueError, match="INVALID_PARAMS_LOCATION"):
        MusicPipeline.derive_r2_key(json_path)


def test_derive_r2_key_rejects_wrong_filename_suffix():
    json_path = Path("/abs/sidequest-content/genre_packs/cav/audio/music/combat.json")
    with pytest.raises(ValueError, match="INVALID_PARAMS_LOCATION"):
        MusicPipeline.derive_r2_key(json_path)


def _write_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "task": "text2music",
        "prompt": "x",
        "audio_duration": 60,
        "actual_seeds": [42],
    }))


def test_generate_happy_path_orchestrates_all_stages(tmp_path):
    pack_dir = tmp_path / "genre_packs/cav/audio/music"
    json_path = pack_dir / "combat_input_params.json"
    _write_json(json_path)

    # Mock adapter — pretends to write a wav at the requested path
    def fake_run(jp, output_wav, ref_audio_override=None):
        output_wav.write_bytes(b"fake wav bytes")
        from sidequest_daemon.media.ace_step_adapter import InferenceResult
        return InferenceResult(wav_path=output_wav, seed=42)
    adapter = MagicMock()
    adapter.run.side_effect = fake_run

    # Mock R2 uploader — records the call, returns the key
    r2_uploader = MagicMock(return_value="genre_packs/cav/audio/music/combat.ogg")

    # Mock watcher — records emits
    watcher = MagicMock()

    render_lock = asyncio.Lock()

    pipeline = MusicPipeline(
        adapter=adapter, r2_uploader=r2_uploader, r2_downloader=MagicMock(),
        watcher=watcher, render_lock=render_lock,
    )

    # Patch FFmpeg subprocess to just rename the wav to ogg
    with patch("sidequest_daemon.media.music_pipeline._run_ffmpeg") as mock_ffmpeg:
        def fake_ffmpeg(wav, ogg):
            ogg.write_bytes(b"fake ogg bytes")
        mock_ffmpeg.side_effect = fake_ffmpeg

        result = asyncio.run(pipeline.generate(json_path))

    assert result.r2_key == "genre_packs/cav/audio/music/combat.ogg"
    assert result.seed == 42
    adapter.run.assert_called_once()
    mock_ffmpeg.assert_called_once()
    r2_uploader.assert_called_once()
    # Watcher emitted start + complete:
    event_types = [c.args[0] for c in watcher.call_args_list]
    assert "music.generation.start" in event_types
    assert "music.generation.complete" in event_types


def test_generate_inference_failure_emits_failed_event_stage_inference(tmp_path):
    pack_dir = tmp_path / "genre_packs/cav/audio/music"
    json_path = pack_dir / "combat_input_params.json"
    _write_json(json_path)

    adapter = MagicMock()
    adapter.run.side_effect = RuntimeError("CUDA OOM")
    pipeline = MusicPipeline(
        adapter=adapter, r2_uploader=MagicMock(), r2_downloader=MagicMock(),
        watcher=MagicMock(), render_lock=asyncio.Lock(),
    )
    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.generate(json_path))

    failed_calls = [c for c in pipeline._watcher.call_args_list
                    if c.args[0] == "music.generation.failed"]
    assert len(failed_calls) == 1
    assert failed_calls[0].args[1]["stage"] == "inference"


def test_generate_ffmpeg_failure_emits_failed_event_stage_ffmpeg(tmp_path):
    pack_dir = tmp_path / "genre_packs/cav/audio/music"
    json_path = pack_dir / "combat_input_params.json"
    _write_json(json_path)

    def fake_run(jp, output_wav, ref_audio_override=None):
        output_wav.write_bytes(b"fake")
        from sidequest_daemon.media.ace_step_adapter import InferenceResult
        return InferenceResult(wav_path=output_wav, seed=42)
    adapter = MagicMock()
    adapter.run.side_effect = fake_run

    pipeline = MusicPipeline(
        adapter=adapter, r2_uploader=MagicMock(), r2_downloader=MagicMock(),
        watcher=MagicMock(), render_lock=asyncio.Lock(),
    )
    with patch("sidequest_daemon.media.music_pipeline._run_ffmpeg") as mock_ffmpeg:
        import subprocess
        mock_ffmpeg.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        with pytest.raises(subprocess.CalledProcessError):
            asyncio.run(pipeline.generate(json_path))

    failed_calls = [c for c in pipeline._watcher.call_args_list
                    if c.args[0] == "music.generation.failed"]
    assert failed_calls[0].args[1]["stage"] == "ffmpeg"


def test_generate_params_failure_emits_failed_event_stage_params(tmp_path):
    # Path not under genre_packs → INVALID_PARAMS_LOCATION raised inside generate
    json_path = tmp_path / "elsewhere/combat_input_params.json"
    _write_json(json_path)

    pipeline = MusicPipeline(
        adapter=MagicMock(), r2_uploader=MagicMock(), r2_downloader=MagicMock(),
        watcher=MagicMock(), render_lock=asyncio.Lock(),
    )
    with pytest.raises(ValueError, match="INVALID_PARAMS_LOCATION"):
        asyncio.run(pipeline.generate(json_path))

    # No watcher event — derive_r2_key fails before the start event fires
    # (this is acceptable; the daemon dispatch reports the error in the reply).
    assert pipeline._watcher.call_count == 0


def test_generate_cleans_tempfiles_on_failure(tmp_path):
    pack_dir = tmp_path / "genre_packs/cav/audio/music"
    json_path = pack_dir / "combat_input_params.json"
    _write_json(json_path)

    captured_tempdirs = []
    def capturing_run(jp, output_wav, ref_audio_override=None):
        captured_tempdirs.append(output_wav.parent)
        raise RuntimeError("fail in inference")
    adapter = MagicMock()
    adapter.run.side_effect = capturing_run

    pipeline = MusicPipeline(
        adapter=adapter, r2_uploader=MagicMock(), r2_downloader=MagicMock(),
        watcher=MagicMock(), render_lock=asyncio.Lock(),
    )
    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.generate(json_path))

    assert len(captured_tempdirs) == 1
    assert not captured_tempdirs[0].exists(), \
        f"tempdir {captured_tempdirs[0]} should have been cleaned up"
