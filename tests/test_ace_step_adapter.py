import json
import pytest
from unittest.mock import MagicMock  # noqa: F401

from sidequest_daemon.media.ace_step_adapter import (  # noqa: F401
    AceStepAdapter,
    prepare_inference_params,
)


def test_prepare_inference_params_strips_output_fields(tmp_path):
    raw = {
        "task": "text2music",
        "format": "ogg",  # daemon should force this to wav
        "prompt": "test prompt",
        "lyrics": "[inst]",
        "audio_duration": 60,
        "actual_seeds": [42, 100, 200],  # only [0] preserved
        "retake_seeds": [123],  # stripped
        "timecosts": {"diffusion": 64.0},  # stripped
        "audio_path": "/Users/keithavery/stale/path.wav",  # overridden
    }
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    output_wav = tmp_path / "out.wav"

    cleaned = prepare_inference_params(json_path, output_wav)

    assert cleaned["format"] == "wav"
    # ACE-Step's __call__ takes save_path/manual_seeds — the JSON's output-style
    # names (audio_path/actual_seeds) are renamed on the way in.
    assert cleaned["save_path"] == str(output_wav)
    assert cleaned["manual_seeds"] == [42]
    assert "audio_path" not in cleaned
    assert "actual_seeds" not in cleaned
    assert "retake_seeds" not in cleaned
    assert "timecosts" not in cleaned
    assert cleaned["prompt"] == "test prompt"
    assert cleaned["audio_duration"] == 60


def test_prepare_inference_params_rejects_missing_seed(tmp_path):
    raw = {"task": "text2music", "prompt": "x", "audio_duration": 60}  # no actual_seeds
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="MISSING_SEED"):
        prepare_inference_params(json_path, tmp_path / "out.wav")


def test_prepare_inference_params_rejects_empty_seed_list(tmp_path):
    raw = {"task": "text2music", "actual_seeds": []}
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="MISSING_SEED"):
        prepare_inference_params(json_path, tmp_path / "out.wav")


def test_prepare_inference_params_rejects_non_integer_seed(tmp_path):
    raw = {"task": "text2music", "actual_seeds": ["abc"]}
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="MISSING_SEED"):
        prepare_inference_params(json_path, tmp_path / "out.wav")


def test_adapter_run_invokes_acestep_pipeline_with_cleaned_params(tmp_path):
    raw = {
        "task": "text2music",
        "prompt": "test",
        "audio_duration": 60,
        "actual_seeds": [42],
    }
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    output_wav = tmp_path / "out.wav"

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = None  # ACE-Step writes the file as a side effect

    adapter = AceStepAdapter(_pipeline=fake_pipeline)
    result = adapter.run(json_path, output_wav)

    assert result.wav_path == output_wav
    assert result.seed == 42
    fake_pipeline.assert_called_once()
    call_kwargs = fake_pipeline.call_args.kwargs
    assert call_kwargs["save_path"] == str(output_wav)
    assert call_kwargs["format"] == "wav"
    assert call_kwargs["manual_seeds"] == [42]


def test_prepare_inference_params_only_emits_kwargs_acestep_accepts(tmp_path):
    """Wiring guard — every key prepare_inference_params produces must be a
    valid kwarg of ACEStepPipeline.__call__. Catches the actual_seeds vs
    manual_seeds rename drift that broke first real-pipeline call after
    daemon-music-tier merged.

    The accepted-kwarg set is a frozen snapshot of ACEStepPipeline.__call__'s
    signature, NOT a live `import acestep` introspection. Importing the real
    package pulls in torch + the full ACE-Step model stack — slow enough to
    hang a unit-test run and starve a concurrent render. If ACE-Step's
    signature changes upstream, refresh this set from the source of truth:
    `../../ACE-Step/acestep/pipeline_ace_step.py` → `def __call__`
    (snapshot taken 2026-06-12)."""
    accepted = {
        "format", "audio_duration", "prompt", "lyrics", "infer_step",
        "guidance_scale", "scheduler_type", "cfg_type", "omega_scale",
        "manual_seeds", "guidance_interval", "guidance_interval_decay",
        "min_guidance_scale", "use_erg_tag", "use_erg_lyric",
        "use_erg_diffusion", "oss_steps", "guidance_scale_text",
        "guidance_scale_lyric", "audio2audio_enable", "ref_audio_strength",
        "ref_audio_input", "lora_name_or_path", "lora_weight", "retake_seeds",
        "retake_variance", "task", "repaint_start", "repaint_end",
        "src_audio_path", "edit_target_prompt", "edit_target_lyrics",
        "edit_n_min", "edit_n_max", "edit_n_avg", "save_path", "batch_size",
        "debug",
    }

    raw = {
        "task": "text2music",
        "format": "wav",
        "prompt": "test",
        "lyrics": "[inst]",
        "audio_duration": 60,
        "infer_step": 60,
        "guidance_scale": 15,
        "scheduler_type": "euler",
        "cfg_type": "apg",
        "omega_scale": 10,
        "actual_seeds": [42],
        "retake_variance": 0.5,
        "guidance_interval": 0.5,
        "guidance_interval_decay": 0.0,
        "min_guidance_scale": 3.0,
        "use_erg_tag": True,
        "use_erg_lyric": False,
        "use_erg_diffusion": True,
        "oss_steps": [],
        "guidance_scale_text": 0.0,
        "guidance_scale_lyric": 0.0,
        "repaint_start": 0,
        "repaint_end": 0,
        "edit_n_min": 0.0,
        "edit_n_max": 1.0,
        "edit_n_avg": 1,
        "src_audio_path": None,
        "edit_target_prompt": None,
        "edit_target_lyrics": None,
        "audio2audio_enable": False,
        "ref_audio_strength": 0.5,
        "ref_audio_input": None,
        "lora_name_or_path": "none",
        "lora_weight": 1.0,
        "audio_path": "/stale/output/path.wav",
        "retake_seeds": [123],
        "timecosts": {"diffusion": 1.0},
    }
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))

    cleaned = prepare_inference_params(json_path, tmp_path / "out.wav")

    extras = set(cleaned) - accepted
    assert not extras, (
        f"prepare_inference_params produced kwargs ACEStepPipeline.__call__ "
        f"does not accept: {sorted(extras)}. Either rename them in "
        f"prepare_inference_params or add them to _OUTPUT_ONLY_FIELDS."
    )


def test_prepare_inference_params_applies_ref_audio_override(tmp_path):
    """When a ref_audio_override is supplied (a locally-fetched base OGG path),
    it replaces ref_audio_input so ACE-Step reads the local file, not the R2 key."""
    raw = {
        "task": "audio2audio",
        "prompt": "war drums",
        "audio_duration": 60,
        "actual_seeds": [42],
        "audio2audio_enable": True,
        "ref_audio_strength": 0.4,
        "ref_audio_input": "genre_packs/rw/audio/music/convoy.ogg",
    }
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    output_wav = tmp_path / "out.wav"
    local = "/tmp/render/ref.ogg"

    cleaned = prepare_inference_params(json_path, output_wav, ref_audio_override=local)

    assert cleaned["ref_audio_input"] == local
    assert cleaned["audio2audio_enable"] is True


def test_prepare_inference_params_override_none_leaves_ref_untouched(tmp_path):
    raw = {
        "task": "text2music",
        "prompt": "x",
        "audio_duration": 60,
        "actual_seeds": [42],
        "ref_audio_input": None,
    }
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    cleaned = prepare_inference_params(json_path, tmp_path / "out.wav")
    assert cleaned["ref_audio_input"] is None
