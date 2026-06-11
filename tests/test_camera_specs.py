from pathlib import Path

import pytest

from sidequest_daemon.media.camera_specs import (
    CameraLoader,
    PostDirective,
)
from sidequest_daemon.media.recipes import CameraPreset

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_loads_all_eighteen_presets():
    loader = CameraLoader.from_file(REPO_ROOT / "cameras.yaml")
    assert len(loader.specs) == 18
    for preset in CameraPreset:
        assert preset in loader.specs


def test_leone_has_crop_post():
    loader = CameraLoader.from_file(REPO_ROOT / "cameras.yaml")
    spec = loader.specs[CameraPreset.extreme_closeup_leone]
    assert isinstance(spec.post, PostDirective)
    assert spec.post.kind == "crop"
    assert spec.post.percent == 0.25


def test_portrait_3q_has_no_post():
    loader = CameraLoader.from_file(REPO_ROOT / "cameras.yaml")
    spec = loader.specs[CameraPreset.portrait_3q]
    assert spec.post is None


def test_missing_preset_in_yaml_raises():
    bad = {"portrait_3q": {"prompt": "..."}}  # only one, 16 missing
    with pytest.raises(ValueError, match="missing"):
        CameraLoader.from_dict(bad)


def test_unknown_preset_in_yaml_raises():
    good = {preset.value: {"prompt": "x"} for preset in CameraPreset}
    good["fabricated_preset"] = {"prompt": "x"}
    with pytest.raises(ValueError, match="unknown"):
        CameraLoader.from_dict(good)
