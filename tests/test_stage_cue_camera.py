from sidequest_daemon.media.recipes import CameraPreset
from sidequest_daemon.renderer.models import RenderTier, StageCue


def test_tactical_sketch_removed() -> None:
    assert not hasattr(RenderTier, "TACTICAL_SKETCH")


def test_stage_cue_accepts_camera() -> None:
    # Story 101-1: StageCue.camera is now plain ``str | None`` (server contract),
    # not the CameraPreset enum. A CameraPreset passed in (it is a str enum)
    # coerces to its string value rather than staying an enum member.
    cue = StageCue(
        tier=RenderTier.SCENE_ILLUSTRATION,
        subject="goblin ambush",
        camera=CameraPreset.topdown_90,
    )
    assert cue.camera == "topdown_90"
    assert isinstance(cue.camera, str)


def test_stage_cue_camera_optional() -> None:
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="rux",
    )
    assert cue.camera is None
