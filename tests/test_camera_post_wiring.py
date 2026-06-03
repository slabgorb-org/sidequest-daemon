"""RED suite for Story 78-1 — wire camera post-processing into the render path.

`apply_post` / `required_render_size` (media/post_processor.py) are fully
implemented and unit-tested in isolation (tests/test_post_processor.py), but
they have NO non-test caller: `ZImageMLXWorker.render` never invokes them, and
`ComposedPrompt` has no field to carry a `CameraSpec.post` directive from the
composer to the worker. So a live camera preset that sets `post:` (today only
``extreme_closeup_leone``: crop / center / 0.25) is silently dropped at render —
the "looks-wired" defect class the daemon CLAUDE.md exists to catch.

These tests pin the WIRE contract end-to-end (all in-daemon, per story scope):

  1. Composer end — `compose()` surfaces the resolved camera's post directive on
     `ComposedPrompt.post` so the daemon can forward it.
  2. Render end — `render()` reads the directive from ``params["post"]``, calls
     `required_render_size` to generate a LARGER source, then `apply_post` to
     crop/rotate the generated image back to the target before save/upload.

Canonical carry form (pinned by these tests): ``params["post"]`` is a JSON-style
dict (``{"kind","mode","percent","degrees"}``) or absent/None — matching the
JSON-scalar shape of the other composed fields the daemon copies into params
(positive_prompt:str, seed:int). The daemon compose seam
(media/daemon.py, where ``composed.positive_prompt`` → ``params["positive_prompt"]``)
must also copy ``composed.post`` → ``params["post"]``; see the session's Delivery
Findings for that middle-link note.

The mflux model is stubbed with a size-faithful fake (returns an image of exactly
the dimensions it is asked to generate) so the assertions distinguish:
  - required_render_size wired  → generate is asked for the larger source size
  - apply_post wired            → the saved image is the cropped 25% of generate
A unit test of `apply_post` in isolation (test_post_processor.py) does NOT prove
either is reached from production code — these drive the real `render()` path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sidequest_daemon.media.camera_specs import CameraLoader, PostDirective
from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    PlaceCatalog,
    StyleCatalog,
)
from sidequest_daemon.media.post_processor import required_render_size
from sidequest_daemon.media.prompt_composer import PromptComposer
from sidequest_daemon.media.recipe_loader import RecipeLoader
from sidequest_daemon.media.recipes import CameraPreset, RenderTarget
from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker
from sidequest_daemon.media.zimage_config import get_zimage_config
from sidequest_daemon.renderer.models import RenderTier

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"

# The crop directive carried by the only live post-bearing preset.
_LEONE_PERCENT = 0.25
_LEONE_POST = {"kind": "crop", "mode": "center", "percent": _LEONE_PERCENT}


class _SizeFaithfulModel:
    """Stub mflux model that returns an image of exactly the requested size.

    Real mflux generates at the (width, height) it is handed, so a faithful
    stub lets us assert *what size render asked to generate* (proves
    required_render_size) and *how the saved image relates to it* (proves
    apply_post cropped the generated source).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def generate_image(self, **kwargs: object) -> Image.Image:
        width = int(kwargs["width"])  # type: ignore[call-overload]
        height = int(kwargs["height"])  # type: ignore[call-overload]
        self.calls.append((width, height))
        return Image.new("RGB", (width, height), color="black")


@pytest.fixture
def worker(tmp_path: Path) -> ZImageMLXWorker:
    # Singleton-slot reset is handled by conftest's autouse _reset_zimage_singleton.
    return ZImageMLXWorker(output_dir=tmp_path)


@pytest.fixture
def composer() -> PromptComposer:
    return PromptComposer(
        recipes=RecipeLoader.from_file(REPO_ROOT / "recipes.yaml"),
        cameras=CameraLoader.from_file(REPO_ROOT / "cameras.yaml"),
        characters=CharacterCatalog.load(
            FIXTURE_ROOT, genre="testgenre", world="testworld"
        ),
        places=PlaceCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld"),
        styles=StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld"),
    )


# ── Render end (AC2): required_render_size + apply_post wired into render() ──


def test_render_generates_at_required_source_size_for_crop(
    worker: ZImageMLXWorker,
) -> None:
    """render() must size the generation via required_render_size so the crop
    has pixels to consume — i.e. generate the LARGER source, not the tier size.

    Fails today: render() generates at tier_cfg.width/height and never calls
    required_render_size, so a crop directive is ignored at generation time.
    """
    model = _SizeFaithfulModel()
    worker.model = model
    tier_cfg = get_zimage_config(RenderTier.PORTRAIT, worker.fidelity)
    expected_w, expected_h = required_render_size(
        (tier_cfg.width, tier_cfg.height),
        PostDirective(kind="crop", mode="center", percent=_LEONE_PERCENT),
    )

    worker.render(
        {
            "tier": "portrait",
            "positive_prompt": "a face",
            "seed": 1,
            "post": dict(_LEONE_POST),
        }
    )

    assert model.calls, "generate_image was never called"
    gen_w, gen_h = model.calls[-1]
    assert (gen_w, gen_h) == (expected_w, expected_h), (
        f"render() must generate at required_render_size {expected_w}x{expected_h} "
        f"(tier {tier_cfg.width}x{tier_cfg.height} ÷ {_LEONE_PERCENT}) so the crop "
        f"has source pixels; generated {gen_w}x{gen_h} — required_render_size is "
        f"not wired into render()."
    )


def test_render_crops_generated_image_after_generate(worker: ZImageMLXWorker) -> None:
    """The saved image must be the post-cropped (25% per axis) version of what
    the model generated — proving apply_post is applied to the generated image
    before save.

    Fails today: the generated image is saved verbatim (no apply_post), so the
    saved size equals the generated size, not 25% of it.
    """
    model = _SizeFaithfulModel()
    worker.model = model

    result = worker.render(
        {
            "tier": "portrait",
            "positive_prompt": "a face",
            "seed": 1,
            "post": dict(_LEONE_POST),
        }
    )

    gen_w, gen_h = model.calls[-1]
    expected = (int(gen_w * _LEONE_PERCENT), int(gen_h * _LEONE_PERCENT))
    with Image.open(result["image_url"]) as saved:
        saved_size = saved.size
    assert saved_size == expected, (
        f"saved image must be the {int(_LEONE_PERCENT * 100)}%-center-crop of the "
        f"generated {gen_w}x{gen_h} source ({expected}); got {saved_size} — "
        f"apply_post is not applied after generate()."
    )


def test_render_without_post_directive_renders_unchanged(
    worker: ZImageMLXWorker,
) -> None:
    """No post directive → generate at the tier size and save verbatim (no
    upsizing, no crop). Guards the no-op edge so wiring the post path does not
    regress ordinary renders.
    """
    model = _SizeFaithfulModel()
    worker.model = model
    tier_cfg = get_zimage_config(RenderTier.PORTRAIT, worker.fidelity)

    result = worker.render(
        {"tier": "portrait", "positive_prompt": "a face", "seed": 1}
    )

    assert model.calls[-1] == (tier_cfg.width, tier_cfg.height), (
        "without a post directive, render() must generate at the tier size — no "
        "required_render_size upsizing."
    )
    with Image.open(result["image_url"]) as saved:
        assert saved.size == (tier_cfg.width, tier_cfg.height), (
            "without a post directive the saved image must equal the tier size — "
            "no crop applied."
        )


def test_render_rejects_invalid_post_directive_loudly(
    worker: ZImageMLXWorker,
) -> None:
    """A malformed post directive in params (external socket input) must fail
    loud, not be silently ignored (No Silent Fallbacks; lang-review #1/#11).

    Fails today: render() ignores ``params["post"]`` entirely, so an invalid
    directive raises nothing. After wiring, the directive is validated
    (PostDirective; pydantic ValidationError is a ValueError subclass) at the
    boundary.
    """
    model = _SizeFaithfulModel()
    worker.model = model

    with pytest.raises(ValueError):
        worker.render(
            {
                "tier": "portrait",
                "positive_prompt": "a face",
                "seed": 1,
                # "warp" is not a valid PostDirective.kind (Literal[crop,rotate]).
                "post": {"kind": "warp", "percent": 0.5},
            }
        )


# ── Composer end: the camera's post directive survives composition ──


def test_compose_carries_camera_post_directive(composer: PromptComposer) -> None:
    """compose() must surface the resolved camera's post directive on
    ComposedPrompt.post so the daemon can forward it to render().

    The illustration recipe binds direction_camera to ``{camera}``, so a target
    naming ``extreme_closeup_leone`` (crop/center/0.25) resolves that spec.

    Fails today: ComposedPrompt has no ``post`` field → AttributeError.
    """
    target = RenderTarget(
        kind="illustration",
        world="testworld",
        genre="testgenre",
        participants=["npc:rux"],
        action="a standoff",
        location="where:testgenre/tavern",
        camera=CameraPreset.extreme_closeup_leone,
    )

    composed = composer.compose(target)

    assert composed.post is not None, (
        "ComposedPrompt.post must carry the resolved camera's post directive "
        "(extreme_closeup_leone sets crop/center/0.25) so render() can apply it."
    )
    assert composed.post.kind == "crop"
    assert composed.post.percent == _LEONE_PERCENT


def test_compose_no_post_when_camera_has_no_directive(
    composer: PromptComposer,
) -> None:
    """A camera preset without a post directive (``scene``) must leave
    ComposedPrompt.post as None — the carry is opt-in, not fabricated.

    Fails today: ComposedPrompt has no ``post`` field → AttributeError.
    """
    target = RenderTarget(
        kind="illustration",
        world="testworld",
        genre="testgenre",
        participants=["npc:rux"],
        action="a quiet drink",
        location="where:testgenre/tavern",
        camera=CameraPreset.scene,
    )

    composed = composer.compose(target)

    assert composed.post is None, (
        "scene has no post directive — ComposedPrompt.post must be None, not a "
        "fabricated directive."
    )
