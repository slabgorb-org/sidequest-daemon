"""Wiring test (per CLAUDE.md): prove daemon config loads cleanly at boot."""

from pathlib import Path

import pytest

from sidequest_daemon.media.recipes import CameraPreset
from sidequest_daemon.media.workers import zimage_mlx_worker
from sidequest_daemon.renderer.models import RenderTier, StageCue

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"


def test_daemon_refuses_to_start_with_invalid_recipes(tmp_path) -> None:
    from sidequest_daemon.media import daemon as daemon_module

    bad = tmp_path / "recipes.yaml"
    bad.write_text("portrait: {kind: portrait, direction_camera: fabricated_shot}")
    with pytest.raises(ValueError):
        daemon_module.validate_startup_config(
            recipes_path=bad,
            cameras_path=Path(__file__).resolve().parents[1] / "cameras.yaml",
        )


def test_daemon_accepts_valid_config() -> None:
    from sidequest_daemon.media import daemon as daemon_module

    root = Path(__file__).resolve().parents[1]
    daemon_module.validate_startup_config(
        recipes_path=root / "recipes.yaml",
        cameras_path=root / "cameras.yaml",
    )


def test_worker_imports_new_composer() -> None:
    """The worker must import PromptComposer from prompt_composer.py."""
    source = Path(zimage_mlx_worker.__file__).read_text()
    assert "from sidequest_daemon.media.prompt_composer import PromptComposer" in source
    assert "RenderTarget" in source


def test_worker_build_render_target_from_cue(monkeypatch) -> None:
    """When the worker receives a StageCue with a CameraPreset, it must
    construct a valid RenderTarget and pass it to the composer."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="npc:rux",
        characters=["npc:rux"],
        camera=CameraPreset.portrait_3q,
        metadata={"world": "testworld", "genre": "testgenre"},
    )
    target = zimage_mlx_worker.build_render_target(cue)
    assert target.kind == "portrait"
    assert target.character == "npc:rux"
    assert target.world == "testworld"
    assert target.genre == "testgenre"


def test_wiring_end_to_end_produces_nonempty_prompt(monkeypatch) -> None:
    """The worker's compose path produces a non-empty positive prompt."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="npc:rux",
        characters=["npc:rux"],
        camera=CameraPreset.portrait_3q,
        metadata={"world": "testworld", "genre": "testgenre"},
    )
    prompt = zimage_mlx_worker.compose_prompt_for(cue)
    assert "inquisitor" in prompt.positive_prompt
    assert prompt.seed != 0


def test_landscape_with_prose_subject_composes_without_participants(
    monkeypatch,
) -> None:
    """Pingpong 2026-04-30: a landscape tier cue with a prose subject
    (no ``where:`` prefix) and no characters must compose successfully
    as an environmental illustration. Pre-fix this raised ``IndexError``
    in ``_character_lod_plan`` because the n>=5 fall-through indexed
    ``participants[0]`` unconditionally — 2 of 2 landscape dispatches
    in the 4P MP playtest silently failed and never reached the
    Scrapbook.

    Post-fix: the n==0 short-circuit returns an empty casting plan;
    ART_SENSIBILITY layers carry the visual. The previous version of
    this test asserted-on-raise (``with pytest.raises(Exception):``),
    which encoded the broken state as the desired contract — that
    assertion was the wrong direction. Replaced with a positive
    assertion: a non-empty positive prompt comes back, no error.
    """
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.LANDSCAPE,
        subject="A stone tavern interior with lamplight on oak beams",
        metadata={"world": "testworld", "genre": "testgenre"},
    )
    prompt = zimage_mlx_worker.compose_prompt_for(cue)
    # Environmental prose lands in the prompt's action / location text;
    # the load-bearing assertion is that compose RETURNS rather than
    # raising IndexError.
    assert prompt.positive_prompt, (
        "Environmental landscape with prose subject must produce a "
        "non-empty positive prompt; an empty prompt would mean compose "
        "silently degraded the environmental render."
    )
    # No CASTING layers expected — empty participants → empty plan.
    casting_layers = [layer for layer in prompt.layers if layer.slot == "CASTING"]
    assert casting_layers == [], (
        f"Empty-participants landscape must produce no CASTING layers; "
        f"got: {[layer.source for layer in casting_layers]}. If non-empty, "
        "the n==0 short-circuit was bypassed and the n>=5 fall-through "
        "fired — pingpong 2026-04-30 daemon-landscape-crash regression."
    )


def test_compose_propagates_catalog_miss(monkeypatch) -> None:
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="npc:no_such_character",
        characters=["npc:no_such_character"],
        camera=CameraPreset.portrait_3q,
        metadata={"world": "testworld", "genre": "testgenre"},
    )
    from sidequest_daemon.media.recipes import CatalogMissError

    with pytest.raises(CatalogMissError):
        zimage_mlx_worker.compose_prompt_for(cue)


def test_build_cue_from_params_forwards_pc_descriptor() -> None:
    """The daemon dispatch loop projects request params into a StageCue via
    ``build_cue_from_params``. The wiring test here pins the projection: when
    the server sends ``pc_descriptor`` alongside ``world``/``genre``, the
    descriptor must land in ``cue.metadata`` so ``compose_prompt_for`` can
    register the PC at runtime. Without this projection, slice 2 is a no-op
    and every portrait keeps falling through to the prose-subject path."""
    descriptor = {
        "id": "rux",
        "appearance": "a wiry highlander in scarred leather",
        "default_pose": "hand on hilt",
        "culture": None,
    }
    params = {
        "tier": "portrait",
        "subject": "pc:rux",
        "characters": ["pc:rux"],
        "world": "flickering_reach",
        "genre": "mutant_wasteland",
        "pc_descriptor": descriptor,
        "mood": "",
        "tags": [],
        "location": "",
    }
    cue = zimage_mlx_worker.build_cue_from_params(params)
    assert cue.tier == RenderTier.PORTRAIT
    assert cue.characters == ["pc:rux"]
    assert cue.metadata["world"] == "flickering_reach"
    assert cue.metadata["genre"] == "mutant_wasteland"
    assert cue.metadata["pc_descriptor"] == descriptor


def test_build_cue_from_params_omits_descriptor_when_absent() -> None:
    """No descriptor in params → no descriptor in metadata. Slice 1 callers
    (no portrait wiring yet) must stay on the legacy path."""
    params = {
        "tier": "scene_illustration",
        "subject": "a courtyard at dusk",
        "world": "flickering_reach",
        "genre": "mutant_wasteland",
    }
    cue = zimage_mlx_worker.build_cue_from_params(params)
    assert "pc_descriptor" not in cue.metadata


def test_compose_with_pc_descriptor_registers_runtime_pc(monkeypatch) -> None:
    """A PORTRAIT cue carrying ``cue.metadata['pc_descriptor']`` must register
    the PC into the freshly-loaded CharacterCatalog before composition. This
    is slice 2 of the catalog-injected compose wiring: the server sends
    ``pc:<slug>`` refs plus a descriptor blob, and the daemon adopts the PC
    into the per-render catalog without a disk lookup. The composed prompt
    must contain the descriptor's appearance prose, proving the runtime
    registration reached the casting layer."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="pc:gladstone",
        characters=["pc:gladstone"],
        camera=CameraPreset.portrait_3q,
        metadata={
            "world": "testworld",
            "genre": "testgenre",
            "pc_descriptor": {
                "id": "gladstone",
                "appearance": "a wiry highlander in scarred leather coat, copper torc",
                "default_pose": "hand resting on belt",
                "culture": None,
            },
        },
    )
    composed = zimage_mlx_worker.compose_prompt_for(cue)
    assert "wiry highlander" in composed.positive_prompt
    # The default_pose from the descriptor must reach DIRECTION_ACTION when
    # the PORTRAIT recipe pulls it from the character's tokens.
    assert "hand resting on belt" in composed.positive_prompt


def test_build_target_portrait_threads_background() -> None:
    """background in StageCue.metadata must surface on the resulting RenderTarget.

    Task 8 wiring pin: ``build_render_target`` must forward
    ``cue.metadata["background"]`` to ``RenderTarget.background`` so the
    PromptComposer can auto-select the ``portrait_in_location`` recipe when a
    backdrop POI is specified.
    """
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="npc:rux",
        characters=["npc:rux"],
        metadata={"world": "w", "genre": "g", "background": "where:w/plaza"},
    )
    target = zimage_mlx_worker.build_render_target(cue)
    assert target.kind == "portrait"
    assert target.background == "where:w/plaza"


def test_build_cue_from_params_forwards_background() -> None:
    """background param must land in StageCue.metadata so it reaches
    build_render_target and ultimately RenderTarget.background."""
    params = {
        "tier": "portrait",
        "subject": "npc:rux",
        "characters": ["npc:rux"],
        "world": "coyote_star",
        "genre": "space_opera",
        "background": "where:coyote_star/landing_pad",
    }
    cue = zimage_mlx_worker.build_cue_from_params(params)
    assert cue.metadata["background"] == "where:coyote_star/landing_pad"


def test_build_cue_from_params_omits_background_when_absent() -> None:
    """No background in params → no background key in metadata."""
    params = {
        "tier": "portrait",
        "subject": "npc:rux",
        "characters": ["npc:rux"],
        "world": "coyote_star",
        "genre": "space_opera",
    }
    cue = zimage_mlx_worker.build_cue_from_params(params)
    assert "background" not in cue.metadata


def test_compose_succeeds_for_pc_ref_with_descriptor(monkeypatch) -> None:
    """compose_prompt_for must succeed when the cue carries a pc:<slug>
    ref AND a matching descriptor — the catalog miss is avoided by the
    runtime add_pc path."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="pc:hero",
        characters=["pc:hero"],
        camera=CameraPreset.portrait_3q,
        metadata={
            "world": "testworld",
            "genre": "testgenre",
            "pc_descriptor": {
                "id": "hero",
                "appearance": "a stoic ranger in oilcloth cloak",
                "default_pose": "",
                "culture": None,
            },
        },
    )
    composed = zimage_mlx_worker.compose_prompt_for(cue)
    assert "stoic ranger" in composed.positive_prompt
