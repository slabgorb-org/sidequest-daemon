from pathlib import Path

import pytest

from sidequest_daemon.media.camera_specs import CameraLoader
from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    PlaceCatalog,
    StyleCatalog,
)
from sidequest_daemon.media.prompt_composer import PromptComposer
from sidequest_daemon.media.recipe_loader import RecipeLoader
from sidequest_daemon.media.recipes import (
    CameraPreset,
    CatalogMissError,
    LOD,
    PlaceLOD,
    RenderTarget,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def composer() -> PromptComposer:
    return PromptComposer(
        recipes=RecipeLoader.from_file(REPO_ROOT / "recipes.yaml"),
        cameras=CameraLoader.from_file(REPO_ROOT / "cameras.yaml"),
        characters=CharacterCatalog.load(
            FIXTURE_ROOT, genre="testgenre", world="testworld"
        ),
        places=PlaceCatalog.load(
            FIXTURE_ROOT, genre="testgenre", world="testworld"
        ),
        styles=StyleCatalog.load(
            FIXTURE_ROOT, genre="testgenre", world="testworld"
        ),
    )


def test_composer_constructs(composer: PromptComposer) -> None:
    assert composer is not None


def test_portrait_lod_plan_is_solo(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    plan = composer._character_lod_plan(t)
    assert plan == {"npc:rux": LOD.SOLO}


def test_illustration_one_participant_is_solo(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="thinking",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    assert composer._character_lod_plan(t) == {"npc:rux": LOD.SOLO}


def test_illustration_two_participants_both_long(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:mira"], action="arguing",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    assert composer._character_lod_plan(t) == {
        "npc:rux": LOD.LONG,
        "npc:mira": LOD.LONG,
    }


def test_illustration_four_participants_focus_long_rest_short(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:mira", "npc:a", "npc:b"],
        action="around a table", location="where:testgenre/tavern",
        camera=CameraPreset.scene,
    )
    plan = composer._character_lod_plan(t)
    assert plan["npc:rux"] == LOD.LONG
    for other in ("npc:mira", "npc:a", "npc:b"):
        assert plan[other] == LOD.SHORT


def test_illustration_zero_participants_returns_empty_plan(
    composer: PromptComposer,
) -> None:
    """Pingpong 2026-04-30: landscape tier with no characters
    (environmental scene, narrator-emitted prose subject — routed to
    ``kind=illustration`` with empty participants per
    ``zimage_mlx_worker.py:137-146``) crashed at ``participants[0]:
    LOD.LONG`` because the n>=5 fall-through branch indexed [0]/[1]/[2]
    unconditionally. 2 of 2 landscape dispatches in the 4P MP playtest
    silently failed — daemon closed socket on IndexError, server logged
    ``render.reply_unavailable``, corridor opening shot + half-landing
    scene missing from Scrapbook for all 4 players.

    Post-fix: ``n == 0`` short-circuits with an empty plan — environmental
    illustrations have no casting layer; the prose subject + ART_SENSIBILITY
    layers carry the visual. Mirrors the POI fall-through at the bottom
    of ``_character_lod_plan`` (POI also has no characters in frame).
    """
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=[],  # the load-bearing repro state
        action="A cramped wrench-house galley under coolant pipes",
        location="where:testworld/the_lookout",
        camera=CameraPreset.scene,
    )
    plan = composer._character_lod_plan(t)
    assert plan == {}, (
        f"Empty participants for illustration tier must return an empty "
        f"plan, not crash with IndexError. Got: {plan!r}. If this fails "
        "with an IndexError, the n==0 short-circuit was removed — "
        "regression of pingpong 2026-04-30 daemon-landscape-crash."
    )


def test_illustration_six_participants_tail_background(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:a", "npc:b", "npc:c", "npc:d", "npc:e"],
        action="tavern brawl", location="where:testgenre/tavern",
        camera=CameraPreset.scene,
    )
    plan = composer._character_lod_plan(t)
    assert plan["npc:rux"] == LOD.LONG
    assert plan["npc:a"] == LOD.SHORT
    assert plan["npc:b"] == LOD.SHORT
    assert plan["npc:c"] == LOD.BACKGROUND
    assert plan["npc:d"] == LOD.BACKGROUND
    assert plan["npc:e"] == LOD.BACKGROUND


def test_poi_lod_solo(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="poi", world="testworld", genre="testgenre",
        place="where:testworld/the_lookout",
    )
    assert composer._place_lod_for(t) == PlaceLOD.SOLO


def test_illustration_place_lod_backdrop(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="arriving",
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    assert composer._place_lod_for(t) == PlaceLOD.BACKDROP


def test_casting_portrait_uses_solo_description(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layers = composer._resolve_casting(t)
    assert len(layers) == 1
    assert layers[0].slot == "CASTING"
    assert layers[0].source == "npc:rux"
    assert "inquisitor" in layers[0].tokens
    assert "grey wool cassock" in layers[0].tokens  # solo is richest


def test_casting_illustration_two_uses_long(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:mira"], action="arguing",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    layers = composer._resolve_casting(t)
    assert len(layers) == 2
    rux = next(layer for layer in layers if layer.source == "npc:rux")
    assert rux.tokens.startswith("gaunt inquisitor in grey wool")


def test_casting_poi_uses_landmark(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="poi", world="testworld", genre="testgenre",
        place="where:testworld/the_lookout",
    )
    layers = composer._resolve_casting(t)
    assert len(layers) == 1
    assert layers[0].source == "where:testworld/the_lookout"
    assert "watchtower" in layers[0].tokens


def test_location_portrait_empty_when_no_background(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layers = composer._resolve_location(t)
    assert layers == []


def test_location_poi_uses_environment(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="poi", world="testworld", genre="testgenre",
        place="where:testworld/the_lookout",
    )
    layers = composer._resolve_location(t)
    assert len(layers) == 1
    assert "upland" in layers[0].tokens


def test_location_illustration_specific_uses_landmark_plus_environment(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="arriving",
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    layers = composer._resolve_location(t)
    combined = " ".join(layer.tokens for layer in layers)
    assert "watchtower" in combined
    assert "upland" in combined


def test_location_illustration_archetypal_uses_environment_only(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="drinking",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    layers = composer._resolve_location(t)
    combined = " ".join(layer.tokens for layer in layers)
    assert "hearth" in combined


def test_location_illustration_empty_skips_layer(
    composer: PromptComposer,
) -> None:
    """An illustration with no `where:` ref must NOT raise — transient
    scenes (corridors mid-transit, breached compartments) carry their
    setting in the action prose. Pre-fix, _resolve_location called
    PlaceCatalog.get("") which raised ValueError("place ref '' must use
    scheme 'where:'"). That ValueError bubbled out of _handle_client
    uncaught and EOF'd the socket — see playtest 2026-04-30.
    """
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["pc:hokulea"], action="prying open a sprung locker",
        location="", camera=CameraPreset.scene,
    )
    layers = composer._resolve_location(t)
    assert layers == []


def test_location_illustration_non_where_ref_still_raises(
    composer: PromptComposer,
) -> None:
    """A non-empty location ref that doesn't use the `where:` scheme is a
    contract violation (server is shipping prose instead of a catalog
    ref). The empty-skip path must NOT swallow this — _handle_client
    converts it to a structured COMPOSE_FAILED reply so the GM panel sees
    the real reason.
    """
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["pc:hokulea"], action="any",
        location="Engine Bay", camera=CameraPreset.scene,
    )
    with pytest.raises(ValueError, match="must use scheme 'where:'"):
        composer._resolve_location(t)


def test_portrait_direction_action_uses_default_pose(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layer = composer._resolve_direction_action(t)
    assert "neutral expression" in layer.tokens


def test_portrait_direction_action_honors_pose_override(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
        pose_override="turning sharply, accusing gesture",
    )
    layer = composer._resolve_direction_action(t)
    assert "accusing" in layer.tokens


def test_illustration_direction_action_is_inline(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="arguing at the door",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    layer = composer._resolve_direction_action(t)
    assert "arguing" in layer.tokens


def test_portrait_camera_uses_recipe_default(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layer = composer._resolve_direction_camera(t)
    # portrait_3q is the default portrait camera and is INTENTIONALLY blank
    # (cameras.yaml, commit e3290d4) — prescriptive tokens like "centered
    # subject"/"detailed face" drove T-pose/photorealism and fought Z-Image's
    # natural framing. The resolver still wires portrait → portrait_3q; the
    # preset just contributes no prompt tokens.
    assert layer.slot == "DIRECTION_CAMERA"
    assert layer.tokens == ""


def test_portrait_in_location_uses_guarded_preset(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux", background="where:testworld/the_lookout",
    )
    result = composer.compose(t)
    p = result.positive_prompt
    assert "single continuous photograph" in p
    assert "telephoto" in p
    cam = composer._resolve_direction_camera(t)
    assert cam.source == "portrait_in_location"


def test_portrait_without_background_keeps_default_camera(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    cam = composer._resolve_direction_camera(t)
    assert cam.source == "portrait_3q"


def test_portrait_camera_passthrough_wins_over_in_location_autoselect(
    composer: PromptComposer,
) -> None:
    # Precedence guard: the `{camera}` recipe binding is checked BEFORE the
    # portrait_in_location auto-select in _resolve_camera_preset. The shipped
    # portrait recipe binds a fixed preset, so rebind it to `{camera}` here
    # (fixture is function-scoped — no cross-test leakage) and verify an
    # explicit target.camera beats the background-driven auto-select.
    composer._recipes.recipes["portrait"] = composer._recipes.recipes[
        "portrait"
    ].model_copy(update={"direction_camera": "{camera}"})
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux", background="where:testworld/the_lookout",
        camera=CameraPreset.portrait_closeup,
    )
    cam = composer._resolve_direction_camera(t)
    assert cam.source == "portrait_closeup"


def test_illustration_camera_from_render_target(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="ambush",
        location="where:testgenre/tavern", camera=CameraPreset.topdown_90,
    )
    layer = composer._resolve_direction_camera(t)
    assert "top-down" in layer.tokens


def test_cascade_world_overrides_genre_portrait(composer: PromptComposer) -> None:
    # testworld ships its own visual_style, so it overrides the genre style
    # entirely — the GENRE layer is suppressed, not stacked on top.
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layers = composer._resolve_art_sensibility(t)
    slots = [layer.slot for layer in layers]
    assert "ART_SENSIBILITY.GENRE" not in slots
    assert "ART_SENSIBILITY.WORLD" in slots
    assert "ART_SENSIBILITY.CULTURE" in slots


def test_cascade_world_empty_still_emitted_as_empty(
    composer: PromptComposer,
) -> None:
    # testworld has a visual_style; WORLD layer should have tokens.
    layers = composer._resolve_art_sensibility(
        RenderTarget(
            kind="portrait", world="testworld", genre="testgenre",
            character="npc:rux",
        ),
    )
    world_layer = next(layer for layer in layers if layer.slot == "ART_SENSIBILITY.WORLD")
    assert world_layer.tokens


def test_cascade_illustration_merges_multiple_cultures(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:mira"], action="standing watch",
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    layers = composer._resolve_art_sensibility(t)
    culture_layers = [layer for layer in layers if layer.slot == "ART_SENSIBILITY.CULTURE"]
    # Rux and Mira share `ironhand`; dedupe produces one culture layer.
    assert len(culture_layers) == 1
    assert "iron-chased" in culture_layers[0].tokens


def test_compose_portrait_assembles_in_order(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    result = composer.compose(t)
    # Assembly order per spec. World overrides genre, so the WORLD style leads
    # instead of a GENRE layer:
    # WORLD, CASTING, LOCATION, DIRECTION_ACTION, DIRECTION_CAMERA,
    # CULTURE, safety clause.
    # DIRECTION_CAMERA contributes no tokens for a portrait — portrait_3q is
    # intentionally blank (cameras.yaml, e3290d4) — so the empty layer drops
    # out of the assembled prompt and is not part of the ordering chain.
    world_idx = result.positive_prompt.find("amber")
    casting_idx = result.positive_prompt.find("inquisitor")
    culture_idx = result.positive_prompt.find("monastic severity")
    safety_idx = result.positive_prompt.find("solo character focus")
    assert 0 <= world_idx < casting_idx < culture_idx < safety_idx
    assert "painterly" not in result.positive_prompt


def test_compose_illustration_specific_location_contains_landmark(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="arriving",
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    result = composer.compose(t)
    assert "watchtower" in result.positive_prompt
    assert "arriving" in result.positive_prompt


def test_compose_populates_layers_list(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    result = composer.compose(t)
    slots = {layer.slot for layer in result.layers}
    assert "CASTING" in slots
    assert "DIRECTION_CAMERA" in slots
    # World overrides genre: the WORLD layer is present, GENRE is suppressed.
    assert "ART_SENSIBILITY.WORLD" in slots
    assert "ART_SENSIBILITY.GENRE" not in slots


def test_compose_seed_is_deterministic(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    a = composer.compose(t)
    b = composer.compose(t)
    assert a.seed == b.seed
    assert a.positive_prompt == b.positive_prompt


def test_eviction_order_drops_location_flourish_first(
    composer: PromptComposer,
) -> None:
    # Inject a deliberately oversized action to force eviction.
    # Build a target that exceeds 512 tokens when every layer is full.
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="x " * 500,  # forces overflow past background LOD
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    result = composer.compose(t)
    # Location flourish should evict before any identity-floor slot.
    assert any(
        "LOCATION" in dl or "DIRECTION_ACTION" in dl
        for dl in result.dropped_layers
    )
    assert not any("CASTING" in dl for dl in result.dropped_layers)


def test_world_style_is_in_identity_floor_not_eviction_order() -> None:
    """Post-2026-04-29 architecture shift: the visual style system was
    decomposed so the world (not the genre) carries the art-movement
    lineage — Mucha for aureate_span, McQuarrie/Leone for coyote_star.
    The genre baseline is now neutral (universal sci-fi framing + the
    Z-Image anti-text-bleed safety clause). If the WORLD layer is ever
    dropped under budget pressure, generated images collapse to generic
    photoreal CG with no styling and visible prose-bleed artifacts.

    This test guards the data tables directly so a future tweak that
    re-introduces WORLD into _EVICTION_ORDER fails immediately, before
    a single image is rendered. Behavioral coverage of the same property
    via integration tests is brittle — the testfixture's flourish layers
    can absorb common overflows before WORLD is reached, and the bug
    only surfaces in production with heavier real-world payloads."""
    assert "ART_SENSIBILITY.WORLD" in PromptComposer._IDENTITY_FLOOR, (
        "ART_SENSIBILITY.WORLD must be in the identity floor — it carries "
        "the load-bearing art-movement lineage post-architecture-shift"
    )
    eviction_slots = [label for label, _ in PromptComposer._EVICTION_ORDER]
    assert "ART_SENSIBILITY.WORLD" not in eviction_slots, (
        "ART_SENSIBILITY.WORLD must not appear in the eviction order — "
        "evicting it produces photoreal CG renders with no painterly styling. "
        f"Current eviction order: {eviction_slots}"
    )

    # The genre layer remains in the floor too; both must be preserved
    # because each carries half of the styling contract (genre = baseline
    # + safety clause; world = art-movement lineage).
    assert "ART_SENSIBILITY.GENRE" in PromptComposer._IDENTITY_FLOOR


def test_identity_floor_breach_raises_budget_error(
    composer: PromptComposer,
) -> None:
    from sidequest_daemon.media.recipes import BudgetError  # noqa: PLC0415
    # npc:verbose has a massive solo description (484+ tokens) that exceeds
    # the identity floor even after all non-floor slots are evicted.
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:verbose"],
        action="word " * 600,
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    with pytest.raises(BudgetError):
        composer.compose(t)


def test_budget_downgrade_participants_before_slot_eviction(
    composer: PromptComposer,
) -> None:
    # Six participants + rich culture/environment — forces downgrade
    # from planned LODs toward `background` for tail participants.
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=[
            "npc:rux", "npc:mira",
            # Reuse existing two characters; participant duplication is OK
            # because LOD downgrade operates on ordinal position not identity.
            "npc:rux", "npc:mira", "npc:rux", "npc:mira",
        ],
        action="argument " * 50,
        location="where:testworld/the_lookout",
        camera=CameraPreset.scene,
    )
    result = composer.compose(t)
    # Even under budget pressure, at least one CASTING layer survives for
    # every participant slot (never drop below background).
    casting_sources = [lay.source for lay in result.layers if lay.slot == "CASTING"]
    assert len(casting_sources) == len(t.participants)


def test_unknown_character_raises_catalog_miss(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:nobody",
    )
    with pytest.raises(CatalogMissError) as exc:
        composer.compose(t)
    assert exc.value.missing_id == "npc:nobody"


def test_unknown_place_raises_catalog_miss(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="poi", world="testworld", genre="testgenre",
        place="where:testworld/fabricated",
    )
    with pytest.raises(CatalogMissError):
        composer.compose(t)


def test_compose_emits_otel_span(composer: PromptComposer, monkeypatch) -> None:
    emitted: list[dict] = []

    def fake_emit(name: str, payload: dict) -> None:
        emitted.append({"name": name, "payload": payload})

    monkeypatch.setattr(
        "sidequest_daemon.media.prompt_composer._emit_watcher_event",
        fake_emit,
    )
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    composer.compose(t)
    assert any(e["name"] == "render.prompt_composed" for e in emitted)
    span = next(e for e in emitted if e["name"] == "render.prompt_composed")
    assert span["payload"]["kind"] == "portrait"
    assert span["payload"]["world"] == "testworld"
    assert "layers" in span["payload"]
    assert any(layer["slot"] == "CASTING" for layer in span["payload"]["layers"])
    # Bug #2a (playtest 2026-04-26) lie-detector flags, updated for
    # world-overrides-genre: testworld ships a visual_style, so it overrides
    # the genre layer. world_style_applied is the load-bearing flag and must be
    # True; genre_style_applied is False because the genre layer is suppressed.
    assert "genre_style_applied" in span["payload"]
    assert "world_style_applied" in span["payload"]
    assert span["payload"]["world_style_applied"] is True
    assert span["payload"]["genre_style_applied"] is False


def test_compose_otel_world_flag_true_for_styled_world(
    composer: PromptComposer, monkeypatch
) -> None:
    """Bug #2a (playtest 2026-04-26): when both genre and world have a
    non-empty ``positive_suffix`` and the recipe includes the WORLD layer,
    the ``world_style_applied`` lie-detector flag must be True. This is
    the test that would have caught the grimvault regression — it asserts
    the flag flips for a fixture where the world style is genuinely
    populated and the recipe consumes it (illustration recipe)."""
    emitted: list[dict] = []

    def fake_emit(name: str, payload: dict) -> None:
        emitted.append({"name": name, "payload": payload})

    monkeypatch.setattr(
        "sidequest_daemon.media.prompt_composer._emit_watcher_event",
        fake_emit,
    )
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="arriving",
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    composer.compose(t)
    span = next(e for e in emitted if e["name"] == "render.prompt_composed")
    assert span["payload"]["world_style_applied"] is True, (
        "world_style_applied must be True when world's visual_style.yaml "
        "has a non-empty positive_suffix and the recipe consumes WORLD; "
        "False here means the regression that hit grimvault is back"
    )
    # World overrides genre, so the genre layer is suppressed when the world
    # ships its own style.
    assert span["payload"]["genre_style_applied"] is False


GOLDEN_DIR = Path(__file__).parent / "golden"


def _assert_golden(name: str, actual: str) -> None:
    golden_path = GOLDEN_DIR / name
    if not golden_path.exists():
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_text(actual)
        pytest.fail(f"Wrote new golden {name}; inspect and re-run.")
    expected = golden_path.read_text()
    assert actual == expected, f"Golden mismatch for {name}. "\
        f"Delete {golden_path} and re-run to regenerate."


def test_golden_portrait(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    _assert_golden("portrait_npc_rux.txt", composer.compose(t).positive_prompt + "\n")


def test_golden_poi(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="poi", world="testworld", genre="testgenre",
        place="where:testworld/the_lookout",
    )
    _assert_golden("poi_the_lookout.txt", composer.compose(t).positive_prompt + "\n")


def test_golden_illustration_specific(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="arriving at dusk",
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    _assert_golden(
        "illustration_specific_location.txt",
        composer.compose(t).positive_prompt + "\n",
    )


def test_golden_illustration_archetypal(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="drinking",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    _assert_golden(
        "illustration_archetypal_location.txt",
        composer.compose(t).positive_prompt + "\n",
    )


def test_golden_illustration_topdown(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="ambush from the doorway",
        location="where:testworld/the_lookout", camera=CameraPreset.topdown_90,
    )
    _assert_golden(
        "illustration_topdown_90.txt",
        composer.compose(t).positive_prompt + "\n",
    )


# ---------------------------------------------------------------------------
# House "solo character focus" clause is a single-subject directive — it must
# reach portraits and illustrations but NOT POI landscapes (2026-06-04 fix:
# the clause was leaking onto every wry_whimsy/wonderland POI and fighting the
# wide establishing framing).
# ---------------------------------------------------------------------------

def test_poi_omits_house_solo_character_clause(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="poi", world="testworld", genre="testgenre",
        place="where:testworld/the_lookout",
    )
    prompt = composer.compose(t).positive_prompt
    assert "solo character focus" not in prompt
    assert "detailed distinctive features" not in prompt


def test_portrait_retains_house_solo_character_clause(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    assert (
        "solo character focus, detailed distinctive features"
        in composer.compose(t).positive_prompt
    )


def test_illustration_retains_house_solo_character_clause(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="thinking",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    assert (
        "solo character focus, detailed distinctive features"
        in composer.compose(t).positive_prompt
    )


# ---------------------------------------------------------------------------
# portrait_positive_suffix — a world may declare a portrait-only style suffix
# (denser engraving + crowd suppression for close subjects) distinct from the
# shared positive_suffix used by POIs/illustrations. Wired into the daemon's
# catalog-compose path (render_common honored it only in the dead legacy path).
# Backwards-compatible: a world without the field falls back to positive_suffix.
# ---------------------------------------------------------------------------

def test_style_catalog_reads_portrait_positive_suffix(tmp_path: Path) -> None:
    world_dir = tmp_path / "g" / "worlds" / "w"
    world_dir.mkdir(parents=True)
    (world_dir / "visual_style.yaml").write_text(
        "positive_suffix: SHARED SUFFIX\n"
        "portrait_positive_suffix: PORTRAIT ONLY SUFFIX\n"
    )
    cat = StyleCatalog.load(tmp_path, genre="g", world="w")
    assert cat.get_world("g", "w") == "SHARED SUFFIX"
    assert cat.get_world_portrait("g", "w") == "PORTRAIT ONLY SUFFIX"


def test_style_catalog_portrait_suffix_absent_returns_none(
    tmp_path: Path,
) -> None:
    world_dir = tmp_path / "g" / "worlds" / "w"
    world_dir.mkdir(parents=True)
    (world_dir / "visual_style.yaml").write_text(
        "positive_suffix: SHARED SUFFIX\n"
    )
    cat = StyleCatalog.load(tmp_path, genre="g", world="w")
    assert cat.get_world_portrait("g", "w") is None


def test_portrait_prefers_portrait_positive_suffix() -> None:
    base = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    styles = StyleCatalog(
        genre_tokens=base._genre,
        world_tokens=base._world,
        culture_tokens=base._culture,
        world_portrait_tokens={
            ("testgenre", "testworld"): "PORTRAIT ENGRAVING SUFFIX"
        },
    )
    comp = PromptComposer(
        recipes=RecipeLoader.from_file(REPO_ROOT / "recipes.yaml"),
        cameras=CameraLoader.from_file(REPO_ROOT / "cameras.yaml"),
        characters=CharacterCatalog.load(
            FIXTURE_ROOT, genre="testgenre", world="testworld"
        ),
        places=PlaceCatalog.load(
            FIXTURE_ROOT, genre="testgenre", world="testworld"
        ),
        styles=styles,
    )
    portrait = comp.compose(
        RenderTarget(
            kind="portrait", world="testworld", genre="testgenre",
            character="npc:rux",
        )
    ).positive_prompt
    assert "PORTRAIT ENGRAVING SUFFIX" in portrait
    assert "weathered surfaces" not in portrait

    poi = comp.compose(
        RenderTarget(
            kind="poi", world="testworld", genre="testgenre",
            place="where:testworld/the_lookout",
        )
    ).positive_prompt
    assert "weathered surfaces" in poi
    assert "PORTRAIT ENGRAVING SUFFIX" not in poi
