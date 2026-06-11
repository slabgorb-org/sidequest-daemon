import pytest
from pydantic import ValidationError

from sidequest_daemon.media.recipes import (
    BudgetError,
    CameraPreset,
    CatalogMissError,
    ComposedPrompt,
    LayerContribution,
    LOD,
    PlaceLOD,
    Recipe,
    RenderTarget,
    Slot,
)


def test_slot_has_required_members():
    assert Slot.CASTING.value == "casting"
    assert Slot.LOCATION.value == "location"
    assert Slot.DIRECTION_ACTION.value == "direction_action"
    assert Slot.DIRECTION_CAMERA.value == "direction_camera"
    assert Slot.ART_SENSIBILITY.value == "art_sensibility"


def test_lod_levels():
    assert [m.value for m in LOD] == ["solo", "long", "short", "background"]


def test_place_lod_levels():
    assert [m.value for m in PlaceLOD] == ["solo", "backdrop"]


def test_camera_preset_count_is_eighteen():
    assert len(CameraPreset) == 18


def test_camera_preset_contains_canary_presets():
    assert CameraPreset.portrait_3q in CameraPreset
    assert CameraPreset.topdown_90 in CameraPreset
    assert CameraPreset.extreme_closeup_leone in CameraPreset
    assert CameraPreset.scene in CameraPreset


def test_portrait_requires_character():
    with pytest.raises(ValidationError):
        RenderTarget(kind="portrait", world="w", genre="g")


def test_portrait_rejects_illustration_fields():
    with pytest.raises(ValidationError):
        RenderTarget(
            kind="portrait",
            world="w",
            genre="g",
            character="npc:rux",
            action="swinging an axe",
        )


def test_poi_requires_specific_place():
    with pytest.raises(ValidationError):
        RenderTarget(kind="poi", world="w", genre="g")
    with pytest.raises(ValidationError):
        # Archetypal place is not allowed for POI renders.
        RenderTarget(
            kind="poi",
            world="w",
            genre="g",
            place="where:low_fantasy/tavern",
        )


def test_poi_accepts_specific_place():
    target = RenderTarget(
        kind="poi",
        world="flickering_reach",
        genre="mutant_wasteland",
        place="where:flickering_reach/the_lookout",
    )
    assert target.place == "where:flickering_reach/the_lookout"


def test_illustration_requires_participants_action_location_camera():
    with pytest.raises(ValidationError):
        RenderTarget(kind="illustration", world="w", genre="g")
    with pytest.raises(ValidationError):
        RenderTarget(
            kind="illustration",
            world="w",
            genre="g",
            participants=["npc:a"],
            action="",
            location="where:w/x",
            camera=CameraPreset.scene,
        )


def test_illustration_accepts_archetypal_or_specific_location():
    specific = RenderTarget(
        kind="illustration",
        world="w",
        genre="g",
        participants=["npc:a"],
        action="talking",
        location="where:w/x",
        camera=CameraPreset.scene,
    )
    archetypal = RenderTarget(
        kind="illustration",
        world="w",
        genre="g",
        participants=["npc:a"],
        action="talking",
        location="where:g/tavern",
        camera=CameraPreset.scene,
    )
    assert specific.location.startswith("where:")
    assert archetypal.location.startswith("where:")


def test_portrait_default_camera_is_portrait_3q():
    target = RenderTarget(
        kind="portrait",
        world="w",
        genre="g",
        character="npc:rux",
    )
    assert target.camera is None  # recipe supplies default — see Task 7


def test_recipe_has_slot_bindings():
    r = Recipe(
        kind="portrait",
        casting="character",
        location="background",
        direction_action="pose",
        direction_camera="portrait_3q",
        art_sensibility=["GENRE", "WORLD", "CULTURE"],
    )
    assert r.direction_camera == "portrait_3q"
    assert r.art_sensibility == ["GENRE", "WORLD", "CULTURE"]


def test_layer_contribution_roundtrip():
    lc = LayerContribution(
        slot="CASTING",
        source="npc:rux",
        tokens="a gaunt inquisitor...",
        estimated_tokens=11,
    )
    assert lc.source == "npc:rux"


def test_composed_prompt_carries_layers_and_warnings():
    cp = ComposedPrompt(
        positive_prompt="...",
        clip_prompt="...",
        negative_prompt="...",
        worker_type="zimage",
        seed=0,
        layers=[],
        dropped_layers=[],
        warnings=[],
    )
    assert cp.dropped_layers == []


def test_catalog_miss_error_carries_source_and_id():
    err = CatalogMissError(source="CharacterCatalog", missing_id="npc:ghost")
    assert "CharacterCatalog" in str(err)
    assert "npc:ghost" in str(err)


def test_budget_error_carries_breakdown():
    err = BudgetError(
        message="identity floor breached",
        breakdown={"CASTING": 200, "ART_SENSIBILITY.GENRE": 320},
    )
    assert "identity floor" in str(err)
    assert err.breakdown["CASTING"] == 200
