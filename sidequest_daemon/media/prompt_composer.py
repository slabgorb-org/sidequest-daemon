"""Catalog-driven prompt composer. See spec:
docs/superpowers/specs/2026-04-24-explicit-visual-recipes-design.md
"""

from __future__ import annotations

import hashlib
import logging

from sidequest_daemon.media.camera_specs import CameraLoader
from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    PlaceCatalog,
    StyleCatalog,
)
from sidequest_daemon.media.recipe_loader import RecipeLoader
from sidequest_daemon.media.recipes import (
    BudgetError,
    CameraPreset,
    ComposedPrompt,
    LayerContribution,
    LOD,
    PlaceLOD,
    RenderTarget,
)
from sidequest_daemon.media.zimage_config import get_zimage_config
from sidequest_daemon.renderer.models import RenderTier
from sidequest_daemon.telemetry import emit_watcher_event as _emit_watcher_event

# Story 45-38: ``RenderTarget.kind`` is portrait/poi/illustration; the
# Z-Image config is keyed by ``RenderTier``. Map kind → a representative
# tier so the composer can surface ``model_variant`` + ``steps`` on the
# render.prompt_composed OTEL span. The mapping is lossy on width/height
# (PORTRAIT_SQUARE vs PORTRAIT, CARTOGRAPHY vs LANDSCAPE) but matches on
# the model_variant + steps axes that actually drive AC5.
_KIND_TO_TIER: dict[str, RenderTier] = {
    "portrait": RenderTier.PORTRAIT,
    "poi": RenderTier.LANDSCAPE,
    "illustration": RenderTier.SCENE_ILLUSTRATION,
}

log = logging.getLogger(__name__)

_TOKEN_LIMIT = 512
_TOKENS_PER_WORD = 1.3
_BASE_NEGATIVES = (
    "watermark, signature, text, blurry, deformed, extra limbs, "
    "photograph, photorealistic, hyperrealistic, smooth skin, CGI"
)
_HOUSE_SAFETY_CLAUSE = "solo character focus, detailed distinctive features"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * _TOKENS_PER_WORD))


class PromptComposer:
    # Eviction order (most-evictable → least). Identity floor is below.
    #
    # ART_SENSIBILITY.WORLD is NOT in this list. The visual style system
    # was decomposed 2026-04-29 so the world (not the genre) carries the
    # art-movement lineage — Mucha for aureate_span, McQuarrie/Leone for
    # coyote_star. Evicting WORLD produces photoreal CG with no painterly
    # styling. The world layer belongs in the identity floor below; if
    # the floor genuinely cannot fit the budget, BudgetError surfaces the
    # real problem instead of silently degrading style.
    _EVICTION_ORDER: list[tuple[str, int]] = [
        # (slot_label, preserve_token_count)
        ("LOCATION.flourish", 8),
        ("DIRECTION_ACTION.flourish", 8),
        ("ART_SENSIBILITY.CULTURE.flourish", 12),
    ]

    # Identity floor — never evict below these.
    _IDENTITY_FLOOR: set[str] = {
        "CASTING",
        "DIRECTION_CAMERA",
        "ART_SENSIBILITY.GENRE",
        "ART_SENSIBILITY.WORLD",
    }

    # LOD degradation ladder — ordered from richest to most minimal.
    _LOD_ORDER = [LOD.SOLO, LOD.LONG, LOD.SHORT, LOD.BACKGROUND]

    def __init__(
        self,
        *,
        recipes: RecipeLoader,
        cameras: CameraLoader,
        characters: CharacterCatalog,
        places: PlaceCatalog,
        styles: StyleCatalog,
    ) -> None:
        self._recipes = recipes
        self._cameras = cameras
        self._characters = characters
        self._places = places
        self._styles = styles

    def compose(self, target: RenderTarget) -> ComposedPrompt:
        plan = self._character_lod_plan(target)
        layers = self._resolve_all_layers_with_plan(target, plan)
        dropped: list[str] = []
        warnings: list[str] = []

        # 1. Participant LOD downgrade (preserves presence of every participant).
        while sum(lay.estimated_tokens for lay in layers) > _TOKEN_LIMIT:
            downgraded = self._downgrade_one_participant(plan)
            if not downgraded:
                break
            layers = self._resolve_all_layers_with_plan(target, plan)

        # 2. Slot-level eviction.
        if sum(lay.estimated_tokens for lay in layers) > _TOKEN_LIMIT:
            layers, dropped = self._apply_slot_eviction(layers)
            warnings.append(
                f"token budget eviction applied: "
                f"{sum(lay.estimated_tokens for lay in layers)}/{_TOKEN_LIMIT}",
            )

        if sum(lay.estimated_tokens for lay in layers) > _TOKEN_LIMIT:
            raise BudgetError(
                "identity floor breached",
                breakdown={lay.slot: lay.estimated_tokens for lay in layers},
            )

        positive = self._assemble(layers)
        clip = self._build_clip(layers)
        negative = self._build_negative(target)

        # Bug #2a (playtest 2026-04-26) lie-detector: record whether the
        # genre and world ART_SENSIBILITY layers actually contributed any
        # tokens. The grimvault regression had the world layer silently
        # empty because the visual_style.yaml used ``style_prompt`` instead
        # of ``positive_suffix``; without an explicit "world style applied?"
        # signal the GM panel could not tell a styled render from a
        # styleless one. Since world overrides genre (see
        # _resolve_art_sensibility), ``world_style_applied`` is the
        # load-bearing health flag — it MUST be true on any styled render.
        # ``genre_style_applied`` is true only in the fallback case where a
        # world ships no visual_style of its own; under normal override it is
        # False because the genre layer is intentionally suppressed.
        genre_layer_applied = any(
            layer.slot == "ART_SENSIBILITY.GENRE" and layer.tokens.strip()
            for layer in layers
        )
        world_layer_applied = any(
            layer.slot == "ART_SENSIBILITY.WORLD" and layer.tokens.strip()
            for layer in layers
        )

        # Story 45-38 AC5: surface the Z-Image variant + step count on the
        # span so the GM panel can tell turbo (in-session) and high-fidelity
        # (pre-gen) renders apart at a glance. The composer reports the
        # *requested* tier — the worker emits its own span with the variant
        # actually loaded at render time, so a divergence between the two
        # is itself diagnostic.
        zimage_cfg = get_zimage_config(
            _KIND_TO_TIER[target.kind], fidelity=target.fidelity,
        )

        _emit_watcher_event(
            "render.prompt_composed",
            {
                "kind": target.kind,
                "world": target.world,
                "genre": target.genre,
                "total_estimated_tokens": sum(
                    layer.estimated_tokens for layer in layers
                ),
                "layers": [
                    {
                        "slot": layer.slot,
                        "source": layer.source,
                        "estimated_tokens": layer.estimated_tokens,
                    }
                    for layer in layers
                ],
                "dropped_layers": dropped,
                "warnings": warnings,
                # Lie-detector flags — see comment above.
                "genre_style_applied": genre_layer_applied,
                "world_style_applied": world_layer_applied,
                # Story 45-38 AC5
                "fidelity": target.fidelity,
                "model_variant": zimage_cfg.model_variant,
                "steps": zimage_cfg.steps,
            },
        )

        return ComposedPrompt(
            positive_prompt=positive,
            clip_prompt=clip,
            negative_prompt=negative,
            worker_type=self._select_worker(target),
            seed=self._derive_seed(target),
            layers=layers,
            dropped_layers=dropped,
            warnings=warnings,
        )

    def _downgrade_one_participant(self, plan: dict[str, LOD]) -> bool:
        """Downgrade the lowest-priority participant one LOD rung. Returns True
        if a downgrade was applied, False if every participant is already at
        background."""
        # Operate in reverse order so tail participants downgrade first.
        for ref in reversed(list(plan.keys())):
            current = plan[ref]
            idx = self._LOD_ORDER.index(current)
            if idx < len(self._LOD_ORDER) - 1:
                plan[ref] = self._LOD_ORDER[idx + 1]
                return True
        return False

    def _resolve_all_layers_with_plan(
        self, target: RenderTarget, plan: dict[str, LOD]
    ) -> list[LayerContribution]:
        art = self._resolve_art_sensibility(target)
        casting = self._resolve_casting_with_plan(target, plan)
        location = self._resolve_location(target)
        action = [self._resolve_direction_action(target)]
        camera = [self._resolve_direction_camera(target)]

        # Split art sensibility: GENRE/WORLD go early, CULTURE goes late.
        genre_world = [
            layer for layer in art
            if layer.slot in ("ART_SENSIBILITY.GENRE", "ART_SENSIBILITY.WORLD")
        ]
        culture = [layer for layer in art if layer.slot == "ART_SENSIBILITY.CULTURE"]

        return genre_world + casting + location + action + camera + culture

    def _resolve_casting_with_plan(
        self, target: RenderTarget, plan: dict[str, LOD]
    ) -> list[LayerContribution]:
        if target.kind == "poi":
            return self._resolve_casting(target)
        layers: list[LayerContribution] = []
        refs: list[str] = (
            [target.character]
            if target.kind == "portrait" and target.character
            else list(target.participants)
        )
        for ref in refs:
            lod = plan.get(ref, LOD.SOLO)
            tokens = self._characters.get(ref)
            text = tokens.descriptions[lod]
            layers.append(
                LayerContribution(
                    slot="CASTING",
                    source=ref,
                    tokens=text,
                    estimated_tokens=_estimate_tokens(text),
                ),
            )
        return layers

    def _apply_slot_eviction(
        self, layers: list[LayerContribution]
    ) -> tuple[list[LayerContribution], list[str]]:
        result = [lay.model_copy() for lay in layers]
        dropped: list[str] = []

        def _truncate(layer: LayerContribution, keep_tokens: int) -> None:
            words = layer.tokens.split()
            keep_words = max(1, int(keep_tokens / _TOKENS_PER_WORD))
            layer.tokens = " ".join(words[:keep_words])
            layer.estimated_tokens = _estimate_tokens(layer.tokens)

        for eviction_label, preserve in self._EVICTION_ORDER:
            if sum(lay.estimated_tokens for lay in result) <= _TOKEN_LIMIT:
                break

            base_slot, _, flourish = eviction_label.partition(".")
            if flourish == "flourish":
                for layer in result:
                    if layer.slot == base_slot or layer.slot.startswith(
                        base_slot + "."
                    ):
                        if layer.estimated_tokens > preserve:
                            _truncate(layer, preserve)
                            dropped.append(
                                f"{layer.slot}:{layer.source}:flourish",
                            )
            else:
                # Drop entirely
                before = len(result)
                result = [lay for lay in result if lay.slot != eviction_label]
                if len(result) < before:
                    dropped.append(eviction_label)

        return result, dropped

    def _character_lod_plan(self, target: RenderTarget) -> dict[str, LOD]:
        if target.kind == "portrait":
            assert target.character is not None
            return {target.character: LOD.SOLO}
        if target.kind == "illustration":
            participants = list(target.participants)
            n = len(participants)
            # Pingpong 2026-04-30: landscape tier with no characters
            # (environmental scene, narrator-emitted prose subject —
            # routed to ``kind=illustration`` with empty participants
            # per ``zimage_mlx_worker.py:137-146``) crashed at
            # ``participants[0]: LOD.LONG`` because the n>=5 fall-
            # through branch unconditionally indexed [0]/[1]/[2].
            # 2 of 2 landscape dispatches in the 4P MP playtest
            # silently failed — daemon closed socket on IndexError,
            # server logged ``render.reply_unavailable``, and the
            # corridor opening shot + half-landing scene were missing
            # from the Scrapbook for all 4 players.
            #
            # Fix: empty-participants is a legitimate state for
            # environmental landscape illustrations — the prose
            # subject + ART_SENSIBILITY layers carry the visual without
            # a casting plan. Return an empty plan so ``_resolve_casting``
            # produces zero CASTING layers and composition continues
            # cleanly. Mirrors the POI fall-through at the bottom of
            # this function (POI also has no characters in frame).
            if n == 0:
                return {}
            if n == 1:
                return {participants[0]: LOD.SOLO}
            if n == 2:
                return {p: LOD.LONG for p in participants}
            if 3 <= n <= 4:
                return {
                    **{participants[0]: LOD.LONG},
                    **{p: LOD.SHORT for p in participants[1:]},
                }
            # n >= 5
            return {
                participants[0]: LOD.LONG,
                participants[1]: LOD.SHORT,
                participants[2]: LOD.SHORT,
                **{p: LOD.BACKGROUND for p in participants[3:]},
            }
        return {}  # POI targets have no character plan

    def _place_lod_for(self, target: RenderTarget) -> PlaceLOD:
        if target.kind == "poi":
            return PlaceLOD.SOLO
        if target.kind == "illustration":
            return PlaceLOD.BACKDROP
        if target.kind == "portrait" and target.background:
            return PlaceLOD.BACKDROP
        return PlaceLOD.SOLO  # unreachable for current targets, safe default

    def _resolve_casting(
        self, target: RenderTarget
    ) -> list[LayerContribution]:
        if target.kind in ("portrait", "illustration"):
            plan = self._character_lod_plan(target)
            layers: list[LayerContribution] = []
            for ref, lod in plan.items():
                tokens = self._characters.get(ref)
                text = tokens.descriptions[lod]
                layers.append(
                    LayerContribution(
                        slot="CASTING",
                        source=ref,
                        tokens=text,
                        estimated_tokens=_estimate_tokens(text),
                    ),
                )
            return layers
        if target.kind == "poi":
            assert target.place is not None
            place = self._places.get(target.place)
            lod = self._place_lod_for(target)
            text = place.landmark[lod]
            return [
                LayerContribution(
                    slot="CASTING",
                    source=target.place,
                    tokens=text,
                    estimated_tokens=_estimate_tokens(text),
                ),
            ]
        return []

    def _resolve_location(
        self, target: RenderTarget
    ) -> list[LayerContribution]:
        if target.kind == "portrait":
            if not target.background:
                return []
            place = self._places.get(target.background)
            lod = PlaceLOD.BACKDROP
            text = place.environment[lod]
            return [
                LayerContribution(
                    slot="LOCATION",
                    source=target.background,
                    tokens=text,
                    estimated_tokens=_estimate_tokens(text),
                ),
            ]
        if target.kind == "poi":
            assert target.place is not None
            place = self._places.get(target.place)
            text = place.environment[PlaceLOD.SOLO]
            return [
                LayerContribution(
                    slot="LOCATION",
                    source=target.place,
                    tokens=text,
                    estimated_tokens=_estimate_tokens(text),
                ),
            ]
        if target.kind == "illustration":
            assert target.location is not None
            # Illustrations support transient scene settings that have not
            # been promoted into the world's PlaceCatalog (corridors mid-
            # transit, breached compartments, ad-hoc encounters). When the
            # caller cannot supply a `where:<slug>` ref, we skip the
            # LOCATION layer rather than raise — the action prose carries
            # the setting. This is documented behaviour, NOT a silent
            # fallback: the empty-location case is the by-design path.
            #
            # A non-empty ref that doesn't use the `where:` scheme is a
            # contract violation (server is shipping free-form prose
            # instead of a catalog ref) and is allowed to surface as
            # ValueError from PlaceCatalog.get — `_handle_client` converts
            # those to structured `COMPOSE_FAILED` JSON-RPC errors so the
            # GM panel sees the real reason instead of an EOF.
            if not target.location:
                return []
            place = self._places.get(target.location)
            lod = PlaceLOD.BACKDROP
            parts: list[str] = []
            if place.landmark[lod]:
                parts.append(place.landmark[lod])
            if place.environment[lod]:
                parts.append(place.environment[lod])
            text = ", ".join(parts)
            return [
                LayerContribution(
                    slot="LOCATION",
                    source=target.location,
                    tokens=text,
                    estimated_tokens=_estimate_tokens(text),
                ),
            ]
        return []

    def _resolve_direction_action(
        self, target: RenderTarget
    ) -> LayerContribution:
        if target.kind == "portrait":
            assert target.character is not None
            if target.pose_override:
                text = target.pose_override
                source = "inline"
            else:
                text = self._characters.get(target.character).default_pose
                source = f"{target.character}.default_pose"
        elif target.kind == "poi":
            assert target.place is not None
            place = self._places.get(target.place)
            text = place.description[PlaceLOD.SOLO]
            source = target.place
        elif target.kind == "illustration":
            text = target.action
            source = "inline"
        else:
            text, source = "", "inline"
        return LayerContribution(
            slot="DIRECTION_ACTION",
            source=source,
            tokens=text,
            estimated_tokens=_estimate_tokens(text),
        )

    def _resolve_direction_camera(
        self, target: RenderTarget
    ) -> LayerContribution:
        recipe = self._recipes.get(target.kind)
        if recipe.direction_camera == "{camera}":
            assert target.camera is not None
            preset = target.camera
        else:
            preset = CameraPreset(recipe.direction_camera)
        spec = self._cameras.get(preset)
        return LayerContribution(
            slot="DIRECTION_CAMERA",
            source=preset.value,
            tokens=spec.prompt,
            estimated_tokens=_estimate_tokens(spec.prompt),
        )

    def _resolve_art_sensibility(
        self, target: RenderTarget
    ) -> list[LayerContribution]:
        recipe = self._recipes.get(target.kind)
        layers: list[LayerContribution] = []

        # World overrides genre — it does NOT blend on top of it. When a world
        # ships its own visual_style.yaml, that style fully replaces the genre
        # style; the genre layer is emitted only as a fallback for a world with
        # no style of its own. Blending stacked the genre's "extreme close-up on
        # weathered face" grammar onto every world render, which hallucinated a
        # face into POI landscapes (the spaghetti_western double-image, 2026-05-25).
        # The 2026-04-29 visual-style decomposition already put the art-movement
        # lineage on the world layer (see _IDENTITY_FLOOR note); override makes
        # that authoritative. World suffixes must therefore be self-complete
        # (carry their own no-text/safety clause).
        world_text = ""
        if "WORLD" in recipe.art_sensibility:
            world_text = self._styles.get_world(target.genre, target.world)

        for layer_name in recipe.art_sensibility:
            if layer_name == "GENRE":
                if world_text.strip():
                    continue
                text = self._styles.get_genre(target.genre)
                layers.append(
                    LayerContribution(
                        slot="ART_SENSIBILITY.GENRE",
                        source=f"genre:{target.genre}",
                        tokens=text,
                        estimated_tokens=_estimate_tokens(text),
                    ),
                )
            elif layer_name == "WORLD":
                layers.append(
                    LayerContribution(
                        slot="ART_SENSIBILITY.WORLD",
                        source=f"world:{target.genre}/{target.world}",
                        tokens=world_text,
                        estimated_tokens=_estimate_tokens(world_text),
                    ),
                )
            elif layer_name == "CULTURE":
                cultures = self._collect_cultures(target)
                for culture in cultures:
                    text = self._styles.get_culture(
                        target.genre, target.world, culture,
                    )
                    layers.append(
                        LayerContribution(
                            slot="ART_SENSIBILITY.CULTURE",
                            source=f"culture:{target.genre}/{target.world}/{culture}",
                            tokens=text,
                            estimated_tokens=_estimate_tokens(text),
                        ),
                    )
        return layers

    def _collect_cultures(self, target: RenderTarget) -> list[str]:
        seen: list[str] = []
        refs: list[str] = []
        if target.kind == "portrait":
            assert target.character is not None
            refs = [target.character]
        elif target.kind == "illustration":
            refs = list(target.participants)
        elif target.kind == "poi":
            assert target.place is not None
            place = self._places.get(target.place)
            if place.controlling_culture:
                return [place.controlling_culture]
            return []
        for ref in refs:
            c = self._characters.get(ref).culture
            if c and c not in seen:
                seen.append(c)
        return seen

    def _assemble(self, layers: list[LayerContribution]) -> str:
        # Order: GENRE, WORLD, CASTING, LOCATION, DIRECTION_ACTION,
        # DIRECTION_CAMERA, CULTURE, safety clause.
        by_slot: dict[str, list[str]] = {}
        for layer in layers:
            if layer.tokens:
                by_slot.setdefault(layer.slot, []).append(layer.tokens)

        ordered: list[str] = []
        for slot in (
            "ART_SENSIBILITY.GENRE",
            "ART_SENSIBILITY.WORLD",
            "CASTING",
            "LOCATION",
            "DIRECTION_ACTION",
            "DIRECTION_CAMERA",
            "ART_SENSIBILITY.CULTURE",
        ):
            if slot in by_slot:
                ordered.extend(by_slot[slot])

        ordered.append(_HOUSE_SAFETY_CLAUSE)
        return ", ".join(ordered)

    def _build_clip(self, layers: list[LayerContribution]) -> str:
        # CLIP gets short style-adjacent keywords — GENRE + CAMERA.
        parts: list[str] = []
        for layer in layers:
            if layer.slot in ("ART_SENSIBILITY.GENRE", "DIRECTION_CAMERA"):
                parts.append(layer.tokens)
        return ", ".join(parts)

    def _build_negative(self, target: RenderTarget) -> str:
        # Preserve base + tier-specific negatives that used to hang on
        # TACTICAL_SKETCH / SCENE_ILLUSTRATION.
        parts = [_BASE_NEGATIVES]
        if (
            target.kind == "illustration"
            and target.camera
            and target.camera.value == "topdown_90"
        ):
            parts.append(
                "illegible text, blurry labels, overlapping tokens, "
                "3D perspective, realistic rendering",
            )
        return ", ".join(parts)

    def _select_worker(self, target: RenderTarget) -> str:  # noqa: ARG002
        # For now, all renders target the zimage worker. Style.preferred_model
        # override can be re-introduced once PR lands — tracked in Task 25.
        return "zimage"

    def _derive_seed(self, target: RenderTarget) -> int:
        key_parts: list[str] = [target.kind, target.world, target.genre]
        if target.character:
            key_parts.append(target.character)
        if target.place:
            key_parts.append(target.place)
        if target.location:
            key_parts.append(target.location)
        key_parts.extend(sorted(target.participants))
        key_parts.append(target.action)
        if target.camera:
            key_parts.append(target.camera.value)
        key = ":".join(key_parts)
        digest = hashlib.sha256(key.encode()).hexdigest()
        return int(digest[:8], 16) % (2**32)
