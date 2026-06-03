"""Recipe-driven prompt composition types."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator


class Slot(str, Enum):
    """Named slots that every recipe declares."""

    CASTING = "casting"
    LOCATION = "location"
    DIRECTION_ACTION = "direction_action"
    DIRECTION_CAMERA = "direction_camera"
    ART_SENSIBILITY = "art_sensibility"


class LOD(str, Enum):
    """Character level-of-detail for prompt contribution."""

    SOLO = "solo"
    LONG = "long"
    SHORT = "short"
    BACKGROUND = "background"


class PlaceLOD(str, Enum):
    """Place level-of-detail — subject or setting."""

    SOLO = "solo"
    BACKDROP = "backdrop"


class CameraPreset(str, Enum):
    """Enumerated camera presets — 17 total, stills-only (no motion)."""

    # Portrait framings
    portrait_3q = "portrait_3q"
    portrait_profile = "portrait_profile"
    portrait_closeup = "portrait_closeup"
    portrait_full_body = "portrait_full_body"
    # POI framings
    wide_establishing = "wide_establishing"
    low_angle_hero = "low_angle_hero"
    interior_wide = "interior_wide"
    aerial_oblique = "aerial_oblique"
    # Illustration framings
    scene = "scene"
    over_shoulder = "over_shoulder"
    wide_action = "wide_action"
    closeup_action = "closeup_action"
    topdown_90 = "topdown_90"
    # Signature shots
    extreme_closeup_leone = "extreme_closeup_leone"
    dutch_tilt = "dutch_tilt"
    single_point_perspective_kubrick = "single_point_perspective_kubrick"
    trunk_shot_tarantino = "trunk_shot_tarantino"


class RenderTarget(BaseModel):
    """The composable render input. One type serves all three kinds."""

    kind: Literal["portrait", "poi", "illustration"]
    world: str
    genre: str

    # Portrait
    character: str | None = None
    pose_override: str | None = None
    background: str | None = None  # optional where:<scope>/<slug>

    # POI
    place: str | None = None  # where:<world>/<slug> — specific only

    # Illustration
    participants: list[str] = []
    location: str | None = None  # where:<scope>/<slug> — specific or archetypal
    action: str = ""
    camera: CameraPreset | None = None

    # Story 45-38: render fidelity selects the Z-Image variant + step count.
    # Default flipped to ``high_fidelity`` 2026-05-02 — base z-image / 20 steps /
    # CFG 4 is the new floor across in-session and pre-gen paths. Pass
    # ``"turbo"`` explicitly when latency wins over painterly quality.
    # See sidequest_daemon.media.zimage_config.get_zimage_config.
    fidelity: Literal["turbo", "high_fidelity"] = "high_fidelity"

    # Debug/preview only
    lod_override: dict[str, LOD] | None = None

    @model_validator(mode="after")
    def _enforce_kind_shape(self) -> "RenderTarget":
        if self.kind == "portrait":
            if self.character is None:
                raise ValueError("portrait targets require `character`")
            if self.place or self.participants or self.action:
                raise ValueError(
                    "portrait targets must not set place/participants/action",
                )
        elif self.kind == "poi":
            if self.place is None:
                raise ValueError("poi targets require `place`")
            # Specific-place guard: the scope segment must match `world`.
            # Full validation (catalog lookup) is enforced by the composer;
            # this guard rejects obviously-archetypal refs at the schema level.
            _, _, scope_slug = self.place.partition(":")
            scope = scope_slug.split("/", 1)[0]
            if scope != self.world:
                raise ValueError(
                    f"poi targets must reference a specific place in world "
                    f"{self.world!r}; got scope {scope!r}",
                )
            if self.character or self.participants or self.action:
                raise ValueError(
                    "poi targets must not set character/participants/action",
                )
        elif self.kind == "illustration":
            # `participants` is OPTIONAL — empty list is valid for
            # environmental illustrations (no PCs in frame). Originally
            # required because illustration was conceived as a
            # "PCs-in-a-scene" kind; playtest 2026-04-30 surfaced the
            # missing case: when the narrator emits ``tier=landscape``
            # with prose subject (not a registered POI ``where:`` ref),
            # there's no participants-bearing kind to route to. ``poi``
            # demands a slug-resolvable place, ``portrait`` demands a
            # character. The right home for "environmental scene without
            # a registered POI" is illustration with empty participants —
            # the action prose carries the setting, ART_SENSIBILITY
            # layers carry the style, no PC casting layer is built.
            # The composer's `_character_lod_plan` iterates participants
            # and produces an empty plan when the list is empty, so this
            # change is internally consistent end-to-end.
            if not self.action:
                raise ValueError("illustration targets require `action`")
            # `location` is optional on illustrations. The original spec
            # required a `where:<scope>/<slug>` ref, but the server has
            # no slug-aware location tracking — it only carries free-form
            # narrator prose ("Engine Bay", "Corridor Deck Three") that
            # cannot be resolved against PlaceCatalog. Transient scenes
            # (corridors mid-transit, breached compartments, ad-hoc
            # encounters) carry their setting through the action prose
            # and the genre/world ART_SENSIBILITY layers; they do not
            # need a registered POI. When `location` IS supplied it must
            # still use the `where:` scheme — `PlaceCatalog.get` enforces
            # that downstream and surfaces a structured `COMPOSE_FAILED`
            # error if the server ships a non-`where:` ref.
            if self.camera is None:
                raise ValueError("illustration targets require `camera`")
        return self


class Recipe(BaseModel):
    """A canonical recipe — names the source bindings for each slot."""

    kind: Literal["portrait", "poi", "illustration"]
    casting: str
    location: str
    direction_action: str
    direction_camera: str  # a CameraPreset member name, or "{camera}" to
                           # pull from RenderTarget.camera
    art_sensibility: list[str]  # ordered cascade: e.g. ["GENRE","WORLD","CULTURE"]


class LayerContribution(BaseModel):
    slot: str
    source: str
    tokens: str
    estimated_tokens: int


class PostDirective(BaseModel):
    """Post-processing applied after the image renders.

    Defined here (rather than in ``camera_specs``) so ``ComposedPrompt`` can
    carry it without a circular import — ``camera_specs`` imports from
    ``recipes``, not the reverse. Re-exported from ``camera_specs`` for
    backward compatibility with existing importers.
    """

    kind: Literal["crop", "rotate"]
    mode: Literal["center", "subject_center"] | None = None
    percent: float | None = None  # crop
    degrees: float | None = None  # rotate


class ComposedPrompt(BaseModel):
    positive_prompt: str
    clip_prompt: str
    negative_prompt: str
    worker_type: str
    seed: int
    layers: list[LayerContribution]
    dropped_layers: list[str]
    warnings: list[str]
    # Camera post-processing directive resolved at compose time (Story 78-1).
    # Forwarded by the daemon into render params so the worker can apply the
    # crop/rotate the camera preset requested. None when the resolved camera
    # sets no post directive.
    post: PostDirective | None = None


class CatalogMissError(Exception):
    """Raised when a catalog reference cannot be resolved. Never silent."""

    def __init__(self, source: str, missing_id: str) -> None:
        super().__init__(f"{source} has no entry for {missing_id!r}")
        self.source = source
        self.missing_id = missing_id


class BudgetError(Exception):
    """Raised when eviction would drop into the identity floor."""

    def __init__(self, message: str, breakdown: dict[str, int]) -> None:
        super().__init__(f"{message}: {breakdown}")
        self.breakdown = breakdown


class RenderConfigError(Exception):
    """Raised when the render pipeline is missing required configuration
    or the compose pipeline failed to produce a usable prompt."""


class StyleMissError(Exception):
    """Raised when a StyleCatalog lookup or load surfaces an unrenderable
    world or genre."""

    def __init__(self, scope: str, identifier: str, reason: str) -> None:
        super().__init__(
            f"StyleCatalog.{scope} unrenderable for {identifier!r}: {reason}"
        )
        self.scope = scope
        self.identifier = identifier
        self.reason = reason
