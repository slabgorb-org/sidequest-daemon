"""Z-Image image generation worker — Apple Silicon native via mflux.

Replaces FluxMLXWorker. Same interface contract: load_model(), warm_up(),
render(), cleanup(). Communicates via JSON-line protocol over stdin/stdout
when run as subprocess. No LoRA support.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from opentelemetry import trace

from sidequest_daemon.media import r2_writer
from sidequest_daemon.media.camera_specs import CameraLoader
from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    CharacterTokens,
    PlaceCatalog,
    StyleCatalog,
)
from sidequest_daemon.media.post_processor import (
    apply_post,
    required_render_size,
    supersample_downscale,
)
from sidequest_daemon.media.prompt_composer import PromptComposer
from sidequest_daemon.media.recipe_loader import RecipeLoader
from sidequest_daemon.media.recipes import (
    LOD,
    CameraPreset,
    ComposedPrompt,
    PostDirective,
    RenderConfigError,
    RenderTarget,
)
from sidequest_daemon.media.zimage_config import (
    VALID_FIDELITIES,
    ZIMAGE_QUANTIZE,
    Fidelity,
    get_zimage_config,
)
from sidequest_daemon.renderer.models import RenderTier, StageCue

_FIDELITY_ENV_VAR = "SIDEQUEST_DAEMON_FIDELITY"

log = logging.getLogger(__name__)

_DAEMON_ROOT = Path(__file__).resolve().parents[3]


def _get_composer(
    genre: str,
    world: str,
    *,
    pc_descriptor: dict | None = None,
) -> PromptComposer:
    """Build a composer scoped to the target's world.

    When ``pc_descriptor`` is supplied (slice 2 of catalog-injected compose
    wiring), the PC is registered into the freshly-loaded ``CharacterCatalog``
    via ``add_pc`` before the composer is returned. The descriptor lets the
    server hand a ``pc:<slug>`` ref to the daemon without first persisting a
    ``portrait_manifest`` entry — a single appearance prose is replicated to
    every ``LOD`` (mirror of ``CharacterCatalog.load``'s production path so
    PCs and NPCs share the same eviction ladder).
    """
    packs_root = Path(os.environ["SIDEQUEST_GENRE_PACKS"])
    characters = CharacterCatalog.load(packs_root, genre=genre, world=world)
    if pc_descriptor is not None:
        pc_id = pc_descriptor["id"]
        appearance = pc_descriptor.get("appearance", "")
        tokens = CharacterTokens(
            kind="pc",
            descriptions=dict.fromkeys(LOD, appearance),
            default_pose=pc_descriptor.get("default_pose", ""),
            culture=pc_descriptor.get("culture"),
            world=world,
        )
        characters.add_pc(pc_id, tokens)
    return PromptComposer(
        recipes=RecipeLoader.from_file(_DAEMON_ROOT / "recipes.yaml"),
        cameras=CameraLoader.from_file(_DAEMON_ROOT / "cameras.yaml"),
        characters=characters,
        places=PlaceCatalog.load(packs_root, genre=genre, world=world),
        styles=StyleCatalog.load(packs_root, genre=genre, world=world),
    )


def build_render_target(cue: StageCue) -> RenderTarget:
    """Translate a StageCue into a RenderTarget.

    `cue.metadata["world"]` and `cue.metadata["genre"]` are required — fail
    loud if either is missing. ``cue.metadata["fidelity"]`` (Story 45-38)
    is optional and defaults to ``"high_fidelity"`` (2026-05-02 flip).
    """
    world = cue.metadata.get("world")
    genre = cue.metadata.get("genre")
    if not world or not genre:
        raise ValueError(
            "StageCue.metadata must carry `world` and `genre` for composer routing",
        )
    fidelity = cue.metadata.get("fidelity", "high_fidelity")

    if cue.tier in (RenderTier.PORTRAIT, RenderTier.PORTRAIT_SQUARE):
        character = cue.characters[0] if cue.characters else cue.subject
        return RenderTarget(
            kind="portrait",
            world=world,
            genre=genre,
            character=character,
            camera=cue.camera,
            fidelity=fidelity,
        )
    if cue.tier == RenderTier.LANDSCAPE:
        # LANDSCAPE has two valid shapes:
        #   1. Explicit POI render: ``cue.subject`` is a ``where:<world>/<slug>``
        #      ref (content-tool / poi-pregeneration path). Routes to
        #      ``kind=poi`` so PlaceCatalog resolves the registered place.
        #   2. Environmental scene from prose: the narrator agent emits
        #      ``tier=landscape`` with a free-form prose subject (e.g.
        #      "Cramped wrench-house galley under a coolant pipe...").
        #      No registered POI exists; routing to ``kind=poi`` fails
        #      validation ("must reference a specific place in world ...
        #      got scope ''"). Route to ``kind=illustration`` with empty
        #      participants — the relaxed validator (recipes.py post
        #      playtest 2026-04-30) allows environmental illustrations.
        #      Action prose + ART_SENSIBILITY layers carry the visual.
        #
        # The discriminator is the `where:` scheme — explicit refs always
        # start with `where:` per the catalog contract; prose never does.
        if cue.subject.startswith("where:"):
            return RenderTarget(
                kind="poi",
                world=world,
                genre=genre,
                place=cue.subject,
                fidelity=fidelity,
            )
        return RenderTarget(
            kind="illustration",
            world=world,
            genre=genre,
            participants=cue.characters,  # may be empty — environmental scene
            location=cue.location or cue.metadata.get("location_ref", ""),
            action=cue.subject,
            camera=cue.camera or CameraPreset.scene,
            fidelity=fidelity,
        )
    if cue.tier == RenderTier.SCENE_ILLUSTRATION:
        return RenderTarget(
            kind="illustration",
            world=world,
            genre=genre,
            participants=cue.characters,
            location=cue.location or cue.metadata.get("location_ref", ""),
            action=cue.subject,
            camera=cue.camera or CameraPreset.scene,
            fidelity=fidelity,
        )
    raise ValueError(f"unsupported tier for composer routing: {cue.tier!r}")


def build_cue_from_params(params: dict) -> StageCue:
    """Project a daemon `render` request's params dict into a ``StageCue``.

    Pulled out of the `_handle_client` dispatch loop so the params→metadata
    projection (specifically: forwarding ``pc_descriptor`` for slice 2 of the
    catalog-injected compose wiring) can be tested without a live socket. The
    dispatch loop's only responsibility on top of this helper is the
    early-out conditional checking that subject/world/genre are present.

    Story 45-38: ``fidelity`` rides along in metadata; ``build_render_target``
    pulls it back out and pins it on the resulting ``RenderTarget``.
    """
    tier_str = params.get("tier", "scene_illustration")
    tier = (
        RenderTier(tier_str)
        if tier_str in {t.value for t in RenderTier}
        else RenderTier.SCENE_ILLUSTRATION
    )
    metadata: dict = {
        "world": params["world"],
        "genre": params["genre"],
        "fidelity": params.get("fidelity", "high_fidelity"),
    }
    pc_descriptor = params.get("pc_descriptor")
    if pc_descriptor is not None:
        metadata["pc_descriptor"] = pc_descriptor
    return StageCue(
        subject=params.get("subject", ""),
        tier=tier,
        location=params.get("location", ""),
        mood=params.get("mood", ""),
        characters=params.get("characters", []),
        tags=params.get("tags", []),
        metadata=metadata,
    )


def compose_prompt_for(cue: StageCue) -> ComposedPrompt:
    """Build a ComposedPrompt from a StageCue end-to-end.

    When ``cue.metadata['pc_descriptor']`` is set, the PC is registered into
    the catalog before composition runs — the server's ``pc:<slug>`` ref will
    resolve without a disk lookup.
    """
    world = cue.metadata["world"]
    genre = cue.metadata["genre"]
    pc_descriptor = cue.metadata.get("pc_descriptor")
    composer = _get_composer(genre, world, pc_descriptor=pc_descriptor)
    target = build_render_target(cue)
    return composer.compose(target)


def upload_render_to_r2(
    *,
    content_bytes: bytes,
    world_slug: str,
    session_id: str,
    kind: r2_writer.ArtifactKind,
    content_type: str,
) -> str:
    """Thin wrapper: upload a freshly-generated render to R2, return the
    relative key. Surfaces upload errors to the caller (no swallowing)."""
    return r2_writer.upload_artifact(
        world_slug=world_slug,
        session_id=session_id,
        kind=kind,
        content_bytes=content_bytes,
        content_type=content_type,
    )


# Map daemon RenderTier values to r2_writer ArtifactKind buckets.
# Portrait tiers go to "portraits"; everything visual else lands under
# "scenes". Cartography/fog-of-war are also "scenes" — they're scene-level
# illustrations, not POIs (POIs are pre-registered places, not runtime
# renders). r2_writer rejects unknown kinds, so the mapping table is the
# single source of truth.
_TIER_TO_R2_KIND: dict[str, r2_writer.ArtifactKind] = {
    "portrait": "portraits",
    "portrait_square": "portraits",
    "landscape": "scenes",
    "scene_illustration": "scenes",
    "text_overlay": "scenes",
    "cartography": "scenes",
    "fog_of_war": "scenes",
}


class ZImageMLXWorker:
    """Z-Image image generation worker using Apple MLX via mflux.

    Story 45-39: the worker reads ``SIDEQUEST_DAEMON_FIDELITY`` at
    construction (default ``"high_fidelity"`` post-2026-05-02) and selects
    its model variant + tier table accordingly. ``"high_fidelity"`` loads
    base ``z-image`` (20 steps, CFG 4.0, ~108s/render) — the new floor.
    ``"turbo"`` loads ``z-image-turbo`` (LCM-distilled, 8 steps, no CFG,
    ~30s/render) and is now opt-in via the env var when latency wins over
    painterly quality. Tier parameters come from ``zimage_config``'s
    ``get_zimage_config(tier, fidelity)`` lookup — there is no
    duplicate-of-truth tier table on the worker class.

    A render request whose ``params["fidelity"]`` is set and does not
    match the loaded fidelity is rejected with a structured ``ValueError``
    (CLAUDE.md "No Silent Fallbacks" — the worker can't satisfy the
    request without silently using the wrong model).

    **Per-process singleton invariant** (Story 43-5): only one
    `ZImageMLXWorker` may exist per Python process. A second construction
    raises ``RuntimeError`` to fail loudly. Z-Image survives a second
    model on the same MPS device, but Flux historically OOM'd the M3 Max
    instantly — the invariant is here to prevent any future renderer
    revert from silently spawning a second model. Production callers
    must route through ``WorkerPool.warm_up_image()``
    (``sidequest_daemon/media/daemon.py``), which also guards via
    ``_image_loaded``. The conftest autouse fixture resets
    ``ZImageMLXWorker._instance = None`` between tests.
    """

    # Per-process singleton handle. Set by __init__ on first construction;
    # raises RuntimeError on second. Tests reset to None between cases.
    _instance: "ZImageMLXWorker | None" = None

    # Quantization level for model loading. 8-bit per mflux's Turbo README
    # example (`--steps 9 --quantize 8`). None = full precision.
    QUANTIZE: int | None = ZIMAGE_QUANTIZE

    # Z-Image's default scheduler per its CLI.
    #
    # Draw Things delta: Draw Things validation used "UniPC Trailing", which
    # empirically produces slightly cleaner detail at 20 steps / CFG 4.0.
    # mflux 0.x does NOT ship UniPC — the only options in
    # mflux/models/common/schedulers/ are:
    #   - linear
    #   - flow_match_euler_discrete  (this one; used for Z-Image with CFG>1)
    #   - seedvr2_euler              (SeedVR2-specific)
    # If/when mflux adds UniPC this is the line to change. For now the daemon
    # is NOT a pixel-perfect match to Draw Things on sampler, only on
    # steps/guidance/resolution-dependent-shift.
    #
    # Resolution-dependent shift ("Resolution Dpt. Shift" in Draw Things) IS
    # enabled automatically: mflux's ModelConfig for "z-image" and
    # "z-image-turbo" both set requires_sigma_shift=True, which causes
    # Config.scheduler to call FlowMatchEulerDiscreteScheduler.set_image_seq_len
    # at construction time — i.e. the sigma schedule is recomputed per call
    # based on (width/16) * (height/16). No extra flag needed here.
    #
    # CFG Zero*: no such toggle exists in mflux. Nothing to disable.
    SCHEDULER: str = "flow_match_euler_discrete"

    def __init__(self, output_dir: Path) -> None:
        if type(self)._instance is not None:
            raise RuntimeError(
                "ZImageMLXWorker is a per-process singleton; a second "
                "construction would load a second model on the same MPS "
                "device. Route through WorkerPool.warm_up_image() instead, "
                "or reset ZImageMLXWorker._instance=None in tests."
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model: object | None = None
        self.fidelity: Fidelity = self._resolve_fidelity()
        # Probe the variant via the canonical lookup so the worker stays in
        # sync with zimage_config — the per-tier tables there all use the
        # same model_variant within a fidelity, so any tier works as a probe.
        self.model_variant: str = get_zimage_config(
            RenderTier.PORTRAIT, self.fidelity
        ).model_variant
        type(self)._instance = self

    @staticmethod
    def _resolve_fidelity() -> Fidelity:
        """Read ``SIDEQUEST_DAEMON_FIDELITY`` and validate it loudly.

        No silent fallback (CLAUDE.md). An unset env var is the explicit
        default of ``"high_fidelity"`` (2026-05-02 flip) — anything else
        must be one of the known fidelity strings or the worker refuses
        to construct.
        """
        raw = os.environ.get(_FIDELITY_ENV_VAR, "high_fidelity")
        if raw not in VALID_FIDELITIES:
            raise ValueError(
                f"{_FIDELITY_ENV_VAR}={raw!r} is not a recognised fidelity; "
                f"expected one of {VALID_FIDELITIES!r}. Refusing to silently "
                f"fall back to 'high_fidelity' — fix the env var or unset it."
            )
        return raw  # type: ignore[return-value]

    def load_model(self) -> None:
        """Load the configured Z-Image variant via mflux."""
        tracer = trace.get_tracer("sidequest_daemon.media.workers.zimage_mlx_worker")
        with tracer.start_as_current_span("zimage_mlx.load_model") as span:
            # `model.name` retains the historical "z-image" value so existing
            # OTEL queries / dashboards keep working. `model.variant` is the
            # new-since-2026-04-26 attribute that distinguishes turbo from
            # base — the GM panel filters on this.
            span.set_attribute("model.name", "z-image")
            span.set_attribute("model.variant", self.model_variant)
            span.set_attribute("worker.fidelity", self.fidelity)
            span.set_attribute("model.quantize", self.QUANTIZE or 0)
            from mflux.models.common.config import ModelConfig
            from mflux.models.z_image.variants.z_image import ZImage

            self.model = ZImage(
                model_config=ModelConfig.from_name(self.model_variant),
                quantize=self.QUANTIZE,
            )

    def warm_up(self) -> dict:
        """MLX graph compilation via dummy generation.

        Caller (``WorkerPool.warm_up_image()``) is contractually required
        to invoke ``load_model()`` first. We raise rather than silently
        lazy-load — `assert` is unsafe under Python `-O` (assertions are
        stripped), so the check is an explicit `raise`.
        """
        if self.model is None:
            raise RuntimeError(
                "warm_up() called before load_model() — caller contract "
                "violation. WorkerPool.warm_up_image() loads the model "
                "before warm_up; do not call warm_up directly."
            )
        tracer = trace.get_tracer("sidequest_daemon.media.workers.zimage_mlx_worker")
        with tracer.start_as_current_span("zimage_mlx.warm_up") as span:
            start = time.monotonic()
            self.model.generate_image(  # type: ignore[attr-defined]
                seed=0,
                prompt="black",
                num_inference_steps=2,
                guidance=None,
                width=512,
                height=512,
                scheduler=self.SCHEDULER,
                negative_prompt=None,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            span.set_attribute("warmup.elapsed_ms", elapsed_ms)
            return {"warmup_ms": elapsed_ms}

    def render(self, params: dict) -> dict:
        """Generate image from StageCue params. Returns result dict."""
        tracer = trace.get_tracer("sidequest_daemon.media.workers.zimage_mlx_worker")
        with tracer.start_as_current_span("zimage_mlx.render") as span:
            try:
                tier_name = params.get("tier", "")
                try:
                    tier = RenderTier(tier_name)
                except ValueError as e:
                    raise ValueError(f"Unsupported tier: {tier_name!r}") from e

                # Story 45-39: a request that names a fidelity must match
                # the loaded one. Mismatch is a misconfiguration the worker
                # cannot satisfy — fail loud rather than silently render
                # with the wrong model. An omitted fidelity means "use what
                # the daemon was launched with" (legacy callers).
                requested_fidelity = params.get("fidelity")
                if (
                    requested_fidelity is not None
                    and requested_fidelity != self.fidelity
                ):
                    raise ValueError(
                        f"fidelity mismatch: daemon loaded fidelity="
                        f"{self.fidelity!r} but request asked for "
                        f"fidelity={requested_fidelity!r}. The worker only "
                        f"holds one model — relaunch the daemon with "
                        f"SIDEQUEST_DAEMON_FIDELITY={requested_fidelity!r} "
                        f"or fix the caller."
                    )

                tier_cfg = get_zimage_config(tier, self.fidelity)

                # Supersample validation (No Silent Fallbacks): the tier
                # config's supersample_factor must be a positive integer; a
                # value ≤ 0 or of the wrong type is a misconfiguration that
                # must be caught loudly here, not swallowed or defaulted.
                ss_factor = tier_cfg.supersample_factor
                if (
                    not isinstance(ss_factor, int)
                    or isinstance(ss_factor, bool)
                    or ss_factor <= 0
                ):
                    raise ValueError(
                        f"tier {tier_name!r} has invalid supersample_factor="
                        f"{ss_factor!r}; must be a positive int.  Fix the "
                        f"ZImageTierConfig entry in zimage_config.py."
                    )

                # Story 78-1: a camera preset may carry a post-processing
                # directive (crop/rotate), forwarded by the daemon as a JSON
                # dict in params["post"]. Validate it at this boundary (No
                # Silent Fallbacks — a malformed directive must fail loud, not
                # be silently ignored) and size the generation up so the crop/
                # rotate has source pixels to consume; apply_post then trims
                # the generated image back to the target after generate().
                post_raw = params.get("post")
                post_directive: PostDirective | None = (
                    PostDirective.model_validate(post_raw)
                    if post_raw is not None
                    else None
                )
                # required_render_size expands to the crop/rotate envelope.
                # We then multiply by supersample_factor so the generator
                # renders at the supersampled resolution; both the post-
                # directive envelope and the supersample scale are applied
                # before the image goes into generate_image().
                post_width, post_height = required_render_size(
                    (tier_cfg.width, tier_cfg.height), post_directive
                )
                gen_width = post_width * ss_factor
                gen_height = post_height * ss_factor

                # Caller-supplied inference-step override (send_render's
                # `--steps` flag, carried as params["steps"]). When present it
                # wins over the (tier, fidelity) default; when absent the tier
                # default stands. Fail loud on a nonsensical value rather than
                # silently falling back to the default (No Silent Fallbacks).
                steps_override = params.get("steps")
                if steps_override is not None and (
                    not isinstance(steps_override, int)
                    or isinstance(steps_override, bool)
                    or steps_override <= 0
                ):
                    raise ValueError(
                        f"render request 'steps' must be a positive int, got "
                        f"{steps_override!r}"
                    )
                effective_steps = (
                    steps_override if steps_override is not None else tier_cfg.steps
                )

                prompt = self._compose_prompt(params)
                negative_prompt = params.get("negative_prompt") or None
                seed = params.get("seed", 0)

                span.set_attribute("model.variant", self.model_variant)
                span.set_attribute("worker.fidelity", self.fidelity)
                span.set_attribute("render.tier", tier_name)
                span.set_attribute("render.seed", seed)
                span.set_attribute("render.width", tier_cfg.width)
                span.set_attribute("render.height", tier_cfg.height)
                # Story 78-1: post-processing visibility for the GM panel —
                # whether a camera post directive fired, and the oversized
                # source it forced (source == target when no directive).
                span.set_attribute("render.post_applied", post_directive is not None)
                span.set_attribute(
                    "render.post_kind",
                    post_directive.kind if post_directive is not None else "",
                )
                span.set_attribute("render.source_width", gen_width)
                span.set_attribute("render.source_height", gen_height)
                # Supersample visibility for the GM panel: whether the tier
                # rendered at a higher internal resolution before Lanczos
                # downscale.  factor=1 means the generator rendered straight
                # to the target (no supersample applied).
                span.set_attribute("render.supersample_factor", ss_factor)
                span.set_attribute("render.supersample_applied", ss_factor > 1)
                # render.steps reports the steps ACTUALLY used; the two extra
                # attributes let the GM panel see whether a caller override was
                # in play and what the tier default would have been.
                span.set_attribute("render.steps", effective_steps)
                span.set_attribute("render.steps_default", tier_cfg.steps)
                span.set_attribute(
                    "render.steps_overridden", steps_override is not None
                )
                span.set_attribute("render.guidance", tier_cfg.guidance)
                span.set_attribute("render.prompt_length", len(prompt))
                span.set_attribute("render.negative_length", len(negative_prompt or ""))

                log.info(
                    "ZIMAGE RENDER [%s] fidelity=%s seed=%s w=%s h=%s steps=%s%s%s",
                    tier_name,
                    self.fidelity,
                    seed,
                    tier_cfg.width,
                    tier_cfg.height,
                    effective_steps,
                    f" (override; default {tier_cfg.steps})"
                    if steps_override is not None
                    else "",
                    f" (supersample×{ss_factor} → {gen_width}×{gen_height})"
                    if ss_factor > 1
                    else "",
                )
                log.info("  prompt: %s", prompt[:150])

                if self.model is None:
                    raise RuntimeError(
                        "render() called before load_model() — caller "
                        "contract violation. Route through WorkerPool which "
                        "guards via _image_loaded."
                    )

                # Z-Image Turbo's ModelConfig sets supports_guidance=False;
                # pass guidance=None (mflux's "disabled" sentinel) when the
                # tier guidance is 0.0 so we don't accidentally activate a
                # CFG path on a distilled model. Base Z-Image (guidance=4.0)
                # passes the float through to drive CFG.
                guidance_arg: float | None = (
                    tier_cfg.guidance if tier_cfg.guidance > 0.0 else None
                )

                start = time.monotonic()
                image = self.model.generate_image(  # type: ignore[attr-defined]
                    seed=seed,
                    prompt=prompt,
                    num_inference_steps=effective_steps,
                    guidance=guidance_arg,
                    width=gen_width,
                    height=gen_height,
                    scheduler=self.SCHEDULER,
                    negative_prompt=negative_prompt,
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)
                span.set_attribute("render.elapsed_ms", elapsed_ms)

                # Story 78-1: apply the camera's crop/rotate to the generated
                # (oversized) image before it is saved/uploaded. No-op when
                # post_directive is None.
                image = apply_post(image, post_directive)

                # Supersample downscale: when the tier's supersample_factor > 1
                # the generator rendered at a higher internal resolution
                # (gen_width × gen_height) and apply_post trimmed it to the
                # post-directive envelope.  We now Lanczos-downscale from the
                # post-directive envelope to the final target resolution
                # (tier_cfg.width × tier_cfg.height).  Factor=1 is a true
                # no-op — this branch is not entered, no resize is performed.
                if ss_factor > 1:
                    image = supersample_downscale(
                        image,
                        target_width=tier_cfg.width,
                        target_height=tier_cfg.height,
                    )

                filename = f"render_{uuid.uuid4().hex[:8]}.png"
                image_path = self.output_dir / filename
                image.save(str(image_path))

                # Cloudflare R2 migration: upload the freshly-rendered image
                # so the server can emit `r2_key` to clients alongside the
                # legacy `image_url` (local disk path) during cutover.
                # Per CLAUDE.md "No Silent Fallbacks": upload errors propagate
                # — the daemon main loop's render error handler emits
                # GENERATION_FAILED and the server surfaces image_unavailable.
                #
                # world_slug / session_id propagation: the params dict is the
                # surface the daemon hands the worker (see
                # ``WorkerPool.render`` in daemon.py). ``world`` is already
                # populated by the server's render dispatch; ``session_id``
                # is added by the daemon dispatcher in this same task. If
                # either is missing at this point the request never matched
                # the pre-render compose gate (which itself requires
                # ``world``), so "unknown" is a debug fallback only — the
                # `r2.params_missing` watcher event flags it for the GM panel.
                world_slug = params.get("world") or "unknown"
                session_id = params.get("session_id") or "unknown"
                tier_value = params.get("tier", "")
                kind = _TIER_TO_R2_KIND.get(tier_value)
                if kind is None:
                    raise ValueError(
                        f"no R2 artifact-kind mapping for tier={tier_value!r}; "
                        f"add it to _TIER_TO_R2_KIND or fix the caller."
                    )
                with open(image_path, "rb") as fh:
                    content = fh.read()
                r2_key = upload_render_to_r2(
                    content_bytes=content,
                    world_slug=world_slug,
                    session_id=session_id,
                    kind=kind,
                    content_type="image/png",
                )

                return {
                    "image_url": str(image_path),  # legacy disk path —
                    # still emitted for in-flight back-compat during cutover
                    "r2_key": r2_key,
                    # Target dimensions. apply_post crops the oversized source
                    # back to the tier size (crop: gen ÷ percent × percent ==
                    # tier), so the saved artifact matches these in production
                    # regardless of the Story 78-1 post directive.
                    "width": tier_cfg.width,
                    "height": tier_cfg.height,
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as exc:
                span.set_status(trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    def _compose_prompt(self, params: dict) -> str:
        if params.get("positive_prompt"):
            return params["positive_prompt"]
        if params.get("prompt"):
            return params["prompt"]
        raise RenderConfigError(
            "compose pipeline failed to produce a prompt; "
            f"params keys={sorted(params.keys())}"
        )

    def cleanup(self) -> None:
        """Unload model, free GPU memory."""
        self.model = None
        # Release the singleton slot so a fresh process or test fixture
        # can construct a new worker without tripping the singleton guard.
        type(self)._instance = None
