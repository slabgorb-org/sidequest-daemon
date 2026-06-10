"""Image render orchestration for the renderer daemon (story 101-7).

Extracted from the inline ``elif method == "render"`` block in
``daemon.py``. ``RenderService.render`` owns the image-tier pipeline —
beat filter, SceneInterpreter / subject extraction, prompt composition,
the worker ``render`` call, and the ``render.completed`` span. It does NOT
own the ``render_lock``, the per-queue heartbeats, or the
``daemon.dispatch.render`` span: those stay in ``_handle_client`` (story
37-23 keeps the lock + dispatch span + ``lock_name`` attribute at the
socket-dispatch site, and the heartbeat emission is per-connection). The
service is invoked through the unified ``dispatch_request`` while the
caller holds ``render_lock``.

Failure contract: ``render`` raises ``RenderError`` (carrying the
JSON-RPC error ``code`` and, for compose failures, ``error_type`` + ``tier``)
instead of writing frames directly — the caller writes the error frame.
``asyncio.CancelledError`` propagates unchanged so the caller can mark its
dispatch span. On success ``render`` returns the worker result dict (which
``dispatch_request`` wraps in the JSON-RPC ``result`` envelope); a
non-visual beat returns ``{"status": "skipped", "reason": "beat_filter"}``.
"""

from __future__ import annotations

import asyncio
import logging

from opentelemetry import trace

from sidequest_daemon.media.recipes import (
    BudgetError,
    CatalogMissError,
    RenderConfigError,
    StyleMissError,
)
from sidequest_daemon.media.tiers import IMAGE_TIERS
from sidequest_daemon.media.worker_pool import WorkerPool
from sidequest_daemon.telemetry import emit_watcher_event as _emit_watcher_event

log = logging.getLogger(__name__)

# Tracer name kept identical to daemon.py so spans land in the same
# instrumentation scope the GM panel already consumes. Resolved at call
# time (not module load) so a test that swaps the global TracerProvider
# after import still captures these spans — see
# test_span_scope_per_call_45_29.py, which resets daemon.tracer but cannot
# reach a module-level tracer cached here.
_TRACER_NAME = "sidequest_daemon.media.daemon"


class RenderError(Exception):
    """A render-pipeline failure carrying its JSON-RPC error code.

    ``_handle_client`` catches this and writes the structured error frame.
    ``error_type`` and ``tier`` are populated for compose failures (the
    frame the server's COMPOSE_FAILED handling expects); they are ``None``
    for generation/extraction failures.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        error_type: str | None = None,
        tier: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.error_type = error_type
        self.tier = tier


class RenderService:
    """Owns the image-tier render pipeline behind the unified dispatcher."""

    def __init__(self, pool: WorkerPool) -> None:
        self._pool = pool

    async def render(self, params: dict) -> dict:
        """Run the image render pipeline and return the worker result.

        Called by ``dispatch_request`` while ``_handle_client`` holds
        ``render_lock``. Raises ``RenderError`` on compose/extraction/
        generation failure; re-raises ``asyncio.CancelledError`` unchanged.
        """
        tracer = trace.get_tracer(_TRACER_NAME)

        # Beat filter: skip non-visual beats before expensive GPU work.
        if params.get("narration") and params.get("game_state"):
            from sidequest_daemon.renderer.beat_filter import should_generate
            from sidequest_daemon.types import (
                ChaseState,
                Character,
                CombatState,
                GameState,
            )

            gs_raw = params["game_state"]
            game_state = GameState(
                location=gs_raw.get("location", ""),
                time_of_day=gs_raw.get("time_of_day", ""),
                characters=[
                    Character(name=c.get("name", ""))
                    for c in gs_raw.get("characters", [])
                ],
                combat=CombatState(
                    in_combat=gs_raw.get("combat", {}).get("in_combat", False)
                ),
                chase=ChaseState(
                    in_chase=gs_raw.get("chase", {}).get("in_chase", False)
                ),
            )
            previous_location = params.get("previous_location")
            if not should_generate(
                params["narration"], game_state, previous_location
            ):
                log.info("beat_filter: skipping non-visual beat")
                return {"status": "skipped", "reason": "beat_filter"}

        # If narration is provided, use SceneInterpreter for fast rule-based
        # StageCue extraction, then fall back to LLM subject extraction.
        if params.get("narration") and not params.get("positive_prompt"):
            from sidequest_daemon.scene_interpreter import SceneInterpreter
            from sidequest_daemon.types import Character, GameState

            narrator_text = params["narration"]

            # Extract documents and strip markers before visual processing
            scene_interp = SceneInterpreter()
            genre = params.get("genre", "unknown")
            doc_events = scene_interp.extract_documents(narrator_text, genre=genre)
            if doc_events:
                log.info(
                    "scene_interpreter: extracted %d document(s)",
                    len(doc_events),
                )
                params.setdefault("document_events", [])
                for doc in doc_events:
                    params["document_events"].append(doc.model_dump())
            narrator_text = scene_interp.strip_document_markers(narrator_text)
            params["narration"] = narrator_text

            # Try rule-based StageCue extraction (fast, no LLM)
            gs_raw = params.get("game_state", {})
            interp_state = GameState(
                location=gs_raw.get("location", ""),
                time_of_day=gs_raw.get("time_of_day", ""),
                characters=[
                    Character(name=c.get("name", ""))
                    for c in gs_raw.get("characters", [])
                ],
            )
            # Only run rule-based interpretation when the caller did NOT
            # already supply a structured visual block. The server-side
            # narrator agent emits {tier, subject, mood, tags} as structured
            # output and the dispatcher forwards those fields verbatim.
            # Overriding them here silently second-guesses the agent's
            # classification — the playtest 2026-04-30 COMPOSE_FAILED
            # signature (a narrator-classified `landscape` rewritten to
            # `scene_illustration` then validated without PC `participants`).
            server_supplied_tier = (
                params.get("tier") in IMAGE_TIERS and bool(params.get("subject"))
            )
            if not server_supplied_tier:
                cues = scene_interp.interpret(narrator_text, interp_state)
                if cues:
                    top_cue = cues[0]
                    params["subject"] = top_cue.subject
                    params["mood"] = top_cue.mood
                    params["tags"] = top_cue.tags
                    params["tier"] = top_cue.tier.value
                    with tracer.start_as_current_span(
                        "scene_interpreter.classified"
                    ) as cls_span:
                        cls_span.set_attribute("tier", top_cue.tier.value)
                        cls_span.set_attribute("subject", top_cue.subject[:120])
                        cls_span.set_attribute("source", "rule_match")
                    log.info(
                        "scene_interpreter — tier=%s subject=%s",
                        top_cue.tier.value,
                        top_cue.subject[:80],
                    )
            else:
                with tracer.start_as_current_span(
                    "scene_interpreter.skipped"
                ) as skip_span:
                    skip_span.set_attribute("reason", "server_supplied_visual_block")
                    skip_span.set_attribute("tier", str(params.get("tier", "")))
                log.info(
                    "scene_interpreter — skipped (server tier=%s subject=%s)",
                    params.get("tier"),
                    str(params.get("subject", ""))[:80],
                )

            # Fall back to LLM subject extraction if SceneInterpreter didn't
            # produce a subject (or for refinement)
            if not params.get("subject"):
                from sidequest_daemon.media.subject_extractor import SubjectExtractor

                extractor = SubjectExtractor()
                extracted = await extractor.extract(params["narration"])
                if not extracted or not extracted.get("subject"):
                    raise RenderError(
                        "EXTRACTION_FAILED",
                        "SubjectExtractor returned no visual subject from "
                        "narration. No fallback — refusing to render narrative "
                        "prose directly.",
                    )
                # Build StageCue-compatible params from extraction
                params["subject"] = extracted["subject"]
                params["mood"] = extracted.get("mood", "")
                params["tags"] = extracted.get("tags", [])
                # Override tier if extractor found a better one
                extracted_tier = extracted.get("tier", "")
                if extracted_tier:
                    tier_lower = extracted_tier.lower()
                    if tier_lower in IMAGE_TIERS:
                        params["tier"] = tier_lower
                log.info(
                    "narration_extracted — subject=%s, mood=%s, tier=%s",
                    extracted["subject"][:80],
                    extracted.get("mood"),
                    params.get("tier"),
                )

        composed = None
        if not params.get("positive_prompt"):
            missing = [k for k in ("subject", "world", "genre") if not params.get(k)]
            try:
                if missing:
                    with tracer.start_as_current_span(
                        "compose.gate_short_circuit"
                    ) as gate_span:
                        gate_span.set_attribute("missing_fields", ",".join(missing))
                        gate_span.set_attribute("tier", params.get("tier", ""))
                    raise RenderConfigError(
                        f"render request missing required field(s): {missing}"
                    )

                from sidequest_daemon.media.workers.zimage_mlx_worker import (
                    build_cue_from_params,
                    compose_prompt_for,
                )

                cue = build_cue_from_params(params)
                composed = compose_prompt_for(cue)
                params["positive_prompt"] = composed.positive_prompt
                params["clip_prompt"] = composed.clip_prompt
                params["negative_prompt"] = composed.negative_prompt
                params["seed"] = composed.seed
                # Story 78-1: forward the resolved camera's post directive
                # (crop/rotate) so the worker applies it after generation.
                params["post"] = (
                    composed.post.model_dump() if composed.post is not None else None
                )
                log.info(
                    "prompt_composed — positive=%s",
                    composed.positive_prompt[:150],
                )
            except (
                RenderConfigError,
                StyleMissError,
                CatalogMissError,
                BudgetError,
                ValueError,
                # Pingpong 2026-04-30: data-shape failures (IndexError from
                # _character_lod_plan on empty participants, etc.) used to
                # leak past the handler and close the socket mid-request.
                # Catch the typical "data shape unexpected" families so a
                # tier-wide compose failure emits the compose.failed span +
                # structured COMPOSE_FAILED frame instead of breaking the
                # JSON-RPC transport.
                IndexError,
                KeyError,
                AttributeError,
                TypeError,
            ) as e:
                # Per CLAUDE.md "OTEL Observability Principle": fail LOUD to
                # the client, not silently to the socket.
                with tracer.start_as_current_span("compose.failed") as fail_span:
                    fail_span.set_attribute("tier", params.get("tier", ""))
                    fail_span.set_attribute("error_type", type(e).__name__)
                    fail_span.set_attribute("error_message", str(e)[:512])
                    fail_span.set_attribute("world", params.get("world", ""))
                    fail_span.set_attribute("genre", params.get("genre", ""))
                # Watcher event for the GM panel (pingpong 2026-04-30
                # daemon-tier-failure ask). The compose.failed OTEL span is
                # for tracer consumers; the watcher event is the path the
                # dashboard's Console / Subsystems tabs read. Both fire so
                # the failure is visible at every observation tier.
                # sync — see sidequest_daemon/telemetry/watcher_bridge.py docstring for trade-off rationale
                _emit_watcher_event(
                    "daemon_compose_failed",
                    {
                        "tier": params.get("tier", ""),
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:512],
                        "world": params.get("world", ""),
                        "genre": params.get("genre", ""),
                        "render_id": params.get("render_id", ""),
                    },
                )
                log.warning(
                    "render.compose_failed — tier=%s err_type=%s err=%s",
                    params.get("tier", ""),
                    type(e).__name__,
                    e,
                )
                raise RenderError(
                    "COMPOSE_FAILED",
                    f"{type(e).__name__}: {e}",
                    error_type=type(e).__name__,
                    tier=params.get("tier", ""),
                ) from e

        try:
            result = await asyncio.to_thread(self._pool.render, params)
        except asyncio.CancelledError:
            # Client disconnect — let the caller mark its dispatch span.
            raise
        except Exception as e:
            log.exception("render.failed — tier=%s", params.get("tier", ""))
            raise RenderError("GENERATION_FAILED", str(e)) from e

        with tracer.start_as_current_span("render.completed") as completed:
            final_prompt = params.get("positive_prompt", "")
            completed.set_attribute("genre", params.get("genre", ""))
            completed.set_attribute("world", params.get("world", ""))
            # R2 migration: surface session_id and the uploaded r2_key on
            # render.completed so the GM panel can verify the artifact landed.
            completed.set_attribute("session_id", params.get("session_id", ""))
            completed.set_attribute("r2_key", str(result.get("r2_key") or ""))
            completed.set_attribute("tier", params.get("tier", ""))
            completed.set_attribute("prompt_length", len(final_prompt))
            genre_applied = False
            world_applied = False
            if composed is not None:
                for layer in composed.layers:
                    tokens = layer.tokens.strip()
                    if not tokens:
                        continue
                    if (
                        layer.slot == "ART_SENSIBILITY.GENRE"
                        and tokens in final_prompt
                    ):
                        genre_applied = True
                    elif (
                        layer.slot == "ART_SENSIBILITY.WORLD"
                        and tokens in final_prompt
                    ):
                        world_applied = True
            completed.set_attribute("genre_style_applied", genre_applied)
            completed.set_attribute("world_style_applied", world_applied)

        return result
