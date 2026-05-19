# CLAUDE.md — SideQuest Daemon (Python)

Python media services sidecar for SideQuest. Handles image generation
(Z-Image Turbo via MLX, ADR-070) and music generation (ACE-Step, ADR-095).
Runs alongside the Python `sidequest-server` (per ADR-082; previously the
archived Rust `sidequest-api`).

**Active renderer:** Z-Image Turbo. The Flux MLX worker has been retired —
`media/workers/zimage_mlx_worker.py` is the sole runtime image worker.

## CRITICAL: Personal Project

This is a personal project under the `slabgorb` GitHub account.
- **No Jira integration.** Never create, reference, or interact with Jira tickets.
- **No 1898 org.** Nothing goes to the work GitHub org. Ever.
- All repos live under `github.com/slabgorb/`.

## SideQuest System Overview

Four repos compose the SideQuest stack (Python backend per ADR-082, ported from the Rust prototype 2026-04):
- **sidequest-server** — Python/FastAPI game engine and WebSocket API on port 8765
- **sidequest-ui** — React/TypeScript game client (Vite, port 5173)
- **sidequest-daemon** — Python media services (image gen, audio library playback)
- **sidequest-content** — Genre packs (YAML configs, audio, images, world data)

Orchestrator repo (`orc-quest`, also cloned as `oq-1` / `oq-2`) coordinates sprint tracking, docs, ADRs, and cross-repo scripts.

## Quality Rules

- No stubs, no hacks, no "we'll fix it later" shortcuts
- No skipping tests to save time
- No half-wired features — connect the full pipeline or don't start
- If something needs 5 connections, make 5 connections. Don't ship 3 and call it done.
- **Never say "the right fix is X" and then do Y.** Do X.
- **Never downgrade to a "quick fix" because you think the context is "just a playtest."**
  Every playtest is production tomorrow. Fix it right.

## Development Principles

### No Silent Fallbacks
If something isn't where it should be, fail loudly. Never silently try an alternative
path, config, or default. Silent fallbacks mask configuration problems and lead to
hours of debugging "why isn't this quite right."

### No Stubbing
Don't create stub implementations, placeholder modules, or skeleton code. If a feature
isn't being implemented now, don't leave empty shells for it. Dead code is worse than
no code.

### Don't Reinvent — Wire Up What Exists
Before building anything new, check if the infrastructure already exists in the codebase.
Many systems are fully implemented but not wired into the server or UI. The fix is
integration, not reimplementation.

### Verify Wiring, Not Just Existence
When checking that something works, verify it's actually connected end-to-end. Tests
passing and files existing means nothing if the component isn't imported, the hook isn't
called, or the endpoint isn't hit in production code. Check that new code has non-test
consumers.

### Every Test Suite Needs a Wiring Test
Unit tests prove a component works in isolation. That's not enough. Every set of tests
must include at least one integration test that verifies the component is wired into the
system — imported, called, and reachable from production code paths.

### Backend Language
The server (`sidequest-server`) is Python/FastAPI per ADR-082, ported from a
Rust prototype in 2026-04. The Rust codebase is preserved read-only at
https://github.com/slabgorb/sidequest-api for historical reference; older ADRs
that show Rust code are historical illustration only — see `docs/adr/README.md`
for the translation table. New backend code goes in Python. Media services
(`sidequest-daemon`) remain Python for inference library maturity (Flux /
Z-Image / ACE-Step). The narrator LLM path uses the Anthropic Python SDK by
default per ADR-101 (supersedes ADR-001; `claude -p`/Ollama are opt-in
non-default backends). This daemon's own Claude usage — subject extraction
in `media/subject_extractor.py` — is a non-narrator job that legitimately
still uses the `claude -p` CLI subprocess. (Kokoro TTS was formerly in this
list; TTS has been removed from the system.)

## OTEL Observability Principle

Every backend fix that touches a subsystem MUST add OTEL watcher events so the GM panel
can verify the fix is working. Claude is excellent at "winging it" — writing convincing
narration with zero mechanical backing. The only way to catch this is OTEL logging on
every subsystem decision:

- **Intent classification** — what was the action classified as, and why?
- **Agent routing** — which agent handled the action?
- **State patches** — what changed in game state (HP, location, inventory)?
- **Inventory mutations** — items added/removed, with source
- **NPC registry** — NPCs detected, names assigned, collisions prevented
- **Trope engine** — tick results, keyword matches, activations
- **Encounter engine** — beat selections, metric changes, resolution

The GM panel is the lie detector. If a subsystem isn't emitting OTEL spans, you can't
tell whether it's engaged or whether Claude is just improvising.

**Not needed for:** Cosmetic UI changes (labels, spacing, colors).

## Architecture Decision Index

ADRs live in the orchestrator repo at `orc-quest/docs/adr/`. See
`orc-quest/docs/adr/README.md` for the canonical index. Before designing
or modifying a subsystem, check the relevant ADR:

Particularly relevant to this daemon repo:

| Domain | ADRs |
|--------|------|
| IPC / transport | 035 (Unix socket IPC for Python sidecar), 046 (GPU memory budget coordinator) |
| Image rendering | 070 (MLX image renderer — replaces PyTorch/diffusers), 086 (image-composition taxonomy: portrait / POI / illustration), 050 (image pacing throttle), 044 (speculative prerender), 083 (multi-LoRA stacking), 084 (LoRA composition dimension), 096 (cavern renderer revival — partial), 089 (cavern template generation) |
| Music tier | 095 (daemon music tier via ACE-Step) |
| Subject / prompts | 056 (script tool generators), 048 (lore RAG store with cross-process embedding) |
| Training | 073 (local fine-tuned model architecture), 069 (scenario fixtures), 032 (genre LoRA style training), 034 (portrait identity consistency) |
| Telemetry | 058 (Claude subprocess OTEL passthrough — *server side superseded by 103*; daemon's `subject_extractor.py` still uses the subprocess passthrough) |

For the full ADR index see `orc-quest/docs/adr/README.md`.
Drift: `orc-quest/docs/adr/DRIFT.md`. Superseded: `orc-quest/docs/adr/SUPERSEDED.md`.

Historical (removed subsystems): TTS / Piper / Kokoro and runtime per-turn music generation were retired 2026-04. The Flux MLX worker was superseded by Z-Image Turbo.

## Spoiler Protection

- **Fully spoilable:** `mutant_wasteland/flickering_reach` only
- **Fully unspoiled:** Everything else
## Why a separate daemon

This repo exists to keep image-generation library state (Flux / Z-Image
weights, CUDA/MLX contexts) out of the request-handling server process. The
server (`sidequest-server`, Python) calls this daemon over a Unix socket per
ADR-035 — the boundary survived the Rust→Python port (ADR-082) because the
isolation benefits (independent restart, GPU lifecycle, slow warmup) still
apply. Music tracks are generated by the daemon's music pipeline on
operator command (see ADR-095 — Daemon Music Tier via ACE-Step).
Per-track JSON params files in `sidequest-content/genre_packs/<pack>/audio/music/`
are the canonical generation spec. Run `python scripts/generate_music.py
--genre <pack>` to regenerate missing audio.

## Build Commands

```bash
uv sync                  # Install dependencies
uv run pytest            # Run tests
uv run python -m sidequest_daemon  # Run daemon
```

## Architecture

- **Unix socket server** (`/tmp/sidequest-renderer.sock`) — JSON-RPC protocol, routes by `tier` field
- **Image gen**: Z-Image Turbo via MLX / mflux (ADR-070, ADR-086) — portraits, POI landscapes, illustrations. No negative prompts (prose bleeds as text in Z-Image); see `sidequest-content/PROMPTING_Z_IMAGE.md`.
- **Music**: ACE-Step generation tier (ADR-095). Operator runs `scripts/generate_music.py --genre <pack>`; the script discovers `*_input_params.json` files and dispatches each to the daemon as `tier=music`. Daemon runs ACE-Step → ffmpeg WAV→OGG (libopus 96k) → R2 upload at `genre_packs/<pack>/audio/music/<track>.ogg`. See `sidequest_daemon/media/music_pipeline.py`.
- **Scene interpretation**: Rules-based narration → visual cue extraction (StageCue)
- **Subject extraction**: Claude CLI invocation that turns narrator prose into visual descriptions
- **Config**: Reads genre pack paths from `SIDEQUEST_GENRE_PACKS` env var

## Git Workflow

- Branch strategy: gitflow
- Default branch: develop
- Feature branches: `feat/{description}`
- PRs target: develop
