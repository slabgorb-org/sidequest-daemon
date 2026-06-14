# sidequest-daemon

Media services daemon for SideQuest — image generation and music generation.
Runs as a standalone Python process communicating via Unix socket, providing
rendering services to the Python `sidequest-server` (port 8765, ADR-082).

> **Active renderer:** Z-Image Turbo (MLX). Flux paths have been retired —
> `zimage_mlx_worker.py` is the sole runtime image worker.
>
> **Music tier (ADR-095):** ACE-Step generation runs on operator command, not
> at every turn. Per-track `*_input_params.json` files in
> `sidequest-content/genre_packs/<pack>/audio/music/` are the canonical spec;
> the daemon produces OGG (libopus 96k) and uploads to R2.

## Architecture

```
sidequest-server (Python)  ──JSON over Unix socket──►  sidequest-daemon (Python)
                                                         ├── Z-Image image generation (MLX, Apple Silicon)
                                                         ├── ACE-Step music generation (operator-triggered)
                                                         ├── Sentence embeddings (ADR-048 lore RAG)
                                                         └── Scene interpretation (narration → StageCue)
```

The daemon is a hot-loaded process — start it once, and it keeps the Z-Image
model warm between requests. The server sends render commands over
`/tmp/sidequest-renderer.sock` and gets back file paths to generated images.

## Services

**Z-Image Worker** (`media/workers/zimage_mlx_worker.py`, ADR-070) — Image
generation across composition tiers from ADR-086: portrait, POI landscape,
illustration. Targets Apple Silicon via MLX / mflux. Z-Image Turbo is the
active model; prose bleeds as text in image prompts, so no negative-prompt
syntax is used. See `sidequest-content/PROMPTING_Z_IMAGE.md`. Per-process
singleton enforced (story 43-5). Fidelity tier swap via
`SIDEQUEST_DAEMON_FIDELITY` (story 45-39): default Turbo (~33s/render);
high-fidelity (story 45-38) for portraits-with-time-to-burn. Renders upload
to R2 alongside the legacy file path (`r2_key` returned in the response).

**Music Pipeline** (`media/music_pipeline.py`, ADR-095) — ACE-Step generation
tier. Operator runs `python scripts/generate_music.py --genre <pack>` from the
orchestrator; the script discovers `*_input_params.json` files and dispatches
each to the daemon as `tier=music`. The daemon runs ACE-Step → ffmpeg
WAV→OGG (libopus 96k) → R2 upload at
`genre_packs/<pack>/audio/music/<track>.ogg`. Generation is **operator-
triggered**, not per-turn — costly and slow, so it's a build-time act.

**Embedding Worker** (`media/embed_worker.py`, ADR-048) — Sentence-embedding
worker behind the `embed` method, feeding the cross-process lore RAG store.
Pre-loaded by `warm_up` alongside the Z-Image model.

> Server-side audio playback once lived here (a `pygame-ce` mixer in `audio/`);
> it was removed with the 2026-04 TTS removal. Music + SFX now play client-side
> in `sidequest-ui`.

**Scene Interpreter** (`scene_interpreter.py`) — Rules-based narration-to-
`StageCue` extractor. Turns narrator prose into structured visual cues for the
image pipeline.

**Subject Extractor** (`media/subject_extractor.py`) — Claude CLI invocation
that turns prose into visual descriptions for portrait/POI prompts.

## Prerequisites

- Python 3.12 (mflux pins below 3.13)
- Apple Silicon (MLX / MPS). CUDA path is no longer the primary target after ADR-070.
- Genre packs directory (shared YAML + audio assets)

## Installation

```bash
# From sidequest-daemon:
uv sync

# Or from the orchestrator root:
just setup
```

## Usage

### Starting the daemon

```bash
# Lazy model loading — no GPU needed until first render:
uv run python -m sidequest_daemon

# With model warmup (recommended for gameplay):
uv run python -m sidequest_daemon --warmup
```

### From the orchestrator

```bash
just daemon          # Start with warmup
just daemon-status   # Check if running
just daemon-stop     # Graceful shutdown
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SIDEQUEST_GENRE_PACKS` | `../sidequest-content/genre_packs` | Path to genre packs |
| `SIDEQUEST_OUTPUT_DIR` | temp dir | Directory for generated output files |
| `SIDEQUEST_DAEMON_FIDELITY` | `turbo` | Z-Image worker tier — `turbo` (8-step preset, ~33s) or `high_fidelity` (20-step, CFG 4, base 1.0) per story 45-38/45-39 |

## Protocol

Newline-delimited JSON over Unix domain socket at `/tmp/sidequest-renderer.sock`.

### Request

```json
{"id": "unique-id", "method": "render", "params": {"tier": "portrait", "subject": "...", ...}}
```

### Response

```json
{"id": "unique-id", "result": {"image_path": "/tmp/...", "width": 1024, "height": 1024, "elapsed_ms": 2800}}
```

### Methods

| Method | Description |
|--------|-------------|
| `ping` | Health check — returns `{"status": "ok"}` |
| `status` | Worker pool status and loaded models |
| `render` | Generate an image, routed by `tier` |
| `music` | Generate a music track via ACE-Step (ADR-095) — operator-triggered |
| `embed` | Generate a sentence embedding for the lore RAG store (ADR-048) |
| `warm_up` | Pre-load the Z-Image model (and embedding model for ADR-048 lore RAG) |
| `shutdown` | Graceful daemon shutdown |

## Package structure

```
sidequest_daemon/
├── media/                       # Daemon server, workers, pipelines
│   ├── daemon.py                # Entry point — Unix socket server + CLI
│   ├── workers/
│   │   └── zimage_mlx_worker.py # Sole runtime image worker (ADR-070)
│   ├── worker_pool.py           # Singleton Z-Image + embedding workers
│   ├── render_service.py        # Render request orchestration, lock, post-process
│   ├── pipeline_factory.py      # Constructs render pipelines
│   ├── tiers.py                 # Render tier constants
│   ├── embed_worker.py          # Sentence-embedding worker (ADR-048 lore RAG)
│   ├── music_pipeline.py        # ACE-Step → ffmpeg → R2 (ADR-095)
│   ├── ace_step_adapter.py
│   ├── prompt_composer.py       # Tier-aware prompt prefixes, token budgeting
│   ├── subject_extractor.py     # Prose → visual description via Claude CLI
│   ├── recipes.py / recipe_loader.py
│   ├── catalogs.py / camera_specs.py
│   ├── post_processor.py        # Post-render adjustments
│   ├── preview.py
│   ├── r2_writer.py             # R2 upload for music artifacts
│   └── zimage_config.py
├── renderer/                    # Data models (StageCue, RenderTier, RenderResult)
├── training/                    # Fine-tune + deploy CLIs: sidequest-train,
│                                #   sidequest-deploy (ADR-073)
├── telemetry/                   # watcher_bridge.py — OTEL HTTP bridge to server (ADR-131)
├── genre/                       # Genre pack model subset (VisualStyle, AudioConfig)
└── scene_interpreter.py         # Narrative → StageCue rules engine
```

> The `ml/` (ADR-046 GPU memory budget) and `audio/` (`pygame-ce` mixer) modules
> were removed as dead code in 2026-06; ADR-046 is retired. Music ducking and TTS
> playback are gone with the 2026-04 TTS removal.

> If a file listed above has moved or been removed, treat the source tree as
> authoritative and update this README rather than the other way around.

## Development

```bash
just daemon-test     # Run tests
just daemon-lint     # ruff check
```

Smoke test without GPU:

```bash
uv run pytest tests/test_daemon_smoke.py -v
```

## Origin

Lifted from [sq-2](https://github.com/slabgorb/sq-2) (the original Python
SideQuest codebase) as part of the now-archived Rust rewrite. The daemon
boundary survived the Rust→Python port (ADR-082) because the isolation
benefits — independent restart, GPU lifecycle, slow model warmup — apply
regardless of server language.

## Training pipeline

See `docs/training.md` for `sidequest-train` usage (ADR-073 Phase 3).

## Related repos

- [orc-quest](https://github.com/slabgorb/orc-quest) — Orchestrator, ADRs
- [sidequest-server](https://github.com/slabgorb/sidequest-server) — Python FastAPI backend
- [sidequest-ui](https://github.com/slabgorb/sidequest-ui) — React client
- [sidequest-content](https://github.com/slabgorb/sidequest-content) — Genre packs
