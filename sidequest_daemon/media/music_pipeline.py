"""MusicPipeline — orchestrates one ACE-Step generation job end-to-end.

Reads a per-track JSON params file, derives the R2 destination key from
the file's path, runs the adapter, converts WAV → OGG, uploads to R2,
emits watcher events at every stage. Per spec
docs/superpowers/specs/2026-05-09-daemon-between-session-music-generation-design.md.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_GENRE_PACKS_RE = re.compile(r".*?(genre_packs/.*?)/audio/music/(.+?)_input_params\.json$")


def _run_ffmpeg(wav_path: Path, ogg_path: Path) -> None:
    """Convert WAV → OGG (libopus, 160kbps). Raises CalledProcessError on
    failure or TimeoutExpired if FFmpeg exceeds 120s (a 60s WAV should
    convert in seconds; a hang means corrupt input).

    Container: OGG. Codec: Opus. Picked over libvorbis because the
    standard Homebrew ffmpeg build ships libopus by default but not
    libvorbis; Opus also produces smaller files at equivalent quality.

    Bitrate: 160k. ACE-Step generates 48 kHz stereo; 160k Opus is
    perceptually transparent for nearly all listeners on dense ambient /
    orchestral material (Opus is highly efficient — ~160k ≈ 320k MP3). The
    original 96k was an undocumented first-write default tuned for size,
    not quality; bumped per durable-retention (storage is cheap, re-encoding
    old saves' broken audio is not).
    """
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path),
         "-c:a", "libopus", "-b:a", "160k", str(ogg_path)],
        check=True, capture_output=True, timeout=120,
    )


@contextmanager
def _tempdir():
    """Yields a tempdir path; deletes everything on exit (success or failure)."""
    d = Path(tempfile.mkdtemp(prefix="music_pipeline_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class RefAudioFetchError(Exception):
    """Raised when an audio2audio variation's base-theme OGG can't be pulled
    from R2. Distinguished so the failure event reports stage=ref_fetch rather
    than being misclassified as an upload/inference error."""


@dataclass
class MusicResult:
    r2_key: str
    duration_ms: int
    seed: int
    elapsed_ms: int


class MusicPipeline:
    """Single-job orchestrator. Constructed once per daemon process,
    reused across requests."""

    def __init__(self, *, adapter, r2_uploader, r2_downloader, watcher, render_lock):
        self._adapter = adapter
        self._r2_uploader = r2_uploader
        self._r2_downloader = r2_downloader
        self._watcher = watcher
        self._render_lock = render_lock

    @staticmethod
    def derive_r2_key(json_path: Path) -> str:
        """Strip `_input_params.json`, append `.ogg`, anchor under
        `genre_packs/<pack>/`. Raises ValueError if path doesn't fit."""
        s = str(json_path)
        m = _GENRE_PACKS_RE.match(s)
        if not m:
            raise ValueError(
                f"INVALID_PARAMS_LOCATION: {json_path} is not under a "
                f"genre_packs/<pack>/audio/music/ directory or is missing "
                f"the _input_params.json suffix"
            )
        pack_path, name = m.group(1), m.group(2)
        return f"{pack_path}/audio/music/{name}.ogg"

    async def generate(self, json_path: Path) -> MusicResult:
        r2_key = self.derive_r2_key(json_path)
        prompt_excerpt = ""
        params_for_log: dict = {}
        try:
            params_for_log = json.loads(json_path.read_text())
            prompt_excerpt = str(params_for_log.get("prompt", ""))[:120]
            duration_s = int(params_for_log.get("audio_duration", 0))
        except Exception:
            duration_s = 0

        # audio2audio leitmotif variations name their base-theme OGG by its R2
        # key in ref_audio_input; resolve it to fetch the bytes at render time.
        ref_audio_key = None
        ref = params_for_log.get("ref_audio_input")
        if params_for_log.get("audio2audio_enable") and isinstance(ref, str) \
                and ref.startswith("genre_packs/"):
            ref_audio_key = ref

        self._watcher("music.generation.start", {
            "r2_key": r2_key,
            "prompt_excerpt": prompt_excerpt,
            "duration_s": duration_s,
            "json_params_path": str(json_path),
        })

        t_start = time.perf_counter()
        async with self._render_lock:
            try:
                with _tempdir() as td:
                    wav_path = td / "out.wav"
                    ogg_path = td / "out.ogg"

                    ref_audio_override = None
                    if ref_audio_key is not None:
                        t0 = time.perf_counter()
                        try:
                            ref_bytes = self._r2_downloader(ref_audio_key)
                        except Exception as exc:
                            raise RefAudioFetchError(
                                f"could not fetch base audio {ref_audio_key!r}: {exc}"
                            ) from exc
                        ref_path = td / "ref.ogg"
                        ref_path.write_bytes(ref_bytes)
                        ref_audio_override = str(ref_path)
                        self._watcher("music.ref_audio.fetch", {
                            "r2_key": r2_key,
                            "ref_audio_key": ref_audio_key,
                            "ref_bytes": len(ref_bytes),
                            "fetch_ms": int((time.perf_counter() - t0) * 1000),
                        })

                    t0 = time.perf_counter()
                    inference = self._adapter.run(
                        json_path, wav_path, ref_audio_override=ref_audio_override
                    )
                    inference_ms = int((time.perf_counter() - t0) * 1000)

                    t0 = time.perf_counter()
                    _run_ffmpeg(wav_path, ogg_path)
                    ffmpeg_ms = int((time.perf_counter() - t0) * 1000)

                    t0 = time.perf_counter()
                    self._r2_uploader(
                        ogg_path.read_bytes(), r2_key, "audio/ogg",
                    )
                    upload_ms = int((time.perf_counter() - t0) * 1000)
                    file_size = ogg_path.stat().st_size

                elapsed_ms = int((time.perf_counter() - t_start) * 1000)
                self._watcher("music.generation.complete", {
                    "r2_key": r2_key,
                    "elapsed_ms": elapsed_ms,
                    "inference_ms": inference_ms,
                    "ffmpeg_ms": ffmpeg_ms,
                    "upload_ms": upload_ms,
                    "seed": inference.seed,
                    "file_size_bytes": file_size,
                })
                return MusicResult(
                    r2_key=r2_key, duration_ms=duration_s * 1000,
                    seed=inference.seed, elapsed_ms=elapsed_ms,
                )

            except Exception as exc:
                stage = self._classify_failure_stage(exc)
                self._watcher("music.generation.failed", {
                    "r2_key": r2_key,
                    "error_code": type(exc).__name__,
                    "stage": stage,
                    "detail": str(exc)[:512],
                })
                raise

    @staticmethod
    def _classify_failure_stage(exc: Exception) -> str:
        if isinstance(exc, RefAudioFetchError):
            return "ref_fetch"
        msg = str(exc).lower()
        if "ffmpeg" in msg or isinstance(exc, subprocess.CalledProcessError):
            return "ffmpeg"
        if "missing_seed" in msg or "invalid_params" in msg:
            return "params"
        if "r2" in msg or "boto" in msg or "s3" in msg:
            return "upload"
        return "inference"
