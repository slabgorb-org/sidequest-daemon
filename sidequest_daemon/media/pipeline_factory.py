"""MediaPipelineFactory — constructs the music pipeline at daemon startup.

Previously constructed an audio playback pipeline (mixer, queue, etc.)
that had no production consumers. That tree was deleted; this factory
now exists solely to wire the music generation pipeline. Image rendering
is constructed elsewhere; this file is music-only for now.
"""
from __future__ import annotations

import asyncio
import logging

from sidequest_daemon.media.ace_step_adapter import AceStepAdapter
from sidequest_daemon.media.music_pipeline import MusicPipeline
from sidequest_daemon.media.r2_writer import download_pack_asset, upload_pack_asset
from sidequest_daemon.telemetry import emit_watcher_event

log = logging.getLogger(__name__)


class MediaPipelineFactory:
    """Lazy constructor for the music generation pipeline."""

    def __init__(self) -> None:
        self.music_pipeline: MusicPipeline | None = None

    def init_music(self, *, render_lock: asyncio.Lock) -> None:
        """Construct the music pipeline. Called once at daemon startup."""
        adapter = AceStepAdapter()  # production: lazy-loads model on first run

        def _r2_uploader(content_bytes: bytes, r2_key: str, content_type: str) -> str:
            return upload_pack_asset(
                r2_key=r2_key,
                content_bytes=content_bytes,
                content_type=content_type,
            )

        def _r2_downloader(r2_key: str) -> bytes:
            return download_pack_asset(r2_key)

        def _watcher(event_type: str, fields: dict) -> None:
            emit_watcher_event(event_type, fields, component="daemon.music")

        self.music_pipeline = MusicPipeline(
            adapter=adapter,
            r2_uploader=_r2_uploader,
            r2_downloader=_r2_downloader,
            watcher=_watcher,
            render_lock=render_lock,
        )
        log.info("MusicPipeline initialized")
