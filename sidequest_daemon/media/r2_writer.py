"""Direct daemon → R2 artifact uploader.

Per CLAUDE.md no-silent-fallback: any boto error propagates. The daemon's
caller is responsible for surfacing the failure to the server, which emits
an image_unavailable event. We never hand back a fake URL.
"""
from __future__ import annotations

import hashlib
import os
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal

import boto3
from botocore.client import BaseClient
from opentelemetry import trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

# Test seam — overridden via patch.object in unit tests to inject an
# in-memory exporter. Production: read the global tracer provider.
# opentelemetry-sdk is a dev-only dep; the import lives under TYPE_CHECKING
# so the daemon can run with main deps alone.
_tracer_provider_for_tests: TracerProvider | None = None


def _get_tracer() -> trace.Tracer:
    if _tracer_provider_for_tests is not None:
        return _tracer_provider_for_tests.get_tracer(
            "sidequest_daemon.media.r2_writer"
        )
    return trace.get_tracer("sidequest_daemon.media.r2_writer")

ArtifactKind = Literal["portraits", "poi", "scenes", "music", "sfx"]
_VALID_KINDS: Final[frozenset[str]] = frozenset(
    {"portraits", "poi", "scenes", "music", "sfx"}
)

_EXT_FOR_CONTENT_TYPE: Final[dict[str, str]] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "audio/ogg": "ogg",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/flac": "flac",
}

CACHE_CONTROL_ARTIFACTS: Final[str] = "public, max-age=86400"
BUCKET: Final[str] = "sidequest"


@lru_cache(maxsize=1)
def _client() -> BaseClient:
    """Lazy boto3 client singleton; respects env mutation across tests
    via the patch.object(r2_writer, "_client", ...) idiom."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_S3_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_artifact(
    *,
    world_slug: str,
    session_id: str,
    kind: ArtifactKind,
    content_bytes: bytes,
    content_type: str,
) -> str:
    """Upload `content_bytes` to R2 under
    ``artifacts/<world>/<session>/<kind>/<sha256>.<ext>``.

    Returns the relative key. Raises ValueError on invalid kind/content_type.
    Raises any boto3/HTTP error verbatim — caller must propagate, not swallow.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
        )
    if content_type not in _EXT_FOR_CONTENT_TYPE:
        raise ValueError(
            f"content_type must be one of {sorted(_EXT_FOR_CONTENT_TYPE)}, "
            f"got {content_type!r}"
        )

    ext = _EXT_FOR_CONTENT_TYPE[content_type]
    sha = hashlib.sha256(content_bytes).hexdigest()
    key = f"artifacts/{world_slug}/{session_id}/{kind}/{sha}.{ext}"
    size = len(content_bytes)
    tracer = _get_tracer()

    with tracer.start_as_current_span("daemon.r2.upload.start") as start_span:
        start_span.set_attribute("upload.kind", kind)
        start_span.set_attribute("upload.world", world_slug)
        start_span.set_attribute("upload.session", session_id)
        start_span.set_attribute("upload.bytes", size)

    t0 = time.perf_counter()
    try:
        _client().put_object(
            Bucket=BUCKET,
            Key=key,
            Body=content_bytes,
            ContentType=content_type,
            CacheControl=CACHE_CONTROL_ARTIFACTS,
        )
    except Exception as exc:
        with tracer.start_as_current_span("daemon.r2.upload.failure") as fail_span:
            fail_span.set_attribute("upload.kind", kind)
            fail_span.set_attribute("upload.error_class", exc.__class__.__name__)
            fail_span.set_attribute("upload.error_message", str(exc))
            fail_span.set_attribute("upload.retry_attempt", 0)
        raise

    dt_ms = int((time.perf_counter() - t0) * 1000)
    with tracer.start_as_current_span("daemon.r2.upload.success") as ok_span:
        ok_span.set_attribute("upload.kind", kind)
        ok_span.set_attribute("upload.key", key)
        ok_span.set_attribute("upload.ms", dt_ms)
        ok_span.set_attribute("upload.bytes", size)
    return key


def download_pack_asset(r2_key: str) -> bytes:
    """Download the pack asset at `r2_key` (must start with `genre_packs/`) and
    return its bytes.

    The mirror of `upload_pack_asset`. Used to fetch a base-theme OGG so an
    audio2audio leitmotif variation can condition on it — the music pipeline
    persists only the OGG to R2, so the variation's `ref_audio_input` names the
    base OGG by its R2 key and we pull it back to a local file at render time.

    Raises ValueError on an invalid key. Propagates any boto3/HTTP error
    verbatim — a missing base OGG must fail loudly, never silently skip the
    conditioning (no silent fallback).
    """
    if not r2_key.startswith("genre_packs/"):
        raise ValueError(f"r2_key must start with 'genre_packs/', got {r2_key!r}")

    tracer = _get_tracer()
    t0 = time.perf_counter()
    with tracer.start_as_current_span("daemon.r2.download.pack_asset") as span:
        span.set_attribute("download.key", r2_key)
        try:
            resp = _client().get_object(Bucket=BUCKET, Key=r2_key)
            data = resp["Body"].read()
        except Exception as exc:
            span.set_attribute("download.error_class", exc.__class__.__name__)
            span.set_attribute("download.error_message", str(exc))
            raise
        dt_ms = int((time.perf_counter() - t0) * 1000)
        span.set_attribute("download.ms", dt_ms)
        span.set_attribute("download.bytes", len(data))

    return data


def upload_pack_asset(
    *,
    r2_key: str,
    content_bytes: bytes,
    content_type: str,
) -> str:
    """Upload `content_bytes` to R2 at `r2_key` (must start with `genre_packs/`).

    Distinct from `upload_artifact`, which writes session-scoped ephemeral
    content under `artifacts/<world>/<session>/...`. Pack assets use the
    raw key the caller provides — the JSON params file's location IS the
    identity (see music_pipeline.derive_r2_key).

    Returns the key. Raises ValueError on invalid key. Propagates any
    boto3 error verbatim — caller surfaces failure (no silent fallback).
    """
    if not r2_key.startswith("genre_packs/"):
        raise ValueError(f"r2_key must start with 'genre_packs/', got {r2_key!r}")
    if content_type not in _EXT_FOR_CONTENT_TYPE:
        raise ValueError(
            f"content_type must be one of {sorted(_EXT_FOR_CONTENT_TYPE)}, "
            f"got {content_type!r}"
        )

    tracer = _get_tracer()
    size = len(content_bytes)
    t0 = time.perf_counter()

    with tracer.start_as_current_span("daemon.r2.upload.pack_asset") as span:
        span.set_attribute("upload.key", r2_key)
        span.set_attribute("upload.bytes", size)
        try:
            _client().put_object(
                Bucket=BUCKET,
                Key=r2_key,
                Body=content_bytes,
                ContentType=content_type,
                CacheControl=CACHE_CONTROL_ARTIFACTS,
            )
        except Exception as exc:
            span.set_attribute("upload.error_class", exc.__class__.__name__)
            span.set_attribute("upload.error_message", str(exc))
            raise
        dt_ms = int((time.perf_counter() - t0) * 1000)
        span.set_attribute("upload.ms", dt_ms)

    return r2_key
