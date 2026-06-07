"""MLX allocator memory management — ping-pong 2026-06-07.

"[BUG] Daemon MLX buffer cache is unbounded — renderer phys_footprint climbs
to 70GB+ across a render sweep, will OOM on long batches."

Root cause (confirmed by grep at filing time): no MLX memory management
anywhere in the daemon — the MLX allocator retains freed buffers forever and
only releases them on process exit. Two controls land here:

  1. ``mx.set_cache_limit`` at worker construction — a hard ceiling on how
     many freed buffers the allocator may retain, configurable via
     ``SIDEQUEST_MLX_CACHE_LIMIT_GB`` (loud failure on a malformed value,
     No Silent Fallbacks).
  2. ``mx.clear_cache()`` after EVERY render (success or failure) and after
     warm_up's dummy generation — returns retained buffers to the OS so the
     footprint between renders is the model weights, not the sweep history.

OTEL (house rule): the render span carries ``mlx.cache_bytes_cleared`` /
``mlx.active_memory_bytes`` / ``mlx.peak_memory_bytes`` so the GM panel can
see the allocator being bounded instead of trusting it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sidequest_daemon.media.workers import zimage_mlx_worker as zw
from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker
from tests.conftest import fake_pil_image


def _make_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ZImageMLXWorker:
    monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
    worker = ZImageMLXWorker(output_dir=tmp_path)
    mock_model = MagicMock(name="ZImage")
    mock_model.generate_image.return_value = fake_pil_image(1024, 1024)
    worker.model = mock_model
    return worker


# ── set_cache_limit at construction ──────────────────────────────────────────


class TestCacheLimitAtConstruction:
    def test_init_sets_default_cache_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constructing the worker must bound the MLX allocator cache —
        default 8 GiB when SIDEQUEST_MLX_CACHE_LIMIT_GB is unset."""
        monkeypatch.delenv("SIDEQUEST_MLX_CACHE_LIMIT_GB", raising=False)
        spy = MagicMock()
        monkeypatch.setattr(zw.mx, "set_cache_limit", spy)

        _make_worker(tmp_path, monkeypatch)

        spy.assert_called_once_with(8 * 2**30)

    def test_init_honors_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIDEQUEST_MLX_CACHE_LIMIT_GB", "2")
        spy = MagicMock()
        monkeypatch.setattr(zw.mx, "set_cache_limit", spy)

        _make_worker(tmp_path, monkeypatch)

        spy.assert_called_once_with(2 * 2**30)

    def test_init_honors_fractional_gb(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIDEQUEST_MLX_CACHE_LIMIT_GB", "0.5")
        spy = MagicMock()
        monkeypatch.setattr(zw.mx, "set_cache_limit", spy)

        _make_worker(tmp_path, monkeypatch)

        spy.assert_called_once_with(int(0.5 * 2**30))

    @pytest.mark.parametrize("bad", ["banana", "-1", ""])
    def test_init_rejects_malformed_limit_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
    ) -> None:
        """No Silent Fallbacks: a malformed SIDEQUEST_MLX_CACHE_LIMIT_GB must
        refuse construction, not silently fall back to the default."""
        monkeypatch.setenv("SIDEQUEST_MLX_CACHE_LIMIT_GB", bad)
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        with pytest.raises(ValueError, match="SIDEQUEST_MLX_CACHE_LIMIT_GB"):
            ZImageMLXWorker(output_dir=tmp_path)

    def test_zero_disables_cache_retention(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """0 is a legitimate operator choice (no buffer retention at all)."""
        monkeypatch.setenv("SIDEQUEST_MLX_CACHE_LIMIT_GB", "0")
        spy = MagicMock()
        monkeypatch.setattr(zw.mx, "set_cache_limit", spy)

        _make_worker(tmp_path, monkeypatch)

        spy.assert_called_once_with(0)


# ── clear_cache per render ───────────────────────────────────────────────────


class TestClearCachePerRender:
    def test_render_clears_cache_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = _make_worker(tmp_path, monkeypatch)
        spy = MagicMock()
        monkeypatch.setattr(zw.mx, "clear_cache", spy)

        worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

        spy.assert_called_once()

    def test_render_clears_cache_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed render must still release retained buffers — a long batch
        with intermittent failures must not leak its way to OOM."""
        worker = _make_worker(tmp_path, monkeypatch)
        worker.model.generate_image.side_effect = RuntimeError("boom")
        spy = MagicMock()
        monkeypatch.setattr(zw.mx, "clear_cache", spy)

        with pytest.raises(RuntimeError, match="boom"):
            worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

        spy.assert_called_once()

    def test_warm_up_clears_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The warm-up dummy generation allocates real buffers too."""
        worker = _make_worker(tmp_path, monkeypatch)
        spy = MagicMock()
        monkeypatch.setattr(zw.mx, "clear_cache", spy)

        worker.warm_up()

        spy.assert_called_once()


# ── OTEL visibility ──────────────────────────────────────────────────────────


class TestMemorySpanAttributes:
    @pytest.fixture
    def otel_exporter(self, monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        monkeypatch.setattr(
            trace,
            "_TRACER_PROVIDER_SET_ONCE",
            trace._TRACER_PROVIDER_SET_ONCE.__class__(),
        )
        trace.set_tracer_provider(provider)
        return exporter

    def test_render_span_carries_memory_attributes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        """The lie-detector for this fix: the GM panel must be able to SEE the
        allocator being cleared, not trust that it is."""
        worker = _make_worker(tmp_path, monkeypatch)

        worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert len(render_spans) == 1
        attrs = render_spans[0].attributes or {}
        assert "mlx.cache_bytes_cleared" in attrs
        assert "mlx.active_memory_bytes" in attrs
        assert "mlx.peak_memory_bytes" in attrs
        assert "mlx.cache_limit_bytes" in attrs
