"""Tests for OTEL span emission in ZImageMLXWorker.

Verifies that the Z-Image MLX render pipeline emits OpenTelemetry spans
so the GM panel can observe model loads, renders, and warm-ups.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def otel_exporter(monkeypatch):
    """Set up an in-memory OTEL span exporter for testing.

    Uses monkeypatch to reset the global TracerProvider cleanly per test,
    avoiding the 'Overriding of current TracerProvider is not allowed' warning.
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Reset the global provider flag so set_tracer_provider works
    monkeypatch.setattr(trace, "_TRACER_PROVIDER_SET_ONCE", trace._TRACER_PROVIDER_SET_ONCE.__class__())
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    provider.shutdown()


def _make_worker_with_mock_model(tmp_path):
    """Build a ZImageMLXWorker whose underlying mflux model is mocked.

    This avoids calling load_model() (which imports mflux) while still
    letting render()/warm_up() exercise the OTEL span code paths.
    """
    from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker

    worker = ZImageMLXWorker(tmp_path)
    mock_model = MagicMock(name="ZImage")
    mock_model.generate_image.return_value = Image.new("RGB", (64, 64))
    worker.model = mock_model
    return worker


# ---------------------------------------------------------------------------
# 1. render() emits a span with correct attributes
# ---------------------------------------------------------------------------

class TestRenderSpan:
    """render() must emit an OTEL span with tier, seed, dimensions, elapsed_ms."""

    def test_render_creates_span(self, tmp_path, otel_exporter):
        """render() must create a span named 'zimage_mlx.render'."""
        worker = _make_worker_with_mock_model(tmp_path)
        worker.render({"tier": "scene_illustration", "positive_prompt": "test", "seed": 42})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert len(render_spans) >= 1, f"Expected 'zimage_mlx.render' span, got: {[s.name for s in spans]}"

    def test_render_span_has_tier(self, tmp_path, otel_exporter):
        """render span must include tier attribute."""
        worker = _make_worker_with_mock_model(tmp_path)
        worker.render({"tier": "portrait", "positive_prompt": "wizard", "seed": 1})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert len(render_spans) >= 1
        attrs = dict(render_spans[-1].attributes)
        assert attrs.get("render.tier") == "portrait"

    def test_render_span_has_seed(self, tmp_path, otel_exporter):
        """render span must include seed attribute."""
        worker = _make_worker_with_mock_model(tmp_path)
        worker.render({"tier": "landscape", "positive_prompt": "hills", "seed": 999})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        assert attrs.get("render.seed") == 999

    def test_render_span_has_dimensions(self, tmp_path, otel_exporter):
        """render span must include width and height."""
        worker = _make_worker_with_mock_model(tmp_path)
        # Story 101-1: was "cartography" (removed); fog_of_war is the surviving
        # 1024x1024 tier.
        worker.render({"tier": "fog_of_war", "positive_prompt": "map", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        assert attrs.get("render.width") == 1024
        assert attrs.get("render.height") == 1024

    def test_render_span_has_elapsed_ms(self, tmp_path, otel_exporter):
        """render span must include elapsed_ms attribute."""
        worker = _make_worker_with_mock_model(tmp_path)
        worker.render({"tier": "scene_illustration", "positive_prompt": "forest", "seed": 7})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        assert "render.elapsed_ms" in attrs
        assert isinstance(attrs["render.elapsed_ms"], int)

    def test_render_span_has_steps(self, tmp_path, otel_exporter):
        """render span must include the step count for the tier."""
        worker = _make_worker_with_mock_model(tmp_path)
        # Story 101-1: was "cartography" (removed); fog_of_war shares the
        # high-fidelity 20-step config under base Z-Image 1.0.
        worker.render({"tier": "fog_of_war", "positive_prompt": "map", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        # Default flipped to high_fidelity 2026-05-02 — fog_of_war uses
        # 20 steps under base Z-Image 1.0 (was 8 under Turbo).
        assert attrs.get("render.steps") == 20

    def test_render_span_has_guidance(self, tmp_path, otel_exporter):
        """render span must include guidance scale for the tier."""
        worker = _make_worker_with_mock_model(tmp_path)
        worker.render({"tier": "text_overlay", "positive_prompt": "title", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        # Default flipped to high_fidelity 2026-05-02 — base Z-Image 1.0
        # supports CFG; guidance recorded as 4.0. (Turbo's 0.0 still applies
        # under SIDEQUEST_DAEMON_FIDELITY=turbo, covered elsewhere.)
        assert attrs.get("render.guidance") == pytest.approx(4.0)

    def test_render_span_has_model_variant(self, tmp_path, otel_exporter):
        """render span must include model.variant so the GM panel can tell
        Turbo apart from base Z-Image."""
        worker = _make_worker_with_mock_model(tmp_path)
        worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        # Default flipped to high_fidelity 2026-05-02 — base ``z-image``
        # variant under no env var (Turbo opt-in via SIDEQUEST_DAEMON_FIDELITY).
        assert attrs.get("model.variant") == "z-image"


# ---------------------------------------------------------------------------
# 2. load_model() emits a span
# ---------------------------------------------------------------------------

class TestLoadModelSpan:
    """load_model() must emit an OTEL span with model.name attribute."""

    def test_load_model_creates_span(self, tmp_path, otel_exporter, monkeypatch):
        from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker

        # Stub out the mflux imports inside load_model by pre-setting self.model
        # and monkeypatching the import machinery via a fake module path is fragile;
        # instead, make load_model short-circuit by patching ZImage construction.
        worker = ZImageMLXWorker(tmp_path)

        # Patch the mflux imports to avoid loading the real model.
        import sys
        import types

        fake_mflux = types.ModuleType("mflux")
        fake_common = types.ModuleType("mflux.models.common")
        fake_common_config = types.ModuleType("mflux.models.common.config")
        fake_common_config.ModelConfig = MagicMock()
        fake_zimage_pkg = types.ModuleType("mflux.models.z_image")
        fake_zimage_variants = types.ModuleType("mflux.models.z_image.variants")
        fake_zimage_mod = types.ModuleType("mflux.models.z_image.variants.z_image")
        fake_zimage_mod.ZImage = MagicMock(return_value=MagicMock(name="ZImageInstance"))

        monkeypatch.setitem(sys.modules, "mflux", fake_mflux)
        monkeypatch.setitem(sys.modules, "mflux.models", types.ModuleType("mflux.models"))
        monkeypatch.setitem(sys.modules, "mflux.models.common", fake_common)
        monkeypatch.setitem(sys.modules, "mflux.models.common.config", fake_common_config)
        monkeypatch.setitem(sys.modules, "mflux.models.z_image", fake_zimage_pkg)
        monkeypatch.setitem(sys.modules, "mflux.models.z_image.variants", fake_zimage_variants)
        monkeypatch.setitem(sys.modules, "mflux.models.z_image.variants.z_image", fake_zimage_mod)

        worker.load_model()

        spans = otel_exporter.get_finished_spans()
        load_spans = [s for s in spans if s.name == "zimage_mlx.load_model"]
        assert len(load_spans) >= 1, f"Expected 'zimage_mlx.load_model' span, got: {[s.name for s in spans]}"

    def test_load_model_span_has_model_name(self, tmp_path, otel_exporter, monkeypatch):
        from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker

        worker = ZImageMLXWorker(tmp_path)

        import sys
        import types

        fake_common_config = types.ModuleType("mflux.models.common.config")
        fake_common_config.ModelConfig = MagicMock()
        fake_zimage_mod = types.ModuleType("mflux.models.z_image.variants.z_image")
        fake_zimage_mod.ZImage = MagicMock(return_value=MagicMock(name="ZImageInstance"))

        monkeypatch.setitem(sys.modules, "mflux", types.ModuleType("mflux"))
        monkeypatch.setitem(sys.modules, "mflux.models", types.ModuleType("mflux.models"))
        monkeypatch.setitem(sys.modules, "mflux.models.common", types.ModuleType("mflux.models.common"))
        monkeypatch.setitem(sys.modules, "mflux.models.common.config", fake_common_config)
        monkeypatch.setitem(sys.modules, "mflux.models.z_image", types.ModuleType("mflux.models.z_image"))
        monkeypatch.setitem(sys.modules, "mflux.models.z_image.variants", types.ModuleType("mflux.models.z_image.variants"))
        monkeypatch.setitem(sys.modules, "mflux.models.z_image.variants.z_image", fake_zimage_mod)

        worker.load_model()

        spans = otel_exporter.get_finished_spans()
        load_spans = [s for s in spans if s.name == "zimage_mlx.load_model"]
        attrs = dict(load_spans[-1].attributes)
        assert attrs.get("model.name") == "z-image"


# ---------------------------------------------------------------------------
# 3. warm_up() emits a span
# ---------------------------------------------------------------------------

class TestWarmUpSpan:
    """warm_up() must emit an OTEL span with warmup.elapsed_ms attribute."""

    def test_warm_up_creates_span(self, tmp_path, otel_exporter):
        worker = _make_worker_with_mock_model(tmp_path)
        worker.warm_up()

        spans = otel_exporter.get_finished_spans()
        warmup_spans = [s for s in spans if s.name == "zimage_mlx.warm_up"]
        assert len(warmup_spans) >= 1, f"Expected 'zimage_mlx.warm_up' span, got: {[s.name for s in spans]}"

    def test_warm_up_span_has_elapsed(self, tmp_path, otel_exporter):
        worker = _make_worker_with_mock_model(tmp_path)
        worker.warm_up()

        spans = otel_exporter.get_finished_spans()
        warmup_spans = [s for s in spans if s.name == "zimage_mlx.warm_up"]
        attrs = dict(warmup_spans[-1].attributes)
        assert "warmup.elapsed_ms" in attrs


# ---------------------------------------------------------------------------
# 4. Error spans — render failure records exception
# ---------------------------------------------------------------------------

class TestErrorSpans:
    """Render failures must record exception on the span."""

    def test_render_error_records_exception(self, tmp_path, otel_exporter):
        """ValueError from unsupported tier should be recorded on the span."""
        worker = _make_worker_with_mock_model(tmp_path)

        with pytest.raises(ValueError):
            worker.render({"tier": "bogus", "positive_prompt": "test"})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert len(render_spans) >= 1, f"Expected error span, got: {[s.name for s in spans]}"
        error_span = render_spans[-1]
        assert error_span.status.is_ok is False, "Error span should have non-OK status"


# ---------------------------------------------------------------------------
# 6. Tracer name — must use correct module name
# ---------------------------------------------------------------------------

class TestTracerName:
    """Spans must use a tracer named for the daemon media module."""

    def test_tracer_name(self, tmp_path, otel_exporter):
        """Tracer instrumentation scope must identify the daemon media module."""
        worker = _make_worker_with_mock_model(tmp_path)
        worker.render({"tier": "scene_illustration", "positive_prompt": "test", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert len(render_spans) >= 1
        scope = render_spans[-1].instrumentation_scope
        assert "sidequest_daemon" in scope.name, (
            f"Tracer name should contain 'sidequest_daemon', got: {scope.name}"
        )


# ---------------------------------------------------------------------------
# 7. Wiring — OTEL dependency exists
# ---------------------------------------------------------------------------

class TestOtelWiring:
    """opentelemetry-api must be in main dependencies (not just dev)."""

    def test_otel_api_in_main_deps(self):
        """pyproject.toml must list opentelemetry-api in main dependencies."""
        source = Path(__file__).parent.parent / "pyproject.toml"
        content = source.read_text()
        # Check main dependencies section (before [project.optional-dependencies])
        main_deps_section = content.split("[project.optional-dependencies]")[0]
        assert "opentelemetry-api" in main_deps_section, (
            "opentelemetry-api must be in main dependencies, not just dev"
        )
