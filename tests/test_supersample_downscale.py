"""Tests for the supersample+Lanczos downscale render-tier feature.

Behavioral tests for:
  - Supersample dimension math: target × factor = generator input dims.
  - Lanczos downscale step: a 2× PIL image is reduced to target dims.
  - Factor=1 is a true no-op: no resize occurs, identity is preserved.
  - Invalid supersample_factor raises loudly (No Silent Fallbacks).
  - OTEL span attributes for supersample are emitted.

Full visual verification (moiré dissolution on a real POI render, e.g.
wonderland/mad_tea_party with the Tenniel pen-and-ink style) requires an
actual MLX render run on Apple GPU and is OUT OF SCOPE for automated tests.
Operator eyeball on a real POI render is the final acceptance gate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sidequest_daemon.media.post_processor import supersample_downscale
from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker
from sidequest_daemon.media.zimage_config import (
    ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS,
    ZIMAGE_TIER_CONFIGS,
    ZImageTierConfig,
)
from sidequest_daemon.renderer.models import RenderTier
from tests.conftest import fake_pil_image


# ── ZImageTierConfig carries supersample_factor ─────────────────────────────


class TestTierConfigSupersampleField:
    """ZImageTierConfig must expose supersample_factor with a default of 1."""

    def test_tier_config_has_supersample_factor_field(self) -> None:
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ZImageTierConfig)}
        assert "supersample_factor" in fields, (
            "ZImageTierConfig must carry supersample_factor"
        )

    def test_default_supersample_factor_is_1(self) -> None:
        cfg = ZImageTierConfig(
            steps=8,
            guidance=0.0,
            width=1024,
            height=768,
            model_variant="z-image-turbo",
        )
        assert cfg.supersample_factor == 1

    def test_turbo_table_defaults_all_factors_to_1(self) -> None:
        for tier, cfg in ZIMAGE_TIER_CONFIGS.items():
            assert cfg.supersample_factor == 1, (
                f"Turbo tier {tier!r} must have supersample_factor=1 "
                f"(factor=1 preserves existing render behavior)"
            )

    def test_high_fidelity_table_defaults_all_factors_to_1(self) -> None:
        for tier, cfg in ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS.items():
            assert cfg.supersample_factor == 1, (
                f"HF tier {tier!r} must have supersample_factor=1 "
                f"(factor=1 preserves existing render behavior)"
            )


# ── Dimension math ───────────────────────────────────────────────────────────


class TestSupersampleDimensionMath:
    """The generator input dimensions must be target × factor."""

    def test_factor_2_doubles_both_axes(self) -> None:
        cfg = ZImageTierConfig(
            steps=8,
            guidance=0.0,
            width=1024,
            height=768,
            model_variant="z-image-turbo",
            supersample_factor=2,
        )
        assert cfg.width * cfg.supersample_factor == 2048
        assert cfg.height * cfg.supersample_factor == 1536

    def test_factor_1_leaves_axes_unchanged(self) -> None:
        cfg = ZImageTierConfig(
            steps=8,
            guidance=0.0,
            width=1024,
            height=768,
            model_variant="z-image-turbo",
            supersample_factor=1,
        )
        assert cfg.width * cfg.supersample_factor == 1024
        assert cfg.height * cfg.supersample_factor == 768

    def test_factor_3_triples_both_axes(self) -> None:
        cfg = ZImageTierConfig(
            steps=20,
            guidance=4.0,
            width=768,
            height=512,
            model_variant="z-image",
            supersample_factor=3,
        )
        assert cfg.width * cfg.supersample_factor == 2304
        assert cfg.height * cfg.supersample_factor == 1536


# ── supersample_downscale unit tests ────────────────────────────────────────


class TestSupersampleDownscale:
    """Tests for post_processor.supersample_downscale."""

    def test_2x_image_downscales_to_target_dims(self) -> None:
        """Feed a 2× PIL image; assert result is exactly target dims."""
        src = Image.new("RGB", (2048, 1536), color=(100, 150, 200))
        result = supersample_downscale(src, target_width=1024, target_height=768)
        assert result.size == (1024, 768)

    def test_downscale_uses_lanczos_resampling(self) -> None:
        """Result is a PIL Image (Lanczos produces no degradation markers
        detectable in a unit test, but the call must return a valid Image)."""
        src = Image.new("RGB", (2048, 2048), color=(50, 50, 50))
        result = supersample_downscale(src, target_width=1024, target_height=1024)
        assert isinstance(result, Image.Image)
        assert result.size == (1024, 1024)

    def test_factor_1_identity_returns_input_unchanged(self) -> None:
        """When source already matches target, the input image is returned as-is."""
        src = Image.new("RGB", (1024, 768), color=(10, 20, 30))
        result = supersample_downscale(src, target_width=1024, target_height=768)
        assert result is src, (
            "supersample_downscale must return the input unchanged when "
            "source and target dims are identical (identity path)"
        )

    def test_target_larger_than_source_raises(self) -> None:
        """Upscaling is not allowed — raises ValueError loudly."""
        src = Image.new("RGB", (512, 512), color=(0, 0, 0))
        with pytest.raises(ValueError, match="downscale"):
            supersample_downscale(src, target_width=1024, target_height=1024)

    def test_target_larger_in_one_axis_raises(self) -> None:
        src = Image.new("RGB", (1024, 512), color=(0, 0, 0))
        with pytest.raises(ValueError, match="downscale"):
            supersample_downscale(src, target_width=1024, target_height=768)


# ── Worker passes supersampled dims to generate_image ───────────────────────


class TestWorkerSupersampleDims:
    """The worker must pass inflated dims to generate_image when factor > 1,
    and the final saved image must have the target (non-inflated) dims."""

    def _worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> ZImageMLXWorker:
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        return ZImageMLXWorker(output_dir=tmp_path)

    def test_factor_1_passes_target_dims_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Factor=1 → no supersample inflation; generate_image receives
        the unmodified tier dims (1024×1024 for portrait HF)."""
        worker = self._worker(tmp_path, monkeypatch)
        mock_model = MagicMock()
        mock_model.generate_image.return_value = fake_pil_image(1024, 1024)
        worker.model = mock_model

        worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

        kwargs = mock_model.generate_image.call_args.kwargs
        assert kwargs["width"] == 1024
        assert kwargs["height"] == 1024

    def test_factor_2_tier_passes_doubled_dims_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a tier's supersample_factor=2, generate_image must receive
        width*2 × height*2 so the model renders at the supersampled resolution.
        We monkeypatch the tier config to inject factor=2."""
        from sidequest_daemon.media import zimage_config as zc

        # Build a modified HF PORTRAIT entry with factor=2.
        original_cfg = zc.ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS[RenderTier.PORTRAIT]
        patched_cfg = ZImageTierConfig(
            steps=original_cfg.steps,
            guidance=original_cfg.guidance,
            width=original_cfg.width,
            height=original_cfg.height,
            model_variant=original_cfg.model_variant,
            supersample_factor=2,
        )

        def patched_get_zimage_config(
            tier: RenderTier, fidelity: str = "high_fidelity"
        ):
            if tier == RenderTier.PORTRAIT and fidelity == "high_fidelity":
                return patched_cfg
            return zc.get_zimage_config(tier, fidelity)

        monkeypatch.setattr(
            "sidequest_daemon.media.workers.zimage_mlx_worker.get_zimage_config",
            patched_get_zimage_config,
        )

        worker = self._worker(tmp_path, monkeypatch)
        # generate_image returns a 2× image (2048×2048 for portrait 1024×1024 × 2)
        mock_model = MagicMock()
        mock_model.generate_image.return_value = fake_pil_image(2048, 2048)
        worker.model = mock_model

        result = worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

        kwargs = mock_model.generate_image.call_args.kwargs
        assert kwargs["width"] == 2048, (
            "factor=2 → generate_image must receive 2048 (1024 × 2)"
        )
        assert kwargs["height"] == 2048, (
            "factor=2 → generate_image must receive 2048 (1024 × 2)"
        )
        # The result dict must still report the TARGET dims (1024×1024)
        assert result["width"] == 1024, (
            "render() result width must be the target tier dim, not the supersampled dim"
        )
        assert result["height"] == 1024, (
            "render() result height must be the target tier dim, not the supersampled dim"
        )


# ── Factor=1 is a true no-op ─────────────────────────────────────────────────


class TestFactor1IsNoop:
    """Factor=1 must be a true no-op: generate_image receives the unmodified
    tier dims, supersample_downscale is never called, and the output file
    matches the expected tier dimensions."""

    def test_factor_1_no_downscale_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify supersample_downscale is not invoked when factor=1."""
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)

        downscale_calls: list = []

        def spy_downscale(img, *, target_width, target_height):
            downscale_calls.append((target_width, target_height))
            return img

        monkeypatch.setattr(
            "sidequest_daemon.media.workers.zimage_mlx_worker.supersample_downscale",
            spy_downscale,
        )

        mock_model = MagicMock()
        mock_model.generate_image.return_value = fake_pil_image(1024, 1024)
        worker.model = mock_model

        worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

        assert downscale_calls == [], (
            "supersample_downscale must NOT be called when factor=1"
        )


# ── Invalid factor raises loudly ─────────────────────────────────────────────


class TestInvalidSupersampleFactorRaisesLoud:
    """A configured supersample_factor ≤ 0 or non-int must fail loud.

    Per CLAUDE.md 'No Silent Fallbacks': the daemon must never silently
    ignore a bad config and fall back to factor=1.
    """

    def _patched_worker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        bad_factor,
    ) -> ZImageMLXWorker:
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        from sidequest_daemon.media import zimage_config as zc

        original_cfg = zc.ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS[RenderTier.PORTRAIT]

        def patched_get_zimage_config(
            tier: RenderTier, fidelity: str = "high_fidelity"
        ):
            if tier == RenderTier.PORTRAIT and fidelity == "high_fidelity":
                # Bypass frozen dataclass to inject bad factor via object.__setattr__
                # is blocked (frozen); we return a fresh config via a monkey-patched
                # object that looks like a ZImageTierConfig but has the bad value.
                class _BadCfg:
                    steps = original_cfg.steps
                    guidance = original_cfg.guidance
                    width = original_cfg.width
                    height = original_cfg.height
                    model_variant = original_cfg.model_variant
                    supersample_factor = bad_factor

                return _BadCfg()
            return zc.get_zimage_config(tier, fidelity)

        monkeypatch.setattr(
            "sidequest_daemon.media.workers.zimage_mlx_worker.get_zimage_config",
            patched_get_zimage_config,
        )
        worker = ZImageMLXWorker(output_dir=tmp_path)
        mock_model = MagicMock()
        mock_model.generate_image.return_value = fake_pil_image(1024, 1024)
        worker.model = mock_model
        return worker

    @pytest.mark.parametrize("bad_factor", [0, -1, -5])
    def test_zero_or_negative_factor_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_factor: int
    ) -> None:
        worker = self._patched_worker(tmp_path, monkeypatch, bad_factor)
        with pytest.raises(ValueError, match="supersample_factor"):
            worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

    def test_non_int_factor_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = self._patched_worker(tmp_path, monkeypatch, 1.5)
        with pytest.raises(ValueError, match="supersample_factor"):
            worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

    def test_bool_factor_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """True == 1 in Python but bool is not int for this contract."""
        worker = self._patched_worker(tmp_path, monkeypatch, True)
        with pytest.raises(ValueError, match="supersample_factor"):
            worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})


# ── OTEL span attributes ─────────────────────────────────────────────────────


class TestOtelSupersampleAttributes:
    """The render span must carry supersample_factor and supersample_applied."""

    def test_factor_1_otel_attributes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        mock_model = MagicMock()
        mock_model.generate_image.return_value = fake_pil_image(1024, 1024)
        worker.model = mock_model

        worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert render_spans
        attrs = render_spans[0].attributes
        assert attrs["render.supersample_factor"] == 1
        assert attrs["render.supersample_applied"] is False

    def test_factor_2_otel_attributes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        from sidequest_daemon.media import zimage_config as zc

        original_cfg = zc.ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS[RenderTier.PORTRAIT]
        patched_cfg = ZImageTierConfig(
            steps=original_cfg.steps,
            guidance=original_cfg.guidance,
            width=original_cfg.width,
            height=original_cfg.height,
            model_variant=original_cfg.model_variant,
            supersample_factor=2,
        )

        def patched_get_zimage_config(
            tier: RenderTier, fidelity: str = "high_fidelity"
        ):
            if tier == RenderTier.PORTRAIT and fidelity == "high_fidelity":
                return patched_cfg
            return zc.get_zimage_config(tier, fidelity)

        monkeypatch.setattr(
            "sidequest_daemon.media.workers.zimage_mlx_worker.get_zimage_config",
            patched_get_zimage_config,
        )

        worker = ZImageMLXWorker(output_dir=tmp_path)
        mock_model = MagicMock()
        mock_model.generate_image.return_value = fake_pil_image(2048, 2048)
        worker.model = mock_model

        worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert render_spans
        attrs = render_spans[0].attributes
        assert attrs["render.supersample_factor"] == 2
        assert attrs["render.supersample_applied"] is True
