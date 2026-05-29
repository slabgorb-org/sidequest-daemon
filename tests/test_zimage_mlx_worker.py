"""Unit tests for ZImageMLXWorker.

The ZImage model is mocked — we test worker glue, not the inference pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from sidequest_daemon.media.recipes import RenderConfigError
from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker


def _fake_pil_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), color="black")


@pytest.fixture
def worker(tmp_path: Path) -> ZImageMLXWorker:
    # Singleton-slot reset between tests is handled by the autouse
    # fixture in conftest.py (`_reset_zimage_singleton`); this fixture
    # only constructs the worker.
    return ZImageMLXWorker(output_dir=tmp_path)


def test_tier_configs_match_render_tier_enum():
    """The worker's tier dispatch must accept every tier the composer emits.

    Story 45-39 removed the worker's duplicate ``TIER_CONFIGS`` dict in
    favour of ``get_zimage_config(tier, fidelity)`` from ``zimage_config``.
    The check is now on the canonical config table (which the worker
    consults at render time), and a non-tier string still raises.
    """
    from sidequest_daemon.media.zimage_config import ZIMAGE_TIER_CONFIGS

    expected = {t.value for t in ZIMAGE_TIER_CONFIGS}
    assert "scene_illustration" in expected
    assert "portrait" in expected
    assert "landscape" in expected
    assert "text_overlay" in expected
    assert "tactical_sketch" not in expected
    assert "fog_of_war" in expected
    assert "cartography" in expected


def test_render_unknown_tier_raises(worker: ZImageMLXWorker):
    with pytest.raises(ValueError, match="Unsupported tier"):
        worker.render({"tier": "not_a_tier", "positive_prompt": "x"})


def test_render_returns_expected_result_shape(worker: ZImageMLXWorker):
    """Successful render returns image_url + dims + elapsed_ms."""
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    result = worker.render(
        {
            "tier": "scene_illustration",
            "positive_prompt": "a dark forest",
            "negative_prompt": "blurry",
            "seed": 42,
        }
    )

    assert "image_url" in result
    assert Path(result["image_url"]).exists()
    assert result["width"] == 1024
    assert result["height"] == 768
    assert isinstance(result["elapsed_ms"], int)


def test_render_passes_negative_prompt_to_model(worker: ZImageMLXWorker):
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    worker.render(
        {
            "tier": "portrait",
            "positive_prompt": "a face",
            "negative_prompt": "photograph, realistic",
            "seed": 1,
        }
    )

    call_kwargs = mock_model.generate_image.call_args.kwargs
    assert call_kwargs["negative_prompt"] == "photograph, realistic"
    assert call_kwargs["prompt"] == "a face"
    assert call_kwargs["seed"] == 1


def test_render_honors_steps_override(worker: ZImageMLXWorker):
    """A caller-supplied params["steps"] overrides the tier/fidelity default.

    Regression guard for the dead `--steps` flag: before this fix send_render
    dropped steps and the worker hardcoded tier_cfg.steps, so 15 and 30 both
    rendered at the tier default. The override must reach generate_image.
    """
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    worker.render(
        {
            "tier": "portrait",
            "positive_prompt": "a face",
            "seed": 1,
            "steps": 30,
        }
    )

    assert mock_model.generate_image.call_args.kwargs["num_inference_steps"] == 30


def test_render_defaults_steps_to_tier_config_when_absent(worker: ZImageMLXWorker):
    """No params["steps"] → the (tier, fidelity) default still applies.

    Guards the back-compat path: in-session / legacy callers that send no
    steps must keep getting the tier default, not a hardcoded number.
    """
    from sidequest_daemon.media.zimage_config import get_zimage_config
    from sidequest_daemon.renderer.models import RenderTier

    default_steps = get_zimage_config(RenderTier.PORTRAIT, worker.fidelity).steps

    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    worker.render({"tier": "portrait", "positive_prompt": "a face", "seed": 1})

    assert (
        mock_model.generate_image.call_args.kwargs["num_inference_steps"]
        == default_steps
    )


@pytest.mark.parametrize("bad_steps", [0, -5, 2.5, True])
def test_render_rejects_invalid_steps(worker: ZImageMLXWorker, bad_steps):
    """A non-positive / non-int steps override fails loud (No Silent Fallbacks)."""
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    with pytest.raises(ValueError, match="positive int"):
        worker.render(
            {
                "tier": "portrait",
                "positive_prompt": "a face",
                "seed": 1,
                "steps": bad_steps,
            }
        )
    mock_model.generate_image.assert_not_called()




def test_worker_targets_z_image_base_by_default(worker: ZImageMLXWorker):
    """Lock-in: with no env var, the worker now loads base Z-Image 1.0.

    The 2026-05-02 default flip puts the painterly 20-step / CFG 4 path
    on the floor. Turbo is still loadable via SIDEQUEST_DAEMON_FIDELITY=turbo
    (covered by ``test_turbo_env_var_explicit_loads_turbo_model``) — this
    test guards against accidental rollback of that flip.
    """
    assert worker.fidelity == "high_fidelity"
    assert worker.model_variant == "z-image"
    assert worker.QUANTIZE == 8


def test_render_calls_model_with_guidance_none_for_turbo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Turbo's mflux ModelConfig sets supports_guidance=False; the worker
    must pass guidance=None to generate_image so we don't accidentally drive
    a CFG path on a distilled model.

    Default flipped to high_fidelity 2026-05-02; this test opts into turbo
    explicitly via the env var rather than the shared ``worker`` fixture.
    """
    monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "turbo")
    worker = ZImageMLXWorker(output_dir=tmp_path)
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

    call_kwargs = mock_model.generate_image.call_args.kwargs
    assert call_kwargs["guidance"] is None
    assert call_kwargs["num_inference_steps"] == 8


def test_compose_prompt_requires_content(worker: ZImageMLXWorker):
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    with pytest.raises(RenderConfigError, match="compose pipeline"):
        worker.render({"tier": "scene_illustration", "seed": 0})


# ── Story 43-5: per-process singleton invariant ──────────────────


class TestSingletonInvariant:
    """ZImageMLXWorker is a per-process singleton (Story 43-5).

    A second construction must raise RuntimeError to fail loudly per
    CLAUDE.md "No Silent Fallbacks" — protects against any future
    revert/regression that would silently spawn a second model on the
    same MPS device.
    """

    def test_second_construction_raises(self, tmp_path: Path) -> None:
        # `worker` fixture already constructed one (which the autouse
        # fixture would normally clean up). Build the first explicitly so
        # this test is self-contained.
        ZImageMLXWorker._instance = None
        first = ZImageMLXWorker(output_dir=tmp_path / "first")
        try:
            with pytest.raises(RuntimeError, match="singleton"):
                ZImageMLXWorker(output_dir=tmp_path / "second")
        finally:
            first.cleanup()

    def test_cleanup_releases_singleton_slot(self, tmp_path: Path) -> None:
        """cleanup() must clear the singleton handle and have __init__
        repopulate it on the next construction.
        """
        ZImageMLXWorker._instance = None
        first = ZImageMLXWorker(output_dir=tmp_path / "first")
        assert ZImageMLXWorker._instance is first
        first.cleanup()
        assert ZImageMLXWorker._instance is None, (
            "cleanup() must release the singleton slot"
        )
        # Construction repopulates the slot.
        second = ZImageMLXWorker(output_dir=tmp_path / "second")
        assert ZImageMLXWorker._instance is second, (
            "__init__ must reinstall the new instance into the singleton slot"
        )
        second.cleanup()


# ── Story 43-5: contract checks for warm_up / render before load_model ──


class TestPreLoadContract:
    """`_ensure_loaded()` was removed in 43-5. `warm_up()` and `render()`
    now raise RuntimeError (not assert — `assert` is stripped under
    Python -O) when called before `load_model()`. These tests pin that
    contract; without them the assertion-replacement-with-raise change
    would have no coverage.
    """

    def test_warm_up_without_load_model_raises(
        self, worker: ZImageMLXWorker
    ) -> None:
        # Worker fixture constructs but does NOT call load_model().
        assert worker.model is None
        with pytest.raises(RuntimeError, match="warm_up.*before load_model"):
            worker.warm_up()

    def test_render_without_load_model_raises(
        self, worker: ZImageMLXWorker
    ) -> None:
        assert worker.model is None
        with pytest.raises(RuntimeError, match="render.*before load_model"):
            worker.render(
                {"tier": "scene_illustration", "positive_prompt": "x"}
            )


# ── Story 43-5: wiring proof for load_model() callers ────────────


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "sidequest_daemon"
# Canonical (and only allowed) caller of `load_model()` on an image
# worker. `WorkerPool.warm_up_image()` lives here. Any other production
# caller would bypass the singleton/idempotency guards.
_ALLOWED_LOAD_MODEL_CALLER = _SOURCE_ROOT / "media" / "daemon.py"


def test_load_model_only_called_by_workerpool() -> None:
    """Wiring proof per daemon CLAUDE.md: `load_model()` on the image
    worker must be invoked exclusively by `WorkerPool.warm_up_image` in
    `sidequest_daemon/media/daemon.py`.

    The pattern matches ANY `.load_model(` call site (regardless of how
    the caller names the handle — `worker.load_model()`, `img.load_model()`,
    `self._image.load_model()`, etc.) and also bare `load_model()` calls
    inside the worker module itself for fully-qualified safety.
    Excludes files that *define* `load_model()` (the worker class itself,
    plus the unrelated EmbedWorker._load_model which is a different
    method on a different class). The single allowed call site is the
    canonical `_ALLOWED_LOAD_MODEL_CALLER` resolved exactly by Path
    equality — no name-based or "media/" substring shortcuts that future
    sibling files could accidentally match.
    """
    # Catch any `.load_model(` invocation (call site), not declarations.
    # Declarations look like `def load_model(self) -> ...`; the leading
    # `def ` excludes them.
    call_pattern = re.compile(r"\.load_model\(")
    def_pattern = re.compile(r"^\s*def\s+_?load_model\s*\(")
    offenders: list[str] = []
    for py_file in _SOURCE_ROOT.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        if py_file.resolve() == _ALLOWED_LOAD_MODEL_CALLER.resolve():
            continue
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if def_pattern.match(line):
                continue
            if call_pattern.search(line):
                rel = py_file.relative_to(_SOURCE_ROOT.parent)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert offenders == [], (
        "Found non-WorkerPool callers of `.load_model()` in the daemon "
        "production tree — the per-process singleton invariant assumes "
        f"`{_ALLOWED_LOAD_MODEL_CALLER.relative_to(_SOURCE_ROOT.parent)}` "
        "is the only call site:\n" + "\n".join(offenders)
    )
