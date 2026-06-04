"""Pillow-based post-processing for camera preset directives."""

from __future__ import annotations

import math

from PIL import Image

from sidequest_daemon.media.camera_specs import PostDirective


def apply_post(img: Image.Image, directive: PostDirective | None) -> Image.Image:
    if directive is None:
        return img
    if directive.kind == "crop":
        return _center_crop(img, directive.percent or 1.0)
    if directive.kind == "rotate":
        return _rotate_inscribed(img, directive.degrees or 0.0)
    raise ValueError(f"unknown post kind: {directive.kind!r}")


def _center_crop(img: Image.Image, percent: float) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(w * percent))
    new_h = max(1, int(h * percent))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def _rotate_inscribed(img: Image.Image, degrees: float) -> Image.Image:
    rotated = img.rotate(degrees, resample=Image.BICUBIC, expand=False)
    w, h = img.size
    theta = math.radians(abs(degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    new_w = int((w * cos_t - h * sin_t) if w >= h else (h * cos_t - w * sin_t))
    new_h = int(new_w * (h / w)) if w >= h else int(new_w * (w / h))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return rotated.crop((left, top, left + new_w, top + new_h))


def supersample_downscale(
    img: Image.Image,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """Downscale ``img`` to ``(target_width, target_height)`` with Lanczos.

    Called after generation when a tier's ``supersample_factor > 1`` has
    caused the generator to render at a higher internal resolution.  Lanczos
    anti-aliases the high-frequency line patterns (engraving, cross-hatch,
    halftone) before they reach the final raster — dissolving moiré.

    Factor=1 callers must NOT invoke this function (the worker skips it as a
    true no-op); the guard here is a defensive belt-and-suspenders check.

    Raises ``ValueError`` if the target dimensions are not smaller than the
    source (enforces the correct call contract — this function is a
    *downscale*, not a resize).
    """
    src_w, src_h = img.size
    if target_width > src_w or target_height > src_h:
        raise ValueError(
            f"supersample_downscale: target ({target_width}×{target_height}) "
            f"is larger than source ({src_w}×{src_h}); this is a downscale "
            f"operation only.  Check that supersample_factor > 1."
        )
    if target_width == src_w and target_height == src_h:
        return img
    return img.resize((target_width, target_height), Image.LANCZOS)


def required_render_size(
    target_size: tuple[int, int],
    directive: PostDirective | None,
) -> tuple[int, int]:
    if directive is None:
        return target_size
    tw, th = target_size
    if directive.kind == "crop":
        percent = directive.percent or 1.0
        if percent <= 0:
            raise ValueError("crop percent must be > 0")
        return (int(tw / percent), int(th / percent))
    if directive.kind == "rotate":
        theta = math.radians(abs(directive.degrees or 0.0))
        denom = max(1e-6, math.cos(theta) - math.sin(theta))
        return (int(tw / denom) + 1, int(th / denom) + 1)
    return target_size
