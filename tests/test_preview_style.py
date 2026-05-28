"""Tests for the `style` subcommand of sidequest-promptpreview.

The `style` subcommand exists as a diagnostic that exercises ONLY the
StyleCatalog — no character/place/recipe loading. This must work against
worlds that lack a portrait_manifest.yaml, because that is exactly the
case where someone reaches for the diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sidequest_daemon.media.preview import main

FIXTURE_ROOT = (
    Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"
)


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_style_text_output_against_fixture(capsys, monkeypatch) -> None:
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    code, out, _ = _run(
        ["style", "--genre", "testgenre", "--world", "testworld"], capsys,
    )
    assert code == 0
    assert "ART_SENSIBILITY.GENRE" in out
    assert "ART_SENSIBILITY.WORLD" in out
    assert "ART_SENSIBILITY.CULTURE" in out
    # Genre tokens from fixture's visual_style.yaml.
    assert "painterly digital illustration" in out
    # World tokens from worlds/testworld/visual_style.yaml.
    assert "bruised amber sky" in out
    # Culture tokens from cultures/ironhand.yaml.
    assert "iron-chased buttons" in out
    # Would-apply summary should concatenate all three.
    assert "Would apply" in out
    assert "monastic severity" in out


def test_style_json_output_against_fixture(capsys, monkeypatch) -> None:
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    code, out, _ = _run(
        ["style", "--genre", "testgenre", "--world", "testworld", "--json"],
        capsys,
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["genre"]["slug"] == "testgenre"
    assert payload["world"]["slug"] == "testworld"
    assert "painterly digital illustration" in payload["genre"]["tokens"]
    assert "bruised amber sky" in payload["world"]["tokens"]
    cultures = {c["slug"]: c for c in payload["cultures"]}
    assert "ironhand" in cultures
    assert "iron-chased buttons" in cultures["ironhand"]["tokens"]
    assert "monastic severity" in payload["would_apply"]


def test_style_world_with_no_style_file_raises_loud(
    tmp_path, capsys, monkeypatch,
) -> None:
    """A world with no visual_style.yaml is unrenderable — the preview
    must surface a StyleMissError, not soft-degrade into a styleless
    stub display."""
    from sidequest_daemon.media.recipes import StyleMissError

    packs = tmp_path / "packs"
    (packs / "g1" / "worlds" / "w1").mkdir(parents=True)
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(packs))

    with pytest.raises(StyleMissError):
        _run(["style", "--genre", "g1", "--world", "w1"], capsys)


def test_style_world_without_portrait_manifest_does_not_crash(
    tmp_path, capsys, monkeypatch,
) -> None:
    """Regression: previously, any preview command for a world without
    portrait_manifest.yaml would crash with FileNotFoundError before the
    style layers could be inspected. The `style` subcommand must NOT
    touch CharacterCatalog / PlaceCatalog at all.
    """
    packs = tmp_path / "packs"
    world_dir = packs / "g1" / "worlds" / "w1"
    world_dir.mkdir(parents=True)
    (packs / "g1" / "visual_style.yaml").write_text(
        'positive_suffix: "diagnostic-genre-style-token"\n',
    )
    (world_dir / "visual_style.yaml").write_text(
        'positive_suffix: "diagnostic-world-style-token"\n',
    )
    # Deliberately NO portrait_manifest.yaml here.
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(packs))

    code, out, _ = _run(
        ["style", "--genre", "g1", "--world", "w1"], capsys,
    )
    assert code == 0
    assert "diagnostic-genre-style-token" in out
    assert "diagnostic-world-style-token" in out


# NOTE: a former wiring test here drove the `style` subcommand against the real
# sidequest-content tree (caverns_and_claudes/mawdeep). It was removed: it
# asserted on live content (signature C&C style tokens) — content-validation
# work that now belongs to validators, not the daemon unit suite — and it
# pointed at the `mawdeep` world, which no longer exists. The entry-point wiring
# is already covered by test_style_text_output_against_fixture, which calls
# `main(...)` against the fixture pack.
