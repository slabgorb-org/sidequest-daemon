from pathlib import Path

import pytest

from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    CharacterTokens,
    PlaceCatalog,
    PlaceTokens,
    StyleCatalog,
)
from sidequest_daemon.media.recipes import (
    LOD,
    CatalogMissError,
    PlaceLOD,
    StyleMissError,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"


def test_loads_world_characters():
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    tokens = cat.get("npc:rux")
    assert isinstance(tokens, CharacterTokens)
    assert tokens.kind == "npc"
    assert tokens.culture == "ironhand"
    assert "inquisitor" in tokens.descriptions[LOD.SOLO]
    assert tokens.default_pose.startswith("standing")


def test_loads_production_schema_characters():
    """Production manifests use flat `name` + `appearance` (no `id`, no LOD dict)."""
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="prodgenre", world="prodworld")

    # Slug derived from name by lowercasing, collapsing whitespace to `_`,
    # and dropping punctuation. "Imperatrix Celestine VII" → "imperatrix_celestine_vii".
    tokens = cat.get("npc:imperatrix_celestine_vii")
    assert isinstance(tokens, CharacterTokens)
    assert tokens.kind == "npc"

    # All four LODs populated — production has no LOD distinctions, so the
    # appearance prose populates every level. Downstream eviction/truncation
    # handles token budgeting per slot.
    for lod in LOD:
        assert tokens.descriptions[lod], f"missing {lod}"
        assert "deep brown skin" in tokens.descriptions[lod]

    # Production schema has no `culture` field; it must resolve to None
    # rather than crashing.
    assert tokens.culture is None
    # Production schema has no `default_pose`; defaults to empty string.
    assert tokens.default_pose == ""

    # Slugifier preserves hyphens and replaces only whitespace.
    two_word = cat.get("npc:two-word_name")
    assert "multi-token names" in two_word.descriptions[LOD.SOLO]


def test_all_four_lods_present():
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    t = cat.get("npc:rux")
    for lod in LOD:
        assert t.descriptions[lod], f"missing {lod}"


def test_missing_character_raises():
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    with pytest.raises(CatalogMissError) as exc:
        cat.get("npc:ghost")
    assert exc.value.source == "CharacterCatalog"
    assert exc.value.missing_id == "npc:ghost"


def test_rejects_non_npc_pc_key():
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    with pytest.raises(ValueError, match="scheme"):
        cat.get("rux")


def test_loads_specific_place_from_history():
    cat = PlaceCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    t = cat.get("where:testworld/the_lookout")
    assert isinstance(t, PlaceTokens)
    assert t.kind == "specific"
    assert t.controlling_culture == "ironhand"
    assert "watchtower" in t.landmark[PlaceLOD.SOLO]
    assert "upland" in t.environment[PlaceLOD.SOLO]


def test_loads_archetypal_place_from_genre():
    cat = PlaceCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    t = cat.get("where:testgenre/tavern")
    assert t.kind == "archetypal"
    assert t.landmark[PlaceLOD.SOLO] == ""
    assert "hearth" in t.environment[PlaceLOD.SOLO]


def test_archetypal_with_populated_landmark_rejected():
    bad_genre = FIXTURE_ROOT / "badgenre"
    bad_genre.mkdir(parents=True, exist_ok=True)
    (bad_genre / "places.yaml").write_text(
        "tavern:\n"
        "  landmark: {solo: 'a specific tavern named The Black Lion', backdrop: ''}\n"
        "  environment: {solo: '...', backdrop: '...'}\n"
        "  description: {solo: 'tavern', backdrop: 'tavern'}\n",
    )
    try:
        with pytest.raises(ValueError, match="landmark"):
            PlaceCatalog.load(FIXTURE_ROOT, genre="badgenre", world="testworld")
    finally:
        (bad_genre / "places.yaml").unlink()
        bad_genre.rmdir()


def test_missing_place_raises():
    cat = PlaceCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    with pytest.raises(CatalogMissError):
        cat.get("where:testworld/no_such_poi")


def test_poi_kind_guard_rejects_archetypal_key():
    cat = PlaceCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    # Archetypal ref is legal in catalog; the caller (composer) enforces
    # poi-recipe-specific constraints. This test documents the catalog accepts both.
    specific = cat.get("where:testworld/the_lookout")
    archetypal = cat.get("where:testgenre/tavern")
    assert specific.kind == "specific"
    assert archetypal.kind == "archetypal"


def test_loads_genre_style():
    cat = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    tokens = cat.get_genre("testgenre")
    assert "painterly" in tokens


def test_loads_world_style():
    cat = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    tokens = cat.get_world("testgenre", "testworld")
    assert "amber" in tokens


def _write_style_tree(
    root: Path,
    *,
    genre: str,
    world: str,
    genre_suffix: str | None,
    world_suffix: str | None,
) -> None:
    """Build a minimal genre_packs tree. ``None`` suffix => omit that file."""
    gdir = root / genre
    wdir = gdir / "worlds" / world
    wdir.mkdir(parents=True, exist_ok=True)
    if genre_suffix is not None:
        (gdir / "visual_style.yaml").write_text(f"positive_suffix: {genre_suffix}\n")
    if world_suffix is not None:
        (wdir / "visual_style.yaml").write_text(f"positive_suffix: {world_suffix}\n")


def test_genre_style_optional_when_world_present(tmp_path: Path) -> None:
    """2026-05-29 directive: pack-level (genre) visual_style is OPTIONAL.

    A world ships its own style; the genre file is absent. Load succeeds and
    the world style is intact. ``get_genre`` raises (genre layer absent) — the
    composer never calls it when a world style is present, so this is the
    correct fail-loud-only-if-actually-needed behavior.
    """
    _write_style_tree(
        tmp_path, genre="g", world="w", genre_suffix=None, world_suffix="amber dusk"
    )
    cat = StyleCatalog.load(tmp_path, genre="g", world="w")
    assert "amber" in cat.get_world("g", "w")
    with pytest.raises(CatalogMissError):
        cat.get_genre("g")


def test_world_style_still_required(tmp_path: Path) -> None:
    """No silent fallback: with neither genre nor world style, load fails loud
    (the world remains the required backstop)."""
    _write_style_tree(
        tmp_path, genre="g", world="w", genre_suffix=None, world_suffix=None
    )
    with pytest.raises(StyleMissError):
        StyleCatalog.load(tmp_path, genre="g", world="w")


def test_present_but_empty_genre_style_fails_loud(tmp_path: Path) -> None:
    """A genre file that EXISTS but has an empty positive_suffix is a broken
    config, not by-design absence — it still fails loud."""
    _write_style_tree(
        tmp_path, genre="g", world="w", genre_suffix="", world_suffix="amber dusk"
    )
    with pytest.raises(StyleMissError):
        StyleCatalog.load(tmp_path, genre="g", world="w")




def test_culture_tokens_loaded():
    cat = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    tokens = cat.get_culture("testgenre", "testworld", "ironhand")
    assert "iron-chased" in tokens


def test_unknown_culture_raises():
    cat = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    with pytest.raises(CatalogMissError):
        cat.get_culture("testgenre", "testworld", "nonexistent")


def test_poi_without_slug_derives_slug_from_name(tmp_path) -> None:
    """Production worlds (e.g. ``tea_and_murder/blackthorn_moor``) author POIs with
    ``name``/``description`` and no ``slug``. Pre-fix the loader did
    ``slug = poi["slug"]`` and crashed with ``KeyError('slug')`` on every
    render, polluting the GM panel with ``compose.skipped reason=KeyError
    error='slug'`` and stripping the entire world from the compose path.

    Mirror the parallel ``CharacterCatalog.load`` rule (commit d787c0a):
    accept both the synthetic ``slug``-keyed schema and the production
    ``name``-only schema, deriving the slug via ``_slugify_name`` when
    absent. Fail loud only when neither is honored."""
    pack = tmp_path / "drifty"
    (pack / "worlds" / "blackthorn_moor").mkdir(parents=True)
    (pack / "visual_style.yaml").write_text("positive_suffix: 'painterly'\n")
    (pack / "worlds" / "blackthorn_moor" / "history.yaml").write_text(
        "chapters:\n"
        "  - name: The Present Age\n"
        "    points_of_interest:\n"
        "      - name: The Morning Room\n"
        "        description: 'Oak panels, leaded windows.'\n"
        "        visual_prompt:\n"
        "          solo: 'an Edwardian morning room, leaded windows'\n"
        "          backdrop: 'an Edwardian morning room'\n"
        "        environment:\n"
        "          solo: 'rain on glass, gas-lamp glow, hush'\n"
        "          backdrop: 'gas-lamp glow'\n"
    )
    cat = PlaceCatalog.load(tmp_path, genre="drifty", world="blackthorn_moor")
    # Slug is derived: "The Morning Room" → "the_morning_room".
    t = cat.get("where:blackthorn_moor/the_morning_room")
    assert t.kind == "specific"
    assert "leaded windows" in t.landmark[PlaceLOD.SOLO]


def test_poi_without_visual_blocks_is_skipped(tmp_path, caplog) -> None:
    """When a POI has neither ``visual_prompt`` nor ``environment`` populated
    (the blackthorn_moor authoring shape), do NOT add it to the catalog with
    empty-prose tokens — that would silently degrade compose for the POI
    (genre style + camera + safety clause only), which is *worse* than the
    prose-subject fallback the safe wrapper produces on a catalog miss.

    Skip with a loud INFO line so the GM panel and content authors can
    see which POIs are unauthored. Compose then catalog-misses → safe
    wrapper logs ``compose.skipped`` → daemon falls through to the rich
    prose subject the narrator already produced."""
    import logging

    pack = tmp_path / "thingenre"
    (pack / "worlds" / "blackthorn_moor").mkdir(parents=True)
    (pack / "worlds" / "blackthorn_moor" / "history.yaml").write_text(
        "chapters:\n"
        "  - name: The Present Age\n"
        "    points_of_interest:\n"
        "      - name: The Study\n"
        "        description: 'Oak-panelled, book-lined.'\n"
    )
    with caplog.at_level(logging.INFO, logger="sidequest_daemon.media.catalogs"):
        cat = PlaceCatalog.load(tmp_path, genre="thingenre", world="blackthorn_moor")
    with pytest.raises(CatalogMissError):
        cat.get("where:blackthorn_moor/the_study")
    assert any(
        "place_catalog.poi_skipped" in record.message
        and "reason=no_visual" in record.message
        and "the_study" in record.message
        for record in caplog.records
    ), f"expected place_catalog.poi_skipped log, got: {[r.message for r in caplog.records]}"


def test_poi_without_slug_or_name_fails_loud(tmp_path) -> None:
    """A POI entry with neither ``slug`` nor ``name`` is unauthored — fail
    loud at load time so content authors learn about it before runtime."""
    pack = tmp_path / "fail"
    (pack / "worlds" / "world").mkdir(parents=True)
    (pack / "worlds" / "world" / "history.yaml").write_text(
        "chapters:\n"
        "  - name: T\n"
        "    points_of_interest:\n"
        "      - description: 'no name, no slug, just vibes'\n"
    )
    with pytest.raises(ValueError, match="slug.*name"):
        PlaceCatalog.load(tmp_path, genre="fail", world="world")


def test_missing_portrait_manifest_yields_empty_catalog(tmp_path) -> None:
    """A world with NO ``portrait_manifest.yaml`` is the contract-correct
    state for worlds whose ``world_register.yaml humanoid_constraint``
    forbids townsfolk-tone named portraits (e.g. ``beneath_sunden``).

    Pre-fix, ``CharacterCatalog.load`` did ``path.read_text()`` with no
    existence guard — a missing manifest raised ``FileNotFoundError``,
    which propagated out of the catalog-composed render path and *closed
    the daemon render socket* (observed live: every POI render in
    ``beneath_sunden`` died until an empty ``characters: []`` file was
    hand-placed as a workaround).

    The sibling ``PlaceCatalog._load_specific`` already treats its
    world artifact (``history.yaml``) as optional with
    ``if not path.exists(): return``. ``portrait_manifest.yaml`` is
    equally optional per the server loader contract
    (``loader.py`` ``_load_portrait_manifest`` is ``.exists()``-guarded).
    Conform to the family standard: a missing manifest is an empty
    catalog, not a socket-killing exception."""
    pack = tmp_path / "graven"
    (pack / "worlds" / "the_deep").mkdir(parents=True)
    # Deliberately author NO portrait_manifest.yaml.
    cat = CharacterCatalog.load(pack.parent, genre="graven", world="the_deep")
    with pytest.raises(CatalogMissError) as exc:
        cat.get("npc:anyone")
    assert exc.value.source == "CharacterCatalog"


def test_empty_portrait_manifest_yields_empty_catalog(tmp_path) -> None:
    """An empty/whitespace ``portrait_manifest.yaml`` parses to ``None``.
    Pre-fix ``data.get("characters", [])`` on ``None`` raised
    ``AttributeError`` — the same socket-killing failure mode as a
    missing file. ``PlaceCatalog`` already hardens this with
    ``yaml.safe_load(...) or {}``; ``CharacterCatalog`` must match."""
    pack = tmp_path / "graven"
    (pack / "worlds" / "the_deep").mkdir(parents=True)
    (pack / "worlds" / "the_deep" / "portrait_manifest.yaml").write_text("\n")
    cat = CharacterCatalog.load(pack.parent, genre="graven", world="the_deep")
    with pytest.raises(CatalogMissError):
        cat.get("npc:anyone")


