"""Catalogs — Character, Place, Style. Load at startup. Fail-loud on miss."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from sidequest_daemon.media.recipes import (
    LOD,
    CatalogMissError,
    PlaceLOD,
    StyleMissError,
)

log = logging.getLogger(__name__)


def _slugify_name(name: str) -> str:
    """Lowercase, collapse whitespace to `_`, drop punctuation except `_`/`-`."""
    lowered = name.strip().lower()
    collapsed = re.sub(r"\s+", "_", lowered)
    return re.sub(r"[^a-z0-9_-]", "", collapsed)


class CharacterTokens(BaseModel):
    kind: Literal["npc", "pc"]
    descriptions: dict[LOD, str]
    default_pose: str
    culture: str | None
    world: str


class CharacterCatalog:
    """World-scoped — every character belongs to exactly one world."""

    def __init__(self, entries: dict[str, CharacterTokens]) -> None:
        self._entries = entries

    @classmethod
    def load(
        cls,
        genre_packs_root: Path,
        *,
        genre: str,
        world: str,
    ) -> "CharacterCatalog":
        path = genre_packs_root / genre / "worlds" / world / "portrait_manifest.yaml"
        # Optional artifact, mirroring PlaceCatalog._load_specific: a world
        # whose world_register humanoid_constraint forbids townsfolk-tone
        # named portraits legitimately ships no manifest. A missing or empty
        # file is an empty catalog, never a render-socket-killing exception.
        if not path.exists():
            return cls({})
        data = yaml.safe_load(path.read_text()) or {}
        entries: dict[str, CharacterTokens] = {}
        for raw in data.get("characters", []):
            # Production manifests use flat `name` + `appearance`; the
            # synthetic test schema uses `id` + LOD-keyed `descriptions`.
            # Accept both and fail loud only when neither shape is honored.
            slug = raw.get("id")
            if not slug:
                name = raw.get("name")
                if not name:
                    raise ValueError(
                        f"character entry in {path}: missing both `id` and `name`",
                    )
                slug = _slugify_name(name)

            descriptions_raw = raw.get("descriptions")
            if descriptions_raw:
                descriptions = {LOD(k): v for k, v in descriptions_raw.items()}
                if set(descriptions) != set(LOD):
                    missing = set(LOD) - set(descriptions)
                    raise ValueError(
                        f"character {slug!r} missing LODs: "
                        f"{sorted(m.value for m in missing)}",
                    )
            else:
                appearance = raw.get("appearance")
                if not appearance:
                    raise ValueError(
                        f"character {slug!r}: provide either `descriptions` "
                        f"(LOD-keyed) or `appearance` prose",
                    )
                # Production schema has no LOD variants. Replicate the prose
                # to every LOD; downstream slot eviction handles truncation.
                descriptions = dict.fromkeys(LOD, appearance)

            entries[f"npc:{slug}"] = CharacterTokens(
                kind="npc",
                descriptions=descriptions,
                default_pose=raw.get("default_pose", ""),
                culture=raw.get("culture"),
                world=world,
            )
        return cls(entries)

    def get(self, ref: str) -> CharacterTokens:
        if not (ref.startswith("npc:") or ref.startswith("pc:")):
            raise ValueError(
                f"character ref {ref!r} must use scheme 'npc:' or 'pc:'",
            )
        if ref not in self._entries:
            raise CatalogMissError(source="CharacterCatalog", missing_id=ref)
        return self._entries[ref]

    def add_pc(self, pc_id: str, tokens: CharacterTokens) -> None:
        """Register a PC at runtime from the character store."""
        self._entries[f"pc:{pc_id}"] = tokens


class PlaceTokens(BaseModel):
    kind: Literal["specific", "archetypal"]
    landmark: dict[PlaceLOD, str]
    environment: dict[PlaceLOD, str]
    description: dict[PlaceLOD, str]
    controlling_culture: str | None
    scope: str  # world slug (specific) or genre slug (archetypal)


class PlaceCatalog:
    def __init__(self, entries: dict[str, PlaceTokens]) -> None:
        self._entries = entries

    @classmethod
    def load(
        cls,
        genre_packs_root: Path,
        *,
        genre: str,
        world: str,
    ) -> "PlaceCatalog":
        entries: dict[str, PlaceTokens] = {}
        cls._load_specific(entries, genre_packs_root, genre, world)
        cls._load_archetypal(entries, genre_packs_root, genre)
        return cls(entries)

    @staticmethod
    def _load_specific(
        entries: dict[str, PlaceTokens],
        root: Path,
        genre: str,
        world: str,
    ) -> None:
        path = root / genre / "worlds" / world / "history.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or {}
        for chapter in data.get("chapters", []):
            for poi in chapter.get("points_of_interest", []):
                # Production worlds (e.g. tea_and_murder/blackthorn_moor) author POIs
                # with `name` only — derive the slug parallel to
                # CharacterCatalog.load. Fail loud only when neither is honored.
                slug = poi.get("slug")
                if not slug:
                    name = poi.get("name")
                    if not name:
                        raise ValueError(
                            f"POI in {path}: needs `slug` or `name`",
                        )
                    slug = _slugify_name(name)
                visual = poi.get("visual_prompt", {}) or {}
                env = poi.get("environment", {}) or {}
                if isinstance(visual, str) or isinstance(env, str):
                    raise ValueError(
                        f"POI {slug!r} has string visual_prompt/environment — "
                        f"migrate to {{solo: ..., backdrop: ...}} LODs "
                        f"(see scripts/migrate_poi_backdrop_lod.py)",
                    )
                # Skip POIs with no visual prose at all. Adding them to the
                # catalog with empty CASTING/LOCATION layers produces a thin
                # compose (genre style + camera + safety only) that is
                # *worse* than the prose-subject prompt the narrator already
                # supplies. Letting them catalog-miss routes through the
                # safe wrapper → prose-subject fallback wins. Logged at INFO
                # so content authors see which POIs are unauthored.
                has_visual = any(visual.get(k) for k in ("solo", "backdrop"))
                has_env = any(env.get(k) for k in ("solo", "backdrop"))
                if not has_visual and not has_env:
                    log.info(
                        "place_catalog.poi_skipped reason=no_visual "
                        "world=%s slug=%s",
                        world,
                        slug,
                    )
                    continue
                landmark = {
                    PlaceLOD.SOLO: visual.get("solo", ""),
                    PlaceLOD.BACKDROP: visual.get("backdrop", ""),
                }
                environment = {
                    PlaceLOD.SOLO: env.get("solo", ""),
                    PlaceLOD.BACKDROP: env.get("backdrop", ""),
                }
                description = {
                    PlaceLOD.SOLO: poi.get("name", slug),
                    PlaceLOD.BACKDROP: poi.get("name", slug),
                }
                entries[f"where:{world}/{slug}"] = PlaceTokens(
                    kind="specific",
                    landmark=landmark,
                    environment=environment,
                    description=description,
                    controlling_culture=poi.get("controlling_culture"),
                    scope=world,
                )

    @staticmethod
    def _load_archetypal(
        entries: dict[str, PlaceTokens],
        root: Path,
        genre: str,
    ) -> None:
        path = root / genre / "places.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or {}
        for slug, raw in data.items():
            landmark = {PlaceLOD(k): v for k, v in raw["landmark"].items()}
            environment = {PlaceLOD(k): v for k, v in raw["environment"].items()}
            description = {PlaceLOD(k): v for k, v in raw["description"].items()}
            for lod, text in landmark.items():
                if text:
                    raise ValueError(
                        f"archetypal place {genre}/{slug!r} has populated "
                        f"landmark.{lod.value}; archetypes have no landmark",
                    )
            entries[f"where:{genre}/{slug}"] = PlaceTokens(
                kind="archetypal",
                landmark=landmark,
                environment=environment,
                description=description,
                controlling_culture=None,
                scope=genre,
            )

    def get(self, ref: str) -> PlaceTokens:
        if not ref.startswith("where:"):
            raise ValueError(f"place ref {ref!r} must use scheme 'where:'")
        if ref not in self._entries:
            raise CatalogMissError(source="PlaceCatalog", missing_id=ref)
        return self._entries[ref]


class StyleCatalog:
    def __init__(
        self,
        genre_tokens: dict[str, str],
        world_tokens: dict[tuple[str, str], str],
        culture_tokens: dict[tuple[str, str, str], str],
        world_portrait_tokens: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self._genre = genre_tokens
        self._world = world_tokens
        self._culture = culture_tokens
        # Optional portrait-only world suffix (``portrait_positive_suffix``);
        # absent for worlds that don't declare one.
        self._world_portrait = world_portrait_tokens or {}

    @classmethod
    def load(
        cls,
        genre_packs_root: Path,
        *,
        genre: str,
        world: str,
    ) -> "StyleCatalog":
        genre_tokens: dict[str, str] = {}
        world_tokens: dict[tuple[str, str], str] = {}
        culture_tokens: dict[tuple[str, str, str], str] = {}
        world_portrait_tokens: dict[tuple[str, str], str] = {}

        genre_style = genre_packs_root / genre / "visual_style.yaml"
        # Pack-level (genre) visual_style is OPTIONAL (2026-05-29 directive —
        # visual prompts live at world level). Every recipe's cascade is
        # [GENRE, WORLD, CULTURE] and the composer skips the GENRE layer
        # whenever a WORLD style is present (always — world style is required
        # below), so the genre layer is a fallback that never fires in
        # practice. An ABSENT genre file is by-design; a PRESENT-but-empty
        # file is a broken config and still fails loud (No Silent Fallbacks).
        if genre_style.exists():
            data = yaml.safe_load(genre_style.read_text()) or {}
            suffix = data.get("positive_suffix", "")
            if not suffix:
                raise StyleMissError(
                    scope="genre",
                    identifier=genre,
                    reason=(
                        f"empty or missing positive_suffix in {genre_style} "
                        f"(known_keys={sorted(data.keys())})"
                    ),
                )
            genre_tokens[genre] = suffix

        world_style = genre_packs_root / genre / "worlds" / world / "visual_style.yaml"
        if not world_style.exists():
            raise StyleMissError(
                scope="world",
                identifier=f"{genre}/{world}",
                reason=f"missing visual_style.yaml at {world_style}",
            )
        data = yaml.safe_load(world_style.read_text()) or {}
        suffix = data.get("positive_suffix", "")
        if not suffix:
            raise StyleMissError(
                scope="world",
                identifier=f"{genre}/{world}",
                reason=(
                    f"empty or missing positive_suffix in {world_style} "
                    f"(known_keys={sorted(data.keys())})"
                ),
            )
        world_tokens[(genre, world)] = suffix

        # Optional portrait-only suffix. A close-subject portrait may want a
        # denser engraving / crowd-suppression style the landscape suffix omits.
        # Absent = fall back to positive_suffix (every existing world unchanged);
        # present-but-empty is a broken config and fails loud (No Silent
        # Fallbacks).
        portrait_suffix = data.get("portrait_positive_suffix")
        if portrait_suffix is not None:
            if not str(portrait_suffix).strip():
                raise StyleMissError(
                    scope="world",
                    identifier=f"{genre}/{world}",
                    reason=(
                        f"empty portrait_positive_suffix in {world_style} "
                        f"(remove the key to fall back to positive_suffix)"
                    ),
                )
            world_portrait_tokens[(genre, world)] = portrait_suffix

        # Cultures (world-scoped — per spec)
        cultures_dir = genre_packs_root / genre / "worlds" / world / "cultures"
        if cultures_dir.is_dir():
            for culture_file in cultures_dir.glob("*.yaml"):
                data = yaml.safe_load(culture_file.read_text()) or {}
                slug = culture_file.stem
                culture_tokens[(genre, world, slug)] = data.get("visual_tokens", "")

        return cls(
            genre_tokens, world_tokens, culture_tokens, world_portrait_tokens
        )

    def get_genre(self, genre: str) -> str:
        if genre not in self._genre:
            raise CatalogMissError(source="StyleCatalog.genre", missing_id=genre)
        return self._genre[genre]

    def get_world(self, genre: str, world: str) -> str:
        if (genre, world) not in self._world:
            raise StyleMissError(
                scope="world",
                identifier=f"{genre}/{world}",
                reason="not present in StyleCatalog",
            )
        return self._world[(genre, world)]

    def get_world_portrait(self, genre: str, world: str) -> str | None:
        """Portrait-only world suffix, or None when the world declares none
        (caller falls back to ``get_world``)."""
        return self._world_portrait.get((genre, world))

    def get_culture(self, genre: str, world: str, culture: str) -> str:
        key = (genre, world, culture)
        if key not in self._culture:
            raise CatalogMissError(
                source="StyleCatalog.culture",
                missing_id=f"{genre}/{world}/{culture}",
            )
        return self._culture[key]
