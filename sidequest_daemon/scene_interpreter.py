"""SceneInterpreter — rules-based narrative to StageCue extraction.

Applies pattern-matching rules to narrative text and game state to produce
structured StageCue objects that the Renderer can consume.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
from typing import TYPE_CHECKING

from sidequest_daemon.media.recipes import CameraPreset
from sidequest_daemon.renderer.models import RenderTier, StageCue

if TYPE_CHECKING:
    from sidequest_daemon.genre.models import GenrePack
    from sidequest_daemon.media.subject_extractor import SubjectExtractor
    from sidequest_daemon.types import DocumentEvent, GameState

log = logging.getLogger(__name__)

_MAX_CUES = 2

# Rules are checked in priority order — highest priority first.
# Each rule returns a StageCue or None.

_LOCATION_PATTERNS = re.compile(
    r"\b(?:enters?|arrives?\s+(?:at|in)|steps?\s+(?:into|through|inside)|"
    r"emerges?\s+(?:into|from)|come[s]?\s+upon|"
    r"reaches?\s+(?:the|a)|walks?\s+(?:into|through)|"
    r"passes?\s+through|heads?\s+(?:to|toward|for)|"
    r"cross(?:es)?\s+(?:into|through)|approaches?\s+(?:the|a))\b",
    re.IGNORECASE,
)

_COMBAT_PATTERNS = re.compile(
    r"\b(?:attacks?|lunges?|charges?|"
    r"(?:draws?|unsheathe[sd]?|brandish(?:es)?)\s+(?:\w+\s+){0,2}(?:weapon|sword|dagger|blade|axe|steel|mace)|"
    r"swings?|strikes?|slash(?:es)?|stabs?|parr(?:y|ies)|blocks?|"
    r"challenges?\s+.+?\s+to\s+(?:a\s+)?(?:fight|duel|combat)|"
    r"spell\s+crackles?|initiative|"
    r"blade\s+flash(?:es)?|steel\s+(?:rings?|sings?|clash(?:es)?))\b",
    re.IGNORECASE,
)

_PORTRAIT_INDICATORS = re.compile(
    # Character nouns — fantasy, sci-fi, wasteland, modern, genre-agnostic
    r"\b(?:woman|man|figure|person|stranger|someone|creature|being|"
    r"elf|dwarf|orc|goblin|troll|halfling|gnome|"
    r"priest|priestess|sister|brother|elder|knight|guard|"
    r"merchant|thief|mage|wizard|witch|nun|monk|soldier|"
    r"captain|lord|lady|queen|king|prince|princess|"
    r"child|girl|boy|old\s+(?:man|woman)|"
    r"drifter|operator|pilot|trader|scavenger|raider|"
    r"synth|mutant|android|cyborg|veteran|survivor|"
    r"bartender|innkeeper|shopkeeper|smith|hunter|healer|"
    r"she|he)\b.{1,300}"
    # Physical descriptors — body, gear, expression, genre-agnostic
    r"\b(?:wear|cloak|armor|armour|scar|eye|face|pendant|"
    r"tall|short|crouch|kneel|stand|lean|sit|hood|robe|"
    r"hair|beard|voice|gaze|stare|look|grin|smile|frown|"
    r"prosthetic|implant|mutation|cybernetic|augment|"
    r"arm|limb|hand|finger|skin|jaw|neck|shoulder|"
    r"tattoo|mark|brand|patch|visor|goggles|helmet|"
    r"compact|wiry|hulk|massive|gaunt|weathered|grizzled|"
    r"calloused|scarred|muscular|built|watching|coil)\b",
    re.IGNORECASE | re.DOTALL,
)

_ATMOSPHERE_PATTERNS = re.compile(
    r"\b(?:fog|mist|darkness|shadow|torchlight|flicker|gloom|fades|"
    r"rolls?\s+in|dusk|dawn|moonlight|starlight|candlelight|"
    r"silence|wind|rain|storm|thunder|lightning|"
    r"cold|chill|warmth|heat|smoke|ash|dust|"
    r"ruins?|crumbl|decay|overgrown|abandoned|"
    r"pulse[sd]?|glow(?:s|ed|ing)?|shimmer|dark(?:ened|ening)?|"
    r"eeri[ely]*|haunted|desolat|bleak|foreboding)\b",
    re.IGNORECASE,
)

_MAGIC_PATTERNS = re.compile(
    r"\b(?:arcane|magic(?:al)?|spell|shimmer|portal|divine|radiance|glyph|"
    r"explosion|enchant|ward|rune|ritual|summon|conjur|"
    r"crystal|amulet|talisman|curse|hex|bless|"
    r"sorcery|incantation|mystic|occult|ethereal|"
    r"standing\s+stone|ley\s+line|ancient\s+power)\b",
    re.IGNORECASE,
)

_DOCUMENT_PATTERNS = re.compile(
    r"\b(?:notice|noticeboard|poster|letter|scroll|parchment|tome|"
    r"book|inscription|sign|placard|decree|proclamation|"
    r"wanted\s+poster|map|journal|diary|note|message|"
    r"reads?\s+(?:the|a)|written|scrawled|carved|etched|"
    r"posted|pinned|nailed)\b",
    re.IGNORECASE,
)

_MAX_SUBJECT_LEN = 350  # ~70 tokens, keeps subjects concise for T5-XXL token budget

_MECHANICS_PATTERN = re.compile(
    r'\[[^\]]*(?:\d+\s*→\s*\d+|\d+\s*HP|d\d+:\s*\d+|vs\s+AC\s+\d+|HIT|MISS|Attack|Damage)[^\]]*\]|'  # brackets with mechanics
    r'\d+\s*→\s*\d+\s*HP|'  # HP changes
    r'd\d+:\s*\d+|'         # dice rolls
    r"^[A-Z]+(?:'S)?\s+TURN\s*$|"  # TURN markers (own line only)
    r'^#+\s+',              # markdown headers
    re.IGNORECASE | re.MULTILINE,
)


def _strip_mechanics(text: str) -> str:
    """Remove game mechanics notation (HP, dice, turns, brackets) from text."""
    cleaned = _MECHANICS_PATTERN.sub('', text)
    # Collapse leftover whitespace
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip(' \t\n—')
    return cleaned


# --- Visual distillation helpers (Story 38-8) ---

_DIALOGUE_QUOTE_PATTERN = re.compile(
    r'["\u201c][^"\u201d]*["\u201d]'      # double / smart-double quotes
    r"|[\u2018][^\u2019]*[\u2019]"          # smart single quotes
    r"|(?<!\w)'.+?'(?=[,.\s!?]|\Z)",       # straight single-quoted speech (handles contractions)
    re.DOTALL,
)

_ATTRIBUTION_VERB_PATTERN = re.compile(
    r',?\s*\b(?:said|says|whispered?|whispers?|muttered?|mutters?|'
    r'shouted?|shouts?|asked?|asks?|replied?|replies?|'
    r'grumbled?|grumbles?|growled?|growls?|exclaimed?|exclaims?|warned?)\b'
    r'[,]?(?:\s+\w+){0,4}[.]?',
    re.IGNORECASE,
)

_ABSTRACTION_PATTERN = re.compile(
    r'\b(?:sense\s+of\s+(?:dread|foreboding|unease|wonder|loss|hope|doom)|'
    r'(?:felt|feeling|feels)\s+(?:a|the|like|as)\b|'
    r'seemed?\s+(?:to|alive|dead|wrong|right)\b|'
    r'as\s+(?:if|though)\s+[^,.]{0,60}|'
    r'darkness\s+of\s+(?:the\s+)?soul|'
    r'weight\s+of\s+(?:despair|sorrow|grief|ages|centuries)|'
    r'couldn[\'\u2019]t\s+(?:help\s+but|shake)|'
    r'shouldn[\'\u2019]t\s+be\s+here|'
    r'like\s+a\s+(?:sleeping\s+giant|burial\s+shroud)|'
    r'(?:heavens?|gods?)\s+(?:themselves\s+)?(?:mourned?|wept|cried)|'
    r'one\s+could\s+(?:almost\s+)?(?:feel|sense|hear)\b[^,.]*|'
    r'sleeping\s+giant|'
    r'its\s+breath\s+the\s+[^,.]+)',
    re.IGNORECASE,
)

_NON_VISUAL_WORDS = frozenset({
    'the', 'a', 'an', 'of', 'in', 'at', 'to', 'and', 'or', 'but', 'for',
    'with', 'on', 'by', 'from', 'into', 'through', 'onto', 'upon', 'over',
    'under', 'between', 'behind', 'beyond', 'where', 'while', 'as', 'its',
    'their', 'his', 'her', 'he', 'she', 'it', 'they', 'you', 'your',
    'who', 'whom', 'that', 'which', 'this', 'these', 'those',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can',
    'not', 'no', 'nor', 'neither',
})


def _strip_dialogue(text: str) -> str:
    """Remove all quoted speech and dialogue attribution verbs from text."""
    result = _DIALOGUE_QUOTE_PATTERN.sub('', text)
    result = _ATTRIBUTION_VERB_PATTERN.sub('', result)
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'[.,]{2,}', '.', result)
    result = re.sub(r'\.\s*,|,\s*\.', '.', result)
    result = re.sub(r'\s*,\s*,+', ',', result)
    result = re.sub(r'^\s*[,.]+\s*|\s*[,.]+\s*$', '', result)
    return result.strip()


def _strip_abstractions(text: str) -> str:
    """Remove emotional and abstract language, keeping concrete visual elements."""
    result = _ABSTRACTION_PATTERN.sub('', text)
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'[.,]{2,}', '.', result)
    result = re.sub(r'\.\s*,|,\s*\.', '.', result)
    result = re.sub(r'\s*,\s*,+', ',', result)
    result = re.sub(r'^\s*[,.]+\s*|\s*[,.]+\s*$', '', result)
    return result.strip()



def _distill_visual(text: str) -> str:
    """Chain visual distillation: mechanics -> dialogue -> abstractions -> cleanup."""
    result = _strip_mechanics(text)
    result = _strip_dialogue(result)
    result = _strip_abstractions(result)
    # Remove stray apostrophes — render prompts don't need possessives or quotes
    result = result.replace("'", "").replace("\u2019", "")
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def _truncate(text: str, max_len: int = _MAX_SUBJECT_LEN) -> str:
    """Truncate text at a word boundary, avoiding mid-word splits."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    # Find the last space to avoid cutting mid-word
    last_space = truncated.rfind(" ")
    if last_space > max_len // 2:
        return truncated[:last_space]
    return truncated


def _is_dialogue_only(text: str) -> bool:
    """Return True if the narrative is purely dialogue with no visual action."""
    stripped = text.strip()
    if not stripped:
        return True
    # Check if the text is primarily quoted speech
    lines = [line.strip() for line in stripped.split("\n") if line.strip()]
    for line in lines:
        if not re.match(
            r'^["\u201c\u2018].*["\u201d\u2019][,.]?\s*(?:\w+\s+)*(?:says?|said|mutter|mutters|whisper|whispers|shout|shouts|ask|asks|replie[sd]?|grumble[sd]?|growl[sd]?|laugh[sed]*)\w*\.?$',
            line,
            re.IGNORECASE,
        ):
            return False
    return True


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _extract_combat_subject(narrative: str, character_names: list[str]) -> str:
    """Extract combatants + action from narrative for SCENE_ILLUSTRATION with topdown_90 camera.

    Collects ALL combatant name+action pairs and formats as comma-separated
    positional tokens with 'positions:' prefix for map-like prompts.
    """
    clean = re.sub(r'\*+|—|_', '', narrative).strip()
    # Collect all combatant name+action pairs
    pairs: list[str] = []
    for name in character_names:
        match = re.search(
            rf'\b{re.escape(name)}\b\s+(attacks?|lunges?\s+at|charges?\s+at|swings?\s+at|strikes?|slashes?\s+at|stabs?\s+at|parr(?:y|ies)|blocks?|challenges?|hurls?|raises?|casts?)\s+(.+?)(?:\.|,|!|\Z)',
            clean,
            re.IGNORECASE,
        )
        if match:
            action = _distill_visual(f"{match.group(1)} {match.group(2)}") or f"{match.group(1)} {match.group(2)}"
            pairs.append(f"{name} {action}")
    if pairs:
        return _truncate("positions: " + ", ".join(pairs))
    # Fallback: find the sentence containing combat keywords
    for sentence in _split_sentences(clean):
        if _COMBAT_PATTERNS.search(sentence):
            return _truncate(_distill_visual(sentence) or sentence)
    # Last resort: character names + "in combat"
    if character_names:
        return _truncate(", ".join(character_names) + " in combat")
    return _truncate(_distill_visual(clean) or clean)


def _strip_proper_nouns(subject: str, known_names: list[str], *, max_keep: int = 1) -> str:
    """Cap named-entity density in a render subject (EH-2 scrapbook craft, 2026-06-05).

    Operator note: scrapbook scene prompts carried too many proper nouns (named
    NPCs like "Padre Ferreira" / "Noriko Nishida"), which mean nothing to
    Z-Image — they consume the token budget and over-constrain the look, crowding
    out the visual/compositional nouns that actually steer the model ("economy of
    names"). Keep at most ``max_keep`` DISTINCT known character names (the first to
    appear, so the subject keeps one anchor), drop the rest, and normalize the
    residual punctuation/whitespace.

    Deterministic and roster-bounded: it only removes names from ``known_names``
    (the scene's present characters), never visual nouns or unknown words. Place
    names — which the daemon cannot enumerate — are handled upstream by the LLM
    extractor prompt's proper-noun guidance. A subject already within the density
    budget is returned unchanged."""
    if not subject or not known_names:
        return subject
    low = subject.lower()
    appearances: list[tuple[int, str]] = []
    for name in known_names:
        name = name.strip()
        if not name:
            continue
        idx = low.find(name.lower())
        if idx >= 0:
            appearances.append((idx, name))
    if len(appearances) <= max_keep:
        return subject
    appearances.sort(key=lambda t: t[0])
    drop = {name for _, name in appearances[max_keep:]}
    out = subject
    for name in drop:
        out = re.sub(rf"\b{re.escape(name)}\b", "", out, flags=re.IGNORECASE)
    # Normalize the gaps the removals leave behind.
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)  # " ," -> ","
    out = re.sub(r"([,;:])(\s*[,;:])+", r"\1", out)  # collapse runs of separators
    out = re.sub(r"(:\s*),", r"\1", out)  # "positions: ," -> "positions:"
    out = re.sub(r"\s{2,}", " ", out)  # collapse double spaces
    return out.strip().strip(",").strip()


def _extract_portrait_subject(narrative: str) -> str:
    """Extract the character description sentence for PORTRAIT."""
    clean = re.sub(r'\*+|—|_', '', narrative).strip()
    for sentence in _split_sentences(clean):
        if _PORTRAIT_INDICATORS.search(sentence):
            return _truncate(_distill_visual(sentence) or sentence)
    return _truncate(_distill_visual(clean) or clean)


def _extract_scene_subject(narrative: str, pattern: re.Pattern) -> str:
    """Extract visual content from narrative for SCENE_ILLUSTRATION."""
    clean = re.sub(r'\*+|—|_', '', narrative).strip()
    distilled = _distill_visual(clean)
    if distilled:
        return _truncate(distilled)
    # Fallback: find sentence matching the pattern
    for sentence in _split_sentences(clean):
        if pattern.search(sentence):
            return _truncate(sentence)
    return _truncate(clean)


def _extract_location_subject(narrative: str) -> str:
    """Pull out the location noun phrase from a location-change sentence."""
    # Try to grab the last noun phrase after location keywords
    match = re.search(
        r"\b(?:enters?|arrives?\s+(?:at|in)|steps?\s+(?:into|through|inside)|"
        r"emerges?\s+(?:into|from)|come[s]?\s+upon|"
        r"reaches?\s+(?:the|a)|walks?\s+(?:into|through)|"
        r"passes?\s+through|heads?\s+(?:to|toward|for)|"
        r"cross(?:es)?\s+(?:into|through)|approaches?\s+(?:the|a))"
        r"\s+(?:the\s+|a\s+|an\s+)?([\w\s]+?)(?:\.|,|!|\Z)",
        narrative,
        re.IGNORECASE,
    )
    if match:
        return _truncate(match.group(1).strip())
    return _truncate(narrative[:80])


def _word_tokens(text: str) -> set[str]:
    """Tokenize text into lowercase word tokens for similarity comparison."""
    return {w.lower() for w in re.findall(r'\w+', text)} - {
        'the', 'a', 'an', 'of', 'in', 'at', 'to', 'and', 'or', 'you', 'your',
    }


def _token_similarity(a: str, b: str) -> float:
    """Jaccard similarity over word tokens."""
    ta, tb = _word_tokens(a), _word_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


_SIMILARITY_THRESHOLD = 0.4


class SceneInterpreter:
    """Extract StageCue objects from narrative text using pattern-matching rules."""

    def __init__(
        self,
        genre_pack: GenrePack | None = None,
        *,
        extractor: SubjectExtractor | None = None,
        location_cooldown_turns: int = 2,
    ) -> None:
        self._genre_pack = genre_pack
        self._extractor = extractor
        self._location_cooldown_turns = location_cooldown_turns
        self.last_rendered_location: str | None = None
        self.location_cooldown_remaining: int = 0

    def _run_extractor(self, narrative: str) -> dict | None:
        """Run the async extractor from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop is not None and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self._extractor.extract(narrative))
                    return future.result(timeout=30)
            else:
                return asyncio.run(self._extractor.extract(narrative))
        except Exception as exc:
            log.warning("Subject extractor failed (falling back to regex): %s", exc)
            return None

    def _substitute_vars(self, narrative: str, state: GameState) -> str:
        """Replace template variables with values from game state."""
        player_name = state.characters[0].name if state.characters else "Adventurer"
        replacements = {
            "player_name": player_name,
            "location": state.location,
            "time_of_day": state.time_of_day,
        }
        for key, value in replacements.items():
            narrative = narrative.replace(f"{{{key}}}", value)
        return narrative

    def interpret(self, narrative: str, state: GameState) -> list[StageCue]:
        """Interpret narrative text and return up to _MAX_CUES StageCue objects.

        Rules are evaluated in priority order:
        1. Location change → LANDSCAPE
        2. Combat initiation → SCENE_ILLUSTRATION with topdown_90 camera
        3. Character introduction → PORTRAIT
        4. Special effects → SCENE_ILLUSTRATION (tagged)
        5. Mood/atmosphere → SCENE_ILLUSTRATION

        Pure dialogue and empty text produce no cues.
        """
        if not narrative.strip():
            log.debug("SCENE_INTERPRET: empty narrative, no cues")
            return []

        narrative = self._substitute_vars(narrative, state)

        if _is_dialogue_only(narrative):
            log.info("SCENE_INTERPRET: dialogue-only narrative, skipping render cues")
            return []

        # Decrement location cooldown each turn
        if self.location_cooldown_remaining > 0:
            self.location_cooldown_remaining -= 1

        # Detect location change
        current_location = state.location or ""
        location_changed = (
            self.last_rendered_location is not None
            and current_location
            and current_location != self.last_rendered_location
            and _token_similarity(current_location, self.last_rendered_location) < _SIMILARITY_THRESHOLD
        )
        if location_changed:
            self.location_cooldown_remaining = 0

        # Try LLM extraction if extractor is available
        llm_result = None
        if self._extractor is not None:
            llm_result = self._run_extractor(narrative)

        character_names = [c.name for c in state.characters]

        # If LLM extraction succeeded, build a cue from it
        if llm_result is not None:
            tier_str = llm_result["tier"]
            try:
                tier = RenderTier(tier_str)
            except ValueError:
                try:
                    tier = RenderTier[tier_str]
                except KeyError:
                    tier = None

            # Reject subjects that look like dialogue
            if tier is not None and re.search(
                r'^["\u201c]|\b(?:said|says|whispered|whisper|asked|asks|replied|shouted|muttered|grumbled|exclaimed)\b',
                llm_result["subject"],
                re.IGNORECASE,
            ):
                tier = None
                llm_result = None

            if tier is not None:
                llm_cue = StageCue(
                    tier=tier,
                    # EH-2 (2026-06-05): cap named-entity density even on the LLM
                    # path — the extractor prompt now discourages proper nouns,
                    # but this is the deterministic backstop for the known roster.
                    subject=_strip_proper_nouns(llm_result["subject"], character_names),
                    mood=llm_result.get("mood", ""),
                    tags=llm_result.get("tags", []),
                    location=state.location,
                    characters=character_names,
                )
                return [llm_cue]

        cues: list[StageCue] = []

        # Rule 0: Document/notice/scroll → TEXT_OVERLAY
        if _DOCUMENT_PATTERNS.search(narrative):
            doc_subject = _extract_scene_subject(narrative, _DOCUMENT_PATTERNS)
            cues.append(StageCue(
                tier=RenderTier.TEXT_OVERLAY,
                subject=doc_subject,
                tags=["document"],
                location=state.location,
            ))

        # Rule 1: Location change → LANDSCAPE
        if _LOCATION_PATTERNS.search(narrative):
            subject = _extract_location_subject(narrative)
            mood = f"{state.time_of_day}, {_detect_mood(narrative)}"
            cues.append(StageCue(
                tier=RenderTier.LANDSCAPE,
                subject=subject,
                mood=mood,
                location=state.location,
                characters=character_names,
            ))

        # Rule 2: Combat → SCENE_ILLUSTRATION with topdown_90 camera
        # FOG_OF_WAR is for /map command only (programmatic PNG, no daemon).
        # Combat goes through Z-Image as SCENE_ILLUSTRATION with a top-down camera preset.
        has_combat = _COMBAT_PATTERNS.search(narrative)
        if has_combat:
            combat_subject = _extract_combat_subject(narrative, character_names)
            cues.append(StageCue(
                tier=RenderTier.SCENE_ILLUSTRATION,
                subject=combat_subject,
                tags=["combat"],
                characters=character_names,
                location=state.location,
                camera=CameraPreset.topdown_90,
            ))

        # Rule 3: Character introduction → PORTRAIT (skip when combat already detected)
        if not has_combat and _PORTRAIT_INDICATORS.search(narrative) and len(narrative) > 60:
            portrait_subject = _extract_portrait_subject(narrative)
            cues.append(StageCue(
                tier=RenderTier.PORTRAIT,
                subject=portrait_subject,
                characters=character_names,
                location=state.location,
            ))

        # Rule 4: Special effects → SCENE_ILLUSTRATION (tagged)
        if _MAGIC_PATTERNS.search(narrative):
            tags = []
            if re.search(r"\b(?:arcane|magic|spell|shimmer|divine|radiance|glyph|enchant)\b", narrative, re.IGNORECASE):
                tags.append("magic")
            if re.search(r"\b(?:explosion|portal)\b", narrative, re.IGNORECASE):
                tags.append("special_effect")
            if not tags:
                tags.append("special_effect")
            scene_subject = _extract_scene_subject(narrative, _MAGIC_PATTERNS)
            cues.append(StageCue(
                tier=RenderTier.SCENE_ILLUSTRATION,
                subject=scene_subject,
                tags=tags,
                mood=_detect_mood(narrative),
                location=state.location,
            ))

        # Rule 5: Atmosphere → SCENE_ILLUSTRATION (only if no special effect already)
        if _ATMOSPHERE_PATTERNS.search(narrative) and not _LOCATION_PATTERNS.search(narrative) and not any(
            c.tier == RenderTier.SCENE_ILLUSTRATION for c in cues
        ):
            mood = _detect_mood(narrative)
            scene_subject = _extract_scene_subject(narrative, _ATMOSPHERE_PATTERNS)
            cues.append(StageCue(
                tier=RenderTier.SCENE_ILLUSTRATION,
                subject=scene_subject,
                mood=mood,
                location=state.location,
                characters=character_names,
            ))

        # Fallback: if no rules matched, distill visual content
        if not cues:
            distilled = _distill_visual(re.sub(r'\*+|—|_', '', narrative).strip())
            if distilled and len(distilled.strip()) > 10:
                cues.append(StageCue(
                    tier=RenderTier.SCENE_ILLUSTRATION,
                    subject=_truncate(distilled),
                    mood=_detect_mood(narrative),
                    location=state.location,
                    characters=character_names,
                ))

        # Location-aware LANDSCAPE dedup
        if self._location_cooldown_turns > 0:
            filtered_cues: list[StageCue] = []
            for cue in cues:
                if cue.tier == RenderTier.LANDSCAPE:
                    same_location = (
                        self.last_rendered_location is not None
                        and current_location
                        and (
                            current_location == self.last_rendered_location
                            or _token_similarity(current_location, self.last_rendered_location) >= _SIMILARITY_THRESHOLD
                        )
                    )
                    if same_location and self.location_cooldown_remaining > 0:
                        continue  # suppress
                    # LANDSCAPE fires — update tracking
                    self.last_rendered_location = current_location if current_location else cue.location
                    self.location_cooldown_remaining = self._location_cooldown_turns
                    filtered_cues.append(cue)
                else:
                    filtered_cues.append(cue)
            cues = filtered_cues
        else:
            # cooldown=0: no suppression, but still track location
            for cue in cues:
                if cue.tier == RenderTier.LANDSCAPE:
                    self.last_rendered_location = current_location if current_location else cue.location
                    break

        # EH-2 (2026-06-05): cap named-entity density on every rule-built subject
        # too (the combat builder deliberately leads with combatant names). Lead
        # with the visual action; keep one name as an anchor, drop the rest.
        for cue in cues:
            cue.subject = _strip_proper_nouns(cue.subject, character_names)

        return cues[:_MAX_CUES]

    # ── Document extraction ───────────────────────────────────────────

    _DOC_PATTERN = re.compile(
        r'\[DOCUMENT:(\w+):"([^"]+)"\]\s*(.*?)\s*\[/DOCUMENT\]',
        re.DOTALL,
    )

    def extract_documents(self, narrative: str, *, genre: str) -> list[DocumentEvent]:
        """Extract DocumentEvent objects from [DOCUMENT:...] markers in narrative."""
        from sidequest_daemon.types import DocumentEvent

        if not narrative:
            return []

        events = []
        for match in self._DOC_PATTERN.finditer(narrative):
            template_name = match.group(1)
            title = match.group(2)
            body_text = match.group(3).strip()
            events.append(
                DocumentEvent(
                    genre=genre,
                    template_name=template_name,
                    title=title,
                    body_text=body_text,
                )
            )
        return events

    def strip_document_markers(self, narrative: str) -> str:
        """Remove [DOCUMENT:...][/DOCUMENT] blocks from narrative text."""
        return self._DOC_PATTERN.sub("", narrative).strip()


def _detect_mood(narrative: str) -> str:
    """Infer a mood string from narrative atmosphere keywords."""
    lower = narrative.lower()
    if any(w in lower for w in ("fog", "mist", "shadow", "gloom", "dark")):
        return "ominous"
    if any(w in lower for w in ("torch", "flicker", "dim", "candle")):
        return "tense"
    if any(w in lower for w in ("radiance", "divine", "light", "shimmer", "arcane")):
        return "mystical"
    if any(w in lower for w in ("explosion", "crack", "burst")):
        return "dramatic"
    if "fade" in lower:
        return "melancholic"
    return "atmospheric"
