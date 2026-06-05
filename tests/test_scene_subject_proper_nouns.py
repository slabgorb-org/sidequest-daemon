"""Cap proper-noun density in scene subjects (EH-2 burning_peace playtest 2026-06-05).

CRAFT NOTE: scrapbook scene prompts carried too many proper nouns (named NPCs
like "Padre Ferreira" / "Noriko Nishida", named places like "Hakone" / "Edo"),
over-constraining Z-Image and drifting the look. Names mean nothing to the image
model — they consume the token budget and crowd out the visual/compositional
nouns that steer it.

Two seams carry names into a scene subject: the LLM extractor (handled by the
``_PROMPT_TEMPLATE`` proper-noun guidance — not unit-testable offline) and the
deterministic rule cascade (the combat builder leads with combatant names). This
suite covers the deterministic guard ``_strip_proper_nouns`` and its wiring into
the real ``SceneInterpreter.interpret`` path (regex / no-LLM)."""

from __future__ import annotations

from sidequest_daemon.scene_interpreter import SceneInterpreter, _strip_proper_nouns
from sidequest_daemon.types import Character, GameState


# ---------------------------------------------------------------------------
# Unit — _strip_proper_nouns
# ---------------------------------------------------------------------------


def test_caps_to_one_known_name_keeping_the_first():
    subject = "positions: Padre Ferreira attacks the guards, Noriko Nishida raises a blade"
    out = _strip_proper_nouns(subject, ["Padre Ferreira", "Noriko Nishida"], max_keep=1)
    assert "Padre Ferreira" in out, "the first-appearing name is kept as the anchor"
    assert "Noriko Nishida" not in out, "names beyond the cap are dropped"
    # The visual action of the dropped combatant survives.
    assert "raises a blade" in out
    # No dangling punctuation/whitespace left behind.
    assert ",  " not in out and not out.endswith(",") and "  " not in out


def test_noop_when_within_budget():
    subject = "a robed priest swings a staff in a lantern-lit street"
    out = _strip_proper_nouns(subject, ["Padre Ferreira"], max_keep=1)
    assert out == subject, "no known name present → unchanged"


def test_noop_when_no_known_names():
    subject = "Padre Ferreira lunges through the rain"
    assert _strip_proper_nouns(subject, [], max_keep=1) == subject


def test_keeps_single_name_anchor():
    subject = "Padre Ferreira raises a glowing staff"
    out = _strip_proper_nouns(subject, ["Padre Ferreira"], max_keep=1)
    assert out == subject, "exactly one name is within budget → kept"


def test_max_keep_two_drops_only_the_third():
    subject = "positions: Ash strikes, Bram blocks, Cael lunges"
    out = _strip_proper_nouns(subject, ["Ash", "Bram", "Cael"], max_keep=2)
    assert "Ash" in out and "Bram" in out
    assert "Cael" not in out
    assert "lunges" in out


# ---------------------------------------------------------------------------
# Wiring — the guard is reached from the real interpret() rule cascade
# ---------------------------------------------------------------------------


def test_interpret_combat_subject_caps_named_combatants():
    """The regex combat builder leads with EVERY combatant name. Driving the real
    interpret() path (no LLM extractor) must yield a subject with at most one
    known combatant name — the deterministic guard is wired in."""
    interp = SceneInterpreter()  # no extractor → regex rule cascade
    state = GameState(
        location="The Hakone Road",
        characters=[Character(name="Padre Ferreira"), Character(name="Noriko Nishida")],
    )
    narrative = (
        "Padre Ferreira attacks the guards as Noriko Nishida raises a blade "
        "against the riders."
    )

    cues = interp.interpret(narrative, state)

    assert cues, "a combat narration must produce at least one cue"
    subject = cues[0].subject
    present = [n for n in ("Padre Ferreira", "Noriko Nishida") if n in subject]
    assert len(present) <= 1, (
        f"the scene subject must lead with visual nouns, not pile up combatant "
        f"names; found {present} in {subject!r}"
    )
