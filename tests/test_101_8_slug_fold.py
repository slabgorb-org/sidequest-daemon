"""RED — Story 101-8 (daemon side): ``_slugify_name`` adopts the unified
NFKD-fold rule.

``sidequest_daemon.media.catalogs._slugify_name`` is the render-side rule that
names portrait/character R2 files. It is the daemon twin of the server's
``slugify_player_name`` (the server docstring literally calls itself "Mirror of
``sidequest_daemon.media.catalogs._slugify_name``"). Story 101-8 unifies the
three slug rules behind one shared NFKD-fold core (session decision, ratified by
Keith): ``unicodedata.normalize("NFKD", …)`` + strip combining marks, then the
existing lowercase / whitespace→``_`` / drop-punctuation steps. ASCII output is
UNCHANGED; diacritics fold to base letters instead of being dropped.

Golden vector ``"Srárný Fyzioloniązka"`` matches the server + orchestrator
suites so the three repos cannot drift on the fold contract.
"""

from __future__ import annotations

import pytest

from sidequest_daemon.media.catalogs import _slugify_name

_DIACRITIC = "Srárný Fyzioloniązka"


# --- AC3: ASCII output unchanged (measured current values; green now + after) ---
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Jane Doe", "jane_doe"),
        ("Old Sten", "old_sten"),
        ("  Mara  Quill  ", "mara_quill"),
        ("Cosh & Run", "cosh__run"),
        ("", ""),
    ],
)
def test_slugify_name_ascii_unchanged(raw: str, expected: str) -> None:
    assert _slugify_name(raw) == expected


# --- AC1: NFKD fold of non-ASCII (RED today — today drops the diacritic) ---
def test_slugify_name_folds_diacritics() -> None:
    # today: "srrn_fyziolonizka"  →  fold: base letters preserved.
    assert _slugify_name(_DIACRITIC) == "srarny_fyzioloniazka"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("café", "cafe"),
        ("naïve", "naive"),
        ("Zoë", "zoe"),
        ("Núñez", "nunez"),
        ("Smörgåsbord", "smorgasbord"),
    ],
)
def test_slugify_name_common_diacritics_fold(raw: str, expected: str) -> None:
    assert _slugify_name(raw) == expected


def test_daemon_matches_server_portrait_rule() -> None:
    # The daemon write-side and the server portrait read-side MUST agree on the
    # folded form (URL == filename). Pin the daemon's output to the same folded
    # slug the server's slugify_player_name produces (asserted in the server
    # suite). If these drift, portraits silently 404.
    assert _slugify_name(_DIACRITIC) == "srarny_fyzioloniazka"
