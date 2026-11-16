"""
extraction/coref.py — lightweight pronoun resolution.

ENB attributes a lot of stances through pronouns: "The Marshall Islands, supported
by Tuvalu, said... She also noted...". Without resolving "She" to the Marshall
Islands, the later clause loses its actor — which matters most for the relation
extractor, where "She opposed X" should count as the country opposing X.

This v1 is a deliberately simple, high-precision heuristic: a subject pronoun is
attributed to the nearest preceding actor mention within the same chunk. Defaults
to {he, she} because those almost always stand for a delegate (hence a country);
"they"/"it" are far more ambiguous and are opt-in. Full name-based coreference
(binding "Ohki" back to "(Japan)") is a later refinement.
"""

from __future__ import annotations

import re

from ..models import Chunk
from .base import Mention


class PronounResolver:
    def __init__(self, pronouns: tuple[str, ...] = ("he", "she")):
        # match subject pronouns as standalone words, case-insensitive
        self._re = re.compile(r"(?<![A-Za-z])(" + "|".join(pronouns) + r")(?![A-Za-z])", re.IGNORECASE)

    def resolve(self, chunk: Chunk, mentions: list[Mention]) -> list[Mention]:
        """Return extra (pronoun-resolved) mentions; does not mutate the input."""
        if not mentions:
            return []
        # only attribute to concrete actors (states / blocs), not orgs/other
        anchors = sorted(
            (m for m in mentions if m.category.value in ("country", "bloc")),
            key=lambda m: m.start_char,
        )
        if not anchors:
            return []

        resolved: list[Mention] = []
        for pm in self._re.finditer(chunk.text):
            prev = None
            for a in anchors:
                if a.end_char <= pm.start():
                    prev = a
                else:
                    break
            if prev is None:
                continue
            resolved.append(
                Mention(
                    actor_id=prev.actor_id,
                    surface=pm.group(0),
                    category=prev.category,
                    start_char=pm.start(),
                    end_char=pm.end(),
                    chunk_id=chunk.chunk_id,
                    is_pronoun=True,
                    source="coref",
                )
            )
        return resolved


__all__ = ["PronounResolver"]
