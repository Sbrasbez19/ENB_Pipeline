"""
extraction/gazetteer.py — the closed-set actor matcher.

Why a gazetteer is the backbone (not NER): the actors that matter — states and
negotiating blocs — are a *closed, known set*. Matching against that set is far
higher precision than a statistical NER model and gives stable canonical IDs for
free. NER is reserved for the open-set tail (NGOs, companies) via a RecallSource.

Two subtleties this module exists to handle:

1. CASE SHIFT. ENB wrote Parties in UPPERCASE in the 1990s ("CHINA") and switched
   to Title Case from ~2000 ("China"). So full names match case-INSENSITIVELY.

2. THE "US" vs "us" TRAP. But case-insensitivity is dangerous for short acronym
   aliases: a case-insensitive "US" would match the pronoun "us". So acronym
   forms (all-caps, short — US, EU, EC, AOSIS, OPEC) match case-SENSITIVELY,
   uppercase only. Each surface form gets its own case rule, combined into one
   scanning pass via Python's scoped inline flags `(?i:...)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils import load_yaml
from .base import ActorCategory, Mention

DEFAULT_GAZETTEER = Path(__file__).resolve().parents[1] / "config" / "gazetteer.yaml"


@dataclass(frozen=True)
class _Form:
    surface: str
    actor_id: str
    category: ActorCategory
    case_sensitive: bool


def _is_acronym(form: str) -> bool:
    """Acronym-like forms match case-sensitively to avoid hitting common words.
    True when the form is short and has no lowercase letters (US, EU, EC, AOSIS,
    OPEC, G-77...). Full names like 'China' or 'Annex I' return False."""
    letters = [c for c in form if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters) and len(form) <= 6


class Gazetteer:
    """Loads actor data and scans text for known actors. One compiled pattern,
    longest-form-first so 'South Korea' wins over a bare 'Korea', and per-form
    case rules so 'CHINA'/'China' both match while 'US' never matches 'us'."""

    def __init__(self, path: Optional[Path] = None):
        cfg = load_yaml(path or DEFAULT_GAZETTEER)
        self.stoplist: set[str] = set(cfg.get("stoplist", []))
        self._forms: dict[str, _Form] = {}          # exact surface -> form (for case-sensitive)
        self._forms_ci: dict[str, _Form] = {}        # lower surface -> form (for case-insensitive)

        def add(surface: str, actor_id: str, category: ActorCategory):
            if surface in self.stoplist:
                return
            form = _Form(surface, actor_id, category, _is_acronym(surface))
            if form.case_sensitive:
                self._forms[surface] = form
            else:
                self._forms_ci[surface.lower()] = form

        for c in cfg.get("countries", []):
            add(c, c, ActorCategory.COUNTRY)
        for b in cfg.get("blocs", []):
            add(b, b, ActorCategory.BLOC)
        for o in cfg.get("organizations", []):
            add(o, o, ActorCategory.ORG)
        # aliases inherit the category of their canonical
        canon_cat = {f.actor_id: f.category for f in (*self._forms.values(), *self._forms_ci.values())}
        for surface, canonical in cfg.get("aliases", {}).items():
            add(surface, canonical, canon_cat.get(canonical, ActorCategory.OTHER))

        self._pattern = self._compile()

    def _compile(self) -> re.Pattern[str]:
        # longest first so multi-word forms win; case-insensitive forms wrapped in (?i:)
        all_forms = list(self._forms.values()) + list(self._forms_ci.values())
        all_forms.sort(key=lambda f: len(f.surface), reverse=True)
        alts = []
        for f in all_forms:
            esc = re.escape(f.surface)
            alts.append(esc if f.case_sensitive else f"(?i:{esc})")
        # not preceded/followed by a letter or digit -> robust word boundary that
        # also works for forms containing '.', '/', '-' (e.g. 'U.S.', 'G-77/China')
        return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(alts) + r")(?![A-Za-z0-9])")

    def _lookup(self, surface: str) -> Optional[_Form]:
        return self._forms.get(surface) or self._forms_ci.get(surface.lower())

    def find(self, text: str, chunk_id: str) -> list[Mention]:
        out: list[Mention] = []
        for m in self._pattern.finditer(text):
            form = self._lookup(m.group(0))
            if form is None:
                continue
            out.append(
                Mention(
                    actor_id=form.actor_id,
                    surface=m.group(0),
                    category=form.category,
                    start_char=m.start(),
                    end_char=m.end(),
                    chunk_id=chunk_id,
                    source="gazetteer",
                )
            )
        return out


__all__ = ["Gazetteer"]
