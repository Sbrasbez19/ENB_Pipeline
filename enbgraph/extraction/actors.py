"""
extraction/actors.py — the actor extractor.

Composition over a monolith: the extractor is a thin orchestrator that runs three
tiers behind the single `Extractor` interface —

    1. Gazetteer (always)   — closed-set states/blocs, high precision, stable IDs
    2. RecallSource (opt.)  — open-set tail (NGOs/companies) via NER, off by default
    3. PronounResolver (opt)— attribute he/she to the last-named actor

Each tier is injected, so you can run gazetteer-only (the robust default), add
recall when you care about the open-set tail, and unit-test each in isolation.

`SpacyRecallSource` is provided but spaCy is imported lazily inside it, so the
package neither requires spaCy nor pays its cost unless you actually enable it.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from ..models import Chunk
from .base import ActorCategory, Extractor, Mention, RecallSource
from .coref import PronounResolver
from .gazetteer import Gazetteer


class ActorExtractor(Extractor):
    def __init__(
        self,
        gazetteer: Optional[Gazetteer] = None,
        recall: Optional[RecallSource] = None,
        coref: Optional[PronounResolver] = None,
        include_orgs: bool = False,
    ):
        self.gazetteer = gazetteer or Gazetteer()
        self.recall = recall
        self.coref = coref
        self.include_orgs = include_orgs  # ORG mentions (IPCC/GEF) excluded from the actor graph by default

    def extract(self, chunk: Chunk) -> list[Mention]:
        mentions = self.gazetteer.find(chunk.text, chunk.chunk_id)
        if self.recall is not None:
            mentions += self.recall.find(chunk.text, chunk.chunk_id, mentions)
        if self.coref is not None:
            mentions += self.coref.resolve(chunk, mentions)
        if not self.include_orgs:
            mentions = [m for m in mentions if m.category != ActorCategory.ORG]
        mentions.sort(key=lambda m: (m.start_char, m.end_char))
        return mentions


# --------------------------------------------------------------------------- #
# Aggregation helper (per-year actor counts feed the graph builder later)
# --------------------------------------------------------------------------- #
def actor_counts(mentions: Iterable[Mention], include_pronouns: bool = True) -> Counter:
    """Count distinct-actor weight across many mentions. Pronoun-resolved mentions
    count by default (a 'She opposed' is still that country acting)."""
    c: Counter = Counter()
    for m in mentions:
        if m.is_pronoun and not include_pronouns:
            continue
        c[m.actor_id] += 1
    return c


# --------------------------------------------------------------------------- #
# Optional NER recall source (spaCy). Imported lazily; never a hard dependency.
# --------------------------------------------------------------------------- #
class SpacyRecallSource:
    """Proposes open-set mentions from spaCy NER (GPE/NORP/ORG) that the gazetteer
    missed and that don't overlap existing mentions or hit the stoplist. Useful
    for NGOs/companies; for known states the gazetteer already wins."""

    def __init__(self, model: str = "en_core_web_sm", stoplist: Optional[set[str]] = None):
        import spacy  # type: ignore  # lazy: only needed when this tier is enabled

        self.nlp = spacy.load(model, disable=["lemmatizer", "tagger", "parser"])
        self.stoplist = stoplist or set()

    def find(self, text: str, chunk_id: str, existing: list[Mention]) -> list[Mention]:
        spans = [(m.start_char, m.end_char) for m in existing]

        def overlaps(a: int, b: int) -> bool:
            return any(a < e and s < b for s, e in spans)

        out: list[Mention] = []
        for ent in self.nlp(text).ents:
            if ent.label_ not in ("GPE", "NORP", "ORG"):
                continue
            if ent.text in self.stoplist or overlaps(ent.start_char, ent.end_char):
                continue
            out.append(
                Mention(
                    actor_id=ent.text,            # provisional id; not canonicalised
                    surface=ent.text,
                    category=ActorCategory.OTHER,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    chunk_id=chunk_id,
                    source="recall",
                )
            )
        return out


__all__ = ["ActorExtractor", "actor_counts", "SpacyRecallSource"]
