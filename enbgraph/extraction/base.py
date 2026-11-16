"""
extraction/base.py — contracts and interfaces for the extraction phase.

A `Mention` is one occurrence of an actor in a chunk: a canonical, year-stable
`actor_id` plus the surface text that produced it and its offsets *into the
chunk's normalised text*. (Chunk-local offsets, not full_text offsets, because we
match on the cleaned `chunk.text`; the chunk's own provenance still ties the
whole span back to the source document.)

`Extractor` is the one interface every extractor implements — the actor extractor
now, relation extractors later — so the graph builder consumes a uniform stream.

`RecallSource` is the optional second tier: a pluggable component (e.g. spaCy NER)
that proposes additional mentions the closed-set gazetteer would miss. It is a
Protocol, not a base class, so any object with a `find` method qualifies and the
core never hard-depends on spaCy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from ..models import Chunk


class ActorCategory(str, Enum):
    COUNTRY = "country"   # a state
    BLOC = "bloc"         # a coalition / group of states (AOSIS, G-77/China, EU)
    ORG = "org"           # an institution / body (IPCC, GEF) — usually excluded from the actor graph
    OTHER = "other"       # open-set candidate from a recall source (NGO, company, person)


class Mention(BaseModel):
    actor_id: str = Field(..., description="Canonical, year-stable node id, e.g. 'United States'.")
    surface: str = Field(..., description="The exact text matched, e.g. 'US' or 'CHINA'.")
    category: ActorCategory
    start_char: int = Field(..., ge=0, description="Offset into the chunk's normalised text.")
    end_char: int = Field(..., ge=0)
    chunk_id: str
    is_pronoun: bool = Field(default=False, description="True if resolved from a pronoun via coref.")
    source: str = Field(default="gazetteer", description="Which tier produced it: gazetteer | recall | coref.")


class Extractor(ABC):
    """Uniform interface for every extractor in the pipeline."""

    @abstractmethod
    def extract(self, chunk: Chunk) -> list[Mention]:  # pragma: no cover - interface
        ...


class RecallSource(Protocol):
    """Optional second-tier mention proposer. Receives the text and the mentions
    the gazetteer already found (to avoid overlaps), returns extra mentions."""

    def find(self, text: str, chunk_id: str, existing: list[Mention]) -> list[Mention]: ...


__all__ = ["ActorCategory", "Mention", "Extractor", "RecallSource"]
