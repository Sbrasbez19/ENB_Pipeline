"""
llm/schema.py — the extraction contract, in two layers.

1. `StanceTriplet` is what the LLM fills: one stance read from the text.
   Its JSON schema is passed to vLLM as a structured-output constraint, so
   responses always parse into this shape. It deliberately contains ONLY
   what must be read from the text.

2. `TripletRecord` is what we store: the triplet wrapped with metadata that
   code attaches deterministically after extraction — where it came from
   (document, passage, pages, offsets, year, COP, date, source type), how it
   was produced (model, revision, prompt/schema versions, run) and the
   grounding checks. The LLM never generates these fields: it could get them
   wrong, and we already know them exactly.

Bump SCHEMA_VERSION whenever a StanceTriplet field changes: it is part of the
cache key, so old cached responses are never parsed against a new schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..models import Passage, Provenance, SourceType

SCHEMA_VERSION = "1"


# --------------------------------------------------------------------------- #
# Layer 1 — filled by the LLM
# --------------------------------------------------------------------------- #
class StanceTriplet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_actor: str = Field(
        ..., description="The party or group expressing the stance, as a country/bloc name "
                         "(e.g. 'G-77/China', 'Saudi Arabia'). For a pronoun or description, "
                         "the actor it refers to.")
    source_mention: str = Field(
        ..., description="How the speaker appears in the FOCUS text, verbatim (e.g. 'She', "
                         "'the G-77/CHINA', 'the delegate').")
    source_role: Literal["party", "officer", "observer"] = Field(
        ..., description="'party' = a country or negotiating group; 'officer' = someone acting "
                         "for the conference (President, Chair, Co-Chair, facilitator); "
                         "'observer' = NGO, IGO, business or expert.")
    on_behalf_of: Optional[str] = Field(
        None, description="Group the speaker spoke for ('on behalf of the G-77/China'), else null.")
    reference_type: Literal["explicit", "pronoun", "description"] = Field(
        ..., description="'explicit' = speaker named; 'pronoun' = he/she/it/they; "
                         "'description' = e.g. 'the delegate', 'this group'.")
    antecedent_evidence: Optional[str] = Field(
        None, description="For pronoun/description: verbatim text (from CONTEXT or FOCUS) "
                          "naming the actor referred to. Null for explicit mentions.")
    verb: str = Field(..., description="The stance verb as written (e.g. 'opposed', 'supported').")
    polarity: Literal["support", "oppose", "neutral"]
    target_type: Literal["actor", "issue"] = Field(
        ..., description="'actor' when the stance is toward another party or group; "
                         "'issue' when it is toward a proposal, text or topic.")
    target: Optional[str] = Field(
        None, description="target_type 'actor': the actor's name, or null if the text keeps it "
                          "anonymous ('a key Annex I country'). target_type 'issue': a short "
                          "label of the issue.")
    proposal: Optional[str] = Field(
        None, description="The specific proposal/text the stance concerns, if stated.")
    proposal_author: Optional[str] = Field(
        None, description="Who made that proposal, only if named in the text; else null.")
    evidence: str = Field(
        ..., description="The sentence(s) from FOCUS supporting this triplet, copied verbatim.")


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triplets: list[StanceTriplet] = Field(default_factory=list)


def json_schema() -> dict:
    return ExtractionResponse.model_json_schema()


def response_format() -> dict:
    """OpenAI-style structured-output request field (accepted by vLLM)."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "stance_extraction", "schema": json_schema(), "strict": True},
    }


# --------------------------------------------------------------------------- #
# Layer 2 — attached by code, stored on disk
# --------------------------------------------------------------------------- #
class SourceInfo(BaseModel):
    """Where the triplet came from (copied from the Passage)."""

    doc_id: str
    passage_id: str
    source_type: SourceType
    provenance: Provenance            # pages + character span in Document.full_text
    section_hint: Optional[str] = None
    cop: Optional[str] = None
    year: Optional[int] = None
    date: Optional[str] = None

    @classmethod
    def from_passage(cls, p: Passage) -> "SourceInfo":
        return cls(doc_id=p.doc_id, passage_id=p.passage_id, source_type=p.source_type,
                   provenance=p.provenance, section_hint=p.section_hint,
                   cop=p.cop, year=p.year, date=p.date)


class ExtractionInfo(BaseModel):
    """How the triplet was produced — enough to reproduce or invalidate it."""

    run_id: str
    model: str                        # served model name
    model_revision: Optional[str] = None  # Hugging Face commit hash
    prompt_version: str
    schema_version: str = SCHEMA_VERSION
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GroundingInfo(BaseModel):
    """Deterministic checks (see llm/grounding.py). Annotates, never drops."""

    evidence_found: bool
    evidence_score: float             # 1.0 = exact match in FOCUS
    source_id: Optional[str] = None   # canonical gazetteer id, None if unresolved
    target_id: Optional[str] = None
    on_behalf_of_id: Optional[str] = None
    proposal_author_id: Optional[str] = None
    source_in_evidence: bool = False
    antecedent_found: Optional[bool] = None   # None when reference_type is explicit
    flags: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Minimum bar for use in a graph: quote found and speaker resolved."""
        return self.evidence_found and self.source_id is not None


class TripletRecord(BaseModel):
    """One line of triplets.jsonl."""

    triplet_id: str                   # "<passage_id>-t<k>", stable within a run
    triplet: StanceTriplet
    source: SourceInfo
    extraction: ExtractionInfo
    grounding: GroundingInfo


__all__ = [
    "SCHEMA_VERSION", "StanceTriplet", "ExtractionResponse", "json_schema", "response_format",
    "SourceInfo", "ExtractionInfo", "GroundingInfo", "TripletRecord",
]
