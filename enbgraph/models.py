"""
models.py — data contracts for the ingest + chunk infrastructure.

These models are the *boundary types* that flow between components:

    ENBIngester.ingest(path)  -> Document      (Document -> Page -> TextBlock)
    StructuralChunker.chunk(doc) -> list[Chunk]  (Chunk -> Sentence, + Provenance)

Design rules followed here:
  * Models describe WHAT the data is, never HOW it was produced. The ingester
    decides column order and boldness; the model just stores the result. This
    keeps the contracts stable while the implementations behind them change.
  * Every unit that can eventually justify a graph edge carries provenance
    (page + character span) so any downstream claim can be traced back to the
    exact text that produced it.
  * Layout signals (font size, bold, all-caps, bbox) live on TextBlock because
    the chunker needs them to recognise agenda-item headers and reading order.
    If the ingester flattened everything to a string, those signals would be
    gone and structural chunking would be impossible.

Scope: this file covers the infrastructure phase only (PDF -> structured text
-> chunks). Actor and Edge models belong to the extraction phase and will be
added in a separate module when we get there.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class BBox(BaseModel):
    """A bounding box in PDF point coordinates (origin top-left, as PyMuPDF
    returns it). Used by the ingester to band blocks into columns and sort
    them into reading order."""

    x0: float
    y0: float
    x1: float
    y1: float

    @field_validator("x1")
    @classmethod
    def _x_ordered(cls, v: float, info):
        if "x0" in info.data and v < info.data["x0"]:
            raise ValueError("x1 must be >= x0")
        return v

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2


# --------------------------------------------------------------------------- #
# Page-level structure (output of the ingester)
# --------------------------------------------------------------------------- #
class TextBlock(BaseModel):
    """One layout block from the PDF, with the typographic signals the chunker
    relies on. `column` and the position in Page.blocks together encode reading
    order; both are assigned by the ingester, not the PDF."""

    text: str
    bbox: BBox
    font_size: float = Field(..., description="Dominant font size of the block, in points.")
    font: Optional[str] = Field(default=None, description="Dominant font name, e.g. 'Times-Bold'.")
    is_bold: bool = False
    column: Optional[int] = Field(
        default=None, description="Column index assigned by the ingester (0=left, 1=right). None until resolved."
    )

    @computed_field  # serialised, because header detection keys off it
    @property
    def is_all_caps(self) -> bool:
        """True if the block's letters are all uppercase. The ENB writes Party
        names and many subheaders in caps, so this is a strong header / actor cue."""
        letters = [c for c in self.text if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    @property
    def char_len(self) -> int:
        return len(self.text)


class Page(BaseModel):
    """A single page: its blocks in reading order, plus the page geometry the
    ingester needs to find the column boundary (page mid-x)."""

    number: int = Field(..., ge=1, description="1-based page number.")
    width: float
    height: float
    blocks: list[TextBlock] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Block text joined in reading order. Relies on `blocks` already being
        ordered by the ingester (column, then top-to-bottom)."""
        return "\n".join(b.text for b in self.blocks)


# --------------------------------------------------------------------------- #
# Document-level structure
# --------------------------------------------------------------------------- #
class SourceType(str, Enum):
    """Which kind of document this is. Drives source-specific options (e.g. the
    ENB-summary region filter) in the LLM passage chunker; the rule-based
    pipeline ignores it."""

    ENB_SUMMARY = "enb_summary"      # ENB end-of-COP summary issue (~15-20 pages)
    ENB_DAILY = "enb_daily"          # ENB daily issue (~4 pages)
    UNFCCC_REPORT = "unfccc_report"  # official FCCC/... session report
    SUBMISSION = "submission"        # party / observer submission
    OTHER = "other"


class DocumentMeta(BaseModel):
    """Bibliographic metadata parsed from the ENB masthead. `year` is the
    temporal sort key for the cross-year analysis; `cop` is a label (kept as a
    string so resumed sessions like '6-2' don't break it)."""

    doc_id: str = Field(..., description="Stable id, e.g. the filename stem 'enb1297e'.")
    source: Path
    source_type: SourceType = SourceType.OTHER
    vol: Optional[str] = None
    number: Optional[str] = Field(default=None, description="ENB issue number, e.g. '97'.")
    cop: Optional[str] = Field(default=None, description="COP session label, e.g. '4'.")
    year: Optional[int] = Field(default=None, ge=1990, le=2100)
    title: Optional[str] = None
    date: Optional[str] = None


class Document(BaseModel):
    """The ingester's output: clean, structured text for one ENB report.
    This is the single hand-off point between ingest and chunk."""

    meta: DocumentMeta
    pages: list[Page] = Field(default_factory=list)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    def iter_blocks(self):
        """Yield (page_number, block) across the whole document in reading order."""
        for page in self.pages:
            for block in page.blocks:
                yield page.number, block


# --------------------------------------------------------------------------- #
# Chunk-level structure (output of the chunker)
# --------------------------------------------------------------------------- #
class SectionKind(str, Enum):
    """What part of the report a chunk came from. The chunker TAGS rather than
    silently deletes boilerplate, so dropped content stays inspectable during
    development; only REPORT_ITEM chunks feed the extractors."""

    MASTHEAD = "masthead"      # editors, donors, contact boilerplate
    HISTORY = "history"        # 'A Brief History of the FCCC...'
    REPORT_ITEM = "report_item"  # the negotiation report — the part we want
    ANALYSIS = "analysis"      # 'A Brief Analysis', 'Things to Look For'
    OTHER = "other"


class Provenance(BaseModel):
    """Pointer back to the exact source text. Character offsets are relative to
    Document.full_text so any chunk, sentence, or future edge can be located and
    shown as evidence."""

    doc_id: str
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _spans_ordered(self) -> "Provenance":
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


class Sentence(BaseModel):
    """One sentence inside a chunk. The baseline extractor works at this
    granularity; offsets are document-relative so each sentence is independently
    addressable as evidence."""

    text: str
    index: int = Field(..., ge=0, description="Position of the sentence within its chunk.")
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)


class Chunk(BaseModel):
    """The chunker's unit of meaning: one agenda-item block of the report,
    carrying its section breadcrumb, its sentences, and full provenance.

    This is the primary unit the extraction phase consumes — the influence layer
    reconstructs 'who responds to whom' *within* a chunk, which is why the
    agenda item (not the sentence) is the natural boundary."""

    chunk_id: str
    kind: SectionKind = SectionKind.REPORT_ITEM
    section_path: list[str] = Field(
        default_factory=list,
        description="Hierarchical breadcrumb, e.g. ['Report of COP-4', 'SBSTA', 'Land use change'].",
    )
    text: str
    sentences: list[Sentence] = Field(default_factory=list)
    provenance: Provenance
    # carried down from the document so a chunk is self-contained for analysis
    cop: Optional[str] = None
    year: Optional[int] = None

    @property
    def section(self) -> str:
        """Human-readable section label (last breadcrumb element)."""
        return self.section_path[-1] if self.section_path else ""

    @property
    def preview(self) -> str:
        return (self.text[:117] + "...") if len(self.text) > 120 else self.text


class Passage(BaseModel):
    """The LLM extractor's unit: a size-bounded window of consecutive sentences,
    cut without relying on document structure (so it works for any source type).

    `focus_text` is what the LLM extracts from; every kept sentence of a document
    lands in exactly one focus window. `context_before` repeats the sentences
    just before the window, shown to the LLM only to resolve references
    ("He said...") — extractions must be grounded in `focus_text`.
    Provenance offsets index Document.full_text (the raw text); `focus_text`
    itself is normalised prose, so grounding checks run against `focus_text`."""

    passage_id: str
    doc_id: str
    source_type: SourceType = SourceType.OTHER
    focus_text: str
    context_before: str = ""
    sentence_start: int = Field(..., ge=0, description="Index of the first focus sentence in the document.")
    sentence_end: int = Field(..., ge=0, description="Index one past the last focus sentence.")
    provenance: Provenance
    section_hint: Optional[str] = None
    cop: Optional[str] = None
    year: Optional[int] = None
    date: Optional[str] = None

    @property
    def n_words(self) -> int:
        return len(self.focus_text.split())


__all__ = [
    "BBox",
    "TextBlock",
    "Page",
    "SourceType",
    "DocumentMeta",
    "Document",
    "SectionKind",
    "Provenance",
    "Sentence",
    "Chunk",
    "Passage",
]
