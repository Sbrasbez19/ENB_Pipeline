"""
passages.py — format-agnostic chunking for the LLM extractor.

The rule-based `StructuralChunker` cuts ENB summary reports into agenda items
using ENB-specific anchors, which breaks on other layouts (ENB dailies, UNFCCC
official reports, submissions). The LLM extractor does not need agenda items:
it needs bounded windows of text with enough preceding context to resolve
references. This module produces those windows from ANY `Document`:

    1. drop running headers/footers (blocks repeated at the top/bottom of many pages)
    2. optionally keep only a source-specific region (ENB summary: the report body)
    3. join blocks into one normalised text stream (text flows across columns and
       pages, so sentences are split on the stream, not per block)
    4. pack sentences greedily into focus windows of at most `max_words`,
       preferring to break at paragraph ends / headings once `min_words` is reached
    5. attach the preceding `context_sentences` sentences as reference context

Invariant: every kept sentence lands in exactly one focus window.

Provenance is block-granular: a passage's offsets span the raw blocks its
sentences come from (normalisation changes lengths, so sentence-exact raw
offsets are not recoverable). Grounding checks therefore run on `focus_text`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from ..models import Document, Passage, Provenance, SectionKind, SourceType
from ..utils import get_logger, normalize_prose, normalize_structural
from .chunker import BoundaryDetector, HeaderClassifier, RegexSentenceSplitter, SentenceSplitter

log = get_logger(__name__)

_ENB = re.compile(r"Earth\s+Negotiations\s+Bulletin", re.IGNORECASE)
_FCCC = re.compile(r"\bFCCC/")
_TERMINAL = re.compile(r"[.!?:;][\"')\]]*$")


# --------------------------------------------------------------------------- #
# Source-type detection
# --------------------------------------------------------------------------- #
def detect_source_type(doc: Document, daily_max_pages: int = 6) -> SourceType:
    """Guess the document type from its first page. ENB issues carry their
    masthead; dailies are short (~4 pages), summaries long. Official UNFCCC
    documents carry an 'FCCC/...' symbol. Anything else is OTHER."""
    head = doc.pages[0].text if doc.pages else ""
    if _ENB.search(head):
        return SourceType.ENB_DAILY if doc.num_pages <= daily_max_pages else SourceType.ENB_SUMMARY
    if _FCCC.search(head):
        return SourceType.UNFCCC_REPORT
    return SourceType.OTHER


def text_quality(text: str) -> float:
    """Share of non-space characters that are letters. Garbled font encodings
    (e.g. 'UNITED NATIONS' extracted as "70+6'& 0#6+105") push this down."""
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


# --------------------------------------------------------------------------- #
# Internal stream representation
# --------------------------------------------------------------------------- #
@dataclass
class _Block:
    raw: str
    start: int          # offset into Document.full_text
    end: int
    page: int
    is_bold: bool
    y0: float
    y1: float
    page_height: float


@dataclass
class _Unit:
    """One sentence (or a standalone heading) of the normalised stream."""
    text: str
    first_block: int
    last_block: int
    para_end: bool      # ends where a block ends with terminal punctuation
    is_header: bool
    section: Optional[str]


def _blocks(doc: Document) -> list[_Block]:
    """Flatten blocks with offsets matching Document.full_text (blocks joined by
    one '\\n' within a page, pages joined by one '\\n')."""
    out, cursor = [], 0
    for page in doc.pages:
        for b in page.blocks:
            out.append(_Block(b.text, cursor, cursor + len(b.text), page.number,
                              b.is_bold, b.bbox.y0, b.bbox.y1, page.height))
            cursor += len(b.text) + 1
    return out


def _header_key(text: str) -> str:
    """Normalise a block for repetition detection: digits vary (page numbers),
    case and spacing are noise."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


class PassageChunker:
    """Cut any Document into size-bounded, context-carrying passages.

    Parameters
    ----------
    max_words / min_words : focus window size bounds (words ~ 0.75 tokens^-1)
    context_sentences     : how many preceding sentences to show as context
    header_min_frac       : a block repeated (digits normalised) at the top or
                            bottom of at least this fraction of pages (and at
                            least 2 pages) is treated as a running header/footer
                            and dropped
    margin_frac           : what counts as top/bottom of the page
    region_filter         : for ENB summaries, keep only the report body
                            (drops masthead, history, analysis)
    """

    def __init__(
        self,
        max_words: int = 600,
        min_words: int = 250,
        context_sentences: int = 3,
        header_min_frac: float = 0.3,  # ENB alternates even/odd-page headers, so each
                                       # variant covers only ~half the pages
        margin_frac: float = 0.15,
        region_filter: bool = True,
        splitter: Optional[SentenceSplitter] = None,
        classifier: Optional[HeaderClassifier] = None,
    ):
        if min_words > max_words:
            raise ValueError("min_words must be <= max_words")
        self.max_words = max_words
        self.min_words = min_words
        self.context_sentences = context_sentences
        self.header_min_frac = header_min_frac
        self.margin_frac = margin_frac
        self.region_filter = region_filter
        self.splitter = splitter or RegexSentenceSplitter()
        self.classifier = classifier or HeaderClassifier()

    def params(self) -> dict:
        """Chunker settings, recorded in run manifests for reproducibility."""
        return {
            "max_words": self.max_words, "min_words": self.min_words,
            "context_sentences": self.context_sentences,
            "header_min_frac": self.header_min_frac, "margin_frac": self.margin_frac,
            "region_filter": self.region_filter,
        }

    # -- step 1: running headers / footers -----------------------------------
    def running_blocks(self, blocks: list[_Block], n_pages: int) -> set[int]:
        if n_pages < 2:
            return set()
        threshold = max(2, math.ceil(self.header_min_frac * n_pages))
        pages_of: dict[str, set[int]] = {}
        for b in blocks:
            in_margin = (b.y0 <= self.margin_frac * b.page_height
                         or b.y1 >= (1 - self.margin_frac) * b.page_height)
            if in_margin and len(b.raw) <= 300:
                pages_of.setdefault(_header_key(b.raw), set()).add(b.page)
        repeated = {k for k, pages in pages_of.items() if len(pages) >= threshold}
        return {i for i, b in enumerate(blocks)
                if len(b.raw) <= 300 and _header_key(b.raw) in repeated}

    # -- step 2: source-specific region ---------------------------------------
    def _region(self, doc: Document) -> Optional[tuple[int, int]]:
        if self.region_filter and doc.meta.source_type == SourceType.ENB_SUMMARY:
            for r in BoundaryDetector().detect(doc.full_text):
                if r.kind == SectionKind.REPORT_ITEM:
                    return r.start, r.end
        return None

    # -- step 3: normalised stream -> sentence units ----------------------------
    def _units(self, blocks: list[_Block]) -> list[_Unit]:
        units: list[_Unit] = []
        section: Optional[str] = None
        buf: list[str] = []          # normalised text pieces of the current run
        buf_blocks: list[int] = []   # block index of each piece

        def flush_run(para_end_at_last: bool):
            """Split the accumulated run into sentences, mapping each back to
            the blocks it overlaps via the running character offsets."""
            nonlocal buf, buf_blocks
            if not buf:
                return
            spans, text, pos = [], "", 0
            for piece, bi in zip(buf, buf_blocks):
                if text and text.endswith("-") and piece[:1].islower() and text[-2:-1].isalpha():
                    text = text[:-1] + piece      # hyphen split across blocks
                else:
                    text = (text + " " + piece) if text else piece
                spans.append((len(text) - len(piece), len(text), bi))
            pos = 0
            sents = self.splitter.split(text)
            for k, s in enumerate(sents):
                idx = text.find(s, pos)
                idx = pos if idx < 0 else idx
                end = idx + len(s)
                pos = end
                over = [bi for (a, z, bi) in spans if a < end and z > idx] or [spans[-1][2]]
                # a sentence ends a paragraph if it ends where its last block ends
                last_span = next(sp for sp in spans if sp[2] == over[-1])
                para_end = (end >= last_span[1] - 1) and bool(_TERMINAL.search(s))
                if k == len(sents) - 1 and para_end_at_last:
                    para_end = True
                units.append(_Unit(s, over[0], over[-1], para_end, False, section))
            buf, buf_blocks = [], []

        for i, b in enumerate(blocks):
            clean = normalize_structural(b.raw)
            if not clean.strip():
                continue
            level = self.classifier.classify(clean, is_bold=b.is_bold)
            if level is not None:
                flush_run(para_end_at_last=True)
                section = clean.replace("\n", " ").strip()
                units.append(_Unit(section, i, i, True, True, section))
                continue
            buf.append(normalize_prose(b.raw))
            buf_blocks.append(i)
        flush_run(para_end_at_last=True)
        return units

    # -- step 4/5: windows ----------------------------------------------------
    def _windows(self, units: list[_Unit]) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        start, words = 0, 0
        for i, u in enumerate(units):
            w = len(u.text.split())
            # break BEFORE this unit if it would overflow, or before a heading
            # once the window is big enough
            if i > start and (words + w > self.max_words
                              or (u.is_header and words >= self.min_words)):
                windows.append((start, i))
                start, words = i, 0
            words += w
            # break AFTER a paragraph end once the window is big enough
            if u.para_end and words >= self.min_words:
                windows.append((start, i + 1))
                start, words = i + 1, 0
        if start < len(units):
            windows.append((start, len(units)))
        # fold a tiny trailing window into its predecessor when it fits
        if len(windows) >= 2:
            (a, b), (c, d) = windows[-2], windows[-1]
            tail = sum(len(u.text.split()) for u in units[c:d])
            prev = sum(len(u.text.split()) for u in units[a:b])
            if tail < self.min_words // 2 and prev + tail <= self.max_words:
                windows[-2:] = [(a, d)]
        return windows

    # -- public -----------------------------------------------------------------
    def chunk(self, doc: Document) -> list[Passage]:
        blocks = _blocks(doc)
        drop = self.running_blocks(blocks, doc.num_pages)
        region = self._region(doc)
        kept = [b for i, b in enumerate(blocks)
                if i not in drop and (region is None or region[0] <= b.start < region[1])]
        if not kept:
            return []

        quality = text_quality(" ".join(b.raw for b in kept))
        if quality < 0.7:
            log.warning("%s: low text quality (%.2f letters/char) - possible garbled font encoding",
                        doc.meta.doc_id, quality)

        units = self._units(kept)
        passages: list[Passage] = []
        for k, (a, z) in enumerate(self._windows(units)):
            focus = units[a:z]
            ctx = units[max(0, a - self.context_sentences):a]
            b0, b1 = kept[focus[0].first_block], kept[focus[-1].last_block]
            section = next((u.section for u in reversed(focus) if u.section), None) \
                or (ctx[-1].section if ctx else None)
            passages.append(Passage(
                passage_id=f"{doc.meta.doc_id}-p{k + 1:04d}",
                doc_id=doc.meta.doc_id,
                source_type=doc.meta.source_type,
                focus_text=" ".join(u.text for u in focus),
                context_before=" ".join(u.text for u in ctx),
                sentence_start=a,
                sentence_end=z,
                provenance=Provenance(doc_id=doc.meta.doc_id, page_start=b0.page,
                                      page_end=max(b0.page, b1.page),
                                      start_char=b0.start, end_char=max(b0.end, b1.end)),
                section_hint=section,
                cop=doc.meta.cop, year=doc.meta.year, date=doc.meta.date,
            ))
        return passages


__all__ = ["PassageChunker", "detect_source_type", "text_quality"]
