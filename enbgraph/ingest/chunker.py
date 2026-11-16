"""
chunker.py — turn a `Document` into a list of `Chunk`s.

Responsibility: understand the report's internal structure and cut it into the
units the extractors consume — one agenda item per chunk, with sentences and
provenance attached. The design follows three decisions argued out against the
real ENB corpus:

1. SUBTRACTIVE, NOT SELECTIVE. We do not try to *find* the report by a fragile
   header that drifts across the decade (it is "REPORT OF COP-4" in 1998 but
   "COP-6 Report" in 2000). Instead we keep everything between two boundaries we
   verified are stable across all issues — the body title / first major header on
   one side, "A Brief Analysis of COP-N" on the other — and TAG the rest
   (masthead, history, analysis) rather than deleting it, so nothing is silently
   lost and misclassification stays inspectable.

2. LAYOUT-FIRST, TEXT-FALLBACK. Header detection uses typographic prominence
   (bold + large font) when the ingester supplies it, but degrades to robust
   text heuristics (all-caps, length, not-a-doc-code, not-an-enumeration) so the
   component is correct — if coarser — even on a layout-free Document.

3. EVERY CHUNK CARRIES PROVENANCE. Character offsets are relative to
   Document.full_text, so any future graph edge traces back to its source span.

(The text normalisation helpers below are intentionally minimal and local; the
fuller `ftfy`/`regex` normaliser belongs in utils.py and will replace them.)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..models import Chunk, Document, Provenance, SectionKind, Sentence
from ..utils import normalize_prose, normalize_structural


# --------------------------------------------------------------------------- #
# Sentence splitting (swappable; spaCy's senter can drop in behind this Protocol)
# --------------------------------------------------------------------------- #
class SentenceSplitter(Protocol):
    def split(self, text: str) -> list[str]: ...


class RegexSentenceSplitter:
    """A pragmatic splitter that guards the abbreviations common in ENB prose
    (titles, initials, 'Art.', 'para.', country/org abbreviations) so it does not
    break mid-sentence. Good enough for v1; replace with spaCy if recall matters."""

    _ABBREV = {
        "mr", "mrs", "ms", "dr", "prof", "art", "para", "paras", "no", "vol",
        "st", "sec", "fig", "e.g", "i.e", "etc", "cf", "al", "approx",
    }
    _BOUNDARY = re.compile(r"([.!?])\s+(?=[\"'(]?[A-Z0-9])")

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        out, start = [], 0
        for m in self._BOUNDARY.finditer(text):
            end = m.end(1)
            candidate = text[start:end].strip()
            last_word = re.split(r"[\s(]", candidate.rstrip("."))[-1].lower()
            # don't split after a known abbreviation or a single capital initial
            if last_word in self._ABBREV or re.fullmatch(r"[A-Z]", candidate[-2:-1] or ""):
                continue
            out.append(candidate)
            start = m.end()
        tail = text[start:].strip()
        if tail:
            out.append(tail)
        return out


# --------------------------------------------------------------------------- #
# Boundary detection — which region of full_text is the negotiation report
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    kind: SectionKind
    start: int
    end: int


class BoundaryDetector:
    """Splits full_text into MASTHEAD / HISTORY / REPORT_ITEM / ANALYSIS regions
    using anchors validated as stable across COP-2..COP-9. Patterns are class
    attributes so they can be tuned per-corpus without touching the algorithm."""

    TITLE = re.compile(r"(REPORT OF COP[-\s]?\d+|COP[-\s]?\d+ REPORT)", re.IGNORECASE)
    END = re.compile(r"(A BRIEF ANALYSIS|ANALYSIS OF COP[-\s]?\d+)", re.IGNORECASE)
    HISTORY = re.compile(r"A BRIEF HISTORY", re.IGNORECASE)
    DOT_LEADER = re.compile(r"\.\s*\.\s*\.\s*\.")  # table-of-contents lines
    MAJOR_FALLBACK = {"PLENARY", "OPENING PLENARY", "ORGANIZATIONAL MATTERS"}
    MIN_BODY_FRAC = 0.04  # body never starts inside the first 4% (masthead/TOC)

    def _line_at(self, text: str, pos: int) -> str:
        a = text.rfind("\n", 0, pos)
        b = text.find("\n", pos)
        return text[(a + 1 if a >= 0 else 0):(b if b >= 0 else len(text))]

    def detect(self, text: str) -> list[Region]:
        n = len(text)

        # body END = LAST analysis anchor (the first copy lives in the TOC)
        end_matches = list(self.END.finditer(text))
        body_end = end_matches[-1].start() if end_matches else n

        # body START = first body-title NOT on a dot-leader (TOC) line, before END
        body_start: Optional[int] = None
        for m in self.TITLE.finditer(text):
            if m.start() < body_end and not self.DOT_LEADER.search(self._line_at(text, m.start())):
                body_start = m.start()
                break
        # fallback (e.g. COP-2 has no 'COP-N Report' title): first major header
        if body_start is None:
            for line_start in (mm.start() + 1 for mm in re.finditer(r"\n", text)):
                line = self._line_at(text, line_start).strip().rstrip(":")
                if line in self.MAJOR_FALLBACK and line_start > self.MIN_BODY_FRAC * n:
                    body_start = line_start
                    break
        if body_start is None:  # last resort: everything up to analysis is report
            body_start = int(self.MIN_BODY_FRAC * n)

        hist = self.HISTORY.search(text)
        hist_start = hist.start() if (hist and hist.start() < body_start) else None

        regions: list[Region] = []
        if hist_start is not None:
            regions.append(Region(SectionKind.MASTHEAD, 0, hist_start))
            regions.append(Region(SectionKind.HISTORY, hist_start, body_start))
        else:
            regions.append(Region(SectionKind.MASTHEAD, 0, body_start))
        regions.append(Region(SectionKind.REPORT_ITEM, body_start, body_end))
        regions.append(Region(SectionKind.ANALYSIS, body_end, n))
        return regions


# --------------------------------------------------------------------------- #
# Header classification — is a text segment a section/agenda header, and at what level
# --------------------------------------------------------------------------- #
@dataclass
class HeaderClassifier:
    """Decide whether a segment is a header and whether it is a MAJOR body
    (PLENARY, SUBSIDIARY BODY FOR ...) or an agenda ITEM. Uses layout bold when
    present; otherwise leans on the ENB convention that headers are uppercase."""

    max_words: int = 9
    max_chars: int = 75
    _MAJOR_HINT = re.compile(
        r"\b(PLENARY|SUBSIDIARY BODY|JOINT SESSION|HIGH-LEVEL SEGMENT|"
        r"PRESIDENTIAL CEREMONY|COMMITTEE OF THE WHOLE|ORGANIZATIONAL MATTERS|"
        r"CLOSING|OPENING)\b"
    )
    # a run-in header: "CAPS PREFIX: running text..." -> the prefix is an item header
    _RUNIN = re.compile(r"^([A-Z][A-Z0-9 ,/&'’()\-]{3,65}):\s+(?=\S)")

    @staticmethod
    def _is_doccode(s: str) -> bool:
        if re.search(r"FCCC/|/(CP|SBI|SBSTA|CRP|MISC|INF|Add|Rev|L)\.?\d", s):
            return True
        noisy = sum(c.isdigit() or c in "/().;" for c in s)
        return len(s) > 0 and noisy / len(s) > 0.30

    @staticmethod
    def _is_enumeration(s: str) -> bool:
        return s.rstrip().endswith(",") or s.count(",") >= 2

    @staticmethod
    def _is_all_caps(s: str) -> bool:
        letters = [c for c in s if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    def runin_split(self, text: str) -> Optional[tuple[str, str]]:
        """If `text` begins with a run-in caps header, return (header, body)."""
        m = self._RUNIN.match(text.strip())
        if not m:
            return None
        header = m.group(1).strip()
        if self._is_doccode(header) or self._is_enumeration(header):
            return None
        body = text.strip()[m.end():]
        return header, body

    def classify(self, text: str, is_bold: bool = False) -> Optional[str]:
        """Return 'major', 'item', or None (not a header)."""
        s = text.strip()
        words = s.split()
        if not (3 <= len(s) <= self.max_chars and len(words) <= self.max_words):
            return None
        if self._is_doccode(s) or self._is_enumeration(s):
            return None
        if not (self._is_all_caps(s) or is_bold):
            return None
        if self._MAJOR_HINT.search(s.upper()) or (self._is_all_caps(s) and ":" not in s):
            return "major"
        return "item"


# --------------------------------------------------------------------------- #
# The chunker
# --------------------------------------------------------------------------- #
@dataclass
class _Segment:
    raw: str           # original block text; start/end offsets index THIS, into full_text
    clean: str         # normalize_structural(raw); used for all header detection/labels
    start: int
    end: int
    page: int
    is_bold: bool = False


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:  # pragma: no cover - interface
        ...


class StructuralChunker(BaseChunker):
    """Structure-aware chunker: tags regions, then segments the report region
    into one chunk per agenda item. `keep_non_report` controls whether masthead /
    history / analysis are emitted as coarse tagged chunks (default True, so the
    partition of the document is complete and auditable)."""

    def __init__(
        self,
        splitter: Optional[SentenceSplitter] = None,
        classifier: Optional[HeaderClassifier] = None,
        boundaries: Optional[BoundaryDetector] = None,
        keep_non_report: bool = True,
    ):
        self.splitter = splitter or RegexSentenceSplitter()
        self.classifier = classifier or HeaderClassifier()
        self.boundaries = boundaries or BoundaryDetector()
        self.keep_non_report = keep_non_report

    # -- build a flat, offset-correct stream of segments from the Document ----
    def _segments(self, doc: Document) -> list[_Segment]:
        """Flatten blocks in reading order. Offsets match Document.full_text,
        where every block is separated by a single '\\n' (page text joins blocks
        with '\\n' and pages are joined with '\\n').

        Each segment keeps the RAW block text (offsets index into it) plus a
        structurally-cleaned copy used only for header detection — so a header
        carrying a soft-hyphen or mojibake is still classified correctly while
        provenance offsets remain exact against the original text."""
        segs: list[_Segment] = []
        cursor = 0
        for page in doc.pages:
            for block in page.blocks:
                raw = block.text
                segs.append(
                    _Segment(
                        raw=raw,
                        clean=normalize_structural(raw),
                        start=cursor,
                        end=cursor + len(raw),  # offset uses RAW length
                        page=page.number,
                        is_bold=block.is_bold,
                    )
                )
                cursor += len(raw) + 1  # +1 for the joining newline
        return segs

    @staticmethod
    def _region_of(pos: int, regions: list[Region]) -> Region:
        for r in regions:
            if r.start <= pos < r.end:
                return r
        return regions[-1]

    _CONNECTOR = {"AND", "OF", "FOR", "THE", "TO", "OR", "ON", "IN", "A", "AN",
                  "WITH", "AS", "FROM", "BY", "AT"}

    def _ends_with_connector(self, text: str) -> bool:
        last = re.sub(r"[^A-Za-z]", "", text.split()[-1]).upper() if text.split() else ""
        return last in self._CONNECTOR

    def _merge_caps_runs(self, segs: list[_Segment]) -> list[_Segment]:
        """Merge consecutive pure-caps header lines ONLY when they form a wrapped
        header — detected because the running text ends on a connector word
        ('SUBSIDIARY BODY FOR SCIENTIFIC AND' + 'TECHNOLOGICAL ADVICE'). Two
        stacked but distinct headers ('REPORT OF COP-4' then 'PLENARY') are left
        separate. Enumerations and doc-codes are never merged."""
        merged: list[_Segment] = []
        i = 0
        c = self.classifier

        def mergeable(seg: _Segment) -> bool:
            return (
                c._is_all_caps(seg.clean)
                and not c._is_doccode(seg.clean)
                and not c._is_enumeration(seg.clean)
                and ":" not in seg.clean
            )

        while i < len(segs):
            cur = segs[i]
            if mergeable(cur):
                j = i + 1
                clean, raw, end = cur.clean, cur.raw, cur.end
                while (
                    j < len(segs)
                    and self._ends_with_connector(clean)
                    and mergeable(segs[j])
                    and len((clean + " " + segs[j].clean).split()) <= c.max_words
                ):
                    clean += " " + segs[j].clean
                    raw += " " + segs[j].raw
                    end = segs[j].end
                    j += 1
                merged.append(_Segment(raw=raw, clean=clean, start=cur.start,
                                       end=end, page=cur.page, is_bold=cur.is_bold))
                i = j
            else:
                merged.append(cur)
                i += 1
        return merged

    def chunk(self, doc: Document) -> list[Chunk]:
        full = doc.full_text
        regions = self.boundaries.detect(full)
        segs = self._merge_caps_runs(self._segments(doc))

        chunks: list[Chunk] = []
        counter = 0

        # body title label = the header text at the report region's start
        body_region = next(r for r in regions if r.kind == SectionKind.REPORT_ITEM)
        body_title = normalize_structural(self.boundaries._line_at(full, body_region.start)).strip()[:60]

        current_major: Optional[str] = None
        current_item: Optional[str] = None
        buf: list[_Segment] = []

        def flush():
            nonlocal counter, buf
            if not buf:
                return
            raw = "\n".join(s.raw for s in buf)
            body = normalize_prose(raw)
            if len(body) < 2:
                buf = []
                return
            path: list[str] = []
            for p in (body_title, current_major, current_item):
                if p and p not in path:
                    path.append(p)
            sents = self.splitter.split(body)
            sentences, off = [], 0
            for k, st in enumerate(sents):
                idx = body.find(st, off)
                idx = idx if idx >= 0 else off
                sentences.append(Sentence(text=st, index=k, start_char=idx, end_char=idx + len(st)))
                off = idx + len(st)
            counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.meta.doc_id}-{counter:04d}",
                    kind=SectionKind.REPORT_ITEM,
                    section_path=path,
                    text=body,
                    sentences=sentences,
                    provenance=Provenance(
                        doc_id=doc.meta.doc_id,
                        page_start=buf[0].page,
                        page_end=buf[-1].page,
                        start_char=buf[0].start,
                        end_char=buf[-1].end,
                    ),
                    cop=doc.meta.cop,
                    year=doc.meta.year,
                )
            )
            buf = []

        non_report_buckets: dict[SectionKind, list[_Segment]] = {}

        for seg in segs:
            region = self._region_of(seg.start, regions)
            if region.kind != SectionKind.REPORT_ITEM:
                if self.keep_non_report:
                    non_report_buckets.setdefault(region.kind, []).append(seg)
                continue

            # run-in header: "ITEM HEADER: running text..." — detect on the clean
            # copy, but slice the remainder from RAW so offsets remain exact.
            runin = self.classifier.runin_split(seg.clean)
            if runin:
                flush()
                current_item = runin[0]
                colon = seg.raw.find(":")
                if colon >= 0:
                    rem_raw = seg.raw[colon + 1:]
                    rem_start = seg.start + colon + 1
                else:  # clean had a colon the raw lacks (rare); fall back to whole segment
                    rem_raw, rem_start = seg.raw, seg.start
                lead = len(rem_raw) - len(rem_raw.lstrip())
                remainder = rem_raw.strip()
                if remainder:
                    start = rem_start + lead
                    buf.append(_Segment(raw=remainder, clean=normalize_structural(remainder),
                                        start=start, end=start + len(remainder),
                                        page=seg.page, is_bold=seg.is_bold))
                continue

            level = self.classifier.classify(seg.clean, is_bold=seg.is_bold)
            if level == "major":
                flush()
                current_major = seg.clean.strip()
                current_item = None
            elif level == "item":
                flush()
                current_item = seg.clean.strip()
            else:
                buf.append(seg)
        flush()

        # emit coarse tagged chunks for the non-report regions (auditable, filterable)
        if self.keep_non_report:
            for kind, segs_ in non_report_buckets.items():
                raw = "\n".join(s.raw for s in segs_)
                body = normalize_prose(raw)
                if len(body) < 2:
                    continue
                counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.meta.doc_id}-{counter:04d}",
                        kind=kind,
                        section_path=[kind.value],
                        text=body,
                        sentences=[],
                        provenance=Provenance(
                            doc_id=doc.meta.doc_id,
                            page_start=segs_[0].page,
                            page_end=segs_[-1].page,
                            start_char=segs_[0].start,
                            end_char=segs_[-1].end,
                        ),
                        cop=doc.meta.cop,
                        year=doc.meta.year,
                    )
                )
        return chunks


__all__ = [
    "SentenceSplitter",
    "RegexSentenceSplitter",
    "BoundaryDetector",
    "HeaderClassifier",
    "BaseChunker",
    "StructuralChunker",
]
