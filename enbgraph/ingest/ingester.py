"""
ingester.py — turn one PDF into a structured `Document`.

Responsibility: extraction only. The ingester gets clean, correctly-ordered,
structured text out of a PDF and nothing more — it does not normalise unicode,
dehyphenate, or strip boilerplate (those belong to utils/chunker). Keeping it
faithful means the structural signals the chunker needs survive intact.

    BaseIngester            abstract contract: ingest(path) -> Document
      └─ TextLayerIngester  generic PyMuPDF extraction + two-column reading order
           └─ ENBIngester   adds ENB masthead parsing (vol / no / COP / year)

Today there is one real source (native-text ENB PDFs), so we instantiate
ENBIngester directly. A factory only becomes worthwhile if a second source
(e.g. webcast transcripts) appears, at which point it slots in behind the same
BaseIngester interface without touching anything downstream.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union, cast

import fitz  # PyMuPDF

from ..models import BBox, Document, DocumentMeta, Page, TextBlock

PathLike = Union[str, Path]

# PyMuPDF span flag bit for bold (2**4). See span["flags"] in get_text("dict").
_BOLD_FLAG = 1 << 4


class BaseIngester(ABC):
    """The single hand-off contract for the ingest stage. Any source format
    (PDF now, transcripts later) implements this and returns the same Document,
    so the rest of the pipeline never knows which subclass produced it."""

    @abstractmethod
    def ingest(self, path: PathLike) -> Document:  # pragma: no cover - interface
        ...


class TextLayerIngester(BaseIngester):
    """Generic extractor for born-digital PDFs via PyMuPDF.

    Two parameters tune layout handling without changing the contract:
      span_ratio  - a block at least this fraction of page width is treated as
                    'spanning' (title / full-width header) and used to split the
                    page into horizontal bands.
      n_columns   - number of body columns (2 for the ENB).
    """

    def __init__(self, span_ratio: float = 0.65, n_columns: int = 2):
        self.span_ratio = span_ratio
        self.n_columns = n_columns

    # -- public -----------------------------------------------------------
    def ingest(self, path: PathLike) -> Document:
        path = Path(path)
        with fitz.open(path) as pdf:
            pages = [
                self._build_page(pdf.load_page(i), i + 1)
                for i in range(pdf.page_count)
            ]
            meta = self._build_meta(path, pages)
        return Document(meta=meta, pages=pages)

    # -- page / block extraction -----------------------------------------
    def _build_page(self, page: "fitz.Page", number: int) -> Page:
        raw = cast("dict[str, Any]", page.get_text("dict"))
        blocks = [
            self._to_block(b)
            for b in raw["blocks"]
            if b.get("type") == 0  # 0 = text block, 1 = image block
        ]
        blocks = [b for b in blocks if b.text.strip()]
        width = page.rect.width
        ordered = self._reading_order(blocks, width)
        return Page(number=number, width=width, height=page.rect.height, blocks=ordered)

    def _to_block(self, raw_block: dict) -> TextBlock:
        """Aggregate PyMuPDF's nested block -> lines -> spans into one TextBlock,
        recording the dominant font size/name and whether the block is mostly bold
        (weighted by character count, so a stray bold word doesn't flip it)."""
        line_texts: list[str] = []
        size_chars: dict[float, int] = {}
        font_chars: dict[str, int] = {}
        bold_chars = total_chars = 0

        for line in raw_block.get("lines", []):
            spans = line.get("spans", [])
            line_texts.append("".join(s["text"] for s in spans))
            for s in spans:
                n = len(s["text"])
                total_chars += n
                size_chars[round(s["size"], 1)] = size_chars.get(round(s["size"], 1), 0) + n
                font_chars[s["font"]] = font_chars.get(s["font"], 0) + n
                if (s["flags"] & _BOLD_FLAG) or "bold" in s["font"].lower():
                    bold_chars += n

        dominant_size = max(size_chars, key=lambda k: size_chars[k]) if size_chars else 0.0
        dominant_font = max(font_chars, key=lambda k: font_chars[k]) if font_chars else None
        is_bold = total_chars > 0 and bold_chars / total_chars > 0.6

        x0, y0, x1, y1 = raw_block["bbox"]
        return TextBlock(
            text="\n".join(line_texts),
            bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
            font_size=dominant_size,
            font=dominant_font,
            is_bold=is_bold,
        )

    # -- reading order (the load-bearing part for two-column layouts) ------
    def _reading_order(self, blocks: list[TextBlock], page_width: float) -> list[TextBlock]:
        """Order blocks the way a human reads a two-column page.

        Algorithm (a lightweight XY-cut):
          1. Blocks wider than `span_ratio * page_width` are 'spanning' (titles,
             full-width headers). Sorted by vertical position, they act as
             dividers that cut the page into horizontal bands.
          2. Within each band the remaining blocks are assigned to a column by
             their horizontal centre, then read left column top-to-bottom, then
             right column top-to-bottom.
          3. The spanning divider is emitted between the band above it and the
             band below it.
        This handles the common ENB shape (a full-width title/section header
        followed by two columns) without a heavy layout-analysis dependency.
        """
        if not blocks:
            return []

        span_threshold = self.span_ratio * page_width
        mid_x = page_width / 2

        spanning = sorted(
            (b for b in blocks if b.bbox.width >= span_threshold),
            key=lambda b: b.bbox.y0,
        )
        columnar = [b for b in blocks if b.bbox.width < span_threshold]
        for b in columnar:
            b.column = 0 if b.bbox.x_center < mid_x else 1

        # band index of a y position = number of dividers above it
        divider_ys = [s.bbox.y_center for s in spanning]

        def band_of(y: float) -> int:
            return sum(1 for dy in divider_ys if y > dy)

        ordered: list[TextBlock] = []
        for band in range(len(spanning) + 1):
            band_blocks = [b for b in columnar if band_of(b.bbox.y_center) == band]
            for col in range(self.n_columns):
                col_blocks = sorted(
                    (b for b in band_blocks if b.column == col),
                    key=lambda b: b.bbox.y0,
                )
                ordered.extend(col_blocks)
            if band < len(spanning):
                spanning[band].column = None  # spanning blocks belong to no column
                ordered.append(spanning[band])
        return ordered

    # -- metadata ---------------------------------------------------------
    def _build_meta(self, path: Path, pages: list[Page]) -> DocumentMeta:
        """Base metadata: just the stable id and source. Subclasses enrich it."""
        return DocumentMeta(doc_id=path.stem, source=path)


class ENBIngester(TextLayerIngester):
    """Earth Negotiations Bulletin specialisation: parses the masthead on page 1
    for volume / issue number / COP session / year / date."""

    _VOL_NO = re.compile(r"Vol\.\s*(\d+)\s*No\.\s*(\d+)", re.IGNORECASE)
    _COP = re.compile(r"COP[-\s]?(\d+)", re.IGNORECASE)
    _YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
    _DATE = re.compile(r"\b\d{1,2}\s+[A-Z][a-z]+\s+(?:19|20)\d{2}\b")

    def _build_meta(self, path: Path, pages: list[Page]) -> DocumentMeta:
        meta = super()._build_meta(path, pages)
        head = pages[0].text if pages else ""

        if m := self._VOL_NO.search(head):
            meta.vol, meta.number = m.group(1), m.group(2)
        if m := self._COP.search(head):
            meta.cop = m.group(1)
        if m := self._DATE.search(head):
            meta.date = m.group(0)
            ym = self._YEAR.search(m.group(0))  # prefer the year inside the date line
            meta.year = int(ym.group(0)) if ym else meta.year
        elif m := self._YEAR.search(head):
            meta.year = int(m.group(0))
        return meta


__all__ = ["BaseIngester", "TextLayerIngester", "ENBIngester"]
