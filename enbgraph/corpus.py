"""
corpus.py — load an ENB report folder into Documents, whatever the on-disk shape.

The corpus arrives in two physical forms: native PDFs (read via the ingester,
with real layout) and a page-packaged ZIP format (per-page text, no layout — we
wrap each line as a block so the chunker's text-fallback path handles it).
Scanned/image-only PDFs yield no text and are flagged, not silently dropped.

This loader is the single place that knows about those formats, shared by the
pipeline driver and the gazetteer audit tool.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .ingest.ingester import ENBIngester
from .models import BBox, Document, DocumentMeta, Page, TextBlock

PathLike = Union[str, Path]

_VOL = re.compile(r"Vol\.\s*(\d+)\s*No\.\s*(\d+)", re.IGNORECASE)
_COP = re.compile(r"COP[-\s]?(\d+)", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass
class LoadedDoc:
    name: str
    fmt: str                       # "zip" | "pdf"
    status: str                    # "ok" | "scanned" | "empty"
    document: Optional[Document] = None


def _document_from_zip(path: Path) -> Document:
    z = zipfile.ZipFile(path)
    manifest = json.loads(z.read("manifest.json"))
    pages: list[Page] = []
    for i, p in enumerate(manifest["pages"], start=1):
        text = z.read(p["text"]["path"]).decode("utf-8", "ignore")
        blocks = [
            TextBlock(text=ln, bbox=BBox(x0=0, y0=0, x1=1, y1=1), font_size=0.0)
            for ln in text.split("\n") if ln.strip()
        ]
        pages.append(Page(number=i, width=612, height=792, blocks=blocks))
    head = pages[0].text if pages else ""
    vol = _VOL.search(head)
    cop = _COP.search(head)
    yr = _YEAR.search(head)
    meta = DocumentMeta(
        doc_id=path.stem, source=path,
        vol=vol.group(1) if vol else None,
        number=vol.group(2) if vol else None,
        cop=cop.group(1) if cop else None,
        year=int(yr.group(0)) if yr else None,
    )
    return Document(meta=meta, pages=pages)


def load_document(path: PathLike) -> LoadedDoc:
    path = Path(path)
    head = path.read_bytes()[:4]
    if head.startswith(b"PK"):
        doc = _document_from_zip(path)
        status = "ok" if len(doc.full_text) > 1000 else "empty"
        return LoadedDoc(path.name, "zip", status, doc if status == "ok" else None)
    doc = ENBIngester().ingest(path)
    status = "ok" if len(doc.full_text) > 1000 else "scanned"
    return LoadedDoc(path.name, "pdf", status, doc if status == "ok" else None)


def load_corpus(folder: PathLike) -> list[LoadedDoc]:
    """Load every *.pdf in a folder, in name order."""
    return [load_document(p) for p in sorted(Path(folder).glob("*.pdf"))]


__all__ = ["LoadedDoc", "load_document", "load_corpus"]
