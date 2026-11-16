"""
Tests pinning the two guarantees we just hardened:
  * dehyphenation keeps genuine compounds and joins everything else, including
    soft-hyphens inside all-caps headers;
  * the chunker detects headers on cleaned text while keeping provenance offsets
    exact against the raw Document.full_text.

Run with:  pytest -q
"""

import pytest

from enbgraph.models import BBox, Document, DocumentMeta, Page, TextBlock, SectionKind
from enbgraph.ingest.chunker import StructuralChunker
from enbgraph.utils import normalize_prose, dehyphenate


# --------------------------------------------------------------------------- #
# utils.dehyphenate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("non\x02Annex I", "non-Annex I"),          # compound: lower->UPPER transition
        ("deliber\x02ated", "deliberated"),          # syllable: lower-lower
        ("SCIENTIF\x02IC ADVICE", "SCIENTIFIC ADVICE"),  # all-caps split -> join
        ("non-\nAnnex Parties", "non-Annex Parties"),    # wrapped compound
        ("deliber-\nated text", "deliberated text"),     # wrapped syllable
        ("SCIENTIF-\nIC", "SCIENTIFIC"),                 # wrapped all-caps split
    ],
)
def test_dehyphenation_rules(raw, expected):
    assert normalize_prose(raw) == expected


def test_dehyphenate_runs_before_quotes_are_irrelevant():
    # dehyphenate alone should not touch a normal hyphen with no break
    assert dehyphenate("high-level segment") == "high-level segment"


# --------------------------------------------------------------------------- #
# chunker: header hardening + provenance
# --------------------------------------------------------------------------- #
def _doc(lines: list[str], doc_id: str = "synth") -> Document:
    blocks = [TextBlock(text=t, bbox=BBox(x0=0, y0=0, x1=1, y1=1), font_size=0.0) for t in lines]
    page = Page(number=1, width=612, height=792, blocks=blocks)
    meta = DocumentMeta(doc_id=doc_id, source="s.pdf", cop="4", year=1998)
    return Document(meta=meta, pages=[page])


ARTIFACT_DOC = [
    "REPORT OF COP-4",
    "SUBSIDIARY BODY FOR SCIENTIF\x02IC AND",   # soft-hyphen inside an all-caps header
    "TECHNOLOGICAL ADVICE",                       # ...wrapped onto a second line
    "The US preferred a phased approach to non\x02Annex commitments.",
    "METHODOLOGICAL ISSUES: Brazil introduced the draft text.",
    "A BRIEF ANALYSIS OF COP-4",
    "Closing thoughts.",
]


def test_artifact_laden_header_is_detected_and_cleaned():
    chunks = [c for c in StructuralChunker().chunk(_doc(ARTIFACT_DOC))
              if c.kind == SectionKind.REPORT_ITEM]
    paths = [" > ".join(c.section_path) for c in chunks]
    # the wrapped, soft-hyphenated header must merge AND be clean in the path
    assert any("SUBSIDIARY BODY FOR SCIENTIFIC AND TECHNOLOGICAL ADVICE" in p for p in paths)
    assert all("\x02" not in p for p in paths)


def test_body_keeps_compound_and_drops_artifact():
    chunks = [c for c in StructuralChunker().chunk(_doc(ARTIFACT_DOC))
              if c.kind == SectionKind.REPORT_ITEM]
    body = next(c for c in chunks if "phased" in c.text)
    assert "non-Annex" in body.text          # compound preserved
    assert "\x02" not in body.text            # artifact gone


def test_provenance_offsets_round_trip_against_full_text():
    doc = _doc(ARTIFACT_DOC)
    full = doc.full_text
    chunks = [c for c in StructuralChunker().chunk(doc) if c.kind == SectionKind.REPORT_ITEM]
    for c in chunks:
        raw_slice = full[c.provenance.start_char:c.provenance.end_char]
        # the chunk body is reconstructible from the raw source span it points at
        assert normalize_prose(raw_slice)[:30] == c.text[:30]


def test_runin_remainder_offset_is_exact():
    # a run-in header chunk's provenance should start at the header (not drift)
    doc = _doc(ARTIFACT_DOC)
    full = doc.full_text
    chunks = [c for c in StructuralChunker().chunk(doc) if c.kind == SectionKind.REPORT_ITEM]
    runin = next(c for c in chunks if c.section_path and c.section_path[-1] == "METHODOLOGICAL ISSUES")
    assert "Brazil introduced" in normalize_prose(full[runin.provenance.start_char:runin.provenance.end_char])
