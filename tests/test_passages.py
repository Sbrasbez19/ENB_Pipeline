"""
Tests for the format-agnostic passage chunker (enbgraph/ingest/passages.py):
  * every kept sentence lands in exactly one focus window, in order
  * windows respect the size bounds; context repeats the preceding sentences
  * running headers/footers are dropped; hyphens split across blocks are joined
  * headings become section hints; ENB summaries keep only the report body
  * source-type detection
"""

import re
from pathlib import Path

from enbgraph.ingest.passages import PassageChunker, detect_source_type, text_quality
from enbgraph.models import BBox, Document, DocumentMeta, Page, SourceType, TextBlock

H = 792.0  # page height


def block(text, y0=300.0, bold=False):
    return TextBlock(text=text, bbox=BBox(x0=50, y0=y0, x1=300, y1=y0 + 40),
                     font_size=10.0, is_bold=bold)


def make_doc(pages, source_type=SourceType.OTHER, doc_id="d1"):
    return Document(
        meta=DocumentMeta(doc_id=doc_id, source=Path("x.pdf"), source_type=source_type),
        pages=[Page(number=i + 1, width=612, height=H, blocks=bl) for i, bl in enumerate(pages)],
    )


def sentences(n, start=0):
    return [f"Sentence {k} reports that the EU supported agenda item {k} at length." for k in range(start, start + n)]


def test_every_sentence_in_exactly_one_window():
    sents = sentences(60)
    # 6 paragraphs of 10 sentences, 2 per page
    paras = [" ".join(sents[i:i + 10]) for i in range(0, 60, 10)]
    doc = make_doc([[block(paras[0], 200), block(paras[1], 400)],
                    [block(paras[2], 200), block(paras[3], 400)],
                    [block(paras[4], 200), block(paras[5], 400)]])
    ps = PassageChunker(max_words=60, min_words=25).chunk(doc)
    assert len(ps) > 1
    joined = " ".join(p.focus_text for p in ps)
    assert joined == " ".join(sents)                      # no gap, no duplicate, in order
    assert ps[0].sentence_start == 0
    for a, b in zip(ps, ps[1:]):
        assert a.sentence_end == b.sentence_start         # contiguous ranges


def test_size_bounds_and_context():
    doc = make_doc([[block(" ".join(sentences(40)))]])
    ch = PassageChunker(max_words=60, min_words=25, context_sentences=2)
    ps = ch.chunk(doc)
    for p in ps:
        assert p.n_words <= 60
    for prev, cur in zip(ps, ps[1:]):
        last_two = re.findall(r"Sentence \d+[^.]*\.", prev.focus_text)[-2:]
        assert cur.context_before == " ".join(last_two)
    assert ps[0].context_before == ""


def test_running_headers_dropped():
    pages = [[block(f"Earth Negotiations Bulletin Vol. 12 No. 97 Page {i}", y0=10),
              block(" ".join(sentences(3, start=10 * i)))] for i in range(1, 5)]
    doc = make_doc(pages)
    text = " ".join(p.focus_text for p in PassageChunker().chunk(doc))
    assert "Negotiations Bulletin" not in text
    assert "Sentence 10 " in text and "Sentence 40 " in text


def test_body_block_repeated_mid_page_is_kept():
    # identical text repeated, but in the middle of the page: not a running header
    pages = [[block("The EU supported the text.", y0=400)] for _ in range(4)]
    text = " ".join(p.focus_text for p in PassageChunker().chunk(make_doc(pages)))
    assert text.count("The EU supported the text.") == 4


def test_hyphen_split_across_blocks_is_joined():
    doc = make_doc([[block("The EU supported the negoti-", 200), block("ation text proposed by Japan.", 400)]])
    text = PassageChunker().chunk(doc)[0].focus_text
    assert "negotiation text" in text


def test_heading_becomes_section_hint():
    doc = make_doc([[block("ADAPTATION", 200), block(" ".join(sentences(3)), 300)]])
    ps = PassageChunker().chunk(doc)
    assert ps[0].section_hint == "ADAPTATION"


def test_enb_summary_keeps_only_report_body():
    pages = [
        [block("Earth Negotiations Bulletin Vol. 12 No. 97 published by IISD.", 200),
         block("A BRIEF HISTORY OF THE FCCC", 300),
         block("The history says the Convention entered into force in 1994.", 400)],
        [block("REPORT OF COP-4", 200), block(" ".join(sentences(4)), 300)],
        [block("A BRIEF ANALYSIS OF COP-4", 200),
         block("Analysts thought the meeting was a modest success.", 300)],
    ]
    doc = make_doc(pages, source_type=SourceType.ENB_SUMMARY)
    text = " ".join(p.focus_text for p in PassageChunker().chunk(doc))
    assert "Sentence 0 " in text
    assert "Analysts" not in text and "entered into force" not in text
    # the same document as OTHER is not filtered
    doc.meta.source_type = SourceType.OTHER
    text_all = " ".join(p.focus_text for p in PassageChunker().chunk(doc))
    assert "Analysts" in text_all


def test_provenance_and_metadata():
    doc = make_doc([[block(" ".join(sentences(3)))], [block(" ".join(sentences(3, 3)))]], doc_id="enb999")
    doc.meta.year, doc.meta.cop = 2000, "6"
    ps = PassageChunker(max_words=1000, min_words=500).chunk(doc)
    assert len(ps) == 1
    p = ps[0]
    assert p.passage_id == "enb999-p0001" and p.doc_id == "enb999"
    assert (p.provenance.page_start, p.provenance.page_end) == (1, 2)
    assert p.year == 2000 and p.cop == "6"
    assert doc.full_text[p.provenance.start_char:p.provenance.start_char + 10] == "Sentence 0"


def test_empty_document():
    assert PassageChunker().chunk(make_doc([[]])) == []


def test_detect_source_type():
    enb = [block("Earth Negotiations Bulletin A Reporting Service", 300)]
    assert detect_source_type(make_doc([enb] + [[block("x")]] * 3)) == SourceType.ENB_DAILY
    assert detect_source_type(make_doc([enb] + [[block("x")]] * 12)) == SourceType.ENB_SUMMARY
    assert detect_source_type(make_doc([[block("Distr. GENERAL FCCC/CP/2001/5")]])) == SourceType.UNFCCC_REPORT
    assert detect_source_type(make_doc([[block("A national submission")]])) == SourceType.OTHER


def test_text_quality_flags_garbled_text():
    assert text_quality("UNITED NATIONS framework convention") > 0.9
    assert text_quality("70+6'& 0#6+105") < 0.5
