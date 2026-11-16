"""
Tests for ParsedStanceExtractor (the spaCy dependency-parse tier-1/3 path).

These SKIP automatically unless spaCy AND en_core_web_sm are installed — so they
are inert in the build sandbox and run in full on your machine, where they are
the verification step for the parser upgrade. Each test targets a failure class
the pattern extractor exhibited in validation, asserting the parse fixes it.

    pip install spacy && python -m spacy download en_core_web_sm
    pytest tests/test_stance_parsed.py -q
"""

import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm", disable=["ner"])
except Exception:  # model not present
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)

from enbgraph.models import Chunk, Provenance, SectionKind, Sentence
from enbgraph.extraction.stance_parsed import ParsedStanceExtractor
from enbgraph.extraction.stance import SUPPORT, OPPOSE


@pytest.fixture(scope="module")
def ex():
    return ParsedStanceExtractor()


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="t", kind=SectionKind.REPORT_ITEM, section_path=["T"], text=text,
        sentences=[Sentence(text=text, index=0, start_char=0, end_char=len(text))],
        provenance=Provenance(doc_id="t", page_start=1, page_end=1, start_char=0, end_char=len(text)),
        cop="6", year=2000,
    )


def _by(rs, **kw):
    return [r for r in rs if all(getattr(r, k) == v for k, v in kw.items())]


# --- source tag -------------------------------------------------------------
def test_records_are_tagged_parsed(ex):
    rs = ex.extract(_chunk("Japan opposed the timetable."))
    assert rs and all(r.source == "parsed" for r in rs)


# --- POS filter: noun 'backing'/'support' must NOT fire (was a tier-1 error) -
def test_noun_backing_does_not_trigger(ex):
    rs = ex.extract(_chunk(
        "The lack of progress by Annex I Parties on financial backing was noted by China."))
    assert not _by(rs, subject="Annex I")          # 'backing' is a noun here


def test_noun_support_phrase_does_not_trigger(ex):
    rs = ex.extract(_chunk("Zimbabwe called for increased support for adaptation projects."))
    assert not _by(rs, verb="support", object_type="actor")


# --- compound subject via conjuncts -----------------------------------------
def test_compound_subject(ex):
    rs = ex.extract(_chunk("The US and the EU opposed a proposal by Argentina."))
    assert _by(rs, subject="United States", object="Argentina", sign=OPPOSE, tier=1)
    assert _by(rs, subject="European Union", object="Argentina", sign=OPPOSE, tier=1)


# --- object resolution: 'proposal by X' and possessive 'X's proposal' -------
def test_object_proposal_by_actor(ex):
    rs = ex.extract(_chunk("China opposed a proposal by Japan."))
    assert _by(rs, subject="China", object="Japan", sign=OPPOSE, tier=1)


def test_object_possessive_actor(ex):
    rs = ex.extract(_chunk("Switzerland supported Canada's text."))
    assert _by(rs, subject="Switzerland", object="Canada", sign=SUPPORT, tier=1)


# --- attachment: a second clause's actor must not become the object ---------
def test_no_cross_clause_object(ex):
    rs = ex.extract(_chunk("Canada supported the text, and Iran offered to host the meeting."))
    assert not _by(rs, subject="Canada", object="Iran")


# --- issue object stays tier 3 ----------------------------------------------
def test_pure_issue_is_tier3(ex):
    rs = ex.extract(_chunk("AOSIS opposed technical assessment processes."))
    assert _by(rs, subject="AOSIS", sign=OPPOSE, object_type="issue", tier=3)


# --- inherited tier-2 appositive still works (not overridden) ---------------
def test_appositive_still_tier2(ex):
    rs = ex.extract(_chunk("Saudi Arabia, supported by Kuwait, objected to the text."))
    assert _by(rs, subject="Kuwait", object="Saudi Arabia", sign=SUPPORT, tier=2)


# --- inherited on-behalf-of still works -------------------------------------
def test_on_behalf_still_attributed(ex):
    rs = ex.extract(_chunk("Nigeria, on behalf of the G-77/China, opposed a proposal by Japan."))
    r = _by(rs, subject="Nigeria", tier=1)
    assert r and r[0].object == "Japan" and r[0].attributed_to == "G-77/China"


# --- coordination: "X stressed ... and opposed ..." — the second verb inherits
#     the subject via conj (the construction that caused ~70% fallback) --------
def test_coordinated_verb_inherits_subject(ex):
    rs = ex.extract(_chunk(
        "The PHILIPPINES, with SAUDI ARABIA and CHINA, stressed that technology "
        "transfer was a commitment and opposed linking it to the CDM."))
    # 'opposed' must attribute to the PHILIPPINES (inherited) plus the 'with'
    # co-asserters, NOT fall back / not drop the head.
    opp = _by(rs, verb="oppose")
    subs = {r.subject for r in opp}
    assert "Philippines" in subs
    assert {"Saudi Arabia", "China"} <= subs        # co-asserters flow through
    assert opp and all(r.source == "parsed" for r in opp)  # parsed, not fallback


def test_plain_coordination_inherits_subject(ex):
    rs = ex.extract(_chunk("The EU supported the text and opposed the timetable."))
    assert _by(rs, subject="European Union", verb="oppose")
    assert _by(rs, subject="European Union", verb="support")