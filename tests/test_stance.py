"""
Tests for the stance extractor (C1). Fixtures are real sentences from the ENB
corpus (quoted in docs/svo_stance_audit.md) — each test pins the handling of
one documented ENB idiom.

Run with:  pytest -q
"""

import pytest

from enbgraph.models import Chunk, Provenance, SectionKind, Sentence
from enbgraph.extraction.stance import StanceExtractor, SUPPORT, OPPOSE


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="t-0001", kind=SectionKind.REPORT_ITEM, section_path=["TEST"],
        text=text,
        sentences=[Sentence(text=text, index=0, start_char=0, end_char=len(text))],
        provenance=Provenance(doc_id="t", page_start=1, page_end=1,
                              start_char=0, end_char=len(text)),
        cop="6", year=2000,
    )


@pytest.fixture(scope="module")
def ex():
    return StanceExtractor()


def _by(records, **kw):
    return [r for r in records
            if all(getattr(r, k) == v for k, v in kw.items())]


# --- compound subject + attributed object (COP-6, 2000) ---------------------
def test_compound_subject_direct_object(ex):
    rs = ex.extract(_chunk(
        "The US and EU opposed a proposal by Argentina to remove brackets."))
    assert _by(rs, subject="United States", object="Argentina",
               sign=OPPOSE, tier=1)
    assert _by(rs, subject="European Union", object="Argentina",
               sign=OPPOSE, tier=1)
    assert len(_by(rs, tier=1)) == 2


# --- coalition appositive "supported by" (COP-2, 1996) ----------------------
def test_supported_by_coalition(ex):
    rs = ex.extract(_chunk(
        "SAUDI ARABIA, supported by the REPUBLIC OF KOREA, the US, IRAN and "
        "BANGLADESH, objected to restricting the Round Table to ministers."))
    # four alignment records member -> head, sign +
    aligns = _by(rs, tier=2, sign=SUPPORT, object="Saudi Arabia")
    assert {r.subject for r in aligns} == {"South Korea", "United States",
                                           "Iran", "Bangladesh"}
    # head + co-asserting members inherit the objection as issue records
    issue = _by(rs, tier=3, sign=OPPOSE)
    assert {r.subject for r in issue} >= {"Saudi Arabia", "United States"}
    assert all("Round Table" in (r.object or "") for r in issue)


# --- "with" coalition, issue object (COP-5, 1999) ----------------------------
def test_with_coalition_issue(ex):
    rs = ex.extract(_chunk(
        "CHINA, with SAUDI ARABIA, VENEZUELA, KUWAIT, INDONESIA, and LIBYA, "
        "opposed the proposals to establish new groups."))
    aligns = _by(rs, tier=2, object="China", sign=SUPPORT)
    assert len(aligns) == 5
    issue = _by(rs, tier=3, sign=OPPOSE)
    assert {r.subject for r in issue} == {"China", "Saudi Arabia", "Venezuela",
                                          "Kuwait", "Indonesia", "Libya"}


# --- passive "opposed by": direction reversed, no co-assertion (COP-5) -------
def test_opposed_by_appositive(ex):
    rs = ex.extract(_chunk(
        "The EU, opposed by CHINA, said the GEF had provided funding."))
    opp = _by(rs, tier=2, sign=OPPOSE)
    assert len(opp) == 1 and opp[0].subject == "China" \
        and opp[0].object == "European Union"
    # 'said' is neutral; CHINA must NOT inherit any main-clause stance
    assert not _by(rs, subject="China", tier=3)
    # and the appositive's own 'opposed' must not double as a main verb
    assert not _by(rs, subject="European Union")


# --- "on behalf of": attribution stored, bloc not a subject (COP-6) ---------
def test_on_behalf_of(ex):
    rs = ex.extract(_chunk(
        "Nigeria, on behalf of the G-77/China, rejected the proposal by a key "
        "Annex I country."))
    r = _by(rs, subject="Nigeria", tier=1)
    assert r and r[0].object == "Annex I" and r[0].sign == OPPOSE
    assert r[0].attributed_to == "G-77/China"
    assert not _by(rs, subject="G-77/China")


# --- issue stance, no actor object (COP-5, 1999) -----------------------------
def test_pure_issue_record(ex):
    rs = ex.extract(_chunk(
        "AOSIS opposed technical assessment processes for individual national "
        "communications."))
    assert len(rs) == 1
    r = rs[0]
    assert (r.subject, r.sign, r.object_type, r.tier) == ("AOSIS", OPPOSE, "issue", 3)
    assert "technical assessment" in r.object


# --- neutral sentences produce nothing ---------------------------------------
def test_neutral_sentence_no_records(ex):
    rs = ex.extract(_chunk("BRAZIL noted that this issue alone could undermine "
                           "the Kyoto Protocol."))
    assert rs == []


# --- provenance round-trip: verb offsets point at the verb -------------------
def test_verb_offsets(ex):
    text = "Japan supported the draft decision."
    rs = ex.extract(_chunk(text))
    assert len(rs) == 1
    r = rs[0]
    assert text[r.verb_start:r.verb_end].lower() == "supported"


# --- regression: the three error classes found in validation (plan C5) -------
def test_and_opposed_by_variant(ex):
    """', and/but opposed by' must be the passive appositive (direction
    reversed), not a main verb (wrong-direction records in validation)."""
    rs = ex.extract(_chunk(
        "The EU, supported by Hungary, and opposed by Japan, underlined its preference."))
    rs2 = ex.extract(_chunk(
        "The EU, supported by Norway, but opposed by Saudi Arabia, suggested a workshop."))
    assert _by(rs2, subject="Saudi Arabia", object="European Union", sign=OPPOSE, tier=2)
    assert not _by(rs2, object="Saudi Arabia")
    assert _by(rs, subject="Hungary", object="European Union", sign=SUPPORT, tier=2)
    assert _by(rs, subject="Japan", object="European Union", sign=OPPOSE, tier=2)
    assert not _by(rs, subject="European Union", object="Japan")


def test_contrast_clause_scopes_are_independent(ex):
    """Actors in a ', while ...' clause must not become objects of the first
    clause's verb (produced spurious AOSIS->Libya support records)."""
    rs = ex.extract(_chunk(
        "AOSIS and Chile supported holding a workshop, while Saudi Arabia and "
        "Libya opposed this."))
    assert not _by(rs, subject="AOSIS", object="Libya")
    assert not _by(rs, subject="AOSIS", object="Saudi Arabia")
    assert _by(rs, subject="AOSIS", sign=SUPPORT, tier=3)      # issue record
    assert _by(rs, subject="Libya", sign=OPPOSE, tier=3)


def test_merged_sentences_are_segmented(ex):
    """The chunker's acronym guard can merge '...CDM. Japan said...'; the
    stance segmenter must not let next-sentence actors become objects."""
    rs = ex.extract(_chunk(
        "Norway and Austria opposed including nuclear energy in the CDM. "
        "Japan said flexibility was needed."))
    assert not _by(rs, object="Japan")
    assert _by(rs, subject="Norway", sign=OPPOSE, object_type="issue")
