"""
Tests for the actor extractor. The headline guarantees:
  * full names match across the case shift (CHINA in the 90s, China later);
  * short acronym aliases match case-sensitively (US matches 'US', never 'us');
  * blocs and aliases canonicalise to stable IDs;
  * the procedural stoplist is excluded;
  * pronouns resolve to the nearest preceding actor.

Run with:  pytest -q
"""

import pytest

from enbgraph.models import BBox, Chunk, Provenance, SectionKind, Sentence
from enbgraph.extraction.actors import ActorExtractor, actor_counts
from enbgraph.extraction.base import ActorCategory
from enbgraph.extraction.coref import PronounResolver


def _chunk(text: str, cid: str = "t-0001") -> Chunk:
    return Chunk(
        chunk_id=cid,
        kind=SectionKind.REPORT_ITEM,
        section_path=["TEST"],
        text=text,
        sentences=[Sentence(text=text, index=0, start_char=0, end_char=len(text))],
        provenance=Provenance(doc_id="t", page_start=1, page_end=1, start_char=0, end_char=len(text)),
        cop="4",
        year=1998,
    )


@pytest.fixture(scope="module")
def extractor():
    return ActorExtractor()


def _ids(mentions):
    return {m.actor_id for m in mentions}


def test_uppercase_era_names_match(extractor):
    ids = _ids(extractor.extract(_chunk("CHINA and SAUDI ARABIA opposed the proposal by JAPAN.")))
    assert {"China", "Saudi Arabia", "Japan"} <= ids


def test_titlecase_era_names_match(extractor):
    ids = _ids(extractor.extract(_chunk("China and Saudi Arabia opposed the proposal by Japan.")))
    assert {"China", "Saudi Arabia", "Japan"} <= ids


def test_us_acronym_matches_but_pronoun_us_does_not(extractor):
    # 'US' (uppercase) -> United States; 'us' (the word) -> nothing
    hit = extractor.extract(_chunk("The US made a submission."))
    assert "United States" in _ids(hit)
    miss = extractor.extract(_chunk("The chair asked us to reconsider the issue."))
    assert "United States" not in _ids(miss)


def test_aliases_canonicalise_to_stable_ids(extractor):
    ids = _ids(extractor.extract(_chunk("The Russian Federation and the EC, with the UK, agreed.")))
    assert {"Russia", "European Union", "United Kingdom"} <= ids


def test_blocs_and_longest_match_wins(extractor):
    ms = extractor.extract(_chunk("G-77/China stressed equity; AOSIS supported the target."))
    ids = _ids(ms)
    assert "G-77/China" in ids and "AOSIS" in ids
    # 'China' inside 'G-77/China' must not be emitted as a separate actor here
    assert not any(m.surface == "China" for m in ms)


def test_stoplist_excluded(extractor):
    ids = _ids(extractor.extract(_chunk("The SBSTA forwarded the text to the COP under the UNFCCC.")))
    assert ids == set()


def test_orgs_excluded_by_default_included_on_request():
    base = ActorExtractor()
    withorg = ActorExtractor(include_orgs=True)
    text = "The IPCC presented findings to the GEF."
    assert _ids(base.extract(_chunk(text))) == set()
    assert {"IPCC", "GEF"} <= _ids(withorg.extract(_chunk(text)))


def test_pronoun_resolves_to_last_actor():
    ex = ActorExtractor(coref=PronounResolver())
    ms = ex.extract(_chunk("Tuvalu introduced the draft. She urged rapid adoption."))
    pron = [m for m in ms if m.is_pronoun]
    assert pron and pron[0].actor_id == "Tuvalu"


def test_actor_counts_aggregates():
    ex = ActorExtractor()
    ms = ex.extract(_chunk("Japan and Brazil spoke; Japan then responded again."))
    counts = actor_counts(ms)
    assert counts["Japan"] == 2 and counts["Brazil"] == 1
