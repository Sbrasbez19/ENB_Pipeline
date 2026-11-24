"""
Tests for the deterministic grounding checks (enbgraph/llm/grounding.py).
"""

import pytest

from enbgraph.extraction.gazetteer import Gazetteer
from enbgraph.llm.grounding import check, evidence_score, resolve
from enbgraph.llm.schema import StanceTriplet
from enbgraph.models import Passage, Provenance

FOCUS = ("Canada, with the support of Japan, and opposed by the G-77/China, proposed replacing "
         "the existing decision. She opposed further delays.")
CONTEXT = "TUVALU called for a review of adequacy."


@pytest.fixture(scope="module")
def gaz():
    return Gazetteer()


def passage():
    return Passage(passage_id="d-p0001", doc_id="d", focus_text=FOCUS, context_before=CONTEXT,
                   sentence_start=0, sentence_end=2,
                   provenance=Provenance(doc_id="d", page_start=1, page_end=1, start_char=0, end_char=1))


def triplet(**kw):
    base = dict(source_actor="G-77/China", source_mention="the G-77/China", source_role="party",
                reference_type="explicit", verb="opposed", polarity="oppose", target_type="actor",
                target="Canada",
                evidence="Canada, with the support of Japan, and opposed by the G-77/China, "
                         "proposed replacing the existing decision.")
    base.update(kw)
    return StanceTriplet.model_validate(base)   # validates the Literal fields at runtime


def test_exact_evidence_and_resolution(gaz):
    g = check(triplet(), passage(), gaz)
    assert g.evidence_found and g.evidence_score == 1.0
    assert g.source_id == "G-77/China" and g.target_id == "Canada"
    assert g.source_in_evidence and g.passed and g.flags == []


def test_evidence_typography_and_whitespace_are_normalised():
    assert evidence_score("Canada,  with the\nsupport of Japan", FOCUS) == 1.0
    assert evidence_score("She opposed “further” delays.".replace("“", "").replace("”", ""), FOCUS) == 1.0


def test_hallucinated_evidence_is_flagged(gaz):
    g = check(triplet(evidence="The G-77/China strongly opposed Canada on finance."), passage(), gaz)
    assert not g.evidence_found and "evidence_not_in_focus" in g.flags and not g.passed


def test_near_verbatim_evidence_is_fuzzy(gaz):
    ev = "Canada, with the support of Japan, and opposed by the G-77/China, proposed replacing the existing decision.)"
    g = check(triplet(evidence=ev), passage(), gaz)
    assert g.evidence_found and "evidence_fuzzy_match" in g.flags


def test_anonymous_target_and_self_loop(gaz):
    assert "target_anonymous" in check(triplet(target=None), passage(), gaz).flags
    assert "self_loop" in check(triplet(target="G-77"), passage(), gaz).flags   # alias of the source


def test_unresolved_and_ambiguous_actors(gaz):
    assert "source_unresolved" in check(triplet(source_actor="Atlantis"), passage(), gaz).flags
    assert "source_ambiguous" in check(triplet(source_actor="Canada and Japan"), passage(), gaz).flags
    assert resolve("the US", gaz) == ("United States", False)


def test_pronoun_antecedent(gaz):
    t = triplet(source_actor="Tuvalu", source_mention="She", reference_type="pronoun",
                antecedent_evidence="TUVALU", target_type="issue", target="further delays",
                evidence="She opposed further delays.")
    g = check(t, passage(), gaz)
    assert g.antecedent_found and g.source_in_evidence and g.source_id == "Tuvalu"
    bad = check(t.model_copy(update={"antecedent_evidence": "SAMOA"}), passage(), gaz)
    assert bad.antecedent_found is False and "antecedent_not_found" in bad.flags


def test_officer_role_flagged(gaz):
    assert "role_officer" in check(triplet(source_role="officer"), passage(), gaz).flags
