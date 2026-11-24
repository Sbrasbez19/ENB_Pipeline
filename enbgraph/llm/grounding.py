"""
llm/grounding.py — deterministic checks on each extracted triplet.

These run before any judge model and answer the objective questions:
  * is the evidence quote actually in the FOCUS text?          (hallucination)
  * do the actor names resolve to canonical gazetteer ids?     (normalisation)
  * is the speaker visible in the quote, or (for a pronoun) is
    its antecedent really in the text?                          (attribution)

They ANNOTATE (flags + canonical ids), never drop: the decision of what to
keep belongs to later stages, and the flags are themselves evaluation data.
"""

from __future__ import annotations

import difflib
import re
from typing import Optional

from ..extraction.gazetteer import Gazetteer
from ..models import Passage
from ..utils import normalize_typography
from .schema import GroundingInfo, StanceTriplet

FUZZY_THRESHOLD = 0.9


def _norm(text: str) -> str:
    """Compare quotes modulo typography, whitespace and case."""
    return re.sub(r"\s+", " ", normalize_typography(text)).strip().casefold()


def evidence_score(evidence: str, focus: str) -> float:
    """1.0 if the quote occurs verbatim (after normalisation); otherwise the
    share of the quote covered by its longest common run with the focus text."""
    ev, fo = _norm(evidence), _norm(focus)
    if not ev:
        return 0.0
    if ev in fo:
        return 1.0
    m = difflib.SequenceMatcher(None, fo, ev, autojunk=False).find_longest_match(
        0, len(fo), 0, len(ev))
    return m.size / len(ev)


def resolve(name: Optional[str], gazetteer: Gazetteer) -> tuple[Optional[str], bool]:
    """Map a free-text actor name to one canonical id.
    Returns (id, ambiguous): id is None when nothing or several actors match."""
    if not name:
        return None, False
    ids = list(dict.fromkeys(m.actor_id for m in gazetteer.find(name, "q")))
    if len(ids) == 1:
        return ids[0], False
    return None, len(ids) > 1


def _actor_ids(text: str, gazetteer: Gazetteer) -> set[str]:
    return {m.actor_id for m in gazetteer.find(text, "q")}


def check(t: StanceTriplet, passage: Passage, gazetteer: Gazetteer) -> GroundingInfo:
    flags: list[str] = []

    score = evidence_score(t.evidence, passage.focus_text)
    found = score >= FUZZY_THRESHOLD
    if not found:
        flags.append("evidence_not_in_focus")
    elif score < 1.0:
        flags.append("evidence_fuzzy_match")

    source_id, amb = resolve(t.source_actor, gazetteer)
    if source_id is None:
        flags.append("source_ambiguous" if amb else "source_unresolved")

    target_id = None
    if t.target_type == "actor":
        if t.target is None:
            flags.append("target_anonymous")
        else:
            target_id, amb = resolve(t.target, gazetteer)
            if target_id is None:
                flags.append("target_ambiguous" if amb else "target_unresolved")
            elif target_id == source_id:
                flags.append("self_loop")

    behalf_id, _ = resolve(t.on_behalf_of, gazetteer)
    if t.on_behalf_of and behalf_id is None:
        flags.append("on_behalf_of_unresolved")
    author_id, _ = resolve(t.proposal_author, gazetteer)

    # attribution: the speaker must be visible in the quote, either by name or
    # through the pronoun/description the model says it resolved
    in_evidence = bool(source_id and source_id in _actor_ids(t.evidence, gazetteer)) \
        or (_norm(t.source_mention) in _norm(t.evidence) if t.source_mention else False)
    if not in_evidence:
        flags.append("source_not_in_evidence")

    antecedent_found: Optional[bool] = None
    if t.reference_type != "explicit":
        text = passage.context_before + " " + passage.focus_text
        antecedent_found = bool(
            t.antecedent_evidence
            and _norm(t.antecedent_evidence) in _norm(text)
            and (source_id is None or source_id in _actor_ids(t.antecedent_evidence, gazetteer))
        )
        if not antecedent_found:
            flags.append("antecedent_not_found")

    if t.source_role != "party":
        flags.append(f"role_{t.source_role}")

    return GroundingInfo(
        evidence_found=found, evidence_score=round(score, 3),
        source_id=source_id, target_id=target_id,
        on_behalf_of_id=behalf_id, proposal_author_id=author_id,
        source_in_evidence=in_evidence, antecedent_found=antecedent_found, flags=flags,
    )


__all__ = ["check", "resolve", "evidence_score", "FUZZY_THRESHOLD"]
