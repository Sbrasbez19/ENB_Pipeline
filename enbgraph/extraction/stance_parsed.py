"""
extraction/stance_parsed.py — dependency-parse main-clause attribution (plan C1
parser upgrade).

`ParsedStanceExtractor` subclasses `StanceExtractor` and overrides ONLY
`_main_clause` — the tier-1/3 step. Everything else (on-behalf-of masking,
coalition appositives = tier 2, clause segmentation, actor mentions) is inherited
unchanged, because the validation showed those are already reliable (tier-2 =
1.00 precision). The parser targets exactly the weak channel (tier-1 = 0.25),
whose errors are attachment and noun/verb confusion — the things a dependency
parse fixes:

  * POS filter        — only tokens tagged VERB count as stance verbs, so the
                        noun "backing"/"support"/"challenges" no longer fire.
  * nsubj / dobj      — subject and object come from the parse, not word order,
                        so cross-clause and cross-appositive mis-attachment stop.
  * conjuncts         — "The US and EU opposed X" -> both are subjects.
  * poss / agent      — "opposed Argentina's proposal" and "a proposal by
                        Argentina" resolve the object to the actor Argentina,
                        converting issue records into actor->actor edges.
  * passive           — "X, opposed by Y" is handled by the inherited appositive
                        rule; a bare passive main clause maps agent->subject.

Records carry source="parsed" so parsed and pattern runs are distinguishable and
directly comparable (re-run the manual validation with each; compare tier-1).

Setup:  pip install spacy && python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import Chunk
from .base import Mention
from .stance import StanceExtractor, StanceLexicon, StanceRecord, SUPPORT, OPPOSE

_SUBJECT_DEPS = {"nsubj", "nsubjpass"}
_OBJECT_DEPS = {"dobj", "obj", "dative", "attr", "oprd"}
_ISSUE_HEAD_POS = {"NOUN", "PROPN", "PRON", "VERB"}


class ParsedStanceExtractor(StanceExtractor):
    def __init__(self, actor_extractor=None, lexicon: Optional[StanceLexicon] = None,
                 model: str = "en_core_web_sm", max_issue_chars: int = 100):
        super().__init__(actor_extractor=actor_extractor, lexicon=lexicon,
                         max_issue_chars=max_issue_chars)
        import spacy  # lazy: only needed when this extractor is used
        # parser + tagger only; NER/lemmatizer defaults are fine, keep it light
        self.nlp = spacy.load(model, disable=["ner"])
        # lemma -> sign, from the same YAML lexicon (its sub-keys are lemmas)
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load((Path(__file__).parent.parent / "config" / "stance_verbs.yaml").read_text())
        self.lemma_sign: dict[str, int] = {}
        for lemma in cfg.get("support", {}):
            self.lemma_sign[lemma] = SUPPORT
        for lemma in cfg.get("oppose", {}):
            self.lemma_sign[lemma] = OPPOSE

    # ------------------------------------------------------------------ #
    def _mention_at(self, sm: list[Mention], lo: int, hi: int) -> Optional[Mention]:
        """The actor mention overlapping the char span [lo, hi), if any."""
        for m in sm:
            if m.start_char < hi and lo < m.end_char:
                return m
        return None

    def _subtree_span(self, tok) -> tuple[int, int]:
        lo = min(t.idx for t in tok.subtree)
        hi = max(t.idx + len(t.text) for t in tok.subtree)
        return lo, hi

    def _object_actor(self, verb, sm: list[Mention]) -> tuple[Optional[Mention], Optional[str]]:
        """Resolve the verb's object to an actor if possible, else an issue
        snippet. Actor may sit as: the direct object head; a possessive inside
        it ("Argentina's proposal"); a prep under the object ("proposal by X");
        or a prep on the VERB ("opposed a proposal by Japan" — the parse
        attaches 'by' to the verb, not to 'proposal"). ALL actor sources are
        tried before falling back to an issue snippet."""
        obj_head = None
        # a) direct/attr/dative object head + its internal modifiers
        for ch in verb.children:
            if ch.dep_ in _OBJECT_DEPS:
                obj_head = ch
                m = self._mention_at(sm, ch.idx, ch.idx + len(ch.text))
                if m:
                    return m, None
                for g in ch.children:
                    if g.dep_ == "poss":                     # "Argentina's proposal"
                        pm = self._mention_at(sm, g.idx, g.idx + len(g.text))
                        if pm:
                            return pm, None
                    if g.dep_ in ("prep", "agent"):          # "proposal by/of X"
                        for gg in g.children:
                            if gg.dep_ in ("pobj", "pcomp"):
                                pm = self._mention_at(sm, gg.idx, gg.idx + len(gg.text))
                                if pm:
                                    return pm, None
                break
        # b) prepositional / agent object on the VERB itself:
        #    "opposed a proposal BY Japan", "objected TO X", passive agent
        for ch in verb.children:
            if ch.dep_ in ("prep", "agent", "dative"):
                for g in ch.children:
                    if g.dep_ in ("pobj", "pcomp"):
                        m = self._mention_at(sm, g.idx, g.idx + len(g.text))
                        if m:
                            return m, None
        # c) no actor object -> issue snippet (prefer the object subtree)
        if obj_head is not None:
            lo, hi = self._subtree_span(obj_head)
            return None, re.sub(r"\s+", " ", verb.doc.text[verb.idx:hi]).strip(" ,.;:")[: self.max_issue_chars]
        return None, None

    def _nsubj_actors(self, verb, sm) -> list[str]:
        """Actor ids from a verb's own nsubj/nsubjpass heads (+ their conjuncts
        for compound subjects). No coordination climb here."""
        out: list[str] = []
        for h in [c for c in verb.children if c.dep_ in _SUBJECT_DEPS]:
            for t in [h, *h.conjuncts]:
                m = self._mention_at(sm, t.idx, t.idx + len(t.text)) \
                    or self._mention_at(sm, *self._subtree_span(t))
                if m:
                    out.append(m.actor_id)
        return out

    def _subjects(self, verb, sm, extra_subjects) -> list[str]:
        out = self._nsubj_actors(verb, sm)
        # Coordinated verbs ("X stressed ... and opposed ...") do not carry their
        # own nsubj; the second verb attaches to the first via dep_='conj' and
        # shares its subject. Climb the conj chain to inherit it — this is the
        # single most common ENB construction that was defeating the parser
        # (~70% of stance verbs), diagnosed from the PHILIPPINES/stressed+opposed
        # sentence. Passive/clausal subjects come through nsubjpass already.
        # Coordinated verbs ("X stressed ... and opposed ...") do not carry their
        # own nsubj; the second verb attaches via dep_='conj' and shares the
        # subject of the coordination. The parse may route through an
        # intermediate head that is NOT a subject-bearer — e.g. the transformer
        # parses "...transfer was a commitment and opposed..." as
        #   opposed --conj--> was(ccomp) --> stressed(ROOT, nsubj=PHILIPPINES),
        # so we must climb the head chain THROUGH the non-subject 'was' until a
        # verb with an actual actor subject is found, not stop at the first hop.
        if not out and verb.dep_ in ("conj", "xcomp", "ccomp", "advcl"):
            hop, guard = verb, 0
            while hop.head is not hop and guard < 6:
                hop = hop.head
                guard += 1
                inherited = self._nsubj_actors(hop, sm)
                if inherited:
                    out = inherited
                    break
                if hop.dep_ == "ROOT":
                    break
        # coalition co-asserters ("X, with A and B, ...") flow to every verb in
        # the segment, including the coordinated one — so all of {head + members}
        # take the coordinated stance, matching the parent 'with' semantics.
        for m in extra_subjects:
            out.append(m.actor_id)
        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------ #
    def _main_clause(self, chunk: Chunk, sidx: int, text: str, sm, in_mask,
                     extra_subjects, attributed) -> list[StanceRecord]:
        records: list[StanceRecord] = []
        doc = self.nlp(text)
        verb_unattached = False           # a VERB-tagged stance token we couldn't attach
        for tok in doc:
            if tok.pos_ != "VERB":
                continue                             # POS filter: kills noun uses
            sign = self.lemma_sign.get(tok.lemma_.lower())
            if sign is None or in_mask(tok.idx):
                continue
            subj_ids = self._subjects(tok, sm, extra_subjects)
            if not subj_ids:
                verb_unattached = True   # e.g. acronym mis-parse: 'opposed' as amod
                continue
            obj_m, issue = self._object_actor(tok, sm)
            vs, ve = tok.idx, tok.idx + len(tok.text)
            if obj_m is not None and obj_m.actor_id not in subj_ids:
                for s in subj_ids:
                    records.append(StanceRecord(
                        subject=s, sign=sign, verb=tok.lemma_.lower(),
                        object_type="actor", object=obj_m.actor_id, tier=1,
                        chunk_id=chunk.chunk_id, sentence_index=sidx,
                        verb_start=vs, verb_end=ve, attributed_to=attributed.get(s),
                        source="parsed",
                    ))
            else:
                snip = issue or re.sub(r"\s+", " ", text[ve:]).strip(" ,.;:")[: self.max_issue_chars]
                for s in subj_ids:
                    records.append(StanceRecord(
                        subject=s, sign=sign, verb=tok.lemma_.lower(),
                        object_type="issue", object=snip or None, tier=3,
                        chunk_id=chunk.chunk_id, sentence_index=sidx,
                        verb_start=vs, verb_end=ve, attributed_to=attributed.get(s),
                        source="parsed",
                    ))
        # Fallback ONLY on a genuine parser failure: a VERB-tagged stance token
        # that we could not attach a subject to (en_core_web_sm mis-parses ENB's
        # ALL-CAPS acronym subjects, e.g. "AOSIS opposed X" -> 'opposed' tagged
        # amod with no nsubj). We do NOT fall back merely because a lexicon
        # surface form appears, since the parser may have correctly rejected it
        # as a noun ("financial backing"). Deferred records keep source="pattern".
        if not records and verb_unattached:
            return super()._main_clause(chunk, sidx, text, sm, in_mask,
                                        extra_subjects, attributed)
        return records


__all__ = ["ParsedStanceExtractor"]