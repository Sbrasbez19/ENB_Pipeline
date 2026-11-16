"""
extraction/stance.py — signed stance extraction (C1 of the stance layer plan).

Produces `StanceRecord`s, not edges: linguistic extraction (what did the
sentence assert?) is separated from graph semantics (what edge does that
justify?), the same contract discipline used for mentions vs. edges throughout
the pipeline. Both `DirectStanceEdges` and `CoStanceEdges` will consume these
records; provenance fields feed the webapp's evidence highlighting.

Method: lexicon-signed stance verbs (config/stance_verbs.yaml, mirroring the
manually curated verb lists of Sudhahar, Veltri & Cristianini 2015, Appendix
Table 8) + explicit handling of ENB's relational idioms, identified in the
corpus audit (docs/svo_stance_audit.md):

  * coalition appositive   "X, with A, B and C, opposed …"
                           "X, supported/backed/joined by A and B, …"
  * passive opposition     "X, opposed by Y, …"        (direction Y→X, sign −)
  * bloc attribution       "Nigeria, on behalf of the G-77/China, rejected …"
  * compound subjects      "The US and EU opposed …"

This implementation is PATTERN-BASED (tier source="pattern"): sentence-local
regex + gazetteer mention geometry. The plan's C1→C2 upgrade swaps in spaCy
dependency attachment on machines where the model is installable (blocked in
this sandbox); records carry `source` so parsed and pattern records can be
distinguished and compared.

Tier semantics (extraction confidence, weighted in aggregation):
  1 = direct: stance verb with an actor object after it
  2 = coalition-derived actor relation (alignment / opposed-by / inherited)
  3 = issue-directed stance (no actor object) — consumed only by projection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from ..models import Chunk
from ..utils import load_yaml
from .actors import ActorExtractor
from .base import Mention

_CONFIG = Path(__file__).parent.parent / "config" / "stance_verbs.yaml"

SUPPORT = +1
OPPOSE = -1

# appositive heads: sign of the alignment they assert, and whether members
# join the main clause's subject set (inherit the head's stance)
_APPOSITIVE_KINDS = {
    "with": (SUPPORT, True),
    "supported by": (SUPPORT, True),
    "backed by": (SUPPORT, True),
    "joined by": (SUPPORT, True),
    "opposed by": (OPPOSE, False),   # Y opposes head; Y does NOT co-assert head's stance
}
_APPOS_RE = re.compile(
    r",\s*(?:and\s+|but\s+)?(with|supported by|backed by|joined by|opposed by)\s+", re.IGNORECASE
)
# clause boundaries: contrast clauses get independent subject/object scopes, and
# a period followed by a capital inside one "sentence" marks a sentence merge
# (the chunker's acronym guard over-merges in the UPPERCASE era: "...CDM. Japan
# said..."). Segmenting here is local to stance extraction and leaves the
# validated chunker (and its provenance guarantees) untouched.
_CLAUSE_RE = re.compile(r";\s+|,\s+(?:while|whereas)\s+|\.\s+(?=[A-Z])")
_BEHALF_RE = re.compile(r",?\s*(?:speaking\s+)?on behalf of\s+(?:the\s+)?", re.IGNORECASE)
_LIST_SEP_RE = re.compile(r"^\s*(?:,|and|,\s*and)\s*$|^\s*$")


@dataclass(frozen=True)
class StanceRecord:
    subject: str                      # canonical actor_id asserting the stance
    sign: int                         # +1 support / -1 oppose
    verb: str                         # lemma from the lexicon, or the appositive head
    object_type: str                  # "actor" | "issue"
    object: Optional[str]             # actor_id, or a short issue snippet
    tier: int                         # 1 | 2 | 3 (see module docstring)
    chunk_id: str
    sentence_index: int
    verb_start: int                   # char offsets within the sentence text
    verb_end: int
    attributed_to: Optional[str] = None   # bloc, when "on behalf of" applies
    source: str = "pattern"


class StanceLexicon:
    """Compiled support/oppose verb matcher from the YAML lexicon. One regex,
    longest-form-first (same discipline as the gazetteer), mapping each matched
    surface form back to (lemma, sign)."""

    def __init__(self, path: Optional[Path] = None):
        cfg = load_yaml(path or _CONFIG)
        self._by_form: dict[str, tuple[str, int]] = {}
        for lemma, forms in cfg.get("support", {}).items():
            for f in forms:
                self._by_form[f.lower()] = (lemma, SUPPORT)
        for lemma, forms in cfg.get("oppose", {}).items():
            for f in forms:
                self._by_form[f.lower()] = (lemma, OPPOSE)
        alts = sorted(self._by_form, key=len, reverse=True)
        self._re = re.compile(r"\b(" + "|".join(map(re.escape, alts)) + r")\b", re.IGNORECASE)

    def finditer(self, text: str):
        for m in self._re.finditer(text):
            lemma, sign = self._by_form[m.group(1).lower()]
            yield m.start(), m.end(), lemma, sign


class StanceExtractor:
    """chunk -> list[StanceRecord]. Reuses the gazetteer ActorExtractor for
    mention geometry; no new actor logic."""

    def __init__(self, actor_extractor: Optional[ActorExtractor] = None,
                 lexicon: Optional[StanceLexicon] = None,
                 max_issue_chars: int = 100):
        self.actors = actor_extractor or ActorExtractor()
        self.lexicon = lexicon or StanceLexicon()
        self.max_issue_chars = max_issue_chars

    # ------------------------------------------------------------------ #
    def extract(self, chunk: Chunk,
                mentions: Optional[list[Mention]] = None) -> list[StanceRecord]:
        """`mentions` may be supplied by callers that already ran the actor
        extractor (the graph builder does); otherwise they are computed here."""
        if mentions is None:
            mentions = self.actors.extract(chunk)
        out: list[StanceRecord] = []
        for sent in chunk.sentences:
            sm = sorted(
                (m for m in mentions
                 if sent.start_char <= m.start_char < sent.end_char),
                key=lambda m: m.start_char,
            )
            if not sm:
                continue
            out.extend(self._sentence(chunk, sent.index, sent.text,
                                      [self._localise(m, sent.start_char) for m in sm]))
        return out

    @staticmethod
    def _localise(m: Mention, sent_start: int) -> Mention:
        return Mention(
            actor_id=m.actor_id, surface=m.surface, category=m.category,
            start_char=m.start_char - sent_start, end_char=m.end_char - sent_start,
            chunk_id=m.chunk_id, is_pronoun=m.is_pronoun, source=m.source,
        )

    # ------------------------------------------------------------------ #
    def _sentence(self, chunk: Chunk, sidx: int, text: str,
                  sm: list[Mention]) -> list[StanceRecord]:
        """Split into clause segments (contrast clauses and merged sentences
        get independent subject/object scopes), then process each segment."""
        records: list[StanceRecord] = []
        bounds = [0] + [m.end() for m in _CLAUSE_RE.finditer(text)] + [len(text)]
        for a, b in zip(bounds, bounds[1:]):
            seg = text[a:b]
            seg_m = [Mention(actor_id=m.actor_id, surface=m.surface,
                             category=m.category, start_char=m.start_char - a,
                             end_char=m.end_char - a, chunk_id=m.chunk_id,
                             is_pronoun=m.is_pronoun, source=m.source)
                     for m in sm if a <= m.start_char < b]
            if seg_m:
                for r in self._segment(chunk, sidx, seg, seg_m, a):
                    records.append(replace(r, verb_start=r.verb_start + a,
                                           verb_end=r.verb_end + a))
        return records

    def _segment(self, chunk: Chunk, sidx: int, text: str,
                 sm: list[Mention], seg_offset: int = 0) -> list[StanceRecord]:
        records: list[StanceRecord] = []
        masked: list[tuple[int, int]] = []       # spans excluded from main-clause logic
        extra_subjects: list[Mention] = []       # coalition members that co-assert
        attributed: dict[str, str] = {}          # speaker actor_id -> bloc actor_id

        def in_mask(pos: int) -> bool:
            return any(a <= pos < b for a, b in masked)

        def actor_at(pos: int, tol: int = 3) -> Optional[Mention]:
            for m in sm:
                if abs(m.start_char - pos) <= tol:
                    return m
            return None

        def actor_list_from(pos: int) -> tuple[list[Mention], int]:
            """Consume 'A, B, C and D' starting at pos; return members and end."""
            members: list[Mention] = []
            cur = pos
            while True:
                nxt = actor_at(cur)
                if nxt is None:
                    # allow a determiner before the actor ("the US")
                    lead = re.match(r"\s*(?:the\s+)?", text[cur:], re.IGNORECASE)
                    nxt = actor_at(cur + (lead.end() if lead else 0))
                if nxt is None:
                    break
                members.append(nxt)
                cur = nxt.end_char
                sep = re.match(r"\s*(?:,\s*(?:and\s+)?|\s+and\s+)", text[cur:])
                if not sep:
                    break
                probe = cur + sep.end()
                lead2 = re.match(r"\s*(?:the\s+)?", text[probe:], re.IGNORECASE)
                if actor_at(probe) is None and \
                        actor_at(probe + (lead2.end() if lead2 else 0)) is None:
                    break                    # separator leads nowhere: don't consume it
                cur = probe
            return members, cur

        # -- 1) "on behalf of": speaker keeps subjecthood; bloc is attribution --
        for bm in _BEHALF_RE.finditer(text):
            speaker = None
            for m in sm:
                if m.end_char <= bm.start():
                    speaker = m
            bloc = actor_at(bm.end())
            if speaker and bloc:
                attributed[speaker.actor_id] = bloc.actor_id
                end = bloc.end_char
                paren = re.match(r"\s*\([^)]{0,50}\)", text[end:])
                if paren:                      # "... Group of 77 and China (G-77/China)"
                    end += paren.end()
                masked.append((bm.start(), end))

        # -- 2) coalition / opposed-by appositives ---------------------------
        for am in _APPOS_RE.finditer(text):
            if in_mask(am.start()):
                continue
            head = None
            for m in sm:
                if m.end_char <= am.start() + 1 and not in_mask(m.start_char):
                    head = m
            if head is None:
                continue
            members, end = actor_list_from(am.end())
            if not members:
                continue
            kind = am.group(1).lower()
            sign, co_assert = _APPOSITIVE_KINDS[kind]
            for mem in members:
                if mem.actor_id == head.actor_id:
                    continue
                records.append(StanceRecord(
                    subject=mem.actor_id, sign=sign, verb=kind,
                    object_type="actor", object=head.actor_id, tier=2,
                    chunk_id=chunk.chunk_id, sentence_index=sidx,
                    verb_start=am.start(1), verb_end=am.end(1),
                    attributed_to=attributed.get(mem.actor_id),
                ))
                if co_assert:
                    extra_subjects.append(mem)
            masked.append((am.start(), end))

        # -- 3) main-clause stance verbs -------------------------------------
        records.extend(self._main_clause(chunk, sidx, text, sm, in_mask,
                                         extra_subjects, attributed))
        return records

    def _main_clause(self, chunk, sidx, text, sm, in_mask, extra_subjects, attributed):
        """Tier-1/3 extraction from the main clause: attribute each stance verb
        to subject actor(s) and (if present) object actor(s), else emit an issue
        record. This is the ONLY step the parsed subclass overrides — the
        appositive (tier-2) and on-behalf-of handling above are already at 1.00
        precision and are reused unchanged. Base implementation is positional
        (pattern)."""
        records: list[StanceRecord] = []
        for vs, ve, lemma, sign in self.lexicon.finditer(text):
            if in_mask(vs):
                continue
            subjects = [m for m in sm
                        if m.end_char <= vs and not in_mask(m.start_char)]
            subjects += [m for m in extra_subjects if m.end_char <= vs]
            if not subjects:
                continue
            # compound subjects: keep ALL distinct pre-verb actors (pattern
            # approximation of spaCy conjuncts; parser upgrade refines this)
            subj_ids = list(dict.fromkeys(m.actor_id for m in subjects))
            objects = [m for m in sm
                       if m.start_char >= ve and not in_mask(m.start_char)]
            obj_ids = [i for i in dict.fromkeys(m.actor_id for m in objects)
                       if i not in subj_ids]
            if obj_ids:
                for s in subj_ids:
                    for o in obj_ids:
                        records.append(StanceRecord(
                            subject=s, sign=sign, verb=lemma,
                            object_type="actor", object=o, tier=1,
                            chunk_id=chunk.chunk_id, sentence_index=sidx,
                            verb_start=vs, verb_end=ve,
                            attributed_to=attributed.get(s),
                        ))
            else:
                issue = re.sub(r"\s+", " ", text[ve:]).strip(" ,.;:")[: self.max_issue_chars]
                for s in subj_ids:
                    records.append(StanceRecord(
                        subject=s, sign=sign, verb=lemma,
                        object_type="issue", object=issue or None, tier=3,
                        chunk_id=chunk.chunk_id, sentence_index=sidx,
                        verb_start=vs, verb_end=ve,
                        attributed_to=attributed.get(s),
                    ))
        return records


__all__ = ["StanceRecord", "StanceLexicon", "StanceExtractor", "SUPPORT", "OPPOSE"]
