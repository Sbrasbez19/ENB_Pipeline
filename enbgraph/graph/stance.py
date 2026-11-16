"""
graph/stance.py — signed edge definitions (C2 of the stance layer plan).

Two definitions behind the existing `EdgeDefinition` seam, consuming the
`StanceRecord` stream from extraction/stance.py:

  * DirectStanceEdges — DIRECTED, signed. Records with an actor object
    (tier 1 parsed/pattern direct objects; tier 2 coalition-derived relations:
    "supported by"/"with" alignments and "opposed by" oppositions).
    This is the graph on which Dambanemuya & Horvát-style centralities
    (in-degree, authority, betweenness) are read as an influence *proxy*.

  * CoStanceEdges — UNDIRECTED, signed. The within-chunk projection of
    tier-3 issue stances: two actors taking a stance in the same agenda item
    are linked, same sign => +1 (aligned), opposite => -1 (opposed).
    Conceptually the signed one-mode projection of the actor-issue bipartite
    graph (duality: Breiger 1974); the agenda item (chunk) is the issue unit —
    the tractable approximation argued in docs/stance_implementation_plan.md.
    Implementation uses itertools.combinations rather than networkx's bipartite
    projection because per-chunk incremental emission through this seam is
    *less* code that way; equivalence with
    networkx.bipartite.generic_weighted_projected_graph is asserted by test.

Tier weights (extraction-confidence weighting, tuned on the training years —
plan C4) are applied at emission so the builder's pos_w/neg_w accumulation
carries effective counts into the Wilson aggregation (plan C3).
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

from ..extraction.stance import StanceExtractor, StanceRecord
from ..models import Chunk
from .base import Edge, EdgeDefinition
from ..extraction.base import Mention

DEFAULT_TIER_WEIGHTS: dict[int, float] = {1: 1.0, 2: 0.75, 3: 0.25}


class _StanceBase(EdgeDefinition):
    def __init__(self, extractor: Optional[StanceExtractor] = None,
                 tier_weights: Optional[dict[int, float]] = None,
                 attribute_to_bloc: bool = False):
        self.extractor = extractor or StanceExtractor()
        self.tier_weights = tier_weights or dict(DEFAULT_TIER_WEIGHTS)
        self.attribute_to_bloc = attribute_to_bloc

    def _subject(self, r: StanceRecord) -> str:
        """The 'on behalf of' modeling switch, surfaced as a flag: by default
        the speaking country keeps subjecthood (attribution retained on the
        record); with attribute_to_bloc=True the bloc becomes the subject."""
        if self.attribute_to_bloc and r.attributed_to:
            return r.attributed_to
        return r.subject

    def _sentence(self, chunk: Chunk, idx: int, limit: int = 150) -> str:
        return chunk.sentences[idx].text[:limit]


class DirectStanceEdges(_StanceBase):
    relation = "stance"
    directed = True

    @property
    def name(self) -> str:
        return "stance:direct"

    def edges(self, chunk: Chunk, mentions: list[Mention]) -> list[Edge]:
        out: list[Edge] = []
        for r in self.extractor.extract(chunk, mentions=mentions):
            if r.object_type != "actor" or r.object is None:
                continue
            subj = self._subject(r)
            if subj == r.object:
                continue
            out.append(Edge(
                source=subj, target=r.object, relation=self.relation,
                directed=True, weight=self.tier_weights.get(r.tier, 1.0),
                chunk_id=chunk.chunk_id,
                evidence=self._sentence(chunk, r.sentence_index),
                sign=r.sign, tier=r.tier,
            ))
        return out


class CoStanceEdges(_StanceBase):
    relation = "costance"
    directed = False

    @property
    def name(self) -> str:
        return "costance:chunk"

    def edges(self, chunk: Chunk, mentions: list[Mention]) -> list[Edge]:
        records = [r for r in self.extractor.extract(chunk, mentions=mentions)
                   if r.object_type == "issue"]
        # stancers of this agenda item: one (actor, sign) tuple each, keeping
        # the first supporting sentence for evidence. An actor appearing with
        # BOTH signs contributes both tuples — the resulting contradictory
        # pairs are exactly what pos_w/neg_w accumulation + the contested flag
        # (plan C3) are designed to absorb.
        stancers: dict[tuple[str, int], int] = {}
        for r in records:
            key = (self._subject(r), r.sign)
            stancers.setdefault(key, r.sentence_index)
        out: list[Edge] = []
        w = self.tier_weights.get(3, 1.0)
        for (a, sa), (b, sb) in combinations(sorted(stancers), 2):
            if a == b:
                continue
            ia, ib = stancers[(a, sa)], stancers[(b, sb)]
            ev = (f"{a}({'+' if sa > 0 else '-'}): {self._sentence(chunk, ia, 60)} | "
                  f"{b}({'+' if sb > 0 else '-'}): {self._sentence(chunk, ib, 60)}")
            out.append(Edge(
                source=a, target=b, relation=self.relation, directed=False,
                weight=w, chunk_id=chunk.chunk_id, evidence=ev,
                sign=+1 if sa == sb else -1, tier=3,
            ))
        return out


__all__ = ["DirectStanceEdges", "CoStanceEdges", "DEFAULT_TIER_WEIGHTS"]
