"""
graph/proximity.py — the baseline edge definition: textual proximity.

"Two actors mentioned within the same window are linked." This is the cheapest
edge definition and exactly the one the brief names ("proximity within the
text"); it is also the null model the later co-stance / influence layers must
beat. It is undirected and symmetric.

The `window` parameter is the substantive choice we surfaced:

  * "sentence" — actors named in the same sentence. Tight, high precision (they
    are genuinely interacting in one statement), sparser graph.
  * "chunk"    — actors named in the same agenda item. Looser: links everyone who
    spoke on a topic, capturing "engaged with the same issue" but blurring allies
    and adversaries together. Denser graph.

Per window, each unordered actor pair contributes weight 1 once (we de-duplicate
actors first), so repeated mentions of the same actor do not inflate weight; the
edge weight a builder accumulates is the count of windows in which the pair
co-occurs — a standard co-occurrence network.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator

from ..extraction.base import Mention
from ..models import Chunk
from .base import Edge, EdgeDefinition

_WINDOWS = ("sentence", "chunk")


class ProximityEdges(EdgeDefinition):
    relation = "proximity"
    directed = False

    def __init__(self, window: str = "sentence"):
        if window not in _WINDOWS:
            raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
        self.window = window

    @property
    def name(self) -> str:
        return f"proximity:{self.window}"

    def _windows(self, chunk: Chunk, mentions: list[Mention]) -> Iterator[tuple[list[Mention], str]]:
        """Yield (mentions-in-window, evidence-snippet) for each window."""
        if self.window == "chunk":
            yield mentions, chunk.preview
        else:
            for s in chunk.sentences:
                group = [m for m in mentions if s.start_char <= m.start_char < s.end_char]
                yield group, s.text

    def edges(self, chunk: Chunk, mentions: list[Mention]) -> list[Edge]:
        out: list[Edge] = []
        for group, evidence in self._windows(chunk, mentions):
            actors = sorted({m.actor_id for m in group})  # de-duplicate per window
            for a, b in combinations(actors, 2):
                out.append(
                    Edge(
                        source=a,
                        target=b,
                        relation=self.relation,
                        directed=False,
                        weight=1.0,
                        chunk_id=chunk.chunk_id,
                        evidence=evidence[:160],
                    )
                )
        return out


__all__ = ["ProximityEdges"]
