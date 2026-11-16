"""
graph/base.py — contracts for the graph phase.

The deliberate separation (argued with the supervisor's brief in mind): deciding
WHAT an edge means is the research contribution and lives here, in plain Python,
with no NetworkX dependency. STORING and analysing the graph is NetworkX's job
and lives in builder.py. This file is the seam between them.

An `EdgeDefinition` takes a chunk and its actor mentions and yields `Edge`s. The
baseline is `ProximityEdges`; co-stance / support / opposition will be further
implementations behind the same interface, so the builder never changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..extraction.base import Mention
from ..models import Chunk


@dataclass(frozen=True)
class Edge:
    """One edge instance emitted from a single window. The builder accumulates
    repeated instances into a weight. `evidence` is a short text snippet kept for
    inspection, consistent with the pipeline's traceability theme.

    `sign` (+1/-1) and `tier` (extraction-confidence tier) are set by signed
    edge definitions (stance layer); None for unsigned definitions (proximity).
    When sign is present the builder additionally accumulates per-pair
    pos_w / neg_w / n_obs — the (k, n) bookkeeping the Bernoulli+Wilson
    aggregation consumes (Sudhahar et al. 2015)."""

    source: str
    target: str
    relation: str
    directed: bool
    weight: float = 1.0
    chunk_id: str = ""
    evidence: str = ""
    sign: Optional[int] = None
    tier: Optional[int] = None


class EdgeDefinition(ABC):
    """Maps (chunk, mentions) -> edges. Declares its own relation label and
    whether it is directed; the builder reads `directed` to choose Graph vs
    DiGraph, so the edge semantics — not the builder — decide graph type."""

    relation: str = "edge"
    directed: bool = False

    @property
    def name(self) -> str:
        """Human-readable identifier stored on the graph (e.g. 'proximity:sentence')."""
        return self.relation

    @abstractmethod
    def edges(self, chunk: Chunk, mentions: list[Mention]) -> list[Edge]:  # pragma: no cover
        ...


__all__ = ["Edge", "EdgeDefinition"]
