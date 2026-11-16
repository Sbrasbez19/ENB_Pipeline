"""
graph/builder.py — assemble edges into a NetworkX graph (storage + analysis substrate).

This layer is deliberately thin: it is bookkeeping, and NetworkX already does
bookkeeping well. It walks (chunk, mentions) pairs, asks the EdgeDefinition for
edges, and accumulates them — incrementing weights on repeats, registering nodes
with their actor category, tagging the graph with year/COP.

Two decisions it enforces, because NetworkX won't make them:
  * Graph type follows the edge definition: DiGraph if directed else Graph.
  * Node identity is the canonical, year-stable actor_id (e.g. 'United States'),
    never a surface form — so the 1996 graph and the 2003 graph are comparable.
"""

from __future__ import annotations

from typing import Iterable, Optional

import networkx as nx

from ..extraction.base import Extractor, Mention
from ..models import Chunk
from .base import EdgeDefinition

ChunkMentions = tuple[Chunk, list[Mention]]


class GraphBuilder:
    def __init__(self, edge_def: EdgeDefinition, max_evidence: int = 3):
        self.edge_def = edge_def
        self.max_evidence = max_evidence

    def build(
        self,
        items: Iterable[ChunkMentions],
        *,
        year: Optional[int] = None,
        cop: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> nx.Graph:
        """Build one graph from (chunk, mentions) pairs — typically one document /
        one year. Decoupled from extraction: the caller supplies mentions, so the
        builder is trivially testable with hand-made inputs."""
        G: nx.Graph = nx.DiGraph() if self.edge_def.directed else nx.Graph()
        G.graph.update(
            year=year, cop=cop, doc_id=doc_id,
            relation=self.edge_def.relation, edge_def=self.edge_def.name,
            directed=self.edge_def.directed,
        )

        for chunk, mentions in items:
            for m in mentions:
                if not G.has_node(m.actor_id):
                    G.add_node(m.actor_id, category=m.category.value)
            for e in self.edge_def.edges(chunk, mentions):
                u, v = (e.source, e.target) if e.directed else tuple(sorted((e.source, e.target)))
                if u == v:
                    continue  # no self-loops
                if G.has_edge(u, v):
                    G[u][v]["weight"] += e.weight
                    ev = G[u][v]["evidence"]
                    if e.evidence and len(ev) < self.max_evidence:
                        ev.append(e.evidence)
                else:
                    G.add_edge(u, v, weight=e.weight, relation=e.relation,
                               evidence=[e.evidence] if e.evidence else [])
                if e.sign is not None:
                    # signed bookkeeping for Bernoulli+Wilson aggregation
                    # (per-pair k,n counts; Sudhahar et al. 2015). Guarded so
                    # unsigned definitions (proximity) are unaffected.
                    d = G[u][v]
                    d["pos_w"] = d.get("pos_w", 0.0) + (e.weight if e.sign > 0 else 0.0)
                    d["neg_w"] = d.get("neg_w", 0.0) + (e.weight if e.sign < 0 else 0.0)
                    d["n_obs"] = d.get("n_obs", 0) + 1
        return G

    def build_from_chunks(
        self, chunks: Iterable[Chunk], extractor: Extractor, **graph_meta
    ) -> nx.Graph:
        """Convenience: extract mentions per chunk, then build. Keeps the common
        path one call while leaving `build` decoupled for testing."""
        return self.build(((c, extractor.extract(c)) for c in chunks), **graph_meta)


__all__ = ["GraphBuilder"]
