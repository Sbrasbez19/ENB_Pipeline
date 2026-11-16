"""
Tests for the signed stance edge definitions (C2).

Includes the equivalence test of CoStanceEdges' pairwise projection against
networkx.bipartite.generic_weighted_projected_graph — the library stays the
correctness oracle for the projection algebra without being forced into the
per-chunk emission seam.

Run with:  pytest -q
"""

import networkx as nx
import pytest

from enbgraph.models import Chunk, Provenance, SectionKind, Sentence
from enbgraph.graph.builder import GraphBuilder
from enbgraph.graph.stance import CoStanceEdges, DirectStanceEdges
from enbgraph.extraction.actors import ActorExtractor


def _chunk(*sentences: str, cid: str = "c1") -> Chunk:
    text = " ".join(sentences)
    sents, pos = [], 0
    for i, s in enumerate(sentences):
        start = text.index(s, pos)
        sents.append(Sentence(text=s, index=i, start_char=start, end_char=start + len(s)))
        pos = start + len(s)
    return Chunk(
        chunk_id=cid, kind=SectionKind.REPORT_ITEM, section_path=["T"], text=text,
        sentences=sents,
        provenance=Provenance(doc_id="d", page_start=1, page_end=1,
                              start_char=0, end_char=len(text)),
        cop="6", year=2000,
    )


def _build(edge_def, *chunks):
    ex = ActorExtractor()
    return GraphBuilder(edge_def).build((c, ex.extract(c)) for c in chunks)


# ---------------------------------------------------------------- direct ----
def test_direct_is_directed_and_signed():
    G = _build(DirectStanceEdges(),
               _chunk("The US and EU opposed a proposal by Argentina."))
    assert G.is_directed()
    assert G.has_edge("United States", "Argentina") and G.has_edge("European Union", "Argentina")
    assert not G.has_edge("Argentina", "United States")
    d = G["United States"]["Argentina"]
    assert d["neg_w"] == 1.0 and d["pos_w"] == 0.0 and d["n_obs"] == 1


def test_direct_sign_accumulation_pos_and_neg():
    c1 = _chunk("Japan supported Brazil.")
    c2 = _chunk("Japan supported Brazil.", cid="c2")
    c3 = _chunk("Japan opposed Brazil.", cid="c3")
    G = _build(DirectStanceEdges(), c1, c2, c3)
    d = G["Japan"]["Brazil"]
    assert d["pos_w"] == 2.0 and d["neg_w"] == 1.0 and d["n_obs"] == 3
    assert d["weight"] == 3.0            # total magnitude still accumulates


def test_coalition_alignment_gets_tier2_weight():
    G = _build(DirectStanceEdges(),
               _chunk("SAUDI ARABIA, supported by KUWAIT, objected to the text."))
    d = G["Kuwait"]["Saudi Arabia"]
    assert d["pos_w"] == pytest.approx(0.75) and d["n_obs"] == 1


def test_attribute_to_bloc_flag():
    c = _chunk("Nigeria, on behalf of the G-77/China, rejected the proposal by Japan.")
    G_default = _build(DirectStanceEdges(), c)
    assert G_default.has_edge("Nigeria", "Japan")
    G_bloc = _build(DirectStanceEdges(attribute_to_bloc=True), c)
    assert G_bloc.has_edge("G-77/China", "Japan")
    assert not G_bloc.has_edge("Nigeria", "Japan")


# -------------------------------------------------------------- costance ----
WORKED = _chunk(
    "The MARSHALL ISLANDS and JAMAICA supported detailed and rigorous reporting.",
    "AOSIS supported rigorous review of communications.",
    "SAUDI ARABIA, with KUWAIT, opposed mandatory formats.",
)


def test_costance_worked_example_counts():
    """3 supporters + 2 opposers in one agenda item:
    C(3,2)+C(2,2) = 4 positive pairs, 3*2 = 6 negative pairs."""
    G = _build(CoStanceEdges(), WORKED)
    assert not G.is_directed()
    pos = [(u, v) for u, v, d in G.edges(data=True) if d["pos_w"] > 0]
    neg = [(u, v) for u, v, d in G.edges(data=True) if d["neg_w"] > 0]
    assert len(pos) == 4 and len(neg) == 6
    assert G.number_of_edges() == 10
    d = G["Marshall Islands"]["Jamaica"]
    assert d["pos_w"] == pytest.approx(0.25)     # tier-3 default weight


def test_costance_equivalent_to_networkx_bipartite_projection():
    """The pairwise loop must reproduce the signed one-mode projection that
    networkx.bipartite.generic_weighted_projected_graph computes with a
    sign-product weight function (single shared issue per chunk)."""
    from networkx.algorithms import bipartite
    G = _build(CoStanceEdges(), WORKED)

    B = nx.Graph()
    stance = {"Marshall Islands": +1, "Jamaica": +1, "AOSIS": +1,
              "Saudi Arabia": -1, "Kuwait": -1}
    B.add_nodes_from(stance, bipartite=0)
    B.add_node("issue:c1", bipartite=1)
    for a, s in stance.items():
        B.add_edge(a, "issue:c1", sign=s)

    def w(BG, u, v):
        shared = set(BG[u]) & set(BG[v])
        return sum(BG[u][i]["sign"] * BG[v][i]["sign"] for i in shared)

    P = bipartite.generic_weighted_projected_graph(B, list(stance), weight_function=w)
    for u, v, d in P.edges(data=True):
        assert G.has_edge(u, v), f"missing {u}-{v}"
        ours = +1 if G[u][v]["pos_w"] > 0 else -1
        assert ours == d["weight"], f"sign mismatch on {u}-{v}"
    assert G.number_of_edges() == P.number_of_edges()


def test_costance_contradictory_actor_yields_both_signs():
    c = _chunk("CANADA supported the mechanism.",
               "CANADA opposed the timetable.",
               "NORWAY supported the mechanism.")
    G = _build(CoStanceEdges(), c)
    d = G["Canada"]["Norway"]
    assert d["pos_w"] > 0 and d["neg_w"] > 0    # aligned once, opposed once


def test_costance_no_cross_chunk_pairs():
    c1 = _chunk("JAPAN supported the fund.")
    c2 = _chunk("BRAZIL opposed the fund.", cid="c2")
    G = _build(CoStanceEdges(), c1, c2)
    assert G.number_of_edges() == 0             # projection is within-chunk only
