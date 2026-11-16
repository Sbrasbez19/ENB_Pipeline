"""
Tests for the graph phase. Built on hand-made (chunk, mentions) inputs so the
builder is exercised independently of the extractor.

Run with:  pytest -q
"""

from enbgraph.models import Chunk, Provenance, SectionKind, Sentence
from enbgraph.extraction.base import ActorCategory, Mention
from enbgraph.graph.proximity import ProximityEdges
from enbgraph.graph.builder import GraphBuilder


def _chunk(text: str, sents: list[tuple[int, int]], cid="c1") -> Chunk:
    sentences = [Sentence(text=text[a:b], index=i, start_char=a, end_char=b) for i, (a, b) in enumerate(sents)]
    return Chunk(
        chunk_id=cid, kind=SectionKind.REPORT_ITEM, section_path=["T"], text=text,
        sentences=sentences,
        provenance=Provenance(doc_id="d", page_start=1, page_end=1, start_char=0, end_char=len(text)),
        cop="6", year=2000,
    )


def _m(actor, start, cat=ActorCategory.COUNTRY, cid="c1"):
    return Mention(actor_id=actor, surface=actor, category=cat, start_char=start, end_char=start + len(actor), chunk_id=cid)


# text:    "China and Japan spoke. Brazil objected."
#           0          ^10        22 23
TEXT = "China and Japan spoke. Brazil objected."
SENTS = [(0, 22), (23, 39)]
MENTIONS = [_m("China", 0), _m("Japan", 10), _m("Brazil", 23)]


def test_sentence_window_links_only_within_sentence():
    G = GraphBuilder(ProximityEdges("sentence")).build([(_chunk(TEXT, SENTS), MENTIONS)])
    assert G.has_edge("China", "Japan")
    assert not G.has_edge("China", "Brazil")   # different sentences
    assert not G.has_edge("Japan", "Brazil")


def test_chunk_window_links_all_pairs():
    G = GraphBuilder(ProximityEdges("chunk")).build([(_chunk(TEXT, SENTS), MENTIONS)])
    for a, b in [("China", "Japan"), ("China", "Brazil"), ("Japan", "Brazil")]:
        assert G.has_edge(a, b)


def test_weight_accumulates_across_windows():
    item = (_chunk(TEXT, SENTS), MENTIONS)
    G = GraphBuilder(ProximityEdges("sentence")).build([item, item])  # same co-occurrence twice
    assert G["China"]["Japan"]["weight"] == 2


def test_no_self_loops_on_repeated_actor():
    text = "Japan and Japan again."
    G = GraphBuilder(ProximityEdges("sentence")).build(
        [(_chunk(text, [(0, len(text))]), [_m("Japan", 0), _m("Japan", 10)])]
    )
    assert not G.has_edge("Japan", "Japan")
    assert G.number_of_edges() == 0


def test_undirected_pair_is_canonical():
    G = GraphBuilder(ProximityEdges("chunk")).build([(_chunk(TEXT, SENTS), MENTIONS)])
    # (A,B) and (B,A) are the same edge regardless of mention order
    assert G.has_edge("Japan", "China") and G.has_edge("China", "Japan")
    assert G.number_of_edges() == 3


def test_node_category_recorded():
    mentions = [_m("AOSIS", 0, ActorCategory.BLOC), _m("Tuvalu", 6, ActorCategory.COUNTRY)]
    text = "AOSIS Tuvalu"
    G = GraphBuilder(ProximityEdges("chunk")).build([(_chunk(text, [(0, len(text))]), mentions)])
    assert G.nodes["AOSIS"]["category"] == "bloc"
    assert G.nodes["Tuvalu"]["category"] == "country"


def test_graph_type_and_metadata_follow_edge_definition():
    G = GraphBuilder(ProximityEdges("sentence")).build([(_chunk(TEXT, SENTS), MENTIONS)], year=2000, cop="6")
    import networkx as nx
    assert isinstance(G, nx.Graph) and not G.is_directed()  # proximity is undirected
    assert G.graph["edge_def"] == "proximity:sentence"
    assert G.graph["year"] == 2000 and G.graph["cop"] == "6"


def test_evidence_attached_and_capped():
    item = (_chunk(TEXT, SENTS), MENTIONS)
    G = GraphBuilder(ProximityEdges("sentence"), max_evidence=2).build([item, item, item])
    ev = G["China"]["Japan"]["evidence"]
    assert ev and len(ev) <= 2 and "China and Japan" in ev[0]
