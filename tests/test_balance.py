"""
Tests for the signed balance analysis (C6). Toy fixtures with known-correct
answers from balance theory (Heider 1946; Cartwright & Harary 1956):
a perfectly balanced two-camp graph must be recovered exactly with zero
frustration; the classic unbalanced triangle (+,+,-) must be counted as such.

Run with:  pytest -q
"""

import networkx as nx

from enbgraph.graph.balance import (
    camp_agreement, frustration_ratio, partition_signed, triangle_balance,
)


def _sg(edges):
    G = nx.Graph()
    for u, v, s in edges:
        G.add_edge(u, v, signed_weight=float(s), weight=abs(s))
    return G


BALANCED = _sg([
    ("A", "B", +1), ("C", "D", +1),          # within-camp positive
    ("A", "C", -1), ("A", "D", -1),
    ("B", "C", -1), ("B", "D", -1),          # across-camp negative
])


def test_perfect_two_camps_recovered():
    camps = partition_signed(BALANCED)
    assert camps["A"] == camps["B"]
    assert camps["C"] == camps["D"]
    assert camps["A"] != camps["C"]


def test_balanced_graph_zero_frustration():
    assert frustration_ratio(BALANCED, partition_signed(BALANCED)) == 0.0


def test_flipping_one_edge_creates_frustration():
    G = _sg([("A", "B", -1), ("C", "D", +1),      # A-B now negative WITHIN camp
             ("A", "C", -1), ("A", "D", -1),
             ("B", "C", -1), ("B", "D", -1)])
    f = frustration_ratio(G, partition_signed(G))
    assert 0.0 < f <= 0.5


def test_triangle_census_heider():
    # (+,+,+) balanced; (+,-,-) balanced ("enemy of my enemy"); (+,+,-) unbalanced
    G = _sg([("A", "B", +1), ("B", "C", +1), ("A", "C", +1),     # balanced
             ("A", "D", +1), ("B", "D", -1)])                     # A-B(+),A-D(+),B-D(-) unbalanced
    c = triangle_balance(G)
    assert c["triangles"] == 2
    assert c["balanced"] == 1 and c["unbalanced"] == 1
    assert c["balance_ratio"] == 0.5


def test_directed_graph_symmetrised():
    D = nx.DiGraph()
    D.add_edge("A", "B", signed_weight=+1.0, weight=1)
    D.add_edge("C", "A", signed_weight=-1.0, weight=1)
    camps = partition_signed(D)
    assert camps["A"] == camps["B"] != camps["C"]


def test_camp_agreement_perfect_split():
    camps = {"United States": 0, "European Union": 0, "Japan": 0,
             "G-77/China": 1, "China": 1, "India": 1}
    assert camp_agreement(camps) == 1.0


def test_camp_agreement_label_invariant():
    camps = {"United States": 1, "G-77/China": 0}
    assert camp_agreement(camps) == 1.0
