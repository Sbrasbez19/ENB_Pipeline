"""
Tests for the Bernoulli+Wilson aggregation pass (C3).

The numeric expectations are the statsmodels Wilson values verified in-session
(and consistent with Wilson 1927 / Sudhahar et al. 2015 Appendix A2):
  1/1 -> lower 0.207   (single sighting: near-worthless, contested)
  4/4 -> lower 0.510   (four unanimous: barely clears 0.5)
  3/4 -> interval (0.301, 0.954)  (one dissent: contested)

Graphs are hand-built with pos_w/neg_w so the pass is tested in isolation
from extraction and building.

Run with:  pytest -q
"""

import networkx as nx
import pytest

from enbgraph.graph.aggregate import aggregate_signed, signed_summary


def _g(pos, neg, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edge("A", "B", weight=pos + neg, pos_w=float(pos), neg_w=float(neg),
               n_obs=int(pos + neg), relation="stance", evidence=[])
    return G


def test_single_sighting_low_confidence_and_contested():
    G = aggregate_signed(_g(1, 0))
    d = G["A"]["B"]
    assert d["sign"] == +1
    assert d["confidence"] == pytest.approx(0.207, abs=1e-3)
    assert d["contested"] is True
    assert d["signed_weight"] == pytest.approx(0.207, abs=1e-3)


def test_four_unanimous_clears_half():
    d = aggregate_signed(_g(4, 0))["A"]["B"]
    assert d["sign"] == +1
    assert d["confidence"] == pytest.approx(0.510, abs=1e-3)
    assert d["contested"] is False


def test_one_dissent_is_contested_but_signed():
    d = aggregate_signed(_g(3, 1))["A"]["B"]
    assert d["sign"] == +1 and d["p_pos"] == pytest.approx(0.75)
    assert d["contested"] is True            # interval (0.301, 0.954) straddles 0.5


def test_exact_tie_has_no_majority():
    d = aggregate_signed(_g(2, 2))["A"]["B"]
    assert d["sign"] == 0 and d["signed_weight"] == 0.0
    assert d["contested"] is True


def test_negative_majority_signed_weight_negative():
    d = aggregate_signed(_g(0, 4))["A"]["B"]
    assert d["sign"] == -1
    assert d["signed_weight"] == pytest.approx(-0.510, abs=1e-3)


def test_float_tier_weighted_counts_accepted():
    d = aggregate_signed(_g(2.25, 0.75))["A"]["B"]   # e.g. 3 tier-1... mixed tiers
    assert d["sign"] == +1 and 0 < d["confidence"] < 1


def test_unsigned_graph_untouched():
    G = nx.Graph()
    G.add_edge("A", "B", weight=5.0, relation="proximity", evidence=[])
    H = aggregate_signed(G)
    assert "sign" not in H["A"]["B"] and "confidence" not in H["A"]["B"]


def test_copy_semantics_default_and_inplace():
    G = _g(4, 0)
    H = aggregate_signed(G)                  # default: copy
    assert "sign" in H["A"]["B"] and "sign" not in G["A"]["B"]
    aggregate_signed(G, inplace=True)
    assert "sign" in G["A"]["B"]


def test_signed_summary_counts():
    G = nx.Graph()
    G.add_edge("A", "B", pos_w=4.0, neg_w=0.0, weight=4)
    G.add_edge("A", "C", pos_w=0.0, neg_w=1.0, weight=1)
    G.add_edge("B", "C", pos_w=1.0, neg_w=1.0, weight=2)
    s = signed_summary(aggregate_signed(G))
    assert s == {"positive": 1, "negative": 1, "tied": 1, "contested": 2}
