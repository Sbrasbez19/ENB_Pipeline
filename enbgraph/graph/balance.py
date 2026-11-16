"""
graph/balance.py — structural balance analysis of signed graphs (plan C6).

Three instruments, all standard, applied to the aggregated signed graphs:

  * partition_signed — spectral bisection: sign pattern of the LEADING
    eigenvector of the (symmetrised) signed adjacency. For a perfectly
    balanced network (two camps, positive within, negative across) this
    recovers the camps exactly — balance is equivalent to the adjacency
    being similar, under a diagonal sign matrix, to an all-positive one
    (structural balance: Heider 1946; Cartwright & Harary 1956, as cited by
    Sudhahar et al. 2015; spectral treatment: Kunegis et al. 2010, SDM —
    general spectral-graph-theory source). Sudhahar et al.'s validation is
    exactly this: partitioning their signed network recovers the
    Democrat/Republican split. Our analogue: Annex I vs G-77/China.

  * frustration_ratio — fraction of signed edges violating a partition
    (positive across camps or negative within). 0.0 = perfectly balanced.

  * triangle_balance — Heider's census: a triangle is balanced iff the
    product of its edge signs is positive ("the enemy of my enemy...").

Linear algebra via numpy.linalg.eigh (existing package; the custom code is
the ~15 lines of glue the implementation plan budgeted). Edge sign values
come from the Wilson pass: signed_weight = sign * confidence, so
low-confidence edges influence the partition proportionally less.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import networkx as nx
import numpy as np


def _signed_value(d: dict) -> float:
    """Edge -> signed scalar. Prefer Wilson signed_weight; fall back to
    sign * weight for graphs that skipped aggregation."""
    if "signed_weight" in d:
        return float(d["signed_weight"])
    s = d.get("sign")
    return float(s) * float(d.get("weight", 1.0)) if s else 0.0


def signed_adjacency(G: nx.Graph, nodes: Optional[list[str]] = None) -> tuple[np.ndarray, list[str]]:
    """Symmetric signed adjacency. Directed graphs are symmetrised (A + A.T):
    balance is a property of the underlying signed relation, and both papers'
    balance analyses operate on the undirected signed structure."""
    nodes = nodes or sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for u, v, d in G.edges(data=True):
        w = _signed_value(d)
        if w == 0.0:
            continue
        i, j = idx[u], idx[v]
        A[i, j] += w
        A[j, i] += w
    return A, nodes


def partition_signed(G: nx.Graph) -> dict[str, int]:
    """Two-camp spectral bisection: camps by the sign of the leading
    eigenvector's components. Isolated-in-sign nodes (all-zero rows) get
    camp 0 by convention but carry no evidence."""
    A, nodes = signed_adjacency(G)
    if len(nodes) < 2 or not A.any():
        return {n: 0 for n in nodes}
    w, V = np.linalg.eigh(A)             # ascending eigenvalues
    lead = V[:, -1]                      # eigenvector of the largest
    return {n: (0 if lead[i] >= 0 else 1) for i, n in enumerate(nodes)}


def frustration_ratio(G: nx.Graph, camps: dict[str, int]) -> float:
    """Fraction of (non-tied) signed edges inconsistent with the partition."""
    bad = tot = 0
    for u, v, d in G.edges(data=True):
        s = np.sign(_signed_value(d))
        if s == 0:
            continue
        tot += 1
        same = camps.get(u) == camps.get(v)
        if (s > 0) != same:
            bad += 1
    return bad / tot if tot else 0.0


def triangle_balance(G: nx.Graph) -> dict:
    """Heider triangle census on the undirected signed structure."""
    H = G.to_undirected() if G.is_directed() else G
    signs = {}
    for u, v, d in H.edges(data=True):
        s = np.sign(_signed_value(d))
        if s != 0:
            signs[frozenset((u, v))] = s
    balanced = unbalanced = 0
    nodes_with = nx.Graph(list(map(tuple, signs)))
    for a in nodes_with:
        for b, c in combinations(sorted(nodes_with[a]), 2):
            if a < b and nodes_with.has_edge(b, c):     # count each triangle once
                p = signs[frozenset((a, b))] * signs[frozenset((a, c))] * signs[frozenset((b, c))]
                balanced += p > 0
                unbalanced += p < 0
    tot = balanced + unbalanced
    return {"triangles": tot, "balanced": balanced, "unbalanced": unbalanced,
            "balance_ratio": round(balanced / tot, 3) if tot else None}


# ---------------------------------------------------------------------- #
# Ground-truth agreement: does the partition recover the known UNFCCC camps?
# Membership sets are modeling data, not code — kept editable here and
# deliberately small (core, uncontroversial members only).
# ---------------------------------------------------------------------- #
ANNEX_I_CORE = {"United States", "European Union", "Japan", "Canada",
                "Australia", "Norway", "Switzerland", "New Zealand", "Russia",
                "Annex I"}
G77_CORE = {"G-77/China", "China", "India", "Brazil", "Saudi Arabia",
            "AOSIS", "LDCs", "non-Annex I", "Nigeria", "Iran"}


def camp_agreement(camps: dict[str, int]) -> Optional[float]:
    """Best-relabelling agreement of the bisection with the Annex-I / G-77
    core sets (only nodes present in the graph count). 1.0 = perfect split."""
    a = [camps[n] for n in ANNEX_I_CORE if n in camps]
    g = [camps[n] for n in G77_CORE if n in camps]
    if not a or not g:
        return None
    n = len(a) + len(g)
    fit0 = (sum(1 for x in a if x == 0) + sum(1 for x in g if x == 1)) / n
    fit1 = (sum(1 for x in a if x == 1) + sum(1 for x in g if x == 0)) / n
    return round(max(fit0, fit1), 3)


__all__ = ["signed_adjacency", "partition_signed", "frustration_ratio",
           "triangle_balance", "camp_agreement", "ANNEX_I_CORE", "G77_CORE"]
