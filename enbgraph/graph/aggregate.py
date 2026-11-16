"""
graph/aggregate.py — Bernoulli + Wilson aggregation over signed edges (plan C3).

WHAT: a post-build pass over a signed graph whose edges carry the builder's
accumulated bookkeeping (pos_w, neg_w, n_obs). For each pair it derives:

  sign          +1 / -1 majority sign (0 on an exact tie)
  p_pos         naive positive share pos_w / n     (kept for comparison)
  confidence    Wilson lower bound of the MAJORITY share — the conservative
                "how strongly can we claim this sign given how little we saw"
  contested     True when the Wilson interval of p_pos straddles 0.5
  signed_weight sign * confidence — the value signed analyses (structural
                balance, spectral partitioning) consume as adjacency entries

WHY this scheme: per ordered pair, sign observations are draws of a Bernoulli
variable (Sudhahar, Veltri & Cristianini 2015, "Positive and negative
relations"; Wilson interval per their Appendix A2 — "a standard approach in
statistics"). The naive estimator k/n is maximally overconfident at n=1; the
Wilson lower bound (Wilson 1927, JASA) is small-sample-correct, so repetition —
not single sightings — creates strong edges. Verified in-session against
statsmodels: 1/1 -> 0.207, 4/4 -> 0.510, 9/10 -> 0.596, 18/20 -> 0.699.
Contested edges are kept and flagged, not dropped: contestedness is the
pair-level analogue of the paper's divisiveness measure.

WHY a post-pass, not builder logic: aggregation is statistical *policy*, not
bookkeeping; the builder stays generic (proximity untouched) and policies stay
swappable (naive vs Wilson can be presented side by side).

Counts are tier-weighted floats (a tier-2 observation contributes 0.75); the
Wilson formula is algebraic in (k, n), and statsmodels' implementation accepts
float pseudo-counts (verified in-session) — scipy's binomtest, which requires
integers, would not work here.
"""

from __future__ import annotations

import networkx as nx
from statsmodels.stats.proportion import proportion_confint


def _wilson(count: float, nobs: float, alpha: float) -> tuple[float, float]:
    """Typed scalar wrapper: statsmodels' stubs advertise array/DataFrame
    returns for array inputs; with scalar inputs the return is a scalar pair
    (verified in-session). The cast localises that knowledge in one place."""
    lo, hi = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return float(lo), float(hi)  # type: ignore[arg-type]


def aggregate_signed(G: nx.Graph, alpha: float = 0.05, inplace: bool = False) -> nx.Graph:
    """Derive sign / confidence / contested / signed_weight on every edge that
    carries signed bookkeeping. Edges without it (unsigned definitions such as
    proximity) are left untouched, so the pass is safe on any graph.

    alpha: 1 - confidence level of the Wilson interval (0.05 -> 95%).
    """
    H = G if inplace else G.copy()
    for u, v, d in H.edges(data=True):
        pos = d.get("pos_w")
        neg = d.get("neg_w")
        if pos is None or neg is None:
            continue                      # unsigned edge: not our business
        n = pos + neg
        if n <= 0:
            continue
        p_pos = pos / n
        lo_pos, hi_pos = _wilson(pos, n, alpha)
        d["p_pos"] = round(p_pos, 4)
        d["contested"] = bool(lo_pos < 0.5 < hi_pos)
        if pos > neg:
            sign, k = +1, pos
        elif neg > pos:
            sign, k = -1, neg
        else:
            sign, k = 0, pos              # exact tie: no majority
        lo_maj, _ = _wilson(k, n, alpha)
        d["sign"] = sign
        d["confidence"] = round(float(lo_maj), 4)
        d["signed_weight"] = round(sign * float(lo_maj), 4)
    return H


def signed_summary(G: nx.Graph) -> dict:
    """Small reporting helper for manifests: edge counts by sign + contested."""
    pos = neg = tie = contested = 0
    for _, _, d in G.edges(data=True):
        s = d.get("sign")
        if s is None:
            continue
        pos += s > 0
        neg += s < 0
        tie += s == 0
        contested += bool(d.get("contested"))
    return {"positive": pos, "negative": neg, "tied": tie, "contested": contested}


__all__ = ["aggregate_signed", "signed_summary"]
