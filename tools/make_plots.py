"""
tools/make_plots.py — generate the analysis figures from the phase artifacts.

Each figure carries ONE argument from the evaluation; none merely re-prints a
small table. Reads:
  test_graphs/metrics.json   (per-graph metrics)
  test_graphs/*.graphml      (the graphs themselves, for network + centrality)
  dev_config.json            (tier precision / weights)

Figures written to <out>/:
  1 camp_agreement_over_time   line: per-year + pooled, direct vs co-stance
  2 adjudication_pooled        grouped bars: camp / frustration / contested
  3 tier_precision             bars: measured precision -> weights (+ a-priori)
  4 wilson_confidence          curve: why per-year is thin & 95% unreachable
  5 network_<name>             the signed graph drawn (green/red, camps, labels)
  6 centrality_pooled          top actors per definition (shows co-stance drift)

Usage:
  python -m tools.make_plots --artifacts phase_artifacts --out plots
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from enbgraph.graph.balance import partition_signed, ANNEX_I_CORE, G77_CORE

# consistent palette
C_DIRECT, C_COST, C_PROX = "#1f77b4", "#ff7f0e", "#7f7f7f"
C_POS, C_NEG = "#2ca02c", "#d62728"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def _load(art: Path):
    tg = art 
    metrics = json.loads((tg / "metrics.json").read_text())
    cfg = json.loads((art / "dev_config.json").read_text())
    return metrics, cfg, tg


def _series(metrics, edge_def, key):
    """(sorted years, values) for a definition, excluding pooled."""
    pts = [(r["year"], r[key]) for r in metrics
           if r["edge_def"] == edge_def and r["year"] != "pooled" and r[key] is not None]
    pts.sort(key=lambda x: int(x[0]))
    return [int(y) for y, _ in pts], [v for _, v in pts]


def _pooled(metrics, edge_def, key):
    for r in metrics:
        if r["edge_def"] == edge_def and r["year"] == "pooled":
            return r[key]
    return None


# 1 ------------------------------------------------------------------------- #
def fig_camp_over_time(metrics, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for ed, c, lab in [("stance:direct", C_DIRECT, "direct stance"),
                       ("costance:chunk", C_COST, "co-stance")]:
        xs, ys = _series(metrics, ed, "camp_agreement")
        ax.plot(xs, ys, "o-", color=c, label=f"{lab} (per-year)")
        p = _pooled(metrics, ed, "camp_agreement")
        if p is not None:
            ax.axhline(p, color=c, ls="--", alpha=0.6,
                       label=f"{lab} pooled = {p}")
    ax.axhline(0.5, color="k", ls=":", alpha=0.5, label="chance (0.5)")
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("COP year"); ax.set_ylabel("camp agreement (Annex I vs G-77/China)")
    ax.set_title("Recovery of the known North–South split\nnoisy per year, strong when pooled")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "1_camp_agreement_over_time.png"); plt.close(fig)


# 2 ------------------------------------------------------------------------- #
def fig_adjudication(metrics, out):
    # signed metrics exist only for the two signed definitions; proximity is
    # shown explicitly as N/A (unsigned) so the baseline is not silently dropped
    # — its inability to take these tests is the motivating result.
    defs = [("proximity:sentence", "proximity\n(baseline)", C_PROX),
            ("stance:direct", "direct\nstance", C_DIRECT),
            ("costance:chunk", "co-stance", C_COST)]
    cats = [("camp_agreement", "camp agreement\n(higher better)"),
            ("frustration", "frustration\n(lower better)"),
            ("contested_frac", "contested frac.\n(lower better)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, (key, title) in zip(axes, cats):
        vals, labs, cols = [], [], []
        for ed, lab, c in defs:
            v = _pooled(metrics, ed, key)
            labs.append(lab); cols.append(c)
            vals.append(v if v is not None else 0)
        bars = ax.bar(labs, vals, color=cols)
        for i, (ed, _, _) in enumerate(defs):
            v = _pooled(metrics, ed, key)
            ax.text(i, (v or 0) + 0.02, f"{v}" if v is not None else "N/A\n(unsigned)",
                    ha="center", fontsize=8.5, color="#555" if v is None else "black")
        ax.set_ylim(0, 1.12); ax.set_title(title, fontsize=9)
    fig.suptitle("Signed metrics — proximity cannot take these tests (its motivating gap)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out / "2_adjudication_pooled.png"); plt.close(fig)


# 2b ------------------------------------------------------------------------ #
def fig_structural_comparison(metrics, out):
    """The metrics all THREE definitions share (nodes, edges, density), pooled."""
    defs = [("proximity:sentence", "proximity", C_PROX),
            ("stance:direct", "direct stance", C_DIRECT),
            ("costance:chunk", "co-stance", C_COST)]
    cats = [("edges", "edges (pooled)"), ("density", "density"),
            ("nodes", "nodes")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, (key, title) in zip(axes, cats):
        vals = [_pooled(metrics, ed, key) for ed, _, _ in defs]
        ax.bar([d[1] for d in defs], vals, color=[d[2] for d in defs])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Structural comparison across all three definitions\n"
                 "proximity is dense but sign-blind · direct stance is sparse but signed · "
                 "co-stance is dense via projection", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "2b_structural_comparison.png"); plt.close(fig)


# 2c ------------------------------------------------------------------------ #
def fig_centrality_agreement(tg: Path, out: Path, topk=12):
    """Do the three definitions agree on WHO is central? Rank heatmap of the
    union of each definition's top actors — the influence read all three can
    give. Shows agreement on the heavyweights and co-stance's drift."""
    import numpy as np
    defs = [("proximity:sentence", "proximity", "POOLED_proximity-sentence.graphml"),
            ("stance:direct", "direct stance", "POOLED_stance-direct.graphml"),
            ("costance:chunk", "co-stance", "POOLED_costance-chunk.graphml")]
    ranks, top_union = {}, set()
    for ed, _, fn in defs:
        G = nx.read_graphml(tg / fn)
        cen = dict(G.in_degree(weight="weight")) if G.is_directed() \
            else dict(G.degree(weight="weight"))
        order = [n for n, _ in sorted(cen.items(), key=lambda x: -x[1])]
        ranks[ed] = {n: i + 1 for i, n in enumerate(order)}
        top_union |= set(order[:topk])
    # order rows by mean rank across definitions
    rows = sorted(top_union,
                  key=lambda n: np.mean([ranks[ed].get(n, 999) for ed, _, _ in defs]))
    M = np.array([[min(ranks[ed].get(n, 40), 40) for ed, _, _ in defs] for n in rows])
    fig, ax = plt.subplots(figsize=(6.5, max(5, 0.42 * len(rows) + 1)))
    im = ax.imshow(M, cmap="YlGnBu_r", aspect="auto", vmin=1, vmax=25)
    ax.set_xticks(range(len(defs))); ax.set_xticklabels([d[1] for d in defs])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(defs)):
            r = M[i, j]
            ax.text(j, i, "—" if r >= 40 else str(int(r)), ha="center", va="center",
                    fontsize=7.5, color="white" if r <= 8 else "#333")
    ax.set_title("Who is central? centrality RANK per definition\n"
                 "(1 = most central; — = outside top 40)", fontsize=10)
    fig.colorbar(im, ax=ax, label="rank (lower = more central)", shrink=0.7)
    fig.tight_layout(); fig.savefig(out / "2c_centrality_agreement.png"); plt.close(fig)


# 3 ------------------------------------------------------------------------- #
def fig_tier_precision(cfg, out):
    tiers = [1, 2, 3]
    prec = [cfg["tier_precision"][str(t)] for t in tiers]
    apriori = [1.0, 0.75, 0.25]
    x = range(len(tiers)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([i - w/2 for i in x], apriori, w, label="a-priori guess", color="#bbbbbb")
    ax.bar([i + w/2 for i in x], prec, w, label="measured precision = weight", color=C_DIRECT)
    for i, v in zip(x, prec):
        ax.text(i + w/2, v + 0.02, f"{v}", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"tier {t}" for t in tiers])
    ax.set_ylim(0, 1.1); ax.set_ylabel("precision / weight")
    ax.set_title("Tier reliability is INVERTED from the a-priori guess\n"
                 "(tier-2 appositives most reliable; tier-1 weakest → parser target)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "3_tier_precision.png"); plt.close(fig)


# 4 ------------------------------------------------------------------------- #
def _wilson_lo(k, n, z):
    if n == 0: return 0.0
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z/d) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return c - h


def fig_wilson(out):
    ns = list(range(1, 21))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for z, lab, c in [(1.959964, "95% lower bound", C_NEG),
                      (0.994458, "68% lower bound", C_DIRECT)]:
        ax.plot(ns, [_wilson_lo(n, n, z) for n in ns], "o-", color=c, label=lab)
    ax.axhline(0.5, color="k", ls=":", label="sign-certainty threshold (0.5)")
    ax.set_xlabel("n = consecutive UNANIMOUS observations of a pair")
    ax.set_ylabel("Wilson lower bound (confidence)")
    ax.set_title("Why per-year is thin and pooling matters\n"
                 "unanimous evidence must accumulate before a sign is confident")
    ax.set_xticks(ns[::2]); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "4_wilson_confidence.png"); plt.close(fig)


# 5 ------------------------------------------------------------------------- #
def fig_network(tg: Path, fname: str, title: str, out: Path, max_nodes=40):
    import numpy as np
    G = nx.read_graphml(tg / fname)
    H = (nx.DiGraph(G) if G.is_directed() else nx.Graph(G))
    deg = dict(H.degree(weight="weight"))
    keep = {n for n, _ in sorted(deg.items(), key=lambda x: -x[1])[:max_nodes]}
    H = H.subgraph(keep).copy()
    H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0])  # drop isolates
    if H.number_of_edges() == 0:
        return
    camps = partition_signed(H)
    ann = [camps[n] for n in ANNEX_I_CORE if n in camps]
    annex_camp = max(set(ann), key=ann.count) if ann else 0

    # --- layout: two clean columns (one per recovered camp), evenly spaced so
    #     labels never collide. Within a column, order by a positive-weight
    #     spring pass so allied actors stay vertically near each other, then
    #     distribute evenly; hubs sit nearer the centre (larger x pull).
    U = nx.Graph(); U.add_nodes_from(H.nodes())
    for u, v, d in H.edges(data=True):
        if d.get("sign") == 1:
            U.add_edge(u, v, weight=U.get_edge_data(u, v, {}).get("weight", 0) + 1.0)
    spring = nx.spring_layout(U, weight="weight", k=1.0, iterations=400, seed=3)
    left = [nd for nd in H if camps[nd] == annex_camp]
    right = [nd for nd in H if camps[nd] != annex_camp]
    pos = {}
    dmax = max(deg.values()) or 1.0
    for col, side in ((left, -1.0), (right, +1.0)):
        col_sorted = sorted(col, key=lambda nd: spring.get(nd, (0, 0))[1])
        m = len(col_sorted)
        for i, nd in enumerate(col_sorted):
            y = 1.0 - 2.0 * (i / max(1, m - 1)) if m > 1 else 0.0
            # hubs pulled toward the centre gap; leaf nodes pushed outward
            x = side * (0.45 + 0.75 * (1.0 - deg.get(nd, 0) / dmax))
            pos[nd] = np.array([x, y])

    node_col = ["#3b6ea5" if camps[nd] == annex_camp else "#c44536" for nd in H.nodes()]
    sizes = [220 + 60 * deg.get(nd, 0) for nd in H.nodes()]

    fig, ax = plt.subplots(figsize=(15, max(9, 0.42 * max(len(left), len(right)) + 3)))
    for u, v, d in H.edges(data=True):
        s = d.get("sign", 0)
        col = C_POS if s == 1 else C_NEG if s == -1 else "#cccccc"
        # curve edges slightly so parallel ties are distinguishable
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-", color=col, alpha=0.32, lw=1.0,
                                    linestyle="dashed" if d.get("contested") else "solid",
                                    connectionstyle="arc3,rad=0.08"), zorder=1)
    nx.draw_networkx_nodes(H, pos, node_color=node_col, node_size=sizes,
                           edgecolors="white", linewidths=1.0, ax=ax)
    import matplotlib.patheffects as pe
    for nd in H.nodes():
        x, y = pos[nd]
        ha = "right" if x < 0 else "left"
        t = ax.text(x + (-0.04 if x < 0 else 0.04), y, nd, fontsize=9, ha=ha,
                    va="center", zorder=3, color="#111111")
        t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

    ax.text(-0.9, 1.16, "Annex I side", color="#3b6ea5", fontsize=14, fontweight="bold", ha="center")
    ax.text(0.9, 1.16, "G-77/China side", color="#c44536", fontsize=14, fontweight="bold", ha="center")
    ax.set_title(title + "\ntwo columns = recovered camps · green = aligned · red = opposed · "
                 "dashed = contested · node size = involvement", fontsize=11)
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.2, 1.32); ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"5_network_{fname.replace('.graphml','')}.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# 6 ------------------------------------------------------------------------- #
def fig_centrality(metrics, out):
    rows = [(r["edge_def"], r["top_central"]) for r in metrics if r["year"] == "pooled"]
    order = {"proximity:sentence": 0, "stance:direct": 1, "costance:chunk": 2}
    rows.sort(key=lambda r: order.get(r[0], 9))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.axis("off")
    lines = ["Top-3 central actors (pooled) — 'influence' read differs by definition\n"]
    label = {"proximity:sentence": "proximity (co-occurrence)",
             "stance:direct": "direct stance (directed)",
             "costance:chunk": "co-stance (projected)"}
    for ed, top in rows:
        lines.append(f"  {label.get(ed, ed):28s}:  {',  '.join(top)}")
    lines.append("\n  → direct stance surfaces the real heavyweights (EU, US, G-77/China);")
    lines.append("     co-stance drifts toward minor actors (projection artifact).")
    ax.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=10,
            transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(out / "6_centrality_pooled.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("phase_artifacts"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    metrics, cfg, tg = _load(a.artifacts)

    fig_camp_over_time(metrics, a.out)
    fig_adjudication(metrics, a.out)
    fig_structural_comparison(metrics, a.out)
    fig_centrality_agreement(tg, a.out)
    fig_tier_precision(cfg, a.out)
    fig_wilson(a.out)
    fig_centrality(metrics, a.out)
    # one drawing per yearly direct-stance graph, plus the pooled one
    yearly = []
    for p in tg.glob("COP*_stance-direct.graphml"):
        m = re.match(r"COP(\d+)-(\d{4})_", p.name)
        if m:
            yearly.append((int(m.group(2)), int(m.group(1)), p.name))
    for year, cop, fname in sorted(yearly):
        fig_network(tg, fname, f"COP-{cop} ({year}) direct-stance network", a.out)
    if (tg / "POOLED_stance-direct.graphml").is_file():
        span = f" ({min(yearly)[0]}–{max(yearly)[0]})" if yearly else ""
        fig_network(tg, "POOLED_stance-direct.graphml",
                    f"Pooled direct-stance network{span}", a.out)
    print(f"wrote figures -> {a.out}")
    for f in sorted(a.out.glob('*.png')):
        print("  ", f.name)


if __name__ == "__main__":
    main()