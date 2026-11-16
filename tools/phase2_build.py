"""
tools/phase2_build.py — PHASE 2 (test) of the train/test pipeline.

Consumes the frozen `dev_config.json` from Phase 1 and applies it, unchanged, to
the TEST years. Nothing here re-tunes anything: tier weights and alpha come
straight from the config. This is the honest separation — the numbers that
shaped the method were fixed on training data before these years were touched.

Subcommands:
  build     — build proximity + stance + costance graphs for the test years
              (per-year at the selected per-year alpha) and a pooled graph
              (at the selected pooled alpha). Writes GraphML + manifest.
  analyze   — compute the pre-registered metric set (validation_protocol.md §2)
              over the built graphs and write metrics.json + a short markdown.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx

from enbgraph.corpus import load_corpus
from enbgraph.extraction.actors import ActorExtractor
from enbgraph.graph.aggregate import aggregate_signed, signed_summary
from enbgraph.graph.balance import (camp_agreement, frustration_ratio,
                                    partition_signed, triangle_balance)
from enbgraph.graph.builder import GraphBuilder
from enbgraph.graph.proximity import ProximityEdges
from enbgraph.graph.stance import CoStanceEdges, DirectStanceEdges
from enbgraph.ingest.chunker import StructuralChunker
from enbgraph.models import SectionKind

EDGES = ("proximity", "stance", "costance")


def _load_cfg(p: Path) -> dict:
    cfg = json.loads(p.read_text())
    cfg["tier_weights"] = {int(k): v for k, v in cfg["tier_weights"].items()}
    return cfg


def _edge_def(edge: str, cfg: dict):
    tw = cfg["tier_weights"]
    if edge == "stance":
        return DirectStanceEdges(tier_weights=tw)
    if edge == "costance":
        return CoStanceEdges(tier_weights=tw)
    return ProximityEdges("sentence")


def _docs(corpus: Path, years):
    keep = set(years)
    for ld in load_corpus(corpus):
        if ld.status == "ok" and ld.document and ld.document.meta.year in keep:
            yield ld.document


def cmd_build(a):
    cfg = _load_cfg(a.config)
    years = cfg["test_years"]
    ch, ax = StructuralChunker(), ActorExtractor()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"phase": "test", "config": str(a.config),
                "tier_weights": cfg["tier_weights"],
                "alpha_selected": cfg["alpha_selected"], "graphs": []}

    # cache (chunk, mentions) per doc so all three definitions share extraction
    items_by_year, meta = {}, {}
    for doc in _docs(a.out.parent if False else Path(a.corpus), years):
        chunks = [c for c in ch.chunk(doc) if c.kind == SectionKind.REPORT_ITEM]
        items_by_year[doc.meta.year] = [(c, ax.extract(c)) for c in chunks]
        meta[doc.meta.year] = doc.meta.cop

    for edge in EDGES:
        ed = _edge_def(edge, cfg)
        signed = edge in ("stance", "costance")
        alpha_py = cfg["alpha_selected"]["per_year"]
        pooled_items = []
        for year in sorted(items_by_year):
            items = items_by_year[year]
            pooled_items += items
            G = GraphBuilder(ed).build(items, year=year, cop=meta[year])
            if signed:
                aggregate_signed(G, alpha=alpha_py, inplace=True)
            f = a.out / f"COP{meta[year]}-{year}_{ed.name.replace(':','-')}.graphml"
            nx.write_graphml(_scalarise(G), f)
            manifest["graphs"].append(_summary(G, edge, year, meta[year], f.name, signed))
        # pooled
        Gp = GraphBuilder(ed).build(pooled_items)
        if signed:
            aggregate_signed(Gp, alpha=cfg["alpha_selected"]["pooled"], inplace=True)
        fp = a.out / f"POOLED_{ed.name.replace(':','-')}.graphml"
        nx.write_graphml(_scalarise(Gp), fp)
        manifest["graphs"].append(_summary(Gp, edge, None, "pooled", fp.name, signed))

    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"built {len(manifest['graphs'])} graphs -> {a.out}")


def _scalarise(G):
    H = G.copy()
    for _, _, d in H.edges(data=True):
        if isinstance(d.get("evidence"), list):
            d["evidence"] = " | ".join(d["evidence"])[:400]
    for k, v in list(H.graph.items()):
        if v is None:
            H.graph[k] = ""
    return H


def _summary(G, edge, year, cop, fname, signed):
    s = {"edge": edge, "year": year, "cop": cop, "file": fname,
         "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
         "density": round(nx.density(G), 4)}
    if signed:
        s["signed"] = signed_summary(G)
    return s


# --------------------------------------------------------------------------- #
def cmd_analyze(a):
    """Pre-registered metric set (validation_protocol.md §2), per graph."""
    rows = []
    for f in sorted(a.out.glob("*.graphml")):
        G = nx.read_graphml(f)
        edge = G.graph.get("edge_def", "?")
        signed = "stance" in edge or "costance" in edge
        row = {"file": f.name, "edge_def": edge,
               "year": G.graph.get("year") or "pooled",
               "directed": G.is_directed(),
               "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
               "density": round(nx.density(G), 4)}
        # §2a universal: centrality top-3 (weighted degree; in-degree if directed)
        if G.is_directed():
            deg = dict(G.in_degree(weight="weight"))
        else:
            deg = dict(G.degree(weight="weight"))
        row["top_central"] = [n for n, _ in sorted(deg.items(), key=lambda x: -x[1])[:3]]
        # §2b signed-only: balance
        if signed and G.number_of_edges():
            camps = partition_signed(G)
            row["frustration"] = round(frustration_ratio(G, camps), 3)
            row["camp_agreement"] = camp_agreement(camps)
            row["triangle_balance"] = triangle_balance(G).get("balance_ratio")
            row["contested_frac"] = round(
                sum(1 for _, _, d in G.edges(data=True) if d.get("contested")) /
                G.number_of_edges(), 3)
        else:
            row["frustration"] = row["camp_agreement"] = None
            row["triangle_balance"] = row["contested_frac"] = None
        rows.append(row)

    (a.out / "metrics.json").write_text(json.dumps(rows, indent=2))
    _write_md(a.out / "metrics_summary.md", rows)
    print(f"analyzed {len(rows)} graphs -> {a.out}/metrics.json, metrics_summary.md")


def _write_md(path: Path, rows):
    lines = ["# Test-phase metrics (pre-registered set)\n",
             "| graph | year | n | e | dens | top central | frustr | camp agr | tri-bal | contested |",
             "|---|---|--:|--:|--:|---|--:|--:|--:|--:|"]
    order = {"proximity": 0, "stance": 1, "costance": 2}
    for r in sorted(rows, key=lambda r: (order.get(r["edge_def"].split(":")[0], 9),
                                         str(r["year"]))):
        lines.append(
            f"| {r['edge_def']} | {r['year']} | {r['nodes']} | {r['edges']} | "
            f"{r['density']} | {', '.join(r['top_central'])} | "
            f"{r['frustration'] if r['frustration'] is not None else '—'} | "
            f"{r['camp_agreement'] if r['camp_agreement'] is not None else '—'} | "
            f"{r['triangle_balance'] if r['triangle_balance'] is not None else '—'} | "
            f"{r['contested_frac'] if r['contested_frac'] is not None else '—'} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Phase 2: build+analyze test graphs from frozen config.")
    sub = ap.add_subparsers(required=True)
    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)
    b.add_argument("--corpus", type=Path, required=True)
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("test_graphs"))
    an = sub.add_parser("analyze"); an.set_defaults(fn=cmd_analyze)
    an.add_argument("--out", type=Path, default=Path("test_graphs"))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
