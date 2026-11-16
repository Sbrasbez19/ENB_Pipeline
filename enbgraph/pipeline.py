"""
pipeline.py — run the full pipeline end to end and save one graph per year.

    ingest/load -> chunk -> extract actors -> build graph -> serialise

Reusable `run(...)` plus an argparse CLI (`python -m enbgraph.pipeline`). The edge
definition is built through a small factory so new definitions (co-stance,
support, ...) register in one place rather than touching the CLI. Serialisation
is format-aware because GraphML/GEXF only accept scalar attributes — the per-edge
`evidence` list is flattened for those formats and kept intact for json/pickle.

Example:
    python -m enbgraph.pipeline --input /data/enb --output graphs \\
        --edge proximity --window sentence --format graphml
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Optional

import networkx as nx

from .corpus import load_corpus
from .extraction.actors import ActorExtractor
from .extraction.coref import PronounResolver
from .extraction.gazetteer import Gazetteer
from .graph.base import EdgeDefinition
from .graph.builder import GraphBuilder
from .graph.proximity import ProximityEdges
from .graph.stance import CoStanceEdges, DirectStanceEdges
from .graph.aggregate import aggregate_signed, signed_summary
from .ingest.chunker import StructuralChunker
from .models import Document, SectionKind
from .utils import get_logger, setup_logging

log = get_logger(__name__)

_EXT = {"graphml": "graphml", "gexf": "gexf", "json": "json", "pickle": "pkl"}


# --------------------------------------------------------------------------- #
# Edge-definition factory — one place to register new definitions
# --------------------------------------------------------------------------- #
def make_edge_definition(edge: str, *, window: str = "sentence",
                         tier_weights: Optional[dict[int, float]] = None,
                         attribute_to_bloc: bool = False) -> EdgeDefinition:
    if edge == "proximity":
        return ProximityEdges(window)
    if edge == "stance":
        return DirectStanceEdges(tier_weights=tier_weights,
                                 attribute_to_bloc=attribute_to_bloc)
    if edge == "costance":
        return CoStanceEdges(tier_weights=tier_weights,
                             attribute_to_bloc=attribute_to_bloc)
    raise ValueError(f"unknown edge type {edge!r} (available: proximity, stance, costance)")


def parse_tier_weights(spec: str) -> dict[int, float]:
    """'1:1.0,2:0.75,3:0.25' -> {1: 1.0, 2: 0.75, 3: 0.25}"""
    out: dict[int, float] = {}
    for part in spec.split(","):
        t, w = part.split(":")
        out[int(t)] = float(w)
    return out


# --------------------------------------------------------------------------- #
# Build + serialise
# --------------------------------------------------------------------------- #
def build_graph(doc: Document, edge_def: EdgeDefinition,
                chunker: StructuralChunker, extractor: ActorExtractor) -> nx.Graph:
    chunks = [c for c in chunker.chunk(doc) if c.kind == SectionKind.REPORT_ITEM]
    return GraphBuilder(edge_def).build_from_chunks(
        chunks, extractor, year=doc.meta.year, cop=doc.meta.cop, doc_id=doc.meta.doc_id
    )


def _scalarise(G: nx.Graph) -> nx.Graph:
    """GraphML/GEXF accept only scalar attributes. Flatten the evidence list to a
    string and replace None graph attributes with ''. Operates on a copy."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        ev = d.get("evidence")
        if isinstance(ev, list):
            d["evidence"] = " | ".join(ev)[:500]
    for k, v in list(H.graph.items()):
        if v is None:
            H.graph[k] = ""
    return H


def save_graph(G: nx.Graph, path: Path, fmt: str) -> None:
    if fmt == "graphml":
        nx.write_graphml(_scalarise(G), path)
    elif fmt == "gexf":
        nx.write_gexf(_scalarise(G), path)
    elif fmt == "json":
        data = nx.node_link_data(G, edges="links")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "pickle":
        with path.open("wb") as fh:
            pickle.dump(G, fh)
    else:
        raise ValueError(f"unknown format {fmt!r}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    input_dir: Path,
    output_dir: Path,
    *,
    edge: str = "proximity",
    window: str = "sentence",
    fmt: str = "graphml",
    alpha: float = 0.05,
    tier_weights: Optional[dict[int, float]] = None,
    attribute_to_bloc: bool = False,
    years: Optional[set[int]] = None,
    include_orgs: bool = False,
    coref: bool = False,
    gazetteer: Optional[Path] = None,
) -> dict:
    setup_logging()
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_def = make_edge_definition(edge, window=window,
                                    tier_weights=tier_weights,
                                    attribute_to_bloc=attribute_to_bloc)
    ext = _EXT[fmt]

    chunker = StructuralChunker()
    extractor = ActorExtractor(
        gazetteer=Gazetteer(gazetteer) if gazetteer else None,
        coref=PronounResolver() if coref else None,
        include_orgs=include_orgs,
    )

    summary: dict = {"edge_def": edge_def.name, "format": fmt, "alpha": alpha,
                     "tier_weights": tier_weights, "attribute_to_bloc": attribute_to_bloc,
                     "graphs": [], "skipped": []}
    for ld in load_corpus(input_dir):
        if ld.status != "ok" or ld.document is None:
            summary["skipped"].append({"name": ld.name, "status": ld.status})
            log.warning("skipped %s (%s)", ld.name, ld.status)
            continue
        doc = ld.document
        if years and doc.meta.year not in years:
            continue

        G = build_graph(doc, edge_def, chunker, extractor)
        if isinstance(edge_def, (DirectStanceEdges, CoStanceEdges)):
            aggregate_signed(G, alpha=alpha, inplace=True)   # Wilson pass (plan C3)
        label = f"COP{doc.meta.cop or '?'}-{doc.meta.year or '?'}_{edge_def.name.replace(':', '-')}"
        label = re.sub(r"[^A-Za-z0-9._-]", "", label)
        path = output_dir / f"{label}.{ext}"
        save_graph(G, path, fmt)

        top = [a for a, _ in sorted(G.degree(weight="weight"), key=lambda x: -x[1])[:5]]
        summary["graphs"].append({
            "file": path.name, "cop": doc.meta.cop, "year": doc.meta.year,
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "density": round(nx.density(G), 4), "top_actors": top,
            **({"signed": signed_summary(G)}
               if isinstance(edge_def, (DirectStanceEdges, CoStanceEdges)) else {}),
        })
        log.info("%s -> %d nodes, %d edges", path.name, G.number_of_nodes(), G.number_of_edges())

    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-year actor graphs from ENB reports.")
    ap.add_argument("--input", required=True, type=Path, help="folder of ENB .pdf reports")
    ap.add_argument("--output", type=Path, default=Path("graphs"), help="output folder")
    ap.add_argument("--edge", default="proximity", choices=["proximity", "stance", "costance"], help="edge definition")
    ap.add_argument("--window", default="sentence", choices=["sentence", "chunk"],
                    help="proximity window")
    ap.add_argument("--format", default="graphml", choices=list(_EXT), help="graph output format")
    ap.add_argument("--alpha", type=float, default=0.05, help="Wilson 1-confidence (0.05 = 95%%)")
    ap.add_argument("--tier-weights", type=str, default=None,
                    help="extraction-confidence weights, e.g. '1:1.0,2:0.75,3:0.25' (plan C4)")
    ap.add_argument("--attribute-to-bloc", action="store_true",
                    help="'on behalf of' statements accrue to the bloc instead of the speaker")
    ap.add_argument("--years", type=str, default=None, help="comma-separated year filter, e.g. 1998,1999")
    ap.add_argument("--include-orgs", action="store_true", help="keep IPCC/GEF as nodes")
    ap.add_argument("--coref", action="store_true", help="enable pronoun resolution")
    ap.add_argument("--gazetteer", type=Path, default=None, help="override gazetteer YAML")
    args = ap.parse_args()

    years = {int(y) for y in args.years.split(",")} if args.years else None
    summary = run(
        args.input, args.output, edge=args.edge, window=args.window, fmt=args.format,
        years=years, include_orgs=args.include_orgs, coref=args.coref, gazetteer=args.gazetteer,
        alpha=args.alpha,
        tier_weights=parse_tier_weights(args.tier_weights) if args.tier_weights else None,
        attribute_to_bloc=args.attribute_to_bloc,
    )
    print(f"built {len(summary['graphs'])} graphs, skipped {len(summary['skipped'])} "
          f"-> {args.output}/ (manifest.json)")


if __name__ == "__main__":
    main()
