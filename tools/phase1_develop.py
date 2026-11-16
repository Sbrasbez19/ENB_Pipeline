"""
tools/phase1_develop.py — PHASE 1 (development) of the train/test pipeline.

Uses ONLY the training years so nothing about the test years leaks into the
choices that will be frozen. Produces `dev_config.json`, which Phase 2 consumes.

What is decided on training data (and nothing else):
  1. extraction method — pattern vs. parsed (spaCy); chosen by measured precision.
     (The parsed path requires spaCy + en_core_web_sm; where unavailable this
     script runs 'pattern' and records that.)
  2. tier weights — set EQUAL TO measured per-tier precision (an observation is
     worth evidence proportional to how often that channel is correct). This is
     the principled objective; we deliberately do NOT tune weights toward the
     balance result, which must stay an independent validation.
  3. alpha policy — NOT fitted. Wilson intervals are closed-form in (k, n); there
     is nothing to estimate. Instead we DECLARE the policy (per-year 68%,
     pooled 95%) and JUSTIFY it with a training-set diagnostic: the contested
     fraction at each level, showing 95% is uninformative per-year but reachable
     when pooled.

Subcommands:
  sample  — write a stratified CSV of training records to hand-label
  freeze  — read the labeled CSV, compute precision -> weights, run the alpha
            diagnostic, and write dev_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from enbgraph.corpus import load_corpus
from enbgraph.extraction.stance import StanceExtractor
from enbgraph.graph.aggregate import aggregate_signed, signed_summary
from enbgraph.graph.builder import GraphBuilder
from enbgraph.graph.stance import CoStanceEdges, DirectStanceEdges
from enbgraph.ingest.chunker import StructuralChunker
from enbgraph.extraction.actors import ActorExtractor
from enbgraph.models import SectionKind

def make_stance_extractor(kind: str):
    if kind == "parsed":
        from enbgraph.extraction.stance_parsed import ParsedStanceExtractor
        return ParsedStanceExtractor()   # requires spaCy + en_core_web_sm
    return StanceExtractor()


TRAIN_DEFAULT = [1996, 1999, 2002, 2005]
TEST_DEFAULT = [1998, 2000, 2001, 2003, 2006]


def _docs(corpus: Path, years: list[int]):
    keep = set(years)
    for ld in load_corpus(corpus):
        if ld.status == "ok" and ld.document and ld.document.meta.year in keep:
            yield ld.document


def _records(corpus: Path, years: list[int], extractor: StanceExtractor):
    ch = StructuralChunker()
    for doc in _docs(corpus, years):
        for c in ch.chunk(doc):
            if c.kind != SectionKind.REPORT_ITEM:
                continue
            for r in extractor.extract(c):
                yield doc.meta.year, c, r


# --------------------------------------------------------------------------- #
def cmd_sample(a):
    ex = make_stance_extractor(a.extractor)
    rows = []
    for year, c, r in _records(a.corpus, a.train, ex):
        rows.append({
            "year": year, "tier": r.tier, "subject": r.subject, "sign": r.sign,
            "verb": r.verb, "object_type": r.object_type, "object": r.object or "",
            "attributed_to": r.attributed_to or "",
            "source": r.source,
            "sentence": c.sentences[r.sentence_index].text[:500], "label": "",
        })
    rng = random.Random(a.seed)
    by_tier = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
    picked = []
    for t in sorted(by_tier):
        picked += rng.sample(by_tier[t], min(a.per_tier, len(by_tier[t])))
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(picked[0].keys()))
        w.writeheader(); w.writerows(picked)
    print(f"training records total: {len(rows)} | sampled {len(picked)} -> {a.out}")
    print("tier counts (all training):", dict(Counter(r['tier'] for r in rows)))


# --------------------------------------------------------------------------- #
def _alpha_diagnostic(corpus: Path, years: list[int], tier_weights, extractor: str):
    """Contested fraction of the DIRECT stance graph at 95% vs 68%, per-year
    (mean) and pooled, on training data — the justification for the alpha policy."""
    ch, ax = StructuralChunker(), ActorExtractor()
    ed = DirectStanceEdges(extractor=make_stance_extractor(extractor), tier_weights=tier_weights)
    per_year = defaultdict(dict)
    pooled_items = []
    for doc in _docs(corpus, years):
        chunks = [c for c in ch.chunk(doc) if c.kind == SectionKind.REPORT_ITEM]
        items = [(c, ax.extract(c)) for c in chunks]
        pooled_items += items
        for lv, al in (("95", 0.05), ("68", 0.32)):
            G = GraphBuilder(ed).build(items)
            aggregate_signed(G, alpha=al, inplace=True)
            s = signed_summary(G)
            tot = s["positive"] + s["negative"] + s["tied"]
            per_year[lv][doc.meta.year] = (s["contested"] / tot) if tot else 0.0
    pooled = {}
    for lv, al in (("95", 0.05), ("68", 0.32)):
        G = GraphBuilder(ed).build(pooled_items)
        aggregate_signed(G, alpha=al, inplace=True)
        s = signed_summary(G)
        tot = s["positive"] + s["negative"] + s["tied"]
        pooled[lv] = (s["contested"] / tot) if tot else 0.0
    mean = {lv: round(sum(d.values()) / len(d), 3) for lv, d in per_year.items()}
    return {"per_year_mean_contested": mean,
            "pooled_contested": {lv: round(v, 3) for lv, v in pooled.items()}}


def cmd_freeze(a):
    # 1. precision per tier from the labeled sample -> tier weights.
    #    Also break down by source (parsed vs pattern) so the config records
    #    how much of each tier's precision the parser actually contributed —
    #    the parsed-vs-pattern comparison the two-phase design exists to measure.
    per = defaultdict(lambda: [0, 0])
    per_src = defaultdict(lambda: [0, 0])            # (source, tier) -> [ok, n]
    for row in csv.DictReader(a.labeled.open()):
        if row["label"] not in ("0", "1"):
            continue
        t = int(row["tier"]); ok = row["label"] == "1"
        per[t][0] += ok; per[t][1] += 1
        src = row.get("source", "pattern") or "pattern"
        per_src[(src, t)][0] += ok; per_src[(src, t)][1] += 1
    precision, weights = {}, {}
    for t in (1, 2, 3):
        ok, n = per.get(t, [0, 0])
        p = round(ok / n, 3) if n else None
        precision[t] = p
        weights[t] = p if p is not None else {1: 1.0, 2: 0.75, 3: 0.25}[t]
    # source x tier table for the config (precision + sample size n)
    src_tier = {}
    for (src, t), (ok, n) in sorted(per_src.items()):
        src_tier[f"{src}:tier{t}"] = {"precision": round(ok / n, 3) if n else None, "n": n}

    # 2. alpha SELECTED on training data by a stated rule: the strictest level
    #    (95% preferred) whose contested fraction <= 0.5; else fall back to 68%.
    diag = _alpha_diagnostic(a.corpus, a.train, weights, a.extractor)

    def _select(contested_by_level: dict) -> float:
        #contested_by_level: {"95": frac, "68": frac}
        if contested_by_level["95"] <= 0.5:
            return 0.05
        return 0.32

    alpha_per_year = _select(diag["per_year_mean_contested"])
    alpha_pooled = _select(diag["pooled_contested"])

    cfg = {
        "training_years": a.train, "test_years": a.test,
        "extractor": a.extractor,
        "tier_precision": precision,
        "precision_by_source_tier": src_tier,
        "tier_weights": {str(k): v for k, v in weights.items()},
        "alpha_selected": {"per_year": alpha_per_year, "pooled": alpha_pooled},
        "alpha_selection_rule": "strictest level with training contested-fraction <= 0.5 (95% preferred, else 68%)",
        "alpha_diagnostic": diag,
        "labeled_sample": str(a.labeled),
        "note": ("tier_weights = measured per-tier precision on TRAINING records; "
                 "alpha SELECTED per regime by the stated rule on the training "
                 "contested-fraction diagnostic (Wilson itself is closed-form, not "
                 "fitted); nothing here used the test years."),
    }
    a.out.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))
    print(f"\nwrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description="Phase 1: develop config on training years.")
    sub = ap.add_subparsers(required=True)
    common = dict()

    s = sub.add_parser("sample"); s.set_defaults(fn=cmd_sample)
    s.add_argument("--corpus", type=Path, required=True)
    s.add_argument("--train", type=int, nargs="+", default=TRAIN_DEFAULT)
    s.add_argument("--per-tier", type=int, default=20)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--extractor", default="pattern", choices=["pattern", "parsed"])
    s.add_argument("--out", type=Path, default=Path("train_sample.csv"))

    f = sub.add_parser("freeze"); f.set_defaults(fn=cmd_freeze)
    f.add_argument("--corpus", type=Path, required=True)
    f.add_argument("--labeled", type=Path, required=True)
    f.add_argument("--train", type=int, nargs="+", default=TRAIN_DEFAULT)
    f.add_argument("--test", type=int, nargs="+", default=TEST_DEFAULT)
    f.add_argument("--extractor", default="pattern", choices=["pattern", "parsed"])
    f.add_argument("--out", type=Path, default=Path("dev_config.json"))

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()