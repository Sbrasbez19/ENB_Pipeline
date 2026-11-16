"""
tools/validate_stance.py — stratified sample of stance records for manual
validation (plan C5), mirroring Sudhahar et al.'s protocol (72 manually coded
triplets; 62% precision / 57% recall reported for unfiltered extraction).

Produces a CSV template: one row per sampled record with its source sentence
and an empty `label` column (1 = correct extraction, 0 = wrong). Precision per
tier is then computed from the filled CSV with --score.

    python -m tools.validate_stance --corpus DIR --per-tier 15 --out sample.csv
    python -m tools.validate_stance --score labeled.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from enbgraph.corpus import load_corpus
from enbgraph.extraction.stance import StanceExtractor
from enbgraph.ingest.chunker import StructuralChunker
from enbgraph.models import SectionKind


def collect(corpus: Path):
    ch, ex = StructuralChunker(), StanceExtractor()
    rows = []
    for ld in load_corpus(corpus):
        if ld.status != "ok" or ld.document is None or not ld.document.meta.year:
            continue
        if ld.document.meta.year > 2010:      # exclude non-ENB PDFs in the folder
            continue
        for c in ch.chunk(ld.document):
            if c.kind != SectionKind.REPORT_ITEM:
                continue
            for r in ex.extract(c):
                rows.append({
                    "year": ld.document.meta.year, "tier": r.tier,
                    "subject": r.subject, "sign": r.sign, "verb": r.verb,
                    "object_type": r.object_type, "object": r.object or "",
                    "attributed_to": r.attributed_to or "",
                    "sentence": c.sentences[r.sentence_index].text[:500],
                    "label": "",
                })
    return rows


def sample(rows, per_tier: int, seed: int):
    rng = random.Random(seed)
    by_tier = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
    out = []
    for t in sorted(by_tier):
        pool = by_tier[t]
        out += rng.sample(pool, min(per_tier, len(pool)))
    return out


def score(path: Path) -> None:
    per = defaultdict(lambda: [0, 0])
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["label"] not in ("0", "1"):
                continue
            per[row["tier"]][0] += row["label"] == "1"
            per[row["tier"]][1] += 1
    tot_ok = tot_n = 0
    for t in sorted(per):
        ok, n = per[t]
        tot_ok += ok; tot_n += n
        print(f"tier {t}: precision {ok}/{n} = {ok/n:.2f}")
    if tot_n:
        print(f"overall: {tot_ok}/{tot_n} = {tot_ok/tot_n:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--per-tier", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("stance_sample.csv"))
    ap.add_argument("--score", type=Path)
    a = ap.parse_args()
    if a.score:
        score(a.score); return
    rows = sample(collect(a.corpus), a.per_tier, a.seed)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} records -> {a.out}")


if __name__ == "__main__":
    main()
