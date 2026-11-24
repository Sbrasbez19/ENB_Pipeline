"""
tools/audit_gazetteer.py — reconcile the gazetteer against a document corpus.

This does NOT generate the gazetteer from scratch. The gazetteer is a curated
artifact that encodes modeling choices (which aliases merge, whether 'Annex I'
counts as an actor), so it stays human-authored. This tool's job is to keep it
HONEST against the data: given a corpus, it reports what the gazetteer covers,
what is dead weight, and — most importantly — which real actors the corpus
contains that the gazetteer is missing, so a human can review and fold them in.

Run it whenever new reports arrive:

    python -m tools.audit_gazetteer --corpus /path/to/pdfs --json audit.json

It prints a summary and, with --json, writes a machine-readable record you can
diff over time. Suggested additions are emitted in YAML-ready form.

Extra dependency (tools only, not the runtime package): `pycountry`.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from enbgraph.extraction.gazetteer import Gazetteer
from enbgraph.ingest.ingester import ENBIngester
from enbgraph.utils import normalize_prose


# --------------------------------------------------------------------------- #
# Corpus loading — handles both the ZIP-packaged page format and real PDFs,
# and flags scanned (image-only) documents that need OCR.
# --------------------------------------------------------------------------- #
@dataclass
class LoadResult:
    name: str
    fmt: str            # "zip" | "pdf"
    status: str         # "ok" | "scanned" | "empty"
    chars: int
    cop: Optional[str]
    year: Optional[int]
    text: str = ""


def load_document(path: Path) -> LoadResult:
    head = path.read_bytes()[:4]
    if head.startswith(b"PK"):
        z = zipfile.ZipFile(path)
        man = json.loads(z.read("manifest.json"))
        text = "\n".join(z.read(p["text"]["path"]).decode("utf-8", "ignore") for p in man["pages"])
        fmt = "zip"
    else:
        text = ENBIngester().ingest(path).full_text
        fmt = "pdf"
    status = "ok" if len(text) > 1000 else ("scanned" if fmt == "pdf" else "empty")
    cop = re.search(r"COP[-\s]?(\d+)", text)
    yr = re.search(r"\b(?:19|20)\d{2}\b", text)
    return LoadResult(path.name, fmt, status, len(text),
                      cop.group(1) if cop else None,
                      int(yr.group(0)) if yr else None,
                      text if status == "ok" else "")


# --------------------------------------------------------------------------- #
# Candidate harvesting (permissive proper-noun runs, both casing eras)
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’.&-]*")
_CONNECT = {"of", "the", "and", "for"}
_DISCOURSE = {
    "the", "this", "these", "however", "following", "during", "after", "several",
    "while", "regarding", "they", "states", "delegates", "many", "for", "conference",
    "bulletin", "fund", "declaration", "internet", "kyoto", "kyoto protocol",
    "protocol article", "protocol articles", "earth negotiations bulletin",
}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december", "monday", "tuesday",
           "wednesday", "thursday", "friday", "saturday", "sunday"}


def _is_cap(tok: str) -> bool:
    return tok[:1].isupper() and any(c.isalpha() for c in tok)


def harvest_candidates(text: str) -> list[tuple[str, int, int]]:
    toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(toks):
        if _is_cap(toks[i][0]):
            j, parts = i, [toks[i]]
            while j + 1 < len(toks):
                nxt = toks[j + 1][0]
                if _is_cap(nxt):
                    parts.append(toks[j + 1]); j += 1
                elif nxt.lower() in _CONNECT and j + 2 < len(toks) and _is_cap(toks[j + 2][0]):
                    parts += [toks[j + 1], toks[j + 2]]; j += 2
                else:
                    break
                if len(parts) >= 6:
                    break
            out.append((" ".join(p[0] for p in parts), parts[0][1], parts[-1][2]))
            i = j + 1
        else:
            i += 1
    return out


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
@dataclass
class AuditReport:
    corpus: list[dict] = field(default_factory=list)
    gazetteer_actors: int = 0
    actors_present: int = 0
    actors_absent: list[str] = field(default_factory=list)
    total_mentions: int = 0
    top_actors: list[tuple[str, int]] = field(default_factory=list)
    missed_countries: list[tuple[str, int]] = field(default_factory=list)
    triage: list[tuple[str, int]] = field(default_factory=list)


def _country_reference() -> dict[str, str]:
    import pycountry  # pyright: ignore[reportMissingImports]  # optional, tools-only: pip install pycountry
    keys: dict[str, str] = {}
    for c in pycountry.countries:
        for attr in ("name", "official_name", "common_name"):
            v = getattr(c, attr, None)
            if v:
                keys[v.lower()] = c.name
    return keys


def run_audit(corpus_dir: Path, gazetteer_path: Optional[Path] = None) -> AuditReport:
    gaz = Gazetteer(gazetteer_path)
    country_ref = _country_reference()
    known_ci = set(gaz._forms_ci)

    rep = AuditReport()
    loaded = [load_document(p) for p in sorted(corpus_dir.glob("*.pdf"))]
    texts = {}
    for lr in loaded:
        rep.corpus.append({"name": lr.name, "fmt": lr.fmt, "status": lr.status,
                           "cop": lr.cop, "year": lr.year, "chars": lr.chars})
        if lr.status == "ok":
            texts[lr.name] = normalize_prose(lr.text)

    cov: Counter = Counter()
    missed: Counter = Counter()
    triage: Counter = Counter()
    for name, t in texts.items():
        spans = [(m.start_char, m.end_char) for m in gaz.find(t, name)]
        for m in gaz.find(t, name):
            cov[m.actor_id] += 1
        for phrase, s, e in harvest_candidates(t):
            if any(s < b and a < e for a, b in spans):
                continue
            key = re.sub(r"\s+", " ", phrase).strip().rstrip(".,;:’'")
            kl = key.lower()
            if (kl in _DISCOURSE or kl in _MONTHS or kl in {x.lower() for x in gaz.stoplist}
                    or re.search(r"fccc|/|\d", kl) or len(kl) < 3):
                continue
            if kl in country_ref and gaz._lookup(kl) is None and kl not in known_ci:
                missed[country_ref[kl]] += 1
            elif kl not in country_ref:
                triage[key] += 1

    all_actors = {f.actor_id for f in (*gaz._forms.values(), *gaz._forms_ci.values())}
    rep.gazetteer_actors = len(all_actors)
    rep.actors_present = len(cov)
    rep.actors_absent = sorted(all_actors - set(cov))
    rep.total_mentions = sum(cov.values())
    rep.top_actors = cov.most_common(15)
    rep.missed_countries = missed.most_common()
    rep.triage = triage.most_common(30)
    return rep


def _print_summary(rep: AuditReport) -> None:
    print("CORPUS")
    for d in rep.corpus:
        flag = "" if d["status"] == "ok" else f"  <-- {d['status'].upper()} (needs OCR)" if d["status"] == "scanned" else "  <-- EMPTY"
        print(f"  {d['name']:14s} {d['fmt']:3s} COP-{d['cop'] or '?':<3} {d['year'] or '?'}  {d['chars']:>7} chars{flag}")
    print(f"\nCOVERAGE: {rep.actors_present}/{rep.gazetteer_actors} actors appear | "
          f"{rep.total_mentions} mentions | {len(rep.actors_absent)} never appear")
    print("  top:", ", ".join(f"{a}({n})" for a, n in rep.top_actors[:10]))
    print("\nMISSED COUNTRIES (real, not in gazetteer):",
          ", ".join(f"{c}({n})" for c, n in rep.missed_countries) or "(none)")
    print("\nTRIAGE (unmatched capitalised phrases — review for blocs/orgs):")
    for k, n in rep.triage[:15]:
        print(f"  {n:4d}  {k}")
    if rep.missed_countries:
        print("\nSUGGESTED YAML ADDITIONS (countries):")
        for c, _ in rep.missed_countries:
            print(f"  - {c}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the actor gazetteer against a corpus.")
    ap.add_argument("--corpus", required=True, type=Path, help="directory of ENB .pdf files")
    ap.add_argument("--gazetteer", type=Path, default=None, help="gazetteer YAML (default: package config)")
    ap.add_argument("--json", type=Path, default=None, help="write the full report as JSON")
    args = ap.parse_args()

    rep = run_audit(args.corpus, args.gazetteer)
    _print_summary(rep)
    if args.json:
        args.json.write_text(json.dumps(asdict(rep), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
