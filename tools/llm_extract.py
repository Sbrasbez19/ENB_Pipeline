"""
tools/llm_extract.py — run the LLM stance extractor over a corpus.

    corpus (PDFs) -> passages -> model server -> grounded TripletRecords

Usage:
  # passages only (no server needed): inspect the chunking
  python -m tools.llm_extract --corpus samples --out runs/dry --dry-run

  # extraction against a running vLLM server
  python -m tools.llm_extract --corpus samples --out runs/qwen-v1 --model qwen3-32b

Outputs in --out:
  passages.jsonl      one Passage per line (what the model saw)
  triplets.jsonl      one TripletRecord per line (triplet + metadata + grounding)
  failures.jsonl      passages whose response failed (bad JSON, truncation, errors)
  run_manifest.json   settings, versions and counts needed to reproduce the run

Responses are cached per model in --cache-dir, so an interrupted or repeated run
only queries what is new.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional

from enbgraph.corpus import LoadedDoc, load_corpus, load_document
from enbgraph.extraction.gazetteer import Gazetteer
from enbgraph.ingest.passages import PassageChunker, detect_source_type
from enbgraph.llm.cache import ResponseCache
from enbgraph.llm.client import ExtractionClient, ExtractionResult, load_model_specs
from enbgraph.llm.grounding import check
from enbgraph.llm.prompts import DEFAULT_PROMPT, PromptTemplate
from enbgraph.llm.schema import (SCHEMA_VERSION, ExtractionInfo, SourceInfo, TripletRecord)
from enbgraph.models import Passage, SourceType
from enbgraph.utils import get_logger, setup_logging

log = get_logger("llm_extract")
DEFAULT_MODELS = Path(__file__).resolve().parents[1] / "enbgraph" / "config" / "llm_models.yaml"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1], timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def load_inputs(corpus: Path) -> list[LoadedDoc]:
    return [load_document(corpus)] if corpus.is_file() else load_corpus(corpus)


def build_passages(docs: list[LoadedDoc], chunker: PassageChunker,
                   source_type: Optional[str]) -> tuple[list[Passage], list[dict]]:
    passages: list[Passage] = []
    skipped: list[dict] = []
    for ld in docs:
        if ld.status != "ok" or ld.document is None:
            skipped.append({"name": ld.name, "status": ld.status})
            log.warning("skipped %s (%s)", ld.name, ld.status)
            continue
        doc = ld.document
        doc.meta.source_type = SourceType(source_type) if source_type else detect_source_type(doc)
        ps = chunker.chunk(doc)
        log.info("%s [%s]: %d passages", doc.meta.doc_id, doc.meta.source_type.value, len(ps))
        passages.extend(ps)
    return passages, skipped


def write_jsonl(path: Path, items) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write((it.model_dump_json() if hasattr(it, "model_dump_json") else json.dumps(it, ensure_ascii=False)) + "\n")


def passage_stats(passages: list[Passage]) -> dict:
    words = [p.n_words for p in passages]
    by_type = Counter(p.source_type.value for p in passages)
    return {
        "passages": len(passages),
        "documents": len({p.doc_id for p in passages}),
        "by_source_type": dict(by_type),
        "words_min": min(words) if words else 0,
        "words_median": median(words) if words else 0,
        "words_max": max(words) if words else 0,
        "words_total": sum(words),
    }


async def extract(passages: list[Passage], args, out: Path, manifest: dict) -> dict:
    specs = load_model_specs(args.models_config)
    if args.model not in specs:
        sys.exit(f"model {args.model!r} not in {args.models_config} (available: {sorted(specs)})")
    spec = specs[args.model]
    prompt = PromptTemplate.load(args.prompt)
    cache = ResponseCache(args.cache_dir / f"{spec.name}.jsonl")
    gazetteer = Gazetteer(args.gazetteer) if args.gazetteer else Gazetteer()
    info = ExtractionInfo(run_id=out.name, model=spec.served_model, model_revision=spec.hf_revision,
                          prompt_version=prompt.version)
    manifest.update({"model": spec.name, "served_model": spec.served_model,
                     "model_revision": spec.hf_revision, "base_url": spec.base_url,
                     "prompt_version": prompt.version, "cache": str(cache.path),
                     "cache_entries_before": len(cache)})

    counts: Counter = Counter()
    flag_counts: Counter = Counter()
    done = 0
    with (out / "triplets.jsonl").open("w", encoding="utf-8") as trip_fh, \
         (out / "failures.jsonl").open("w", encoding="utf-8") as fail_fh:

        def on_result(p: Passage, res: ExtractionResult) -> None:
            nonlocal done
            done += 1
            counts["cached" if res.cached else "queried"] += 1
            if not res.ok:
                counts["failed"] += 1
                fail_fh.write(json.dumps({"passage_id": p.passage_id, "error": res.error,
                                          "raw": res.raw}, ensure_ascii=False) + "\n")
            for k, t in enumerate(res.triplets, start=1):
                g = check(t, p, gazetteer)
                counts["triplets"] += 1
                counts["grounded"] += g.passed
                flag_counts.update(g.flags)
                rec = TripletRecord(triplet_id=f"{p.passage_id}-t{k:02d}", triplet=t,
                                    source=SourceInfo.from_passage(p), extraction=info, grounding=g)
                trip_fh.write(rec.model_dump_json() + "\n")
            if done % 25 == 0 or done == len(passages):
                trip_fh.flush(); fail_fh.flush()
                log.info("%d/%d passages done (%d triplets so far)", done, len(passages), counts["triplets"])

        client = ExtractionClient(spec, prompt, cache, concurrency=args.concurrency,
                                  max_tokens=args.max_tokens)
        try:
            if not args.no_wait:
                await client.wait_until_ready(timeout=args.wait_timeout)
            await client.extract_all(passages, on_result=on_result)
        finally:
            await client.aclose()

    return {"counts": dict(counts), "grounding_flags": dict(flag_counts.most_common())}


def main(argv: Optional[list[str]] = None) -> None:
    # __doc__ is None under `python -OO` (docstrings stripped)
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("Usage:")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path, help="a PDF file or a folder of PDFs")
    ap.add_argument("--out", required=True, type=Path, help="run folder (created)")
    ap.add_argument("--model", help="model name from --models-config (not needed with --dry-run)")
    ap.add_argument("--models-config", type=Path, default=DEFAULT_MODELS)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt version (file in enbgraph/llm/prompts)")
    ap.add_argument("--cache-dir", type=Path, default=Path("runs/cache"))
    ap.add_argument("--gazetteer", type=Path, default=None, help="override gazetteer YAML")
    ap.add_argument("--source-type", choices=[s.value for s in SourceType], default=None,
                    help="force a source type instead of auto-detecting it per document")
    ap.add_argument("--dry-run", action="store_true", help="write passages only, no model calls")
    ap.add_argument("--limit", type=int, default=None, help="only the first N passages")
    ap.add_argument("--max-words", type=int, default=600)
    ap.add_argument("--min-words", type=int, default=250)
    ap.add_argument("--context-sentences", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--wait-timeout", type=float, default=1800, help="seconds to wait for the server")
    ap.add_argument("--no-wait", action="store_true", help="skip the readiness check")
    args = ap.parse_args(argv)
    setup_logging()

    if not args.dry_run and not args.model:
        ap.error("--model is required unless --dry-run")
    args.out.mkdir(parents=True, exist_ok=True)

    chunker = PassageChunker(max_words=args.max_words, min_words=args.min_words,
                             context_sentences=args.context_sentences)
    passages, skipped = build_passages(load_inputs(args.corpus), chunker, args.source_type)
    if args.limit:
        passages = passages[:args.limit]
    write_jsonl(args.out / "passages.jsonl", passages)

    manifest: dict = {
        "run_id": args.out.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "corpus": str(args.corpus),
        "chunker": chunker.params(),
        "schema_version": SCHEMA_VERSION,
        "skipped_documents": skipped,
        "passage_stats": passage_stats(passages),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        manifest.update(asyncio.run(extract(passages, args, args.out, manifest)))
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("passage_stats",) + (("counts",) if "counts" in manifest else ())},
                     indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
