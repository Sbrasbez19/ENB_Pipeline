"""
End-to-end test of tools/llm_extract.py: a generated PDF goes through the
real ingester, chunker, client (over HTTP to a local fake OpenAI-compatible
server), grounding and output writers. A second run must be served entirely
from the cache.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import fitz
import pytest

from tools.llm_extract import main

SENTENCES = [
    "Canada, with the support of Japan, proposed replacing the existing decision.",
    "The European Union opposed the proposal by Canada on procedural grounds.",
    "Tuvalu called for an urgent review of adequacy of commitments.",
] * 10   # > 1000 chars, or corpus.load_document treats the PDF as scanned

RESPONSE = {"triplets": [{
    "source_actor": "Japan", "source_mention": "Japan", "source_role": "party",
    "on_behalf_of": None, "reference_type": "explicit", "antecedent_evidence": None,
    "verb": "support", "polarity": "support", "target_type": "actor", "target": "Canada",
    "proposal": "replacing the existing decision", "proposal_author": "Canada",
    "evidence": "Canada, with the support of Japan, proposed replacing the existing decision.",
}, {
    "source_actor": "European Union", "source_mention": "The European Union", "source_role": "party",
    "on_behalf_of": None, "reference_type": "explicit", "antecedent_evidence": None,
    "verb": "opposed", "polarity": "oppose", "target_type": "actor", "target": "Canada",
    "proposal": None, "proposal_author": "Canada",
    "evidence": "The European Union opposed the proposal by Canada in a closed session.",  # not in text
}]}


class _Handler(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, format, *args):  # keep test output quiet
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "fake-model", "object": "model",
                                                 "created": 0, "owned_by": "x"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).calls += 1
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": "fake-model",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": json.dumps(RESPONSE)}}]})


@pytest.fixture
def server():
    _Handler.calls = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    pdf = fitz.open()
    page = pdf.new_page()
    y = 72
    for s in SENTENCES:
        page.insert_text((50, y), s, fontsize=9)
        y += 14
    pdf.save(folder / "test_doc.pdf")
    return folder


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_end_to_end_with_cache(tmp_path, server, corpus):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(f"models:\n  fake:\n    base_url: {server}\n    served_model: fake-model\n",
                   encoding="utf-8")
    common = ["--corpus", str(corpus), "--model", "fake", "--models-config", str(cfg),
              # windows of >= 4 sentences, so each contains the Japan/Canada sentence
              "--cache-dir", str(tmp_path / "cache"), "--min-words", "40", "--max-words", "80",
              "--wait-timeout", "5"]

    out1 = tmp_path / "run1"
    main(common + ["--out", str(out1)])
    passages = read_jsonl(out1 / "passages.jsonl")
    records = read_jsonl(out1 / "triplets.jsonl")
    manifest = json.loads((out1 / "run_manifest.json").read_text(encoding="utf-8"))

    assert len(passages) >= 2
    assert _Handler.calls == len(passages)
    assert len(records) == 2 * len(passages)
    rec = records[0]
    assert rec["source"]["doc_id"] == "test_doc" and rec["source"]["passage_id"] == passages[0]["passage_id"]
    assert rec["extraction"]["model"] == "fake-model" and rec["extraction"]["prompt_version"] == "extract_v1"
    assert rec["grounding"]["source_id"] == "Japan" and rec["grounding"]["target_id"] == "Canada"
    assert rec["grounding"]["evidence_found"] is True
    # the second triplet's quote was altered by the "model": grounding must catch it
    assert records[1]["grounding"]["evidence_found"] is False
    assert manifest["counts"]["queried"] == len(passages) and manifest["counts"].get("failed", 0) == 0
    # every passage's EU quote is fake; the Japan quote is real only where the sentence occurs
    japan_missing = sum(SENTENCES[0] not in p["focus_text"] for p in passages)
    assert manifest["grounding_flags"]["evidence_not_in_focus"] == len(passages) + japan_missing
    assert manifest["counts"]["grounded"] == len(passages) - japan_missing

    # rerun: everything from cache, identical triplets
    out2 = tmp_path / "run2"
    main(common + ["--out", str(out2)])
    assert _Handler.calls == len(passages)
    manifest2 = json.loads((out2 / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest2["counts"]["cached"] == len(passages) and "queried" not in manifest2["counts"]
    def key(rs):
        return sorted((r["triplet_id"], json.dumps(r["triplet"], sort_keys=True)) for r in rs)
    assert key(read_jsonl(out2 / "triplets.jsonl")) == key(records)
