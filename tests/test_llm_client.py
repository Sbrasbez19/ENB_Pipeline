"""
Tests for the extraction client (enbgraph/llm/client.py) against a fake
OpenAI-compatible server (httpx.MockTransport): request shape, parsing,
caching, retries and failure handling. No network, no GPU.
"""

import asyncio
import json

import httpx
import pytest

from enbgraph.llm.cache import ResponseCache
from enbgraph.llm.client import ExtractionClient, ModelSpec, load_model_specs
from enbgraph.llm.prompts import PromptTemplate
from enbgraph.models import Passage, Provenance

SPEC = ModelSpec(name="test", base_url="http://fake/v1", served_model="fake-model")
GOOD = json.dumps({"triplets": [{
    "source_actor": "Japan", "source_mention": "Japan", "source_role": "party",
    "on_behalf_of": None, "reference_type": "explicit", "antecedent_evidence": None,
    "verb": "supported", "polarity": "support", "target_type": "actor", "target": "Canada",
    "proposal": None, "proposal_author": None, "evidence": "Japan supported Canada.",
}]})


def passage(i=1, text="Japan supported Canada."):
    return Passage(passage_id=f"d-p{i:04d}", doc_id="d", focus_text=text, sentence_start=0,
                   sentence_end=1,
                   provenance=Provenance(doc_id="d", page_start=1, page_end=1, start_char=0, end_char=1))


def completion(content, finish="stop"):
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "fake-model",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}]}


class FakeServer:
    """Serves a scripted sequence of responses and records requests."""

    def __init__(self, responses, models=("fake-model",)):
        self.responses = list(responses)
        self.models = models
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": "x"} for m in self.models]})
        self.requests.append(json.loads(request.content))
        status, body = self.responses.pop(0)
        return httpx.Response(status, json=body)


def make_client(server, cache=None, **kw):
    http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    return ExtractionClient(SPEC, PromptTemplate.load(), cache, http_client=http,
                            backoff_base=0, **kw)


def run(coro):
    return asyncio.run(coro)


def test_success_and_request_shape():
    server = FakeServer([(200, completion(GOOD))])

    async def go():
        async with make_client(server) as c:
            return await c.extract_one(passage())

    res = run(go())
    assert res.ok and res.triplets[0].target == "Canada" and res.attempts == 1
    req = server.requests[0]
    assert req["model"] == "fake-model" and req["temperature"] == 0
    assert req["response_format"]["type"] == "json_schema"
    assert "Japan supported Canada." in req["messages"][1]["content"]


def test_cache_hit_skips_request(tmp_path):
    server = FakeServer([(200, completion(GOOD))])
    cache = ResponseCache(tmp_path / "test.jsonl")

    async def go():
        async with make_client(server, cache) as c:
            first = await c.extract_one(passage())
            second = await c.extract_one(passage())
        return first, second

    first, second = run(go())
    assert not first.cached and second.cached and second.ok
    assert len(server.requests) == 1
    # the cache persists across instances (i.e. across runs)
    assert len(ResponseCache(tmp_path / "test.jsonl")) == 1


def test_cache_survives_torn_last_line(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"key": "a", "raw": "x"}\n{"key": "b", "ra', encoding="utf-8")
    assert len(ResponseCache(path)) == 1


def test_retries_on_server_errors():
    server = FakeServer([(500, {"error": "boom"}), (503, {"error": "busy"}), (200, completion(GOOD))])

    async def go():
        async with make_client(server, max_retries=3) as c:
            return await c.extract_one(passage())

    res = run(go())
    assert res.ok and res.attempts == 3


def test_gives_up_after_max_retries():
    server = FakeServer([(500, {"error": "boom"})] * 3)

    async def go():
        async with make_client(server, max_retries=2) as c:
            return await c.extract_one(passage())

    res = run(go())
    assert not res.ok and res.error is not None
    assert "request failed" in res.error and len(server.requests) == 3


def test_client_error_is_not_retried():
    server = FakeServer([(400, {"error": "bad request"})])

    async def go():
        async with make_client(server) as c:
            return await c.extract_one(passage())

    res = run(go())
    assert not res.ok and len(server.requests) == 1


@pytest.mark.parametrize("content, finish, msg", [
    ("not json at all", "stop", "invalid response"),
    (json.dumps({"triplets": [{"source_actor": "Japan"}]}), "stop", "invalid response"),
    (GOOD, "length", "truncated"),
])
def test_bad_responses_are_returned_not_raised(content, finish, msg):
    server = FakeServer([(200, completion(content, finish))])

    async def go():
        async with make_client(server) as c:
            return await c.extract_one(passage())

    res = run(go())
    assert not res.ok and res.error is not None
    assert msg in res.error and res.raw == content


def test_extract_all_calls_back_for_each_passage():
    server = FakeServer([(200, completion(GOOD))] * 3)
    seen = []

    async def go():
        async with make_client(server, concurrency=2) as c:
            return await c.extract_all([passage(i, f"Japan supported Canada {i}.") for i in range(3)],
                                       on_result=lambda p, r: seen.append(p.passage_id))

    results = run(go())
    assert len(results) == 3 and all(r.ok for r in results)
    assert sorted(seen) == ["d-p0000", "d-p0001", "d-p0002"]


def test_wait_until_ready():
    async def go(models):
        async with make_client(FakeServer([], models=models)) as c:
            await c.wait_until_ready(timeout=0, interval=0)

    run(go(("fake-model",)))
    with pytest.raises(TimeoutError):
        run(go(("other-model",)))


def test_load_model_specs(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text("models:\n  a:\n    base_url: http://h:8000/v1\n    served_model: org/A\n", encoding="utf-8")
    specs = load_model_specs(cfg)
    assert specs["a"].served_model == "org/A" and specs["a"].hf_revision is None
