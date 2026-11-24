"""
llm/client.py — send passages to an OpenAI-compatible server (vLLM) and parse
the structured responses.

Nothing here is vLLM-specific: the same client works against any server that
speaks the OpenAI chat-completions API, including a paid API later. Design:

  * concurrency-capped async requests (the server batches them on the GPUs)
  * temperature 0 + fixed seed (as deterministic as the server allows)
  * retries with backoff on connection errors / timeouts / 5xx / 429 only
  * every response goes through the cache first; failures (bad JSON, schema
    violation, truncation, client errors) are RETURNED, never raised, so one
    bad passage cannot stop a corpus run
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional, cast

import httpx
import openai
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import ValidationError

from ..models import Passage
from ..utils import get_logger, load_yaml
from .cache import ResponseCache, cache_key
from .prompts import PromptTemplate
from .schema import SCHEMA_VERSION, ExtractionResponse, StanceTriplet, response_format

log = get_logger(__name__)

_RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError,
              openai.InternalServerError, openai.RateLimitError)


@dataclass(frozen=True)
class ModelSpec:
    name: str                          # our label, e.g. "qwen3-32b"
    base_url: str                      # e.g. "http://gpu-node-03:8000/v1"
    served_model: str                  # the model id the server answers to
    hf_revision: Optional[str] = None  # Hugging Face commit hash, for provenance
    api_key: str = "EMPTY"             # vLLM ignores it unless started with --api-key


def load_model_specs(path: Path) -> dict[str, ModelSpec]:
    cfg = load_yaml(path)
    return {name: ModelSpec(name=name, **spec) for name, spec in (cfg.get("models") or {}).items()}


@dataclass
class ExtractionResult:
    passage_id: str
    ok: bool
    triplets: list[StanceTriplet] = field(default_factory=list)
    raw: Optional[str] = None          # the model's raw text output
    error: Optional[str] = None
    cached: bool = False
    attempts: int = 0


def parse_response(raw: Optional[str], finish_reason: Optional[str]) -> tuple[list[StanceTriplet], Optional[str]]:
    """Return (triplets, error). Truncated output is an error even if it parses."""
    if finish_reason == "length":
        return [], "truncated: hit max_tokens"
    if not raw:
        return [], "empty response"
    try:
        return ExtractionResponse.model_validate_json(raw).triplets, None
    except ValidationError as e:
        return [], f"invalid response: {e.error_count()} validation error(s): {e.errors()[0]['msg']}"


class ExtractionClient:
    def __init__(
        self,
        spec: ModelSpec,
        prompt: PromptTemplate,
        cache: Optional[ResponseCache] = None,
        concurrency: int = 32,
        max_retries: int = 4,
        backoff_base: float = 1.0,     # seconds before the first retry; doubles each time
        timeout: float = 600.0,
        max_tokens: int = 4096,
        seed: int = 0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.spec = spec
        self.prompt = prompt
        self.cache = cache
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_tokens = max_tokens
        self.seed = seed
        self._sem = asyncio.Semaphore(concurrency)
        self._client = openai.AsyncOpenAI(
            base_url=spec.base_url, api_key=spec.api_key, timeout=timeout,
            max_retries=0,  # we retry ourselves, with logging
            http_client=http_client,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "ExtractionClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # -- readiness --------------------------------------------------------------
    async def wait_until_ready(self, timeout: float = 1800, interval: float = 10) -> None:
        """Poll /v1/models until the server lists our model. A 70B model takes
        minutes to load after the SLURM job starts."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            try:
                ids = [m.id async for m in self._client.models.list()]
                if self.spec.served_model in ids:
                    log.info("%s ready at %s", self.spec.served_model, self.spec.base_url)
                    return
                log.info("server up but serving %s, waiting for %s", ids, self.spec.served_model)
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
                log.info("waiting for %s ...", self.spec.base_url)
            if loop.time() >= deadline:
                raise TimeoutError(f"{self.spec.served_model} not ready at {self.spec.base_url} "
                                   f"after {timeout:.0f}s")
            await asyncio.sleep(interval)

    # -- extraction -------------------------------------------------------------
    def key_for(self, passage: Passage) -> str:
        return cache_key(self.spec.served_model, self.spec.hf_revision, self.prompt.version,
                         SCHEMA_VERSION, passage.context_before, passage.focus_text)

    async def _request(self, passage: Passage) -> tuple[Optional[str], Optional[str], int]:
        """Return (raw_text, finish_reason, attempts). Raises on non-retryable errors."""
        # prompts/schema build plain dicts (provider-agnostic); cast them to the
        # openai package's TypedDicts here, where they enter the openai call
        messages = cast(list[ChatCompletionMessageParam],
                        self.prompt.messages(passage.context_before, passage.focus_text))
        fmt = cast(ResponseFormat, response_format())
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.spec.served_model, messages=messages,
                    temperature=0, seed=self.seed, max_tokens=self.max_tokens,
                    response_format=fmt,
                )
                choice = resp.choices[0]
                return choice.message.content, choice.finish_reason, attempt
            except _RETRYABLE as e:
                if attempt > self.max_retries:
                    raise
                delay = self.backoff_base * 2 ** (attempt - 1)
                log.warning("%s: %s (attempt %d), retrying in %.0fs",
                            passage.passage_id, type(e).__name__, attempt, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def extract_one(self, passage: Passage) -> ExtractionResult:
        key = self.key_for(passage)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            triplets, err = parse_response(hit.get("raw"), hit.get("finish_reason"))
            return ExtractionResult(passage.passage_id, err is None, triplets,
                                    hit.get("raw"), err, cached=True)
        async with self._sem:
            try:
                raw, finish, attempts = await self._request(passage)
            except openai.OpenAIError as e:   # retries exhausted or a 4xx
                return ExtractionResult(passage.passage_id, False,
                                        error=f"request failed: {type(e).__name__}: {e}")
        if self.cache is not None:
            self.cache.put(key, {"passage_id": passage.passage_id, "model": self.spec.served_model,
                                 "raw": raw, "finish_reason": finish})
        triplets, err = parse_response(raw, finish)
        return ExtractionResult(passage.passage_id, err is None, triplets, raw, err,
                                attempts=attempts)

    async def extract_all(
        self,
        passages: Iterable[Passage],
        on_result: Optional[Callable[[Passage, ExtractionResult], Awaitable[None] | None]] = None,
    ) -> list[ExtractionResult]:
        """Run all passages concurrently (bounded by the semaphore). `on_result`
        is called as each one finishes, so results can be written incrementally."""
        passages = list(passages)

        async def run(p: Passage) -> ExtractionResult:
            res = await self.extract_one(p)
            if on_result is not None:
                maybe = on_result(p, res)
                if asyncio.iscoroutine(maybe):
                    await maybe
            return res

        return list(await asyncio.gather(*(run(p) for p in passages)))


__all__ = ["ModelSpec", "load_model_specs", "ExtractionResult", "ExtractionClient", "parse_response"]
