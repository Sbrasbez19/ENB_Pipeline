"""
enbgraph.llm — LLM-based stance extraction.

    Passage (ingest/passages.py)
      -> ExtractionClient (client.py)   OpenAI-compatible request, structured output
      -> ResponseCache (cache.py)       append-only JSONL, one file per model
      -> grounding.check (grounding.py) deterministic checks + gazetteer ids
      -> TripletRecord (schema.py)      triplet + source/extraction metadata

Driven end to end by tools/llm_extract.py. The rule-based extractors in
enbgraph.extraction remain the baseline.
"""

from .cache import ResponseCache, cache_key
from .client import ExtractionClient, ExtractionResult, ModelSpec, load_model_specs
from .grounding import check as check_grounding
from .prompts import PromptTemplate
from .schema import (SCHEMA_VERSION, ExtractionInfo, ExtractionResponse, GroundingInfo,
                     SourceInfo, StanceTriplet, TripletRecord)

__all__ = [
    "ResponseCache", "cache_key", "ExtractionClient", "ExtractionResult", "ModelSpec",
    "load_model_specs", "check_grounding", "PromptTemplate", "SCHEMA_VERSION",
    "ExtractionInfo", "ExtractionResponse", "GroundingInfo", "SourceInfo",
    "StanceTriplet", "TripletRecord",
]
