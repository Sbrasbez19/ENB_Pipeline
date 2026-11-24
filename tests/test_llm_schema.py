"""
Tests for the extraction contract (enbgraph/llm/schema.py) and prompt files:
the JSON schema sent to the server is well formed, the prompt's worked
examples are valid responses, and stored records round-trip.
"""

import json
import re

import pytest
from pydantic import ValidationError

from enbgraph.llm.prompts import PROMPT_DIR, PromptTemplate
from enbgraph.llm.schema import (ExtractionInfo, ExtractionResponse, GroundingInfo, SourceInfo,
                                 StanceTriplet, TripletRecord, response_format)
from enbgraph.models import Passage, Provenance

TRIPLET = {
    "source_actor": "G-77/China", "source_mention": "the G-77/China", "source_role": "party",
    "on_behalf_of": None, "reference_type": "explicit", "antecedent_evidence": None,
    "verb": "opposed", "polarity": "oppose", "target_type": "actor", "target": "Canada",
    "proposal": None, "proposal_author": None,
    "evidence": "Canada, opposed by the G-77/China, proposed replacing the decision.",
}


def test_response_format_is_json_schema():
    rf = response_format()
    assert rf["type"] == "json_schema"
    schema = rf["json_schema"]["schema"]
    json.dumps(schema)                                   # serialisable
    triplet_schema = schema["$defs"]["StanceTriplet"]
    assert triplet_schema["additionalProperties"] is False
    assert {"source_actor", "polarity", "target_type", "evidence"} <= set(triplet_schema["required"])


def test_valid_triplet_parses_and_rejects_extras():
    StanceTriplet.model_validate(TRIPLET)
    with pytest.raises(ValidationError):
        StanceTriplet.model_validate({**TRIPLET, "doc_id": "x"})       # metadata is not the LLM's job
    with pytest.raises(ValidationError):
        StanceTriplet.model_validate({**TRIPLET, "polarity": "maybe"})


@pytest.mark.parametrize("prompt_file", sorted(PROMPT_DIR.glob("*.txt")))
def test_prompt_examples_are_valid_responses(prompt_file):
    text = prompt_file.read_text(encoding="utf-8")
    examples = re.findall(r"Output:\n(\{\"triplets\".*?\n\]\})", text, flags=re.DOTALL)
    assert examples, "prompt has no worked examples"
    for ex in examples:
        ExtractionResponse.model_validate_json(ex)


def test_prompt_messages():
    msgs = PromptTemplate.load().messages("Earlier text.", "Focus text.")
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "Earlier text." in msgs[1]["content"] and "Focus text." in msgs[1]["content"]
    with pytest.raises(FileNotFoundError):
        PromptTemplate.load("no_such_prompt")


def test_record_round_trip():
    p = Passage(passage_id="d-p0001", doc_id="d", focus_text="x", sentence_start=0, sentence_end=1,
                provenance=Provenance(doc_id="d", page_start=1, page_end=1, start_char=0, end_char=1),
                year=2000, cop="6")
    rec = TripletRecord(
        triplet_id="d-p0001-t01", triplet=StanceTriplet.model_validate(TRIPLET),
        source=SourceInfo.from_passage(p),
        extraction=ExtractionInfo(run_id="r", model="m", prompt_version="extract_v1"),
        grounding=GroundingInfo(evidence_found=True, evidence_score=1.0, source_id="G-77/China"),
    )
    back = TripletRecord.model_validate_json(rec.model_dump_json())
    assert back == rec
    assert back.source.year == 2000 and back.grounding.passed
