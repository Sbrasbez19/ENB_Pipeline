"""
llm/prompts — versioned prompt templates.

A template is a plain text file holding the system prompt; its file stem is
its version (e.g. 'extract_v1'). The version goes into the cache key and the
run manifest, so never edit a released template in place: copy it to a new
version instead, or cached results will be silently mixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = "extract_v1"


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str

    @classmethod
    def load(cls, version: str = DEFAULT_PROMPT) -> "PromptTemplate":
        path = PROMPT_DIR / f"{version}.txt"
        if not path.is_file():
            available = sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))
            raise FileNotFoundError(f"prompt {version!r} not found (available: {available})")
        return cls(version=version, system=path.read_text(encoding="utf-8").strip())

    def messages(self, context: str, focus: str) -> list[dict]:
        user = (f"CONTEXT (only for resolving references):\n{context or '(none)'}\n\n"
                f"FOCUS (extract from here):\n{focus}")
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": user}]


__all__ = ["PromptTemplate", "PROMPT_DIR", "DEFAULT_PROMPT"]
