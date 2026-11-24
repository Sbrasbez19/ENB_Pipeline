"""
llm/cache.py — append-only response cache, one JSONL file per model.

Why JSONL and not SQLite: the cluster's /home is NFS, where SQLite's file
locking is unreliable. An append-only file written by one process is safe,
survives crashes (a torn last line is skipped on load) and is human-readable.

The key covers everything that changes the model's answer — model, revision,
prompt version, schema version and the exact passage text — so a rerun only
sends requests for what is new, and changing any of them invalidates the cache.
Raw responses are stored (not parsed triplets), so improving the parsing or
grounding code never requires re-querying the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from ..utils import get_logger

log = get_logger(__name__)


def cache_key(model: str, revision: Optional[str], prompt_version: str,
              schema_version: str, context: str, focus: str) -> str:
    payload = json.dumps([model, revision, prompt_version, schema_version, context, focus],
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            bad = 0
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        bad += 1      # torn line from an interrupted write
            if bad:
                log.warning("%s: skipped %d unreadable cache line(s)", self.path, bad)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    def put(self, key: str, record: dict) -> None:
        rec = {"key": key, **record}
        self._data[key] = rec
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()


__all__ = ["ResponseCache", "cache_key"]
