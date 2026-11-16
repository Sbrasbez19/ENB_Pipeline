"""
utils.py — shared, dependency-light helpers for the ingest + chunk layers.

Three groups of helpers live here, all leaf-level (they import only third-party
libraries, never our own modules, so there is no import cycle):

  * text normalisation — repair the ENB text layer into clean prose
  * config loading      — read YAML rule files and layer overrides
  * logging             — one place to configure log format/level

Normalisation ordering note (the non-obvious part): domain-aware dehyphenation
runs BEFORE ftfy. ftfy is excellent at mojibake, unicode composition and curly
quotes, but it collapses a soft-hyphen indiscriminately, turning "non-Annex"
into "nonAnnex". Since "Annex I" vs "non-Annex I" is a core distinction in these
negotiations, we fix hyphenation ourselves first (syllable break -> join;
compound -> keep the hyphen) and only then hand off to ftfy.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Union

import ftfy
import yaml

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #
# control chars to drop, EXCEPT tab/newline/carriage-return which carry layout
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# soft-hyphen artefact (U+0002 here) and end-of-line hyphenation. The signal for
# a genuine compound (keep the hyphen) is a lowercase->UPPERCASE transition, as in
# "non-Annex". A break with no such transition — including one inside an all-caps
# header like "SCIENTIF<sh>IC" — is a syllable split, so we join. (Residual ambiguity
# remains for lowercase compounds wrapped at a line end, e.g. "cost-\neffective";
# resolving those needs a lexicon and is out of scope.)
_SOFT_COMPOUND = re.compile(r"(?<=[a-z])\x02(?=[A-Z])")
_SOFT_JOIN = re.compile(r"\x02")
_WRAP_COMPOUND = re.compile(r"([a-z])-\n\s*(?=[A-Z])")
_WRAP_JOIN = re.compile(r"(\w)-\n\s*")

# explicit, deterministic quote/space normalisation (don't rely on ftfy defaults)
_TYPOGRAPHY = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u00a0": " ",  # non-breaking space
}
_TYPO_TABLE = {ord(k): v for k, v in _TYPOGRAPHY.items()}


def dehyphenate(text: str) -> str:
    """Repair words split by a soft-hyphen or end-of-line hyphen, keeping genuine
    compounds (lowercase->UPPERCASE transition) and joining everything else.
    Must run before ftfy, which would otherwise collapse the compound hyphen."""
    text = _SOFT_COMPOUND.sub("-", text)   # "non<sh>Annex" -> "non-Annex"
    text = _SOFT_JOIN.sub("", text)         # "SCIENTIF<sh>IC" -> "SCIENTIFIC"
    text = _WRAP_COMPOUND.sub(r"\1-", text)  # "non-\nAnnex"   -> "non-Annex"
    text = _WRAP_JOIN.sub(r"\1", text)        # "deliber-\nated" -> "deliberated"
    return text


def fix_encoding(text: str) -> str:
    """ftfy: repair mojibake and normalise unicode (e.g. combining accents)."""
    return ftfy.fix_text(text)


def normalize_typography(text: str) -> str:
    """Map curly quotes/apostrophes and nbsp to ASCII. Idempotent, so it is a
    no-op if ftfy already straightened them — we keep it for determinism across
    ftfy versions and because gazetteer matching needs stable apostrophes
    (e.g. \"Cote d'Ivoire\")."""
    return text.translate(_TYPO_TABLE)


def collapse_whitespace(text: str, keep_newlines: bool = False) -> str:
    """Collapse runs of whitespace. With keep_newlines, preserve single newlines
    (and trim trailing spaces per line) so structural/header analysis still sees
    line boundaries; otherwise flatten everything to single spaces."""
    if keep_newlines:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    return re.sub(r"\s+", " ", text).strip()


def _base_clean(text: str) -> str:
    """Shared front of both pipelines: dehyphenate -> ftfy -> typography -> drop
    stray control chars. Newlines are still intact after this step."""
    text = dehyphenate(text)
    text = fix_encoding(text)
    text = normalize_typography(text)
    return _CONTROL.sub("", text)


def normalize_prose(text: str) -> str:
    """Produce flowing prose: wrapped lines become spaces. Use for chunk body
    text and anything fed to the sentence splitter / extractors."""
    text = _base_clean(text)
    text = text.replace("\n", " ")
    return collapse_whitespace(text)


def normalize_structural(text: str) -> str:
    """Clean the text but KEEP newlines, so header/line detection still works.
    Use when line boundaries are a signal (e.g. raw page text before chunking)."""
    text = _base_clean(text)
    return collapse_whitespace(text, keep_newlines=True)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_yaml(path: PathLike) -> dict[str, Any]:
    """Read a YAML file into a dict, with clear errors. Used for the section-rule,
    stoplist and (later) gazetteer config we externalise from code."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {p}, got {type(data).__name__}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively layer `override` onto `base` (override wins on conflicts).
    Lets a small user config sit on top of shipped defaults without mutating
    either input."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once (idempotent-ish). Call from the entry point."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=_LOG_FORMAT, datefmt="%H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    """Module-level logger accessor: `log = get_logger(__name__)`."""
    return logging.getLogger(name)


__all__ = [
    "dehyphenate", "fix_encoding", "normalize_typography", "collapse_whitespace",
    "normalize_prose", "normalize_structural",
    "load_yaml", "deep_merge",
    "setup_logging", "get_logger",
]
