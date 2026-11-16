# Migration notes — enbgraph/ingest  ->  src/ingestion

This tree uses the canonical names the tests run against: package `enbgraph`,
subpackage `ingest`. Your machine uses `src` / `ingestion`. To adopt this
snapshot under your names (mechanical, verifiable with `pytest -q`):

## 1. Rename folders
```
enbgraph/          -> src/
enbgraph/ingest/   -> src/ingestion/
```

## 2. Rewrite imports
Absolute (in tests/, tools/, conftest.py):
  from enbgraph.X            -> from src.X
  from enbgraph.ingest.Y     -> from src.ingestion.Y

Relative inside the package (dot-counts DO NOT change; only the `ingest` name):
  from ..ingest.ingester     -> from ..ingestion.ingester
  from .ingest.chunker       -> from .ingestion.chunker
Everything else (from ..models, from .stance, from ..graph...) is unchanged.

## 3. One-liner (from the archive root, BEFORE renaming folders)
```bash
grep -rl --include='*.py' -e 'enbgraph' -e 'ingest\.' . | while read f; do
  sed -i -e 's/enbgraph\.ingest\./src.ingestion./g' \
         -e 's/\.ingest\.ingester/.ingestion.ingester/g' \
         -e 's/\.ingest\.chunker/.ingestion.chunker/g' \
         -e 's/enbgraph/src/g' "$f"
done
mv enbgraph/ingest enbgraph/ingestion && mv enbgraph src
```

Config paths use `Path(__file__).parent...`, which is name-independent — renaming
does not break gazetteer/lexicon loading.

## Note: TWO stance.py files (see FILE_MAP.md)
`extraction/stance.py` (records) and `graph/stance.py` (edges) are different.
Never overwrite one with the other.
