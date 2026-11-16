# FILE MAP — where each file lives (read this before moving files)

Two files share the name `stance.py`. They are DIFFERENT and must NOT be
swapped — overwriting one with the other deletes the classes the rest of the
package imports, which breaks many imports at once.

| Path | Defines | Role |
|---|---|---|
| `enbgraph/extraction/stance.py` | `StanceRecord`, `StanceLexicon`, `StanceExtractor` | turns sentences into signed stance RECORDS |
| `enbgraph/graph/stance.py` | `DirectStanceEdges`, `CoStanceEdges` | turns records into signed EDGE definitions |

Quick check after any edit:
```bash
grep -oE "class [A-Za-z]+" enbgraph/extraction/stance.py   # StanceRecord/Lexicon/Extractor
grep -oE "class [A-Za-z]+" enbgraph/graph/stance.py        # DirectStanceEdges/CoStanceEdges
```

## Full layout

```
enbgraph/                      the package  (your machine: rename -> src/)
├── models.py utils.py corpus.py pipeline.py
├── ingest/        ingester.py chunker.py         (your machine: -> ingestion/)
├── extraction/    gazetteer.py coref.py actors.py stance.py stance_parsed.py base.py
├── graph/         base.py proximity.py stance.py builder.py aggregate.py balance.py
└── config/        gazetteer.yaml stance_verbs.yaml
tests/                         (sibling of the package)
tools/                         audit_gazetteer.py validate_stance.py phase1_develop.py phase2_build.py
docs/                          all .md docs + train_labeled.csv
phase_artifacts/               dev_config.json, train_labeled.csv, test_graphs/  (GENERATED — regenerate, don't hand-edit)
conftest.py  requirements.txt
```

`stance_parsed.py` is the spaCy parser upgrade; it needs
`pip install spacy && python -m spacy download en_core_web_sm`. Its test
(`tests/test_stance_parsed.py`) skips until the model is installed.

## Naming reminder
This tree uses `enbgraph`/`ingest`; your machine uses `src`/`ingestion`. See
`MIGRATION_NOTES.md` for the rename map. Relative imports (`from ..models`,
`from .stance`) do not change; only absolute `enbgraph.` -> `src.` and the
`ingest` -> `ingestion` folder/import references do.
```
