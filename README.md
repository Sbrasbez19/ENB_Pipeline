# enbgraph — ENB negotiation network pipeline

Complete, current package + tests + tools + docs + generated artifacts.
Verified before packaging: `pytest -q` -> 63 passed, 1 skipped (the spaCy parser
test skips until en_core_web_sm is installed).

Read **FILE_MAP.md** first (it explains the two same-named stance.py files and the
full layout) and **MIGRATION_NOTES.md** if you use src/ingestion names.

## Install & test
```bash
pip install -r requirements.txt
pytest -q                       # 63 passed, 1 skipped
# optional parser upgrade:
pip install spacy && python -m spacy download en_core_web_sm
pytest tests/test_stance_parsed.py -q
```

## Pipelines
```bash
# one-shot graphs
python -m enbgraph.pipeline --input <pdfs> --output out --edge stance --alpha 0.32

# train/test evaluation
#   all generated files live under phase_artifacts/ (git-ignored)
python -m tools.phase1_develop sample --corpus <pdfs> --extractor pattern --out phase_artifacts/train_sample.csv
#   ...hand-label train_sample.csv -> phase_artifacts/train_labeled.csv
python -m tools.phase1_develop freeze --corpus <pdfs> --labeled phase_artifacts/train_labeled.csv --out phase_artifacts/dev_config.json
python -m tools.phase2_build build   --corpus <pdfs> --config phase_artifacts/dev_config.json --out phase_artifacts/test_graphs
python -m tools.phase2_build analyze --out phase_artifacts/test_graphs
python -m tools.make_plots --artifacts phase_artifacts --out plots
```

## Repository layout / what is tracked
- Tracked: `enbgraph/`, `tests/`, `tools/`, `docs/`, `phase_artifacts/dev_config.example.json`.
- Ignored (regenerate): everything else in `phase_artifacts/`, `plots/`, `*.graphml`, ENB `*.pdf`
  inputs, `.venv/`. Start from the example config:
  `cp phase_artifacts/dev_config.example.json phase_artifacts/dev_config.json`.
- `make_plots` expects `phase_artifacts/dev_config.json` and `phase_artifacts/test_graphs/`
  (`metrics.json` + `*.graphml`).
Webapp is a separate component (enbgraph-webapp.zip).
