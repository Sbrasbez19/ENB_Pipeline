# Actor Gazetteer — Construction & Corpus Audit

*Project: Modeling Influence and Deliberation in UN Climate Negotiations*
*Last updated: 30 June 2026*

## Purpose

The actor gazetteer is the closed-set list of negotiation actors (states,
coalitions, institutions) that the extractor matches against. This document
records how it was built, how it was checked against the ENB corpus, and exactly
what was changed — so the gazetteer is defensible and its evolution is traceable.

## Method

The gazetteer is **not** generated from the documents. It is a curated artifact
(it encodes modeling choices — which spellings merge to one node, whether
groupings like *Annex I* count as actors), authored from prior knowledge of
UNFCCC membership and negotiating coalitions. It is then **reconciled against the
corpus** by an audit tool (`tools/audit_gazetteer.py`):

1. Load every report; normalise text exactly as the extraction pipeline does
   (so line-wrapped names like "Saudi\nArabia" are not counted as misses).
2. Run the gazetteer for coverage (which entries appear, how often).
3. Harvest all capitalised phrases, subtract what the gazetteer already matches.
4. Cross-reference the remainder against `pycountry` (an independent country
   reference) to separate **genuine missed countries** from blocs/orgs/person
   noise, which a human then reviews.

This keeps modeling control with the author while grounding the contents in data.

## Corpus

Ten ENB final reports were provided. **Nine are machine-readable; one is a
scanned image-only PDF** (COP-3 / Kyoto, 1997 — `enb1221e.pdf`) and is excluded
pending OCR. Seven arrived as a page-packaged format, three as native PDFs.

| File | COP | Year | Format | Status |
|---|---|---|---|---|
| enb1238e | COP-2 | 1996 | packaged | ok |
| enb1221e | COP-3 | 1997 | PDF | **scanned — needs OCR** |
| enb1297e | COP-4 | 1998 | packaged | ok |
| enb12123e | COP-5 | 1999 | packaged | ok |
| enb12163e | COP-6 | 2000 | packaged | ok |
| enb12189e | COP-7 | 2001 | packaged | ok |
| enb12209e | COP-8 | 2002 | packaged | ok |
| enb12231e | COP-9 | 2003 | packaged | ok |
| enb12291e | COP-11 | 2005 | PDF | ok |
| enb12318e | COP-12 | 2006 | PDF | ok |

## Gazetteer composition: a-priori vs. augmented

| Category | A-priori (prior knowledge) | Additions from audit | Augmented |
|---|---:|---:|---:|
| Countries | 174 | +4 | 178 |
| Blocs / coalitions | 16 | +1 | 17 |
| Organizations | 6 | 0 | 6 |
| Aliases (surface → canonical) | 21 | +3 | 24 |
| Stoplist (non-actor terms) | 34 | +4 | 38 |
| **Distinct actor nodes** (countries+blocs+orgs) | **196** | **+5** | **201** |

## Additions from the corpus audit

**Countries (4)** — real UNFCCC parties present in the text but absent from the
prior:

- Antigua and Barbuda *(AOSIS member; appeared several times)*
- Central African Republic
- Saint Lucia
- Moldova

**Bloc (1)**:

- Economies in Transition (EITs) — a recognised negotiating grouping (19 mentions)

**Aliases (3)**: `EITs` → Economies in Transition, `EIT` → Economies in
Transition, `Antigua` → Antigua and Barbuda.

**Stoplist (4)** — capitalised terms that look like actors but are not, found
leaking into candidates: `IISD` (the bulletin's publisher), `JWG` (Joint Working
Group), `BAPA` (Buenos Aires Plan of Action), `PaMs` (Policies and Measures).

After these changes the audit reports **zero remaining missed countries**.

## Coverage (augmented gazetteer, 9 readable reports)

- **175 / 201** actor nodes appear in the corpus.
- **5,122** total actor mentions.
- Most frequent: Annex I (361), United States (353), European Union (341),
  G-77/China (285), GEF (251), non-Annex I (211), IPCC (192), Canada (181),
  Japan (154), Australia (146).

### Dead weight — present in the gazetteer, absent from these 9 reports (26)

Kept deliberately: an unused entry costs nothing and may appear once COP-1/COP-3
or later years are added. These are mostly small states:

> Angola, Cambodia, Cape Verde, Comoros, Congo, Dominica, Dominican Republic,
> Eritrea, Estonia, Gabon, Guinea, Guyana, Iraq, Kyrgyzstan, Liberia,
> Liechtenstein, Luxembourg, Madagascar, Niger, San Marino, Sierra Leone,
> Singapore, Somalia, Swaziland, Tajikistan, Tonga

## Caveats

- **COP-3 (Kyoto, 1997) is not yet included** — it is a scanned PDF and requires
  OCR before it can be audited or processed.
- Coverage is validated **1996–2006**. ENB's house style has shifted before
  (Party names moved from UPPERCASE to Title Case around 2000); the boundary and
  casing assumptions are confirmed only through 2006 and should be re-audited if
  later years are added.
- `GEF` and `IPCC` are matched but classified as organizations and excluded from
  the actor graph by default; whether institutions count as nodes is a
  graph-definition choice to revisit in the relation phase.

## How to update the gazetteer

When new reports arrive, re-run the audit and review its suggestions before
editing `enbgraph/config/gazetteer.yaml`:

```
python -m tools.audit_gazetteer --corpus /path/to/reports --json audit.json
```

The tool prints coverage, any newly missed countries (with YAML-ready lines), and
a triage list of unmatched capitalised phrases to review for new blocs or
stoplist terms. It never edits the gazetteer itself — additions stay a reviewed,
human decision.
