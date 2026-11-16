# Stance Extraction — Manual Validation (plan C5)

*Date: 5 July 2026 · Companion to `stance_implementation_plan.md` and
`svo_stance_audit.md`*

> **STATUS — superseded figures below; final frozen numbers here.** This document
> records the *first* validation round (an exploratory 45-record sample over all
> nine years, tier-1 0.53 / tier-2 1.00 / tier-3 0.80). It was later superseded by
> the **train/test freeze**, which re-labelled a *training-only* sample (two review
> passes) and split precision by extractor. The **authoritative final figures** are
> in `phase_artifacts/dev_config.json` and the report:
> **tier-1 0.533, tier-2 1.00, tier-3 0.933**; by source,
> **parsed tier-1 = 1.00 (n=7)** versus **pattern tier-1 = 0.125 (n=8)** — the
> quantified value of the dependency parser (n=7 small, report as directional).
> The process detail and error taxonomy below remain valid and are kept as the
> methodological record.

## Protocol

Mirrors Sudhahar, Veltri & Cristianini (2015), who manually coded 72 triplets
and reported **62% precision / 57% recall** for unfiltered extraction (uploaded
PDF, Appendix). Here: a stratified random sample of **45 records (15 per tier,
seed 7**, reproducible via `tools/validate_stance.py`) drawn from all nine
readable ENB reports, each labeled against its full source sentence
(1 = subject, sign, direction and object all correct; 0 = any error). A
**12-sentence recall probe** (random stance-verb sentences with ≥2 actors)
checked coverage.

**Annotator disclosure:** labels were assigned by the AI assistant in a single
careful pass and are delivered for review in
`stance_sample_labeled.csv`; they should be treated as a first annotation, not
ground truth, until spot-checked. Two ambiguous items were resolved
conservatively against the extractor.

**Two-round protocol.** A first sample (seed 42) surfaced three systematic
error classes; these were **fixed before** the reported round (seed 7), and the
fixes are pinned by regression tests (`tests/test_stance.py`, final three
tests):
1. `", and opposed/supported by"` appositives (with the connective *and*)
   fell through to the main-verb path, producing wrong-direction edges.
2. Contrast clauses (*"…supported X, while Saudi Arabia … opposed…"*) leaked
   second-clause actors into first-clause objects → clause segmentation.
3. The chunker's acronym guard merges UPPERCASE-era sentences
   (*"…CDM. Japan said…"*), letting next-sentence actors become objects →
   period-before-capital treated as a clause boundary inside stance extraction
   (local; the validated chunker is untouched).

## Results (post-fix sample, frozen labels)

| Tier | Meaning | Precision |
|---|---|---:|
| 1 | main-verb direct actor object | **8/15 = 0.53** |
| 2 | coalition appositive relations | **15/15 = 1.00** |
| 3 | issue-directed stance | **12/15 = 0.80** |
| — | **overall** | **35/45 = 0.78** |

Recall probe: **11/12** stance-verb sentences correctly covered (the one miss
was a wrong tier-1 record, not an absence).

Reference point: our 0.78 overall exceeds the reference paper's 0.62
unfiltered precision; their error-suppression machinery (repetition, 0.38^k;
Wilson weighting) applies **on top** of these figures, as it does in their
pipeline.

## Error taxonomy (the 10 wrong labels)

| Class | Cases | Example | Status |
|---|---|---|---|
| Noun/verb confusion (*"financial support"*, *"backing"*, *"the challenges"*, *"in support of"*) | 4 | Annex I +> Cuba from "financial **backing**" | needs POS — the planned spaCy upgrade's clearest payoff |
| `", but opposed by"` — third connective variant | 1 | AOSIS→Saudi (direction flipped) | **fixed post-validation** + regression test (labels frozen pre-fix) |
| Coordinated independent clause (*"…, and Iran offered to host"*) | 1 | Canada +> Iran | parser territory (conjunct clause boundaries) |
| Option-contrast (*"as proposed by X … or … as supported by Y"*) | 1 | US +> Kuwait | parser territory |
| Issue text containing an actor name (*"addressing future **Annex I** commitments"*) | 1 | G-77 +> Annex I | needs object-NP head analysis (parser) |
| Coordinated VP with intervening object (*"yielded to the G-77 and agreed…"*) | 1 | G-77 as false co-subject | parser territory |
| Semantic overtrigger of *agree* outside negotiation stance | 1 | Germany "agreed to offer accommodation" | lexicon refinement candidate |

## Implications

1. **The tier hierarchy is empirically validated — but inverted at the top.**
   Appositive-derived relations (tier 2) are the *most* reliable channel
   (15/15), main-verb objects (tier 1) the least (0.53). The C4 tier weights
   should be tuned accordingly on the training years; a precision-proportional
   starting point is ≈ {1: 0.55, 2: 1.0, 3: 0.8} rather than the a-priori
   {1: 1.0, 2: 0.75, 3: 0.25}. This is exactly the kind of correction the
   development/held-out split exists to absorb.
2. **Most residual tier-1 errors are attachment errors** (noun/verb, clause
   coordination) — the specific failure modes dependency parsing resolves,
   confirming the plan's C1→parser upgrade path and quantifying its expected
   payoff (~5–6 of the 7 tier-1 errors).
3. **Sampling note:** record-level sampling over-represents complex sentences
   (each yields many records), so these figures are conservative for the
   corpus's typical short stance sentences.

## Files

- `tools/validate_stance.py` — reusable sampler + scorer
  (`--corpus … --per-tier N --seed S`, then `--score labeled.csv`).
- `stance_sample_labeled.csv` — the 45 labeled records (this round).
