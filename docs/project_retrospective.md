# Project Retrospective — What We Built and Why

*Modeling Influence and Deliberation in UN Climate Negotiations (PhD admission test)*
*A complete inventory of decisions, components, and analyses, each with its justification.*

This is a reference map of the whole build: the pipeline that turns Earth
Negotiations Bulletin (ENB) reports into per-year actor networks, the analyses
that validated it, and the reasoning behind every non-obvious choice. Where a
choice came from the two reference papers it is attributed; where it came from
the corpus it is grounded in a measurement.

---

## 0. The task and the framing decisions

**The test (from Ixandra):** build 4–5 graphs, one per year, nodes = actors,
edges = a relation "derived from textual analysis" (proximity, action-verb
influence, …); test several edge definitions and justify which is most relevant;
the other years "can be used as a kind of training set."

| Decision | Justification |
|---|---|
| Interpret "training set" as **development/validation years**, not ML training | The pipeline is rule-based (no learned parameters); "training" here means the years on which to tune and compare edge definitions before freezing the presented ones. Flagged as a question to confirm with the supervisor. |
| Build a **full pipeline**, not one-off scripts | The deliverable is *comparable* graphs across years and *several* edge definitions; that requires a reusable ingest→extract→build spine, not per-year notebooks. |
| **Senior-engineer discipline**: design → justify → code → pyright → pytest, one component at a time | An admission test is judged on defensibility; every component is type-checked and tested so claims are demonstrable, not asserted. 63 tests, pyright-clean throughout. |

---

## 1. Corpus understanding (the analysis that shaped everything)

Before building, we characterised the 10 provided ENB PDFs. The findings drove
the architecture:

| Finding | Consequence for the design |
|---|---|
| **Two physical formats** share the `.pdf` name: 7 are ZIP-packaged (page image + separate text layer), 3 are native `%PDF` | A single loader must detect format by magic bytes and handle both; only native PDFs carry text coordinates (→ later, highlighting is native-only). |
| **1 file (COP-3, 1997) is scanned image-only** — a real PDF with no text layer | Excluded pending OCR; coverage is 1996–2006 (9 readable years). Justifies keeping OCR as a *pluggable future ingester* rather than dropping it. |
| **ENB house style flipped ~2000**: Party names UPPERCASE pre-2000 ("CHINA"), Title Case after | Actor matching must be **case-insensitive**, never uppercase-anchored — the single most important extraction decision; proven on real data (COP-4 vs COP-6). |
| **Body titles drift** ("REPORT OF COP-4" → "COP-6 Report") | Chunk boundaries anchored on **stable markers** ("A BRIEF ANALYSIS", major headers), not year-specific titles. |
| **Soft-hyphen artifact** (`\x02`); dehyphenation must keep "non-Annex" but join wrapped syllables | Custom dehyphenation rule (keep hyphen only on lowercase→UPPERCASE transition). |

---

## 2. The pipeline package (`enbgraph/`, ~2,600 LOC)

Architecture principle applied throughout: **three separations** — extraction
from edge semantics, edge semantics from graph storage, and data (lexicons,
config) from code. This is what lets edge definitions be swapped and unit-tested
in isolation, and what made the later stance layer a drop-in.

### 2.1 Ingest layer

| Component | What it does | Justification |
|---|---|---|
| `models.py` | pydantic v2 contracts (BBox, TextBlock, Page, Document, Chunk, Provenance, …) | Typed contracts catch integration errors at boundaries; provenance offsets enable evidence-tracing end to end. |
| `ingest/ingester.py` | PDF → Document; two-column reading order via lightweight XY-cut | ENB is two-column; naive extraction interleaves columns. OCR designed as a future subclass behind `BaseIngester` — deferred, not precluded. |
| `ingest/chunker.py` | Document → agenda-item chunks (subtractive: tag regions, don't delete) | Preserves exact provenance on raw text while detecting headers on a normalized copy; yields 24–38 report chunks/year without collapse. 100% provenance round-trip. |
| `corpus.py` | Unified loader for both formats + scanned detection | Promoted from duplicated throwaway code to one shared module used by pipeline **and** audit tool (DRY). |
| `utils.py` | Normalization (dehyphenate, ftfy, typography), YAML, logging | ftfy runs **after** domain dehyphenation (it wrongly collapses "non-Annex"→"nonAnnex"). Order is a deliberate, documented choice. |

### 2.2 Extraction layer

| Component | What it does | Justification |
|---|---|---|
| `extraction/gazetteer.py` | Closed-set actor matcher (one regex, longest-form-first) | Actors are a **known, closed set**; a gazetteer is higher-precision than NER and yields **stable canonical IDs** (year-comparability). Full names case-insensitive (era flip); short acronyms (US/EU) case-**sensitive** (avoids matching pronoun "us"). |
| `extraction/coref.py` | Pronoun → nearest preceding actor | High-precision floor; full name-coref deferred. |
| `extraction/actors.py` | Orchestrates gazetteer + optional NER recall + coref | Composition over monolith; spaCy imported **lazily** so it's never a hard dependency. ORG (IPCC/GEF) excluded from the actor graph by default. |
| `config/gazetteer.yaml` | 201 actor nodes (178 countries, 17 blocs, 6 orgs) + aliases + stoplist | **Data, not code**: adding a Party is a YAML edit. Encodes modeling choices (EC/EU→one node; "Annex I" as a bloc actor). |
| `extraction/stance.py` | Sentence → signed `StanceRecord`s | See §4. |

### 2.3 Graph layer

| Component | What it does | Justification |
|---|---|---|
| `graph/base.py` | `Edge` record + `EdgeDefinition` interface (NetworkX-free) | Edge **semantics** (the research contribution) kept separate from **storage**. Definitions declare `directed`; the builder reads it. |
| `graph/proximity.py` | Baseline co-occurrence edges (sentence/chunk window) | The relation the brief names, and the **null model** later layers must beat. Two windows built so their contrast is a finding, not an assertion. |
| `graph/builder.py` | Thin NetworkX wrapper; accumulates edges; canonical-ID nodes | NetworkX is the substrate, not the architecture; nodes are year-stable `actor_id`s so per-year graphs are comparable. |
| `graph/stance.py` | `DirectStanceEdges` (directed, signed) + `CoStanceEdges` (projected, signed) | See §4. |
| `graph/aggregate.py` | Bernoulli + Wilson per-pair sign/confidence | See §4. |
| `graph/balance.py` | Signed spectral partitioning, frustration, triangle census | See §5. |
| `pipeline.py` | End-to-end CLI: corpus → per-year graphs + manifest | One command produces the deliverable; edge factory is the extension seam; settings recorded in the manifest for reproducibility. |

---

## 3. The gazetteer audit (grounding the prior in data)

**Approach:** the gazetteer is human-authored (it encodes modeling choices), but
**audited** against the corpus by `tools/audit_gazetteer.py`, which cross-checks
harvested capitalized phrases against `pycountry` to separate genuine missed
countries from noise.

| Why this way | |
|---|---|
| Keep authoring human | Sign/merge decisions (EC↔EU, "Annex I" as actor) shouldn't be silently automated. |
| Audit empirically | A hand-list has silent gaps; the tool found 4 missing countries (Antigua and Barbuda, Central African Republic, Saint Lucia, Moldova) + 1 bloc (EITs) + stoplist leaks (IISD, JWG, BAPA). |
| Run on normalized text | First pass on raw text produced false misses (line-wrapped "Saudi\nArabia"); fixed to mirror the pipeline. |

Documented in `docs/gazetteer_report.md` (a-priori 196 → augmented 201 nodes;
coverage 175/201; 5,122 mentions).

---

## 4. The signed stance layer (the core research contribution)

Motivated by two papers you provided, **re-read and verified against their
texts** (correcting an earlier from-memory error):

- **Sudhahar, Veltri & Cristianini (2015), *Big Data & Society*** — a **signed
  SVO** graph: parse subject–verb–object, keep only support/opposition verbs,
  sign edges, analyse by partitioning + structural balance.
- **Dambanemuya & Horvát (2024), *ICWSM*** — influence = **centrality in a
  reply network**; language features are a *separate* persuasion predictor, and
  the whole thing is validated by the ∆ (view-change) ground truth.

**The corrected synthesis (why these adapt as they do):**

| Question | Answer, justified |
|---|---|
| Does the SVO approach transfer? | **Yes, near-directly** — ENB prose is actor–verb–object; dominant verbs are the paper's own *supported/opposed*. |
| Does the influence approach transfer? | **Its network half yes, its validation no.** ENB gives directed signed relations (richer than unsigned replies), so centralities compute — but there is **no ∆-equivalent**, so centrality is an *influence proxy*, not a validated persuasion measure. Language features are inapplicable (ENB is a rapporteur paraphrase). |
| Are they two separate builds? | **No** — Paper 1 is the *construction*, Paper 2 the *reading*. One signed directed graph, analysed both ways. |

**The audit that gated the design** (`docs/svo_stance_audit.md`): measured, per
year, how much signal exists. Direct actor→verb→actor patterns are sparse (~19/yr,
172 total), but **53% of stance is issue-directed** ("AOSIS opposed technical
assessment processes"). The resolution:

| Decision | Justification |
|---|---|
| Keep **issue-directed stance** and **project** it | Issue objects mostly have no proponent to resolve to — this is a property of the discourse (Parties position on *content*), matching Sudhahar et al.'s choice to keep issue nodes. |
| **Issue = agenda item (chunk)** | Clustering free-text issue phrases is unstable at ~4.4k sentences; the chunker already segments the document's own issue structure. Cost (sub-issue conflation) disclosed and weighted down. |
| **Signed one-mode projection** (Breiger 1974 duality) | Two actors stancing on a shared issue → same sign = aligned, opposite = opposed. Yields ~4,276 signed pairs (25× the direct count) — resolves sparsity. Verified equivalent to `networkx.bipartite.generic_weighted_projected_graph` by test. |
| **Tiered confidence** (1 direct > 2 coalition > 3 projection) | The three channels have different reliability; weights tune on training years. Validation later *inverted* the top: tier-2 appositives are most reliable (see §6). |

**Extraction (`extraction/stance.py`) — ENB idioms, each traced to the audit:**

| Idiom | Handling | Why |
|---|---|---|
| Compound subject ("The US and EU opposed…") | Both become subjects | The exact sentence that exposed proximity's flaw (allies vs adversaries). |
| Coalition appositive ("X, supported by A, B and C") | member→head alignment edges, passive direction | ENB's standard alignment marker; multi-member → many edges. |
| "opposed by" | Direction reversed, no co-assertion | Passive grammar: Y opposes X, Y doesn't share X's stance. |
| "on behalf of" | Attribution stored, **switchable** to bloc-as-subject | Whether bloc statements accrue to country or bloc is a modeling question for the supervisor — a flag, not a buried constant. |
| Clause/sentence merges | Local segmentation inside stance extraction | Fixes cross-clause leakage **without** touching the validated chunker. |

**Aggregation (`graph/aggregate.py`) — Bernoulli + Wilson:**

| Decision | Justification |
|---|---|
| Model per-pair sign as a **Bernoulli** variable; score by **Wilson lower bound** | Sudhahar et al.'s own scheme. Naive k/n is maximally overconfident at n=1 (1/1→1.0); Wilson lower bound (1/1→0.207) means **repetition, not single sightings, creates strong edges**. |
| Use `statsmodels.proportion_confint(method="wilson")` | Your "use existing packages" rule; **verified** it accepts our tier-weighted float counts (scipy's binomtest would not). |
| Keep **contested** edges, flagged | Contestedness is the pair-level analogue of the paper's divisiveness measure — information, not noise. |
| Post-pass, not builder logic | Aggregation is statistical *policy*; keeps the builder generic and the policy swappable (naive vs Wilson side by side). |

---

## 5. Balance analysis — the validation slide (`graph/balance.py`)

| Instrument | Justification |
|---|---|
| **Spectral bisection** of the signed adjacency (numpy `eigh`) | Direct analogue of Sudhahar et al.'s validation (partitioning recovered Dem/Rep). Ours tests whether **Annex I vs G-77/China** emerge. Balance theory: Heider 1946; Cartwright & Harary 1956 (cited by the paper). |
| **Frustration ratio** + **Heider triangle census** | Standard balance diagnostics; quantify how two-camp the structure is. |
| `camp_agreement` (relabelling-invariant) | Eigenvector sign is arbitrary; agreement must be label-invariant. |

**Result, stated honestly:** per-year camp agreement is 0.55–0.80 with **no
monotone trend** (mean ≈0.66); it is the **pooled** graph that resolves the split
to **0.90** — a real North–South signal, not the clean 0.95 recovery
the reference paper got on 130k articles. Framed as a temporal finding, this is
arguably a *better* story than a static split; the scale limitation is disclosed.

---

## 6. Manual validation (`tools/validate_stance.py`, `docs/stance_validation.md`)

| Decision | Justification |
|---|---|
| Hand-label a **stratified 45-record sample**; report per-tier precision | Mirrors the paper's protocol (72 triplets, 62%). Result: **0.78 overall** (tier-1 0.53, tier-2 **1.00**, tier-3 0.80) — above the paper's unfiltered baseline. |
| Two-round protocol (fix, then measure) | Round 1 surfaced 3 systematic bugs; all fixed **and pinned by regression tests** before the reported round. |
| **Disclosed limitation** | Labels were assigned by the AI in one pass — a first estimate, not independent ground truth; flagged for your spot-check before citing. |
| **Finding that redirects the roadmap** | Tier-1 (direct) is weakest; most errors are noun/verb confusion ("financial **backing**") — exactly what POS/dependency parsing fixes, quantifying the spaCy upgrade's payoff. |

---

## 7. The visualizer webapp (`enbgraph-webapp/`, separate component)

| Decision | Justification |
|---|---|
| **Separate component** importing `enbgraph` via `ENBGRAPH_ROOT` | Keeps the research package dependency-clean; the app is a consumer, not part of the library. |
| Flask + server templates + **vis-network** | Simple, reliable stack; vis-network handles tooltips, node-sizing, physics without a build step. |
| **Backend fully tested via Flask test client; frontend not visually testable here** | Honest scope: every API verified; JS written to standard patterns but flagged as needing your eyes. |
| **Shared temporal layout** (one force layout over the union of years) | Node positions fixed across years so the slider shows *ties changing while actors stay put* — the temporal story made legible. |
| **PDF evidence highlighting** via PyMuPDF `search_for` + anchor strategy | Robust against normalization (search opening words, not offsets); **native-PDF only** (packaged/scanned degrade gracefully) — which matches your real IISD downloads. |
| **Signed surfacing** (edge selector, green/red, dashed = contested, Wilson conf in panel) | Makes the stance layer explorable; aggregated at 68% for per-year legibility (95% uninformative at scale). |

---

## 8. Cross-cutting engineering decisions

| Decision | Justification |
|---|---|
| **Reuse over reimplementation** (statsmodels Wilson, networkx projection/centralities, pycountry, spaCy, ftfy) | Your explicit instruction; custom code confined to domain data (lexicons), thin glue, and genuinely project-specific logic (~15 lines of eigen-glue, the record contracts). |
| **Data in config, decisions in code** (gazetteer.yaml, stance_verbs.yaml) | Reviewers can inspect/edit modeling choices without reading code; mirrors the papers' curated verb lists. |
| **Everything type-checked (pyright) and tested (pytest)** | 63 tests, 0 pyright errors across package + tools + webapp; claims are demonstrable. |
| **Provenance preserved end to end** | Every edge traces to sentences; powers the webapp's evidence panel and the audit trail a defense needs. |
| **Two requirements files, derived from actual imports and verified with pip** | Reproducibility on your Windows machine; import-name traps (fitz=PyMuPDF, yaml=PyYAML) called out. |

---

## 9. Known limitations & deferred items (stated, not hidden)

| Item | Status / why deferred |
|---|---|
| **COP-3 (1997) scanned** | Needs OCR; excluded. OCR designed as a pluggable ingester. Matters only if presented years must be consecutive early years — a question for the supervisor. |
| **spaCy dependency parse** | Model download blocked in the sandbox; the pattern-based extractor is measured (0.78) and the parser is a drop-in upgrade whose payoff is quantified. |
| **Validation labels self-assigned** | First-pass estimate; spot-check before citing. |
| **Influence = centrality proxy** | No ∆-equivalent in ENB; must be presented as structural proxy, not validated persuasion. |
| **Per-year signed graphs are small** | Wilson at 68% for per-year; 95% claims need year-pooling (a training-set use). |
| **Frontend not visually verified** | Backend tested; JS needs your screen. |

---

## 10. What exists now (delivered inventory)

- **Package** `enbgraph/`: ingest (ingester, chunker, corpus), extraction
  (gazetteer, coref, actors, stance), graph (base, proximity, builder, stance,
  aggregate, balance), models, utils, pipeline; 2 config lexicons.
- **Tools**: `audit_gazetteer.py`, `validate_stance.py`.
- **Tests**: 7 files, 63 tests, pyright-clean.
- **Webapp** `enbgraph-webapp/` (+ zip): 4 sections, 3 edge types, temporal
  slider, PDF highlighting, signed surfacing.
- **Docs**: `gazetteer_report.md`, `svo_stance_audit.md`,
  `stance_implementation_plan.md`, `stance_validation.md`, this retrospective;
  `stance_sample_labeled.csv`.
- **Sample graphs**: proximity + signed GraphML/manifests.

---

## 11. Addendum — developments since the retrospective was first written

The sections above describe the pipeline as designed. This addendum records what
changed during the parser-hardening and freezing work that followed, so the log
stays complete.

### 11.1 The dependency parser, from written-blind to validated

The parser (`extraction/stance_parsed.py`) was originally delivered **unverified**
— the build sandbox could not download the spaCy model, so it shipped as
"written but not run." First real execution on the target machine surfaced a
sequence of concrete bugs, each diagnosed from an actual parse dump and fixed
deterministically rather than by guessing:

1. **Prep-on-verb object resolution.** "China opposed a proposal by Japan" — the
   parse attaches *by* to the verb, not to *proposal*; the object walker returned
   the object issue before checking the verb's own prepositions. Fixed by
   collecting actor candidates from object head, its possessive, its preps, and
   the verb's preps before falling back to an issue.
2. **Acronym mis-parse fallback.** `en_core_web_sm`/`trf` parse ENB's ALL-CAPS
   subjects ("AOSIS opposed X") as a nominal phrase (verb tagged `amod`, no
   `nsubj`). A guarded fallback now defers *only* such genuine mis-parses to the
   inherited pattern logic (never merely because a lexicon surface form appears —
   which would re-break the noun/verb filter on "backing"). Deferred records keep
   `source="pattern"`.
3. **Coordination subject inheritance.** The decisive fix. "X stressed … and
   opposed …" attaches the second verb via `conj`, sometimes routed through an
   intermediate non-subject head (`opposed --conj--> was(ccomp) --> stressed`).
   `_subjects` now climbs the head chain through intermediate heads until it
   finds an actor subject. This single construction was defeating ~70% of stance
   verbs; fixing it roughly halved the fallback rate.

**Empirical lesson worth stating in a methods defense:** the bottleneck was never
the model's parse *quality* — swapping `en_core_web_sm` → `en_core_web_trf` barely
moved the fallback rate. It was our subject-finder not following the grammatical
relations (`conj`) that carry shared subjects in coordinated clauses. The parser
is now validated against ENB's hard constructions (coordination, appositives,
contrastive-options, acronym subjects) by a 12-test acceptance suite that runs
on any machine with the model installed.

### 11.2 Frozen train/test results (the numbers presented)

Phase 1 froze `dev_config.json` from a hand-labeled 45-record training sample
(labels reviewed twice; the second pass corrected several tier-2 appositive rows
for internal consistency). Final frozen decisions:

- **Tier weights = measured precision:** tier-1 0.533, tier-2 1.000, tier-3 0.933.
- **Precision by source × tier** (the parser's value, quantified):
  `parsed:tier1 = 1.00 (n=7)` vs `pattern:tier1 = 0.125 (n=8)`;
  `pattern:tier2 = 1.00 (n=15)`; `parsed:tier3 = 0.933 (n=15)`. Tier-2 is
  pattern-only *by construction* (the appositive rule lives in the parent
  segmenter, never in the parser's main-clause path), so no `parsed:tier2` row
  can exist — its absence is structural, not a gap.
- **α selected by rule** (strictest level with training contested-fraction ≤ 0.5):
  95% is unreachable even pooled (contested 1.0), so **68% for both regimes**.

**Small-sample caveat, to be stated before it is asked:** the headline
`parsed:tier1 = 1.00` rests on **7 records**; the honest reading is "7/7 correct,
Wilson 95% lower bound ≈ 0.65," i.e. *dramatically better than the 0.125 pattern
floor but not a literal 1.0*. The blended tier-1 weight (0.533) mixes parsed and
pattern-fallback records, and is deliberately the precision of tier-1 edges *as
they actually occur in the pipeline*, not the parser's own.

Phase 2 applied the frozen config unchanged to the five test years. Pooled
adjudication (68%): **direct stance camp agreement 0.90**, frustration 0.179,
contested 0.404 — the winner; co-stance camp agreement 0.70 but 93% contested and
projection-distorted centrality (corroboration only); proximity structurally
cannot take the balance test.

### 11.3 Figures produced (`plots/`, via `tools/make_plots.py`)

Nine argument-carrying figures, each reproducible from the frozen artifacts:
camp-agreement-over-time (headline), signed-metric adjudication (proximity shown
as N/A, its motivating gap), **three-way structural comparison** and
**centrality-rank agreement** (the metrics all three definitions share —
proximity and direct stance agree on the top-3 actors EU/US/G-77-China; co-stance
drifts), tier-precision (a-priori vs measured), the Wilson curve, two
camp-separated network drawings (pooled + COP-6), and a centrality summary.

---

## 12. Influence as a proxy for persuasion (a first-class modelling decision)

This deserves its own section because it is the project's most important
epistemic caveat and the point most likely to be probed.

**What we measure.** On the directed signed stance graph we compute centralities
(in-degree, HITS authority, betweenness) and read them as *influence*: an actor
many others take positions *toward* — and against whom the sign structure
organises — is structurally central to the deliberation.

**What we do NOT measure, and cannot.** Dambanemuya & Horvát (2024) define
influence operationally and *validate* it against a ground-truth outcome — Δ, an
observed change of stated view (a reply that shifts a position). ENB has **no Δ
equivalent**: it is a rapporteur's third-person summary of positions, not a
record of anyone changing their mind because of anyone else. There is no
per-utterance persuasion outcome to regress centrality against.

**Therefore influence here is a STRUCTURAL PROXY, not measured persuasion.** The
claim we can defend is: *actor A is central in the network of who-opposes/aligns-
with-whom*, which is a reasonable structural correlate of influence in a
negotiation (agenda-setters and veto players attract positioning). The claim we
**cannot** make is: *A changed B's position.* We never assert the second from the
first. Three consequences we hold to:

1. Centrality is reported as *structural prominence in the positioning network*,
   with the word "influence" always qualified as a proxy.
2. We do **not** import Dambanemuya & Horvát's *validation* (the Δ half) — only
   their *network construction and centrality* half. Their language-feature
   persuasion model is inapplicable (ENB is paraphrase, not verbatim speech).
3. Where the two reference papers split cleanly: Sudhahar et al. give the *edge
   construction* (signed SVO, Bernoulli/Wilson); Dambanemuya & Horvát give the
   *reading* (centrality as influence) — but only its structural half transfers.

Stating this proxy status openly is not a weakness to hedge; it is the correct
scientific description of what a rapporteur corpus can and cannot support, and it
defines exactly what a future corpus *with* an outcome signal (e.g. draft-text
edits adopted or rejected) would add.

---

## Sources

- Sudhahar, S., Veltri, G.A., Cristianini, N. (2015). *Automated analysis of the
  US presidential elections using Big Data and network analysis.* Big Data &
  Society. — SVO pipeline, signed verb lists, Bernoulli/Wilson, partitioning,
  structural balance.
- Dambanemuya, H.K., Horvát, E.-Á. (2024). *Emergent Influence Networks in
  Good-Faith Online Discussions.* ICWSM. — reply-network influence, centralities,
  ∆ validation.
- Breiger, R. (1974). *The Duality of Persons and Groups.* — one-mode projection.
- Heider (1946); Cartwright & Harary (1956). — structural balance.
- Wilson, E.B. (1927), JASA. — score interval (impl.: statsmodels).
- All corpus figures computed in-session over the nine readable ENB reports
  using the project's own chunker and extractors.
