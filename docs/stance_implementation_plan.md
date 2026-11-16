# Design — Stance / SVO Layer: Implementation Plan

*Project: Modeling Influence and Deliberation in UN Climate Negotiations*
*Date: 5 July 2026 · Companion to `svo_stance_audit.md` (empirical basis) *

## 0. Governing principles

1. **Reuse over reimplementation.** Where an existing, maintained package
   resolves a component, use it; custom code is confined to (a) domain data
   (lexicons), (b) thin glue, (c) genuinely project-specific logic. Each
   component below names its resolving package or declares why it must be
   custom.
2. **Same architecture discipline as the rest of the pipeline**: extraction is
   separated from edge semantics, edge semantics from graph storage; data
   (lexicons, weights) lives in config, decisions live in code; every component
   lands with pyright + pytest; provenance is preserved end to end.
3. **Empirically grounded**: design targets follow the audit's findings — 172
   direct actor→actor stance patterns, 53% issue-directed stance, ~4,276
   projected signed pairs, ENB's coalition-appositive and "on behalf of" idioms.

## 1. Background concepts (summarised; full exposition in the audit discussion)

**Signed one-mode projection.** Stance data is naturally bipartite: actors take
signed positions on issues. Projecting onto the actor side links two actors that
share an issue; the sign algebra is kept (same sign ⇒ positive/aligned tie,
opposite ⇒ negative/opposed). Duality of the two modes: Breiger (1974), *The
Duality of Persons and Groups* (general network-science source). We use the
**agenda item (chunk)** as the issue unit — the document's own issue structure,
already segmented by our validated chunker — instead of clustering free-text
issue phrases, which is unstable at our corpus size (~4.4k sentences vs the
130k articles of Sudhahar et al.). Known cost (accepted, tier-weighted down):
sub-issue conflation within an agenda item.

**Bernoulli + Wilson aggregation.** Per ordered actor pair, observations of
sign are draws of a Bernoulli variable (Sudhahar et al. 2015, §"Positive and
negative relations", uploaded PDF). Naive p̂=k/n is maximally overconfident at
n=1. The Wilson score interval (Wilson 1927, JASA; "a standard approach in
statistics" per the paper's Appendix A2) gives a small-sample-correct interval;
we use its **lower bound** as a conservative edge confidence, so repetition —not
single sightings — creates strong edges. Verified behaviour
(statsmodels, 95%): 1/1 → 0.207; 4/4 → 0.510; 9/10 → 0.596; 18/20 → 0.699.
Pairs whose interval straddles 0.5 are kept and flagged *contested* — the
pair-level analogue of the paper's divisiveness measure.

## 2. Package resolutions (decided up front)

| Need | Resolved by | Verified | Custom code remaining |
|---|---|---|---|
| Dependency parsing; subject/object attachment | **spaCy** (`en_core_web_sm`) — successor to the paper's Minipar | install blocked in sandbox (network allowlist); runs on target machine | none |
| Compound subjects ("The US and EU opposed…") | **spaCy** `token.conjuncts` / `dep_ == "conj"` | — | thin mapping to records |
| Coalition appositive ("X, with A, B and C, …") | **spaCy** prep/appositive attachment + conjuncts | — | idiom rule (~20 lines) |
| Wilson interval | **statsmodels** `stats.proportion.proportion_confint(k, n, method="wilson")` | ✅ reproduces reference table above | none — do **not** hand-roll the formula |
| Signed bipartite projection | **networkx** `bipartite.generic_weighted_projected_graph` with a sign-product weight function | ✅ 6-line weight fn reproduces the worked example | the 6-line weight fn |
| Graph storage / centralities (in-degree, HITS, betweenness) | **networkx** (already the project substrate) | in use since the proximity graph | none |
| Spectral partitioning of the **signed** graph | **numpy/scipy** `eigh` on the signed adjacency / signed Laplacian (networkx has no signed-graph partitioner) | — | ~15 lines of linear-algebra glue |
| Balance triangle census | **networkx** triangle enumeration + sign product | — | ~10 lines |
| Lexicon/config loading | existing `enbgraph.utils.load_yaml` | in use | none |
| Actor identity, sentences, provenance | existing gazetteer extractor + chunker | validated earlier | none |

Genuinely custom (justified): the **ENB stance-verb lexicon** (domain data —
the paper's own method is manually curated verb lists, Appendix Table 8); the
**`StanceRecord` contract and extraction orchestration** (project-specific
semantics); **tier policy** (research choice); **"on behalf of" handling**
(domain idiom).

## 3. Components

### C1 — `extraction/stance.py`: `StanceExtractor` → `StanceRecord`s
Record = (subject_actor, sign, verb_lemma, object_type ∈ {actor |
attributed_issue | issue}, object_actor?, tier, chunk_id, sentence index/offsets).
**Why records, not edges**: same separation that structured the whole pipeline
(mentions vs edges); both edge definitions consume identical records; testable
per-sentence without graphs; provenance feeds the webapp's existing evidence
highlighting.
Sub-parts, each answering a corpus finding:
- **Verb lexicon** in `config/stance_verbs.yaml` (support/oppose lists +
  lemma variants). Mirrors the paper's hand-built lists; YAML because sign
  assignments are reviewable modeling choices (same argument as the gazetteer).
- **Compound-subject expansion** via spaCy conjuncts — fixes the exact COP-6
  sentence that exposed the proximity baseline ("The US and EU opposed…").
- **Coalition appositive** rule: ", with / supported by A, B and C" →
  one record per member, direction member→head, sign from the appositive verb
  (passive: A supports X). Mandatory: ENB's standard alignment idiom; the
  COP-5 six-member example is one sentence → six records.
- **"on behalf of"**: store `speaker` and `attributed_to`, do **not** rewrite —
  whether bloc statements accrue to country, bloc, or both is a modeling
  decision to surface (question for the supervisor), not to bury in extraction.
- Tier tagging: 1 = parsed direct actor object; 2 = same-sentence coalition/
  compound alignment; 3 = issue stance (consumed only by projection).

### C2 — `graph/stance.py`: two `EdgeDefinition`s (existing seam; builder untouched)
- `DirectStanceEdges` — directed, signed; records with actor objects (tiers 1–2).
- `CoStanceEdges` — undirected, signed; within-chunk projection of tier-3
  records via the networkx bipartite projection above.
**Why two, not one merged graph**: the brief asks to "test several approaches
and explain"; separate definitions ⇒ separately comparable graphs; merging
later is trivial, unmerging is not.

### C3 — `graph/aggregate.py`: Wilson pass
Post-build pass over accumulated (k, n) per pair → sign, confidence
(Wilson lower bound via statsmodels), `contested` flag.
**Why a post-pass, not builder logic**: aggregation is statistical policy, not
bookkeeping; builder stays generic (proximity untouched); policies swappable
(naive vs Wilson side-by-side for the presentation).

### C4 — Tier weights in config, tuned on training years
Defaults 1.0 / 0.75 / 0.25 for tiers 1/2/3 — explicitly *hyperparameters with
no principled a-priori values*; tuned on the development years (this is what
"the others can be used as a kind of training set" is for), frozen for the
presented years.

### C5 — Manual validation (~50–70 records)
Hand-label extracted records against source sentences; report precision/recall.
Mirrors the paper's own protocol (72 triplets; 62%/57%) and supplies the error
rate its 0.38^k repetition argument needs.

### C6 — Headline validation: signed partitioning / structural balance
Spectral bisection of each year's signed graph (scipy `eigh` glue); test whether
Annex I vs G-77/China camps emerge — the exact analogue of the paper's
Democrat/Republican recovery. Balance theory: Heider (1946); Cartwright &
Harary (1956) — cited by the paper itself. A negative result is still a
finding (coalition fluidity), and either way it's the key slide.

### C7 — Pipeline registration
`make_edge_definition`: add `stance` and `costance` (+ `--tier-weights`,
`--stance-lexicon` args). One-line seam, by design.

### C8 — Webapp surfacing
Signed edges coloured (green/red), sign + tier + confidence in the evidence
panel; reuses the existing evidence/highlight machinery. Small `services.py` /
`graph.js` delta.

## 4. Sequencing

1. C1 lexicon + records with **pattern-based** compound/coalition handling
   (buildable and testable in the sandbox now; parser-independent).
2. Parser attachment upgrade to C1 (target machine; spaCy model installable
   there). Pattern fallback retained for environments without the model.
3. C2 edge definitions + tests (hand-made records).
4. C3 Wilson pass (+ statsmodels dependency added to requirements files).
5. C7 registration; run on all nine years; inspect manifests.
6. C5 manual validation → error rate.
7. C6 balance/partitioning analysis.
8. C4 tier tuning on training years; C8 webapp last (cosmetic).

Rationale: value-ordering under uncertainty — everything testable-now first,
machine-dependent second, analysis after graphs exist, cosmetics last.

## 5. Risks & mitigations

- **Parse quality on ENB's long appositive sentences** — mitigated by tier
  weighting, Wilson aggregation (paper's own remedy for 62% precision), and C5
  measurement rather than assumption.
- **Chunk-level issue conflation** — accepted, weighted down (tier 3),
  disclosed; sentence-level co-stance (tier 2) provides the high-precision
  alignment signal.
- **2005 sparsity** (7 direct patterns) — projection supplies 320 pairs; still
  flag when choosing presentation years.
- **COP-3 remains OCR-blocked** — unchanged known gap.

## 6. Sources

- Sudhahar, Veltri & Cristianini (2015), *Big Data & Society* (uploaded PDF):
  SVO pipeline, manual verb lists, Bernoulli/Wilson weighting, 62%/57%
  validation, partitioning validation, structural balance, divisiveness.
- Dambanemuya & Horvát (2024), *ICWSM* (uploaded PDF): reply-network influence,
  centrality set; motivates reading centralities on the stance graph as an
  influence **proxy** (no ∆-style outcome exists in ENB — stated limitation).
- Breiger (1974), *Social Forces*: duality/one-mode projection (general
  network-science knowledge, named source).
- Heider (1946); Cartwright & Harary (1956): structural balance (as cited by
  Sudhahar et al.).
- Wilson (1927), *JASA*: score interval (general statistics; implementation:
  statsmodels, verified in-session).
- Corpus figures: `svo_stance_audit.md`, computed in-session over the nine
  readable ENB reports.
