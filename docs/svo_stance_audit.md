# Audit — Stance/SVO Signal in the ENB Corpus vs. the Two Reference Papers

*Project: Modeling Influence and Deliberation in UN Climate Negotiations*
*Date: 5 July 2026*

## Purpose

Two reference papers propose network constructions we want to adapt:

1. **Sudhahar, S., Veltri, G.A., Cristianini, N. (2015).** *Automated analysis of the
   US presidential elections using Big Data and network analysis.* Big Data &
   Society. — a **signed SVO semantic graph** (support/opposition).
2. **Dambanemuya, H.K., Horvát, E.-Á. (2024).** *Emergent Influence Networks in
   Good-Faith Online Discussions.* Proc. ICWSM 2024. — a **directed reply network**
   whose centralities quantify influence.

This audit (a) verifies, against the uploaded texts, what each paper's mechanism
actually is; (b) measures **empirically, on the nine readable ENB reports**, how
much of the raw signal each construction needs is present; and (c) states the
resulting design verdict for our graphs.

## 1. What the papers actually do (verified against the texts)

### Paper 1 — Sudhahar et al. 2015 (signed SVO graph)

Pipeline as stated in the paper and its appendix: co-reference and anaphora
resolution (GATE/ANNIE), Minipar dependency parsing, extraction of
subject–verb–object triplets ("nodes are formed by noun phrases and links by
verbs"); triplets whose verb is **not** in hand-built positive/negative verb
lists are **discarded** ("the remaining triplets are removed as they are not
relevant for our goal of detecting political positions"). Edge sign strength for
an ordered pair (a,b) is modelled as a **Bernoulli variable** (probability of a
positive vs negative verb) with a **Wilson score** confidence correction.
Analyses: spectral partitioning of the signed network (recovers the
Democrat/Republican split — their key validation), centrality (incl. HITS
authority read as agenda-setting), assortativity, hierarchy, **structural
balance**, and a "divisiveness" score from the variance of incoming signs.
Their verb lists (Appendix Table 8) include *support, endorse, favor, defend,
applaud, approve, commend…* vs *oppose, reject, criticize, disagree, condemn,
blame, deny, disapprove…*.
Reported extraction quality without filtering: **62% precision, 57% recall**
(validated on 72 manually coded triplets); errors suppressed by requiring
repeated observation (error ≈ 0.38^k for k independent sightings) and the
Wilson score.

### Paper 2 — Dambanemuya & Horvát 2024 (influence = reply structure + centrality)

Their influence network is **structural, not linguistic**: nodes are
r/ChangeMyView users, edges are **"in-reply-to" relations**, weighted by reply
counts. Influence is a node's **centrality** in this network: in-degree,
out-degree, degree ratio, HITS authority & hubbiness, betweenness. Language
features (word counts, hedges, positive/negative words, lexical diversity,
URLs, similarity to the OP) are a **separate predictor set** for a supervised
classifier of persuasion success; adding the network feature (degree ratio) to
language features raises Random-Forest AUC from ≈0.648 to **0.701** (+7.95%).
Ground truth is the **∆ (delta)** the original poster awards when actually
persuaded; a difference-in-differences analysis shows winning a ∆ subsequently
*raises* the winner's centrality.

**Correction recorded:** an earlier from-memory reconstruction in this project
described Paper 2's influence network as built from "linguistic features of the
responses." That is wrong: the influence network is reply-topology; language
features only enter the persuasion classifier. This changes the adaptation
question (see §4).

## 2. Method of this audit

- **Corpus:** the 9 machine-readable ENB reports (COP-2…COP-12, 1996–2006).
  The two paper PDFs sit in the same folder and were auto-detected by the
  loader (as "2015"/"2024"); they are **excluded** from all counts below.
  COP-3 (1997) remains excluded (scanned, no text layer).
- **Unit:** sentences of REPORT_ITEM chunks from our pipeline; actor mentions
  from our gazetteer extractor (201 canonical actors).
- **Stance lexicon:** Sudhahar et al.'s Table-8 verbs that occur in diplomatic
  register, plus ENB-register equivalents (*welcomed, concurred, objected,
  questioned, cautioned…*), inflection-tolerant regex. Neutral reporting verbs
  (*said, noted, stressed, proposed, urged, called for…*) counted separately.
- **Directed candidate ("A-v-A"):** a stance verb occurring **strictly between
  two different actor mentions** in one sentence — a conservative proxy for a
  parseable actor→actor stance triplet.
- **Coalition constructions:** actor immediately followed by ", with/supported
  by" + actor; and "on behalf of" phrases.
- **Tooling caveat:** a dependency parser (spaCy) could not be installed in
  this environment (model download blocked at the network layer), so counts are
  **pattern-based, not parsed**. This measures *signal availability*; per-triplet
  precision/recall (the analogue of Sudhahar's 62%/57%) will only be measurable
  once parsing is enabled. Multi-actor coalition lists ("supported by A, B, C
  and D") are counted **once**, so the coalition column is a lower bound.

## 3. Results

| Year | Sentences | ≥1 actor | ≥2 actors | Stance verb present | **A-verb-A** | support | oppose | Neutral, ≥2 actors | Coalition ",with/supported by" | "on behalf of" |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1996 | 416 | 206 | 66 | 80 | **11** | 8 | 3 | 32 | 0 | 8 |
| 1998 | 555 | 307 | 120 | 102 | **23** | 17 | 6 | 54 | 13 | 5 |
| 1999 | 519 | 296 | 113 | 84 | **24** | 16 | 8 | 60 | 17 | 0 |
| 2000 | 543 | 217 | 112 | 105 | **33** | 16 | 17 | 44 | 6 | 12 |
| 2001 | 401 | 177 | 77 | 91 | **16** | 11 | 5 | 27 | 7 | 5 |
| 2002 | 462 | 224 | 83 | 107 | **21** | 14 | 7 | 30 | 6 | 0 |
| 2003 | 536 | 248 | 108 | 110 | **20** | 7 | 13 | 66 | 6 | 4 |
| 2005 | 518 | 185 | 87 | 81 | **7** | 6 | 1 | 42 | 1 | 8 |
| 2006 | 475 | 186 | 85 | 102 | **17** | 14 | 3 | 33 | 3 | 3 |
| **Total** | **4425** | **2046** | **851** | **862** | **172** | **109** | **63** | **388** | **59** | **45** |

Mean **19.1 directed stance patterns per year**; stance patterns cover **20.2%**
of multi-actor sentences; support:oppose ≈ **109:63**.

Top stance verbs inside A-verb-A patterns: *supported* (76), *opposed* (49),
*support* (14), *objected* (6), *agreed* (5), *welcomed* (3), *rejected*,
*questioned*, *endorsed* (2 each).

Representative sentences found (verbatim from the corpus):

> "SAUDI ARABIA, supported by the REPUBLIC OF KOREA, the US, IRAN and
> BANGLADESH, objected to restricting the Round Table to ministers…" (COP-2, 1996)

> "The EU, opposed by CHINA, said the Global Environment Facility (GEF) had
> provided most non-Annex I Parties with funding…" (COP-5, 1999)

> "Nigeria, on behalf of the Group of 77 and China (G-77/China), rejected the
> proposal by a 'key Annex I country'…" (COP-6, 2000)

> "The US and EU opposed a proposal by Argentina to remove brackets from text on
> national communications…" (COP-6, 2000)

## 4. Interpretation

**The SVO construction (Paper 1) is viable and is the right primary build.**
ENB prose is written in exactly the actor–verb–object register QNA assumes, and
the dominant stance verbs are the paper's own core pair (*supported/opposed* =
125 of 172 patterns). Two ENB-specific constructions carry much of the signal
and must be handled explicitly, beyond plain active-voice SVO:
(i) the **"X, supported by A, B, C, …" appositive** — ENB's standard alignment
marker, which yields *multiple* support edges per sentence (our count of 59 is
once-per-sentence, so a true lower bound), and whose grammar is **passive**
(edge direction A→X, not X→A); (ii) **"Y, on behalf of Z"** (45 cases) — bloc
representation, which is a *speaks-for* relation, not a stance, and should
update the speaker's attribution rather than add a stance edge.

**A per-year signed actor↔actor network is real but sparse.** ~19 directed
stance patterns/year (7–33), before coalition-list expansion, versus 365 edges
in the COP-6 sentence-proximity graph. Consequences:
- Present the stance layer as a **signed overlay on the proximity baseline**,
  not a free-standing dense network.
- Aggregate signs per ordered pair with **Sudhahar's Bernoulli + Wilson-score
  scheme** — designed precisely for small, noisy per-pair counts — and consider
  pooling adjacent years when a single year's graph is too thin.
- Expanding coalition lists to all members ("supported by A, B, C and D" → 4
  edges) will materially raise edge counts; that expansion is the single
  highest-yield extraction improvement available.

**Paper 2 transfers in its network half, not its validation.** Its influence
network is who-replies-to-whom + centrality; our directed stance edges play the
structural role of reply edges (and are *richer*: signed). The same
centralities (in-degree, degree ratio, authority, betweenness) can be computed
and read as influence — **but only as a structural proxy**: ENB provides no
analogue of the ∆ ground truth, so the persuasion-outcome validation that
anchors their claim is not reproducible on our data and must not be claimed.
The language-feature half of their paper is inapplicable in principle: ENB text
is a rapporteur's paraphrase; delegates' actual wording (what hedges/lexical
diversity are measured on) is not in the source.

**Neutral-verb sentences are not waste.** 388 multi-actor neutral sentences
(*said, noted, proposed…*) are the substrate of the proximity baseline and of a
possible actor+issue graph in Paper 1's fuller sense (they retain issue nodes;
their HITS-authority-as-agenda-setting analysis needs them). Discarding
non-stance triplets, as Paper 1 does, is a choice we can afford only for the
signed layer specifically.

## 5. Design verdict

1. Build **`SVOEdges`** (directed, verb-typed) with a dependency parser once
   available; sign edges with a curated ENB verb lexicon (method mirrors
   Paper 1's manual lists; our lexicon draft is in §2).
2. Handle ENB's two idioms explicitly: **passive "supported/opposed by"
   appositives** (reverse direction, expand full coalition lists) and **"on
   behalf of"** (attribution, not stance).
3. Derive **`StanceEdges`** as the signed actor→actor subset; aggregate with
   Bernoulli + Wilson per Paper 1; analyse structural balance (do Annex I vs
   G-77/China emerge as balanced camps, as the Democrat/Republican split did?).
4. Compute Paper 2's centralities (in-degree, degree ratio, authority,
   betweenness) on the directed stance graph and report them as an
   **influence proxy**, explicitly noting the absence of outcome validation.
5. Keep the proximity graph as the **null model**; the stance layer's value
   claim is precisely that it separates allies from adversaries where proximity
   cannot (demonstrated by the COP-6 US–EU/Argentina sentence, which appears in
   §3 verbatim).

## 6. Limitations of this audit

- Pattern-based counts, not parsed triplets: figures measure signal
  availability, not extraction quality. Paper 1's own unfiltered quality was
  62% precision / 57% recall; ours is unknown until parsing is enabled.
- The A-verb-A rule (verb strictly between two different actors) misses
  stance verbs at sentence edges and passive constructions where both actors
  precede the verb; coalition lists counted once per sentence. Net effect:
  **these are conservative lower bounds.**
- COP-3 (1997) still excluded pending OCR; coverage is 1996–2006.
- Sentence splitting and actor extraction inherit the pipeline's known
  behaviour (validated earlier in the project); errors there propagate here.

## Addendum (same date) — Actor–stanceverb–ISSUE patterns

Follow-up question: how much stance signal has an *issue* object rather than an
actor object, can it be resolved to actor→actor, and what does it contribute?

**Decomposition of all 403 stance sentences with an actor subject (9 years):**

| Category | Count | Share | Resolvable to actor→actor? |
|---|---:|---:|---|
| Direct actor object ("X opposed CHINA", "X, opposed by Y") | 146 | 36% | already actor→actor (in the A-v-A count) |
| Issue object with explicit attribution ("opposed a proposal **by Argentina**") | 18 | 4% | **yes — and already captured** by the A-v-A rule, since the attributed actor follows the verb |
| Issue object, proponent in a *previous* sentence ("…the proposal") | 8 | 2% | yes, via discourse resolution (two-hop); heuristic count, noisy |
| Definite issue, no identifiable proponent within 3 sentences | 18 | 4% | no |
| Anonymous attribution ("a proposal by one Party") | ~1 | <1% | no, by editorial design (rare) |
| **Pure issue object** ("AOSIS opposed technical assessment processes") | **212** | **53%** | **no — no proponent exists; stance is about policy content** |

Key findings:

1. **Issue-directed stance is the majority of ENB positioning (~64%).** This is
   a property of the discourse, not an extraction gap: Parties mostly position
   on *policy content*, not on each other's proposals. It matches Sudhahar et
   al.'s design choice to keep **issues as nodes** (their positive-object-ratio,
   divisiveness, and authority-as-agenda-setting analyses all require issue
   nodes).
2. **Attribution-based resolution adds almost nothing new**: explicit "by
   ACTOR" attributions (18) are already inside the A-v-A count because the
   attributed actor follows the verb; discourse resolution adds ~8 noisy cases;
   anonymized proponents are negligible (~1).
3. **Projection is where issue stances pay off.** Treating the agenda item
   (chunk) as the shared issue and projecting: two actors taking a stance in
   the same agenda item → same sign = alignment (+), opposite sign =
   opposition (−). This induces **4,276 signed actor–actor pairs across the 9
   years (~475/year: 2,580 aligned, 1,696 opposed) vs 172 direct A-v-A — a
   ~25× multiplier** that resolves the sparsity concern of §4.

| Year | 96 | 98 | 99 | 00 | 01 | 02 | 03 | 05 | 06 | Tot |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| projected signed pairs | 533 | 1108 | 258 | 524 | 149 | 296 | 110 | 320 | 978 | 4276 |
| direct A-v-A | 11 | 23 | 24 | 33 | 16 | 21 | 20 | 7 | 17 | 172 |

Caveats on the projection: chunk-level windows conflate sub-issues (two actors
stancing on different points of the same agenda item get linked); an actor with
both signs in one chunk generates contradictory pairs; the discourse/anonymous
heuristics above showed false positives on inspection. Recommended confidence
hierarchy for `StanceEdges`: **direct A-v-A > same-sentence co-stance (compound
subjects, expanded coalition lists) > within-chunk projection**, with weights
descending and per-pair aggregation via Bernoulli + Wilson (Paper 1's scheme).
Coalition expansion remains mandatory and compounds with projection: "CHINA,
with SAUDI ARABIA, VENEZUELA, KUWAIT, INDONESIA, and LIBYA, opposed the
proposals…" (COP-5, 1999) is one sentence → six actor→issue oppose edges plus
fifteen alignment pairs.

## 7. Sources

- Sudhahar, Veltri & Cristianini (2015), *Big Data & Society* — uploaded PDF
  (2053951715572916_full.pdf): SVO pipeline, verb lists (Table 8), Bernoulli/
  Wilson weighting, partitioning validation, structural balance, divisiveness,
  62%/57% triplet validation.
- Dambanemuya & Horvát (2024), *ICWSM* — uploaded PDF
  (41232ICWSM_DambanemuyaH.pdf): reply-network construction, centrality set,
  language-vs-network feature comparison (AUC 0.648 → 0.701), ∆ ground truth,
  difference-in-differences effect of ∆ on centrality.
- All corpus figures: computed 5 July 2026 by the audit script in this project
  over the nine readable ENB reports, using the project's own chunker and
  gazetteer extractor.
