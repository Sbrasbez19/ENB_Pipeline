# Understanding the Signed Stance Layer — Tiers, Wilson Aggregation, and Balance

*A from-first-principles walkthrough of the three pieces that tend to blur
together: (1) the tiered stance edges, (2) the Bernoulli+Wilson aggregation,
(3) the structural-balance validation. Kept as a companion to the code
(`enbgraph/extraction/stance.py`, `graph/stance.py`, `graph/aggregate.py`,
`graph/balance.py`).*

The clearest way to see how these fit is to **follow one edge through the
pipeline**, from a sentence to a validated signed graph. That is the spine of
this document.

```
sentence ──▶ StanceRecord(s)  ──▶  signed Edge(s)  ──▶  accumulated (pos,neg,n)
             [TIER assigned]       [tier→weight]        per actor pair
                                                             │
                                              Bernoulli+Wilson pass
                                                             ▼
                                        sign + confidence + contested
                                                             │
                                                structural-balance analysis
                                                             ▼
                                     "do Annex I vs G-77/China emerge?"
```

---

## Part 1 — Why tiers exist, and what they are

### 1.1 The problem tiers solve

A stance between two actors can be stated in ENB in very different grammatical
forms, and **those forms are not equally trustworthy to extract**. Compare:

- *"The US opposed Japan."* — subject, verb, object. If our pattern matcher
  finds an actor before the verb and an actor after it, the reading is almost
  certainly right.
- *"Saudi Arabia, supported by Kuwait, objected…"* — the relation *Kuwait
  supports Saudi Arabia* comes from the appositive "supported by", a fixed
  idiom. Once you recognise the idiom, it is essentially never wrong.
- *"AOSIS opposed technical assessment processes."* — there is **no second
  actor**. The only way this contributes an actor–actor edge is *indirectly*:
  if some other actor also took a position on the same agenda item, we infer a
  relation between them (projection). That inference is useful but weaker — the
  two actors were never actually mentioned together.

If we treated all three as identical evidence, the reliable signals would be
diluted by the speculative ones. **Tiers are simply a label recording which of
these three situations produced a given edge**, so that later stages can weight
them differently. Nothing more mysterious than that.

### 1.2 The three tiers (as assigned in `extraction/stance.py`)

| Tier | Situation | Example | Produced by |
|---|---|---|---|
| **1** | main verb with an actor object | "The US **opposed** a proposal by Argentina" | the main-clause scan: actor(s) before the verb, actor(s) after it |
| **2** | coalition appositive | "Saudi Arabia, **supported by** Kuwait, …" | the appositive rule: member → head |
| **3** | issue-directed stance (no actor object) | "AOSIS **opposed** technical assessment processes" | recorded now, turned into edges later by *projection* |

Each becomes a `StanceRecord` carrying `tier`, the `sign` (+1 support / −1
oppose), the subject, the object (an actor for tiers 1–2, an issue snippet for
tier 3), and provenance. Tier-1 and tier-2 records feed the **directed** graph
(`DirectStanceEdges`); tier-3 records feed the **projected undirected** graph
(`CoStanceEdges`).

### 1.3 What "projection" means (tier 3), concretely

Tier-3 records have no actor object, so on their own they are actor→*issue*, not
actor→actor. We recover actor–actor structure by treating the **agenda item (the
chunk) as the shared issue**: if two actors both took a stance inside the same
agenda item, we connect them — **same sign ⇒ aligned (+), opposite sign ⇒
opposed (−)**. One paragraph can therefore yield many edges. Worked example (a
single agenda item):

> "The Marshall Islands and Jamaica **supported** rigorous reporting. AOSIS
> **supported** review. Saudi Arabia, with Kuwait, **opposed** mandatory
> formats."

Stancers: Marshall Islands(+), Jamaica(+), AOSIS(+), Saudi Arabia(−),
Kuwait(−). Pairs:

- among the 3 supporters: 3 **positive** pairs (MI–JAM, MI–AOSIS, JAM–AOSIS)
- among the 2 opposers: 1 **positive** pair (SAU–KUW)
- supporter × opposer: 3×2 = 6 **negative** pairs

→ 10 signed edges from one paragraph. (This is the classical *signed one-mode
projection* of an actor–issue bipartite graph; the code's result is asserted
equal to `networkx`'s bipartite projection in `tests/test_stance_graph.py`.)

**Why the chunk, not the exact sentence-level issue?** Because clustering the
free-text issue phrases ("rigorous reporting" vs "detailed review") into stable
issue nodes is unreliable on a corpus this small, whereas the chunker already
segments the report into the document's *own* agenda items. The cost — two
actors stancing on different sub-points of the same item get linked — is real,
and is exactly why tier 3 is down-weighted.

### 1.4 Tier → weight (the number that flows onward)

`graph/stance.py` attaches a **weight** to every edge from its tier
(`DEFAULT_TIER_WEIGHTS = {1: 1.0, 2: 0.75, 3: 0.25}`). This weight is not the
edge's importance in the final graph; it is **how much one observation counts**
when evidence is accumulated. A tier-3 projected pair adds 0.25 to the tally;
a tier-1 direct object adds 1.0. Think of it as *fractional evidence*: weaker
channels contribute fractional "sightings".

**These defaults are a starting point, not a verdict.** The manual validation
(`docs/stance_validation.md`) measured per-tier precision and found the ordering
partly *inverted*: tier-2 appositives were the most reliable (1.00), tier-1 the
least (0.533 blended; 0.125 pattern vs 1.00 parsed), tier-3 in between (0.933). So the empirically-motivated weights are
closer to `{1: 0.55, 2: 1.0, 3: 0.8}`, which you set on the command line
(`--tier-weights "1:0.55,2:1.0,3:0.8"`) and tune on the training years. The
point of making them configuration rather than hard-coded constants is exactly
this: they are hyperparameters to be fitted on development data.

---

## Part 2 — Bernoulli + Wilson: turning a pile of observations into a signed edge

### 2.1 The problem

After building, each actor pair has accumulated some signed evidence. In the
builder this is three numbers per edge (`graph/builder.py`): `pos_w` (weighted
support observations), `neg_w` (weighted oppose observations), and `n_obs`
(raw count). Example: US–EU across a year might be `pos_w=0, neg_w=2.75`
(two direct opposes at 1.0 + one coalition oppose at 0.75).

Two questions remain: **what sign does the edge get, and how much should we
trust it?**

### 2.2 Why the naive answer fails

The obvious estimate of "how positive is this relation" is the share
`p = pos_w / (pos_w + neg_w)`. Its fatal flaw is at **small samples**, which is
our normal case (a pair is often seen once or twice per year):

- seen **once**, support: `p = 1/1 = 1.0` → looks *maximally certain*.
- seen **twenty times**, 18 support: `p = 18/20 = 0.9` → looks *less* certain.

That is backwards. One sighting should not outrank twenty. The naive share
throws away *how much evidence there is*.

### 2.3 The model and the fix

Model each observation of the pair as a flip of a biased coin — a **Bernoulli
trial** with unknown probability *p* of "support". We have *k* supports out of
*n* observations and want a trustworthy estimate of *p*. This is exactly the
framing of Sudhahar, Veltri & Cristianini (2015).

The **Wilson score interval** (Wilson 1927) gives a confidence interval for *p*
that behaves correctly at small *n* (unlike the textbook normal approximation,
which misbehaves near 0 and 1). The interval, at confidence level with
z-score *z*:

```
              p̂ + z²/2n        z          ┌─────────────────────┐
center =  ─────────────── ,  half = ───────── · √ p̂(1−p̂)/n + z²/4n²
               1 + z²/n         1 + z²/n     └─────────────────────┘

interval = center ± half        (z = 1.96 for 95%, ≈0.994 for 68%)
```

We take the **lower bound of the majority side** as the edge's `confidence`:
"given how little we have seen, how strongly can we *defend* this sign?" This is
the same trick used to rank items by rating — it rewards evidence, not luck.

### 2.4 The numbers (computed exactly — keep this table)

| supports *k* | total *n* | naive *p̂* | 95% lower bound | contested at 95%? | 68% lower bound |
|---:|---:|---:|---:|:---:|---:|
| 1 | 1 | 1.00 | **0.207** | yes | 0.503 |
| 2 | 2 | 1.00 | 0.342 | yes | 0.669 |
| 3 | 3 | 1.00 | 0.439 | yes | 0.752 |
| 4 | 4 | 1.00 | **0.510** | no | 0.802 |
| 3 | 4 | 0.75 | 0.301 | yes | 0.501 |
| 9 | 10 | 0.90 | 0.596 | no | 0.767 |
| 18 | 20 | 0.90 | **0.699** | no | 0.813 |

Read the first four rows: a single unanimous sighting scores only 0.207 — almost
worthless — and it takes **four** unanimous observations to clear 0.5 at 95%
confidence. Repetition, not a lone mention, is what builds a strong edge. Row 5
shows dissent: 3-support/1-oppose is *contested* even though the naive share is
0.75, because one counter-example widens the interval below 0.5.

### 2.5 "Contested", and why we keep such edges

`contested = True` when the interval of the positive share **straddles 0.5** —
the evidence cannot confidently call the sign. We do **not** delete these; we
flag them. A pair that different sentences describe as sometimes-aligned,
sometimes-opposed is genuinely ambivalent, and that ambivalence is information
(the paper calls the same idea "divisiveness"). In the webapp, contested edges
render **dashed**.

### 2.6 Why tier weights are floats here

Because a tier-2 observation contributes 0.75 and a tier-3 contributes 0.25, the
accumulated `pos_w`/`neg_w` are **fractional**. The Wilson formula is algebraic
in (k, n) — nothing requires whole numbers — and `statsmodels`' implementation
accepts float pseudo-counts (verified; `graph/aggregate.py` uses it). So a
low-tier observation literally counts as a *fraction of a sighting*, which is the
precise mathematical meaning we wanted for "weaker evidence". Worked example,
US–EU with `neg_w = 2.75`: 95% confidence lower bound = **0.417** (contested at
95%), but **0.736** at 68% — which is why per-year graphs are read at 68%.

### 2.7 The 95% vs 68% point (important, and honest)

At the scale of *one report per year*, most pairs recur only a handful of times,
so **almost nothing clears 0.5 at 95%** — three unanimous oppositions still
can't, as row 3 shows. This is not a flaw in the method; it is the corpus
telling you a single year is thin evidence. Two honest responses, both used:
(1) read per-year graphs at **68% confidence** (the `--alpha 0.32` default in the
webapp/pipeline for signed graphs), treating them as suggestive; (2) for strong
claims, **pool years** so *n* per pair grows — exactly what the "training set"
years are for.

---

## Part 3 — Structural balance: validating the signed graph (`graph/balance.py`)

### 3.1 What we are validating and why it's the key slide

We now have a signed directed graph per year. Is it *meaningful*, or noise? We
have no labelled "correct graph" to compare against — but we do have strong
outside knowledge of UNFCCC politics: there are two broad camps, **Annex I**
(industrialised) and **G-77/China** (developing), that tend to align internally
and oppose across. If the extracted signs recover that split, the whole pipeline
is doing something real. This mirrors the reference paper's own validation:
partitioning their signed network recovered the Democrat/Republican divide.

### 3.2 Structural balance theory in one idea

Balance theory (Heider 1946; Cartwright & Harary 1956) is the formalisation of
"the friend of my friend is my friend; the enemy of my enemy is my friend." A
signed network is **balanced** if its nodes split into two groups with all
**positive** ties *inside* groups and all **negative** ties *between* them. Real
data is never perfectly balanced; we measure *how close* it is, three ways.

### 3.3 The three instruments

**(a) Spectral partition — `partition_signed`.** Build the signed adjacency
matrix (entry = the edge's `signed_weight` = sign × confidence; directed graphs
are symmetrised because balance is about the underlying relation). Take the
**eigenvector of its largest eigenvalue** and split nodes by the *sign of their
component*. For a perfectly balanced graph this recovers the two camps exactly.
(The linear algebra is `numpy.linalg.eigh`; the intuition: the leading
eigenvector is the direction of strongest coherent positive pull, so opposite
signs mark the two poles.)

**(b) Frustration ratio — `frustration_ratio`.** Given the two-camp split, count
the **edges that violate it**: a positive edge across camps, or a negative edge
within a camp. `frustration = violations / signed edges`. **0.0 = perfectly
balanced**; higher = messier. It is the direct, interpretable "how two-camp is
this really?" number.

**(c) Triangle census — `triangle_balance`.** For every triangle of three
mutually-connected actors, multiply its three signs. **Positive product =
balanced** triangle (+++ or +−−); **negative = unbalanced** (++− or −−−). The
share of balanced triangles is a local view of the same property.

**(d) Ground-truth agreement — `camp_agreement`.** Finally, compare the spectral
split against small, uncontroversial membership lists for Annex I and G-77/China
(kept as editable data in `balance.py`). Because eigenvector sign is arbitrary,
the comparison is **relabelling-invariant** (`max` over both assignments). 1.0 =
the split perfectly matches the known camps.

### 3.4 How to read our actual results (honestly)

Across the nine years, camp agreement runs **0.53–0.80 and strengthens over the
decade**. That is a *real* North–South signal — well above the 0.5 you'd get by
chance — but **not** the near-perfect recovery the reference paper obtained. The
reason is the same scale issue from Part 2: they had 130k articles, we have one
report per year, so per-pair evidence is thin and some signs are shaky. The
right framing for the presentation is therefore a **temporal** one: the
two-camp structure is present and *sharpens* as the negotiations mature — which
is arguably a more interesting story for a project about *dynamics* than a static
split would be. Stating the scale limitation plainly is part of the result, not
an apology for it.

---

## One-paragraph summary

Tiers label **how a stance was stated** (direct / coalition / issue-projected),
which sets **how much one observation counts** (its weight). The builder tallies
weighted support vs oppose per actor pair. The **Bernoulli+Wilson** pass turns
those tallies into a **sign plus a confidence** that rewards repeated evidence
and honestly flags **contested** pairs — necessary because per-year counts are
small. **Structural balance** then asks whether the resulting signs recover the
known Annex I / G-77 camps; they do, increasingly, over the decade, at a
strength consistent with the modest corpus size. Every step is a defensible,
sourced choice, and every number above is reproducible from the code.

## Sources

- Sudhahar, Veltri & Cristianini (2015), *Big Data & Society* — signed SVO
  network, Bernoulli/Wilson edge weighting, partitioning validation.
- Wilson, E. B. (1927), *JASA* — the score interval (implementation: statsmodels).
- Heider (1946); Cartwright & Harary (1956) — structural balance.
- Breiger (1974) — one-mode projection of a bipartite (actor–issue) graph.
- Per-tier precision figures: `docs/stance_validation.md`; corpus balance
  figures: computed over the nine readable ENB reports.
