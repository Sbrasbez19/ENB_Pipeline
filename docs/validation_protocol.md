# Validation & Comparison Protocol

*Modeling Influence and Deliberation in UN Climate Negotiations*
*Status: pre-registration — fixed BEFORE any comparison figure is generated.*

## 0. Why this document exists

The task asks us to "test several edge definitions and explain which seems most
relevant." A comparison is only credible if the **yardstick is fixed in advance**
and applied **identically** to every definition — otherwise differences reflect
which metric we chose to report, not which definition is better. This protocol
declares, up front: the definitions under test, the common metric set, which
instruments are universal vs. definition-specific, the confidence policy, the two
evaluation regimes, and the adjudication rubric. Nothing here depends on results
we have already seen.

Two principles govern everything below:

- **Two lenses, not one.** *Balance* tells us whether the edge **signs cohere**;
  *extraction precision* tells us whether the edges are **real**. A graph can be
  balanced yet built on wrong extractions. Both are required; neither substitutes
  for the other.
- **Applicability is a finding, not a fudge.** Some instruments are undefined for
  some definitions (balance needs signs; proximity has none). Reporting "N/A —
  and here is why" is part of the result, not a gap in it.

---

## 1. Definitions under comparison

| ID | Edge definition | Directed? | Signed? | Tiers | Code |
|---|---|:--:|:--:|---|---|
| **B** | proximity (co-occurrence) | no | no | — | `graph/proximity.py` |
| **D** | direct stance | yes | yes | 1 (direct object) + 2 (coalition) | `graph/stance.py` `DirectStanceEdges` |
| **C** | co-stance (issue projection) | no | yes | 3 (projected) | `graph/stance.py` `CoStanceEdges` |

B is the **null model**. D and C are the signed contributions. (Within D, tiers 1
and 2 are pooled and Wilson-aggregated into one signed edge per ordered pair; see
`docs/signed_layer_explained.md`.)

---

## 2. The common yardstick (declared now, applied identically)

### 2a. Universal metrics — computed for **B, D, C**

Structural descriptors:
- node count, edge count, density
- degree distribution (mean, max, Gini or top-k share)
- connected-component structure (giant-component fraction, # isolates)

Centrality (the Dambanemuya–Horvát influence read), computed identically:
- degree / weighted degree
- for **D** (directed): in-degree, out-degree, HITS authority, betweenness
- for **B, C** (undirected): degree, betweenness, eigenvector
- report the **top-k actors** by each, per graph

Temporal trajectory (the project's spine): every descriptor above tracked across
the nine years, plus **node-set stability** (are the same actors central over
time?) and **rank churn** of the top-k.

### 2b. Signed-only metrics — computed for **D, C**; **N/A for B**

From `graph/balance.py` / `graph/aggregate.py`:
- sign distribution (# positive / negative / tied) and **contested fraction**
- spectral partition → **frustration ratio** (0 = perfectly two-camp)
- **triangle balance ratio** (Heider census)
- **camp agreement** vs the Annex I / G-77-China reference sets

> **B is intentionally exempt.** Proximity is unsigned; balance is undefined on
> it — not merely uninformative. This exemption **is the motivating result**:
> proximity cannot separate allies from adversaries. The canonical exhibit is the
> COP-6 edge from *"The US and EU opposed a proposal by Argentina"*, where B links
> two allies exactly as it would link adversaries. B's job in the comparison is to
> be the definition that structurally **cannot** take the balance test.

### 2c. Extraction validation — computed for **D, C**; trivial for **B**

- per-tier precision from a stratified manual sample (protocol and current
  figures: tier-1 0.533, tier-2 1.00, tier-3 0.933 on the frozen training sample, with the parser scoring 1.00 (n=7) vs the pattern floor 0.125 (n=8) on tier-1; see `dev_config.json`)
- B's extraction is actor co-occurrence only (gazetteer), already validated
  upstream; no stance decision to mis-make.

---

## 3. Confidence-level policy (fixed in advance)

Signed edges are Wilson-aggregated (`graph/aggregate.py`). Because per-year
per-pair counts are small:

- **Per-year graphs:** read at **68%** (`--alpha 0.32`). Treat as suggestive;
  always report the **contested fraction** alongside.
- **Pooled graph:** read at **95%** (`--alpha 0.05`), where pooling has grown n
  enough for the stricter bound to be informative.

State both the level and the contested fraction on every signed figure; never
present a signed graph without them.

---

## 4. Two evaluation regimes (run both)

1. **Pooled regime — to adjudicate.** Aggregate all nine years into one graph per
   definition (adequate n), apply §2, read signed graphs at 95%. This produces the
   soundest single verdict on which definition is strongest, with evidence thick
   enough to trust.
2. **Per-year regime — to tell the temporal story.** Nine slices per definition,
   read at 68%, tracking §2a trajectories and §2b balance-over-time. This is where
   the dynamics live (e.g., does camp separation sharpen across the decade?).

Rule: **adjudicate on the pooled regime; narrate on the per-year regime with the
adjudicated winner.** Do not pick a winner from a single noisy year.

---

## 5. Per-definition procedure

For **each** definition, in this order:

**Step 1 — build** (per-year ×9 and pooled). Commands in §7.
**Step 2 — universal metrics** (§2a), both regimes.
**Step 3 — signed metrics** (§2b) — D and C only; record "N/A: unsigned" for B
with the one-line justification.
**Step 4 — extraction precision** (§2c) — D and C.
**Step 5 — record limitations** specific to that definition (below).

Definition-specific caveats to foreground:
- **B (proximity):** conflates alliance and opposition (by construction);
  window choice (sentence vs chunk) changes density materially — report both.
- **D (direct stance):** highest precision but **sparse per year**; most edges
  contested at 95% per-year (hence pooling); tier-1 is the weakest channel
  (attachment errors) pending the spaCy parse upgrade.
- **C (co-stance):** densest alignment picture but most inferential — chunk-as-
  issue conflation links actors stancing on different sub-points; its balance
  result is read **against**, not merged with, D's.

---

## 6. Adjudication rubric

The brief asks which is "most relevant," but the honest verdict is expected to be
**relational, not a single podium**. Weigh definitions on four pre-declared
criteria:

| Criterion | Why it matters | Favours |
|---|---|---|
| **Directionality** | influence/agenda reads need direction | D |
| **Extraction precision** | edges must be real | D (validated), C (0.80) |
| **Balance recovery** | do known camps emerge? | D and C (test both) |
| **Density / coverage per year** | thin graphs tell weak temporal stories | C, then B |

Expected shape of the conclusion (to be confirmed by results, not assumed): **B**
is the interpretable null that exposes the problem; **D** is the highest-precision,
directional, best-validated signal and the one to **lead with**; **C** is the
densest, most inferential alignment view, used as **corroboration**. If results
contradict this, follow the results — the rubric, not the guess, is what is
pre-registered.

---

## 7. Reproducibility

```bash
# per-year, all three definitions
for E in proximity stance costance; do
  python -m enbgraph.pipeline --input <enb_pdfs> --output out/$E \
      --edge $E --alpha 0.32 --format graphml
done
# recommended tuned tier weights for the signed runs (from stance_validation.md)
python -m enbgraph.pipeline --input <enb_pdfs> --output out/stance_tuned \
    --edge stance --alpha 0.32 --tier-weights "1:0.55,2:1.0,3:0.8"

# pooled regime: point --input at the same folder but aggregate across the
# per-year graphs in analysis, reading signed graphs at 95% (alpha 0.05).
```

Balance/centrality metrics: `graph/balance.py`, `graph/aggregate.py`, and
networkx centralities, applied per §2. Record the exact `--alpha`,
`--tier-weights`, and window in each manifest (the pipeline already stamps these).

---

## 8. Global limitations to declare in the write-up

- **No ground-truth outcome (∆).** Influence is a **structural proxy** (centrality
  in a positioning network), not validated persuasion — ENB has no view-change
  signal.
- **Per-year sparsity** forces 68% reads; 95% claims require pooling.
- **COP-3 (1997) excluded** (scanned, no text layer).
- **Pattern-based stance extraction**; the dependency-parser upgrade is pending,
  its payoff (mainly tier-1) already quantified.
- **Manual-precision labels** are a first pass (self-assigned) — spot-check before
  citing.
- **Coverage 1996–2006**; house-style assumptions validated only in that window.

---

## 9. Sources

- Sudhahar, Veltri & Cristianini (2015), *Big Data & Society* — signed SVO
  network, partitioning validation, Bernoulli/Wilson weighting.
- Dambanemuya & Horvát (2024), *ICWSM* — reply-network influence via centrality;
  ∆ ground truth (absent here → proxy framing).
- Heider (1946); Cartwright & Harary (1956) — structural balance.
- Breiger (1974) — one-mode projection.
- Wilson (1927), *JASA* — score interval.
- Companion docs: `signed_layer_explained.md`, `stance_validation.md`,
  `svo_stance_audit.md`, `project_retrospective.md`.
