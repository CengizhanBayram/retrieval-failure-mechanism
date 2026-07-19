# Findings - full results (draft for the paper)

> Drafted by the assistant now that E1-E5 are complete. States the measured
> numbers and the **pre-registered decisions** (computed by E4, not chosen here),
> with every **exploratory** result and every **interpretive synthesis** flagged as
> such. Companion documents: `AMENDMENT_2026-07-13.md` (status of each run),
> `LIMITATIONS.md` (caveats). Measurement/record - the paper argues significance.

**Bottom line.** Distractor-capture (M2) is the common failure *mechanism* (6/7),
but the *causal weight* of the retrieval heads is **graded by model and uncorrelated
with the mechanism strength** - from few-heads-sufficient (qwen2.5-3b, k=5) through
many-heads-redundant (mistral k=30; gemma2 only at its **full 77-head set**) to
statistically-significant-but-below-the-20 pp-margin **even when the entire detected
head set is patched** (llama3.1, 39/39 heads). Every model's break is BH-significant
(no model is causally inert); what differs is *how much* you must patch before it
matters, and the M2 rate does not predict it (gemma2 0.14-M2 and mistral 0.84-M2 are
**both** redundant; llama3.1 0.62-M2 stays sub-margin at the full set; qwen2.5-3b
0.73-M2 needs 5 heads). That gap - visible signature vs graded causal weight - is
what profiling (which measures only the signature) cannot see. **The full-set patch
(exploratory) is decisive against the under-dosing objection: llama3.1 is the unique
dissociation that survives patching all 39 heads; gemma2 was under-dosed and is
redundant, not dissociated.**

**Panel (7 instruct models, 6 families).** llama3.1-8b, qwen2.5-7b, gemma2-9b,
mistral-7b-v0.3, olmo2-7b, phi3.5-mini, qwen2.5-3b. Difficulty axis: `shell_share`
(distractor reuses the needle's ADJ or NOUN). Detector: the prior detector's argmax heads, seed
42. Decoding: greedy, deterministic. Pre-registration frozen (M2 floor 0.10, flip
margin 20 pp over control, α 0.05).

---

## 0. Validity gates (all pass - the results rest on these)

| gate | result |
|---|---|
| **Capture correctness** (manual attention row vs the model's own eager `output_attentions`, fp32) | **7/7 PASS**, worst 5.66e-07 (qwen2.5-7b), best 1.86e-08 (mistral) - covers gemma2 softcap, olmo2 QK-norm, phi3 partial rotary |
| **Self-patch no-op** (patch a recipient with its own donor → token-identical) | `self_patch_ok = true` for every model, every k |
| **No-patch determinism** (two no-patch runs agree) | flip ≈ 0.000 throughout |
| **E2 ↔ E3 same sample** (`pairs_by_cell == pairs_used`) | **self-consistent 7/7** |
| **Control margin** (a causal effect must beat the random-control flip by ≥ 20 pp) | enforced in E4 `effect = bh_rejected AND margin_met` |

---

## 1. E1 - Breaking surface

`shell_share` breaks all seven models **in-window**; **zero** beyond-window
(positional-OOD) cells entered the breaking band, so the confound the design
guards against did not materialise.

| model | breaking cells | E2 matched pairs |
|---|---|---|
| mistral | 29 | 956 |
| olmo2 | 24 | 839 |
| phi3.5 | 20 | 695 |
| llama3.1 | 19 | 701 |
| gemma2 | 18 | 625 |
| qwen2.5-3b | 13 | 450 |
| qwen2.5-7b | 8 | **113** |

qwen2.5-7b yields the fewest pairs (*n*=113) - the least-powered model (see
`LIMITATIONS.md` §B).

---

## 2. E2 - Failure mechanism (argmax heads, sample-level bucket rates)

| model | **M2** capture | M1 silence | correct-attend | residual | dominant |
|---|---|---|---|---|---|
| mistral | **0.84** | 0.00 | 0.08 | 0.08 | M2 |
| qwen2.5-3b | **0.73** | 0.08 | 0.04 | 0.15 | M2 |
| llama3.1 | **0.62** | 0.18 | 0.02 | 0.18 | M2 |
| qwen2.5-7b | **0.62** | 0.16 | 0.01 | 0.21 | M2 |
| olmo2 | **0.62** | 0.03 | 0.15 | 0.21 | M2 |
| phi3.5 | **0.57** | 0.12 | 0.05 | 0.26 | M2 |
| gemma2 | 0.14 | 0.34 | 0.15 | **0.36** | residual (outlier) |

**M2 (distractor-capture) is the dominant failure signature in 6/7 models.** gemma2
is the outlier - no single dominant bucket, residual/M1 highest, M2 lowest. (gemma2
also has the head-set tie problem, `LIMITATIONS.md` §D.)

Reproducibility: the E2 grades carry ~1-2 % marginal-sample noise across GPUs
(bucket-rate drift ≤ 2.9 pp; the three-regime carriers ≤ 1.2 pp; `LIMITATIONS.md`
§A). Self-consistency (the property E3/E4 depend on) holds for all 7.

---

## 3. E3 / E4 - Causality (bidirectional patching, k-sweep)

**Break-flip rate** (patch a success recipient with a failure donor over the top-k
argmax heads; higher = more disruption). Pre-registered k ∈ {1,5,10}; **k ∈ {20,30}
exploratory** (`AMENDMENT` §H).

| model | k=1 | k=5 | k=10 | k=20 | k=30 |
|---|---|---|---|---|---|
| qwen2.5-3b | 0.169 | 0.333 | 0.531 | 0.669 | 0.929 |
| phi3.5 | 0.004 | 0.076 | 0.391 | 0.845 | 0.839 |
| olmo2 | 0.013 | 0.132 | 0.219 | 0.327 | 0.639 |
| qwen2.5-7b | 0.009 | 0.062 | 0.053 | 0.106 | 0.425 |
| mistral | 0.002 | 0.030 | 0.051 | 0.058 | 0.287 |
| gemma2 | 0.016 | 0.014 | 0.040 | 0.101 | 0.117 |
| llama3.1 | 0.001 | 0.073 | 0.067 | 0.077 | 0.100 |

The raw flip rises with k for **all** seven. But the raw flip is not the effect -
E4 subtracts the random-control flip and requires the ≥ 20 pp margin **and** BH
significance.

**Two things must be read separately** and the earlier draft conflated them:
* **BH significance** - is the break flip distinguishable from the random control
  at all? (`bh_rejected`)
* **The pre-registered `effect` gate** - is it *also* ≥ 20 pp above the control?
  (`effect = bh_rejected AND margin_met`)

The margin exists precisely to separate "real but small" from "large enough to be
the bottleneck". Reading `bh_rejected` as the causal-weight axis:

| model | break BH-significant from | clears 20 pp margin at | causal weight |
|---|---|---|---|
| qwen2.5-3b | k = 1 | **k = 5** | large - few heads suffice |
| phi3.5 | k = 5 | **k = 10** | large |
| olmo2 | k = 5 | **k = 10** | large |
| qwen2.5-7b | k = 5 | **k = 30** (exploratory) | redundant - many heads |
| mistral | k = 5 | **k = 30** (exploratory) | redundant - many heads |
| gemma2 | k = 10 (tie-confounded; clean from k=20) | **k = 77 = full set** (exploratory) | redundant - needs the whole set |
| llama3.1 | **k = 5** | **never - sub-margin even at k = 39 = full set** | **significant but sub-margin - the unique dissociation** |

**Full-set patch (exploratory, `--extra-k`) - the decisive test of the under-dosing
objection.** Patching the *entire* detected head set for the two sub-margin models:

| model | full set | break flip | control | margin | clears 20 pp? | verdict |
|---|---|---|---|---|---|---|
| llama3.1 | k = 39 (all 39) | 0.126 | 0.043 | **+0.083** | **no** | **dissociation - holds** |
| gemma2 | k = 77 (all 77) | 0.478 | 0.040 | **+0.438** | **yes** | **redundancy - reclassified** |

* **Every model's break is BH-significant** by k = 5-10 (llama and gemma included,
  p ≈ 1e-4): patching the retrieval heads measurably disrupts retrieval in all
  seven, distinct from the random control. **No model is causally inert.**
* The **full-set patch settles the under-dosing objection decisively**:
  * **llama3.1** stays sub-margin even patching **all 39 detected heads** (margin
    +0.083 < 0.20). Its dissociation is **not** an artifact of patching too few
    heads - it is the **unique genuine dissociation**: maximal visible signature
    (0.62 M2), causal weight that never reaches threshold.
  * **gemma2** *was* under-dosed: at its full 77-head set the break flip jumps to
    0.478 (margin +0.438), clearing the margin decisively. gemma2 is **redundant-
    causal**, not sub-margin - reclassified out of the dissociation bin.
* **The pre-registered decision (k ≤ 10) is binary**: 3/7 clear the margin
  (qwen2.5-3b, olmo2, phi3.5); the other four are significant-but-sub-margin. The
  redundancy regime (mistral & qwen2.5-7b at k = 30; **gemma2 at the full set**) and
  the confirmed llama3.1 dissociation at the full set are **exploratory refinements**
  from the k-extension, not part of the pre-registered result.

**Break vs repair - necessity ≠ sufficiency.** The `break` flip rises monotonically
with k for every model; the `repair` flip (inject a success donor into a failure)
is sparse and **collapses at high k** (qwen2.5-3b repair 0.33 at k=10 → 0.04 at
k=30). Patching the retrieval heads is enough to *disrupt* a working retrieval
(break); it is generally **not** enough to *restore* a broken one (repair). Break is
the robust causal signal and leads; the break/repair asymmetry is the empirical body
of the necessity-vs-sufficiency distinction.

---

## 4. The synthesis - mechanism strength does not predict causal weight

Crossing the mechanism (§2, the M2 rate) with the causal weight (§3, how much you
must patch to clear the margin):

| model | M2 rate | causal weight | regime |
|---|---|---|---|
| qwen2.5-3b | 0.73 | large - 5 heads suffice | causal (pre-registered) |
| olmo2 | 0.62 | large - 10 heads | causal (pre-registered) |
| phi3.5 | 0.57 | large - 10 heads | causal (pre-registered) |
| mistral | **0.84** | redundant - 30 heads | redundant (exploratory) |
| qwen2.5-7b | 0.62 | redundant - 30 heads | redundant (exploratory) |
| gemma2 | **0.14** | redundant - full 77-head set | redundant (exploratory) |
| llama3.1 | 0.62 | **sub-margin even at the full 39-head set** | **dissociation (unique)** |

**The central result is a dissociation between two magnitudes, not between presence
and absence - and the M2 rate tells you nothing about the causal architecture:**

* the **lowest**-M2 model (gemma2, 0.14) and the **highest**-M2 model (mistral, 0.84)
  are **both redundant** - many heads needed, but a clear effect once patched;
* llama3.1 (mid, 0.62 M2) is the **unique dissociation**: significant but sub-margin
  **even patching all 39 detected heads** - maximal visible signature, causal weight
  that never reaches threshold;
* qwen2.5-3b (0.73 M2) has the **strongest** causal weight (5 heads suffice).

M2 rate and causal architecture are **uncorrelated across the panel** (0.14→redundant,
0.62→dissociation, 0.62→redundant, 0.73→5-heads, 0.84→redundant).

So "the retrieval heads attend the distractor" (the profiling observable) does not
predict "patching them matters, and how much" (the causal fact). That gap - visible
signature vs graded causal weight, **uncorrelated across the panel** - is the
contribution over profiling work that measures only the correlation. llama3.1 is the
sharpest case: **maximal visible signature, causal weight that stays below threshold
even at the full detected set** - yet still BH-significant. A *strength* dissociation
that survives the strongest form of the under-dosing objection, not a
mechanism-without-causality one.

gemma2 is the one architecture that is not M2-dominant at all (residual/M1). Its
causal signal is tie-confounded at low k (§D) and sub-margin through k = 30, but at
its **full 77-head set it clears the margin decisively** (break 0.478, margin +0.438)
- so gemma2 is **redundant-causal**, not dissociated. Its distinct non-M2 profile is
**plausibly related to** its softcap + sliding-window attention, but that link is not
tested here and is stated as conjecture, not result.

### 4.1 Where is llama3.1's bottleneck? - value / MLP patch sites (exploratory)

Since llama3.1's retrieval-head *outputs* (`z_h`) are sub-margin even at the full
39-head set, the patch site was moved downstream (amendment §M):

| site | qwen2.5-3b (causal ref.) | llama3.1 | reading |
|---|---|---|---|
| **v** (value cache-swap) | break 0.99, repair 0.00 | break **1.00**, repair **0.00** | **confounded** - break-only = content transport, not a value pathway; a **caution**, not evidence |
| **mlp** (layer MLP) | **bidirectional** (break 0.28→0.38, repair ~0.34, ≫ control) | **null** (break 0.13 ≈ ctrl 0.13; repair 0.10 ≈ ctrl) over all **12** head-layers | mlp is causal in the causal model, **null** in llama3.1 |

**llama3.1's failure is not localised to the retrieval-head circuit at either
interpretable site** - not the head outputs (`z_h`) and not their layers' MLP. The
dissociation holds across two intervention points; the value site is uninterpretable
(content-transport confound). This *strengthens* the dissociation rather than turning
it positive. **Exploratory.**

---

## 5. E5 - Robustness

### 5.1 Detector choice (`wu` - argmax vs Wu/copy heads)

Dominant bucket under each detector (argmax M2 → copy M2):

| model | argmax M2 | copy M2 | copy dominant | M2 detector-robust? |
|---|---|---|---|---|
| qwen2.5-3b | 0.73 | 0.66 | m2_capture | **yes** |
| qwen2.5-7b | 0.62 | 0.56 | m2_capture | **yes** |
| mistral | 0.84 | 0.20 | residual | no |
| olmo2 | 0.62 | 0.13 | residual | no |
| phi3.5 | 0.57 | 0.26 | residual | no |
| llama3.1 | 0.62 | ~0 | m1_silence (1.00) | no |
| gemma2 | 0.14 | 0.04 | m1_silence (0.94) | no |

**M2 is detector-robust only for the Qwen family (2/7).** For the other five it is
specific to the argmax (attention-based) heads; the copy-score heads fail by
residual attenuation (mistral, olmo2, phi3.5) or silence (llama3.1, gemma2). So the
"detector-independent" robustness claim is **not supported** - the M2 label belongs
to the argmax head population, not to retrieval heads defined arbitrarily. The
causal result (argmax) is unaffected. See `LIMITATIONS.md` §F. This
argmax-vs-copy heterogeneity is itself a reportable mechanistic observation.

### 5.2 Run-to-run reproducibility (`check_e2_repro.py`)

**Self-consistency PASS 7/7** (`pairs_by_cell == pairs_used`) → E3/E4 measure
causality on exactly the sample E2 classified. Run-to-run bucket-rate drift vs the
pre-fix artifacts: **≤ 2.9 pp** (phi 0.0, mistral 0.5, olmo2 0.6, llama 0.6,
qwen2.5-3b 1.2, gemma2 2.6, qwen2.5-7b 2.9) - cross-session GPU-kernel
nondeterminism, immaterial to the conclusions. See `AMENDMENT` §I, `LIMITATIONS` §A.

### 5.3 Not yet run (optional)

`seeds` (formal grading-noise repeat: mean ± range, R_self) and `m2sens` (M2 floor
0.10 vs 0.15 sensitivity) - the pre-registered secondary robustness variants. wu is
the primary detector check and is complete.

---

## 6. Confirmatory vs exploratory (what can be claimed how)

| claim | status |
|---|---|
| M2-dominant failure in 6/7 (argmax heads) | confirmatory |
| every model's break is BH-significant by k = 5-10 (no model causally inert) | confirmatory |
| 3/7 clear the 20 pp margin at k ≤ 10 (qwen2.5-3b, olmo2, phi3.5) | **confirmatory** (pre-registered) |
| the other 4/7 are BH-significant but **sub-margin** at k ≤ 10 | **confirmatory** (pre-registered decision is binary) |
| mistral & qwen2.5-7b clear the margin at k = 30 → redundancy | **exploratory** (k > 10, post-hoc) |
| **llama3.1 stays sub-margin even at the full 39-head set → unique dissociation, under-dosing ruled out** | **exploratory** (full-set patch) - the decisive robustness result |
| **gemma2 clears the margin at the full 77-head set → redundant, reclassified out of dissociation** | **exploratory** (full-set patch) |
| llama3.1: dominant mechanism, causal weight below threshold at every k incl. full set - a *strength* dissociation | confirmatory (sub-margin) + exploratory (full-set robustness) |
| **M2 rate does not predict causal weight** (0.14 & 0.84 both redundant; 0.62 dissociates) | confirmatory direction, sharpened by the exploratory full-set |
| gemma2 behaviour ↔ softcap/sliding-window | **conjecture** - not tested |
| M2 detector-robust | **not supported** - Qwen-only; argmax-specific for 5/7 (§F) |

---

## 7. Status

E1-E4 complete; E5 `wu` and the reproduction check complete; E5 `seeds`/`m2sens`
optional and outstanding. No result is invalidated by any caveat in
`LIMITATIONS.md`. The remaining work is the write-up.
