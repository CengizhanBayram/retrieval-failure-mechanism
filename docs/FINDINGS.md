# Findings — full results (draft for the paper)

> Drafted by the assistant now that E1–E5 are complete. States the measured
> numbers and the **pre-registered decisions** (computed by E4, not chosen here),
> with every **exploratory** result and every **interpretive synthesis** flagged as
> such. Companion documents: `AMENDMENT_2026-07-13.md` (status of each run),
> `LIMITATIONS.md` (caveats). Measurement/record — the paper argues significance.

**Panel (7 instruct models, 6 families).** llama3.1-8b, qwen2.5-7b, gemma2-9b,
mistral-7b-v0.3, olmo2-7b, phi3.5-mini, qwen2.5-3b. Difficulty axis: `shell_share`
(distractor reuses the needle's ADJ or NOUN). Detector: Part-2 argmax heads, seed
42. Decoding: greedy, deterministic. Pre-registration frozen (M2 floor 0.10, flip
margin 20 pp over control, α 0.05).

---

## 0. Validity gates (all pass — the results rest on these)

| gate | result |
|---|---|
| **Capture correctness** (manual attention row vs the model's own eager `output_attentions`, fp32) | **7/7 PASS**, worst 5.66e-07 (qwen2.5-7b), best 1.86e-08 (mistral) — covers gemma2 softcap, olmo2 QK-norm, phi3 partial rotary |
| **Self-patch no-op** (patch a recipient with its own donor → token-identical) | `self_patch_ok = true` for every model, every k |
| **No-patch determinism** (two no-patch runs agree) | flip ≈ 0.000 throughout |
| **E2 ↔ E3 same sample** (`pairs_by_cell == pairs_used`) | **self-consistent 7/7** |
| **Control margin** (a causal effect must beat the random-control flip by ≥ 20 pp) | enforced in E4 `effect = bh_rejected AND margin_met` |

---

## 1. E1 — Breaking surface

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

qwen2.5-7b yields the fewest pairs (*n*=113) — the least-powered model (see
`LIMITATIONS.md` §B).

---

## 2. E2 — Failure mechanism (argmax heads, sample-level bucket rates)

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
is the outlier — no single dominant bucket, residual/M1 highest, M2 lowest. (gemma2
also has the head-set tie problem, `LIMITATIONS.md` §D.)

Reproducibility: the E2 grades carry ~1–2 % marginal-sample noise across GPUs
(bucket-rate drift ≤ 2.9 pp; the three-regime carriers ≤ 1.2 pp; `LIMITATIONS.md`
§A). Self-consistency (the property E3/E4 depend on) holds for all 7.

---

## 3. E3 / E4 — Causality (bidirectional patching, k-sweep)

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

The raw flip rises with k for **all** seven. But the raw flip is not the effect —
E4 subtracts the random-control flip and requires the ≥ 20 pp margin **and** BH
significance.

**Causal `break` effect** (`effect = bh_rejected AND margin_met`), by the smallest
k at which it holds:

| model | first k with break effect | verdict |
|---|---|---|
| qwen2.5-3b | **k = 5** | causal (pre-registered) |
| olmo2 | **k = 10** | causal (pre-registered) |
| phi3.5 | **k = 10** | causal (pre-registered) |
| qwen2.5-7b | **k = 30** (exploratory) | redundant-causal |
| mistral | **k = 30** (exploratory) | redundant-causal |
| llama3.1 | never (through k = 30) | no causal effect |
| gemma2 | never (through k = 30) | no causal effect |

* **Pre-registered result (k ≤ 10): 3/7 causal** — qwen2.5-3b, olmo2, phi3.5.
* **Exploratory extension (k ≤ 30): +2** — mistral and qwen2.5-7b cross the margin
  only at k = 30. Their k ≤ 10 "null" was **under-dosing** (they have 82 / 50
  retrieval heads; patching 10 was too few) — a **redundancy / distributed-
  retrieval** result, reported as exploratory.
* **llama3.1 and gemma2 never meet the margin**, even patching 30 heads.

The `repair` direction (patch a failure recipient with a success donor → does it
recover?) is **weak and inconsistent** — sparse effects (qwen2.5-3b at k≤10,
qwen2.5-7b/olmo2 at mid-k) that *fall* at high k. **Break is the robust causal
signal** and should lead.

---

## 4. The three regimes (interpretive synthesis — the paper's spine)

Combining the mechanism (§2) with the causal decision (§3):

| regime | models | mechanism (E2) | causality (E4) |
|---|---|---|---|
| **1 · mechanism = cause** | qwen2.5-3b, olmo2, phi3.5 | M2-dominant | causal at low k (pre-registered) |
| **2 · redundant-causal** | mistral, qwen2.5-7b | M2-dominant (mistral 0.84!) | causal only at k = 30 (exploratory) — distributed across many heads |
| **3 · dissociation** | llama3.1 | **M2-dominant (0.62)** | **no causal effect through k = 30** — the M2 signature is a *correlate*, not the bottleneck |
| **4 · non-M2** | gemma2 | residual/M1 (M2 only 0.14) | no causal effect; head-set tie-confounded at low k |

The headline is the **spectrum**, not a single verdict: retrieval failure is causal
in some models, causal-but-redundant in others, and — for **llama3.1** — a genuine
**mechanism-without-causality dissociation** (distractor-capture is visible but
patching the retrieval heads does not break retrieval). This is the contribution
over profiling work that measures only the correlation.

---

## 5. E5 — Robustness

### 5.1 Detector choice (`wu` — argmax vs Wu/copy heads)

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
"detector-independent" robustness claim is **not supported** — the M2 label belongs
to the argmax head population, not to retrieval heads defined arbitrarily. The
causal result (argmax) is unaffected. See `LIMITATIONS.md` §F. This
argmax-vs-copy heterogeneity is itself a reportable mechanistic observation.

### 5.2 Run-to-run reproducibility (`check_e2_repro.py`)

**Self-consistency PASS 7/7** (`pairs_by_cell == pairs_used`) → E3/E4 measure
causality on exactly the sample E2 classified. Run-to-run bucket-rate drift vs the
pre-fix artifacts: **≤ 2.9 pp** (phi 0.0, mistral 0.5, olmo2 0.6, llama 0.6,
qwen2.5-3b 1.2, gemma2 2.6, qwen2.5-7b 2.9) — cross-session GPU-kernel
nondeterminism, immaterial to the conclusions. See `AMENDMENT` §I, `LIMITATIONS` §A.

### 5.3 Not yet run (optional)

`seeds` (formal grading-noise repeat: mean ± range, R_self) and `m2sens` (M2 floor
0.10 vs 0.15 sensitivity) — the pre-registered secondary robustness variants. wu is
the primary detector check and is complete.

---

## 6. Confirmatory vs exploratory (what can be claimed how)

| claim | status |
|---|---|
| M2-dominant failure in 6/7 (argmax heads) | confirmatory |
| 3/7 causal at k ≤ 10 (qwen2.5-3b, olmo2, phi3.5) | **confirmatory** (pre-registered) |
| +2 causal at k = 30 (mistral, qwen2.5-7b) → redundancy | **exploratory** (k > 10, post-hoc) |
| llama3.1 mechanism-without-causality dissociation | confirmatory (no effect through the pre-registered k, and none at k=30) |
| gemma2 non-M2 / not causal | confirmatory, **but** read only past its head-set tie (k ≥ 20; §D) |
| M2 detector-robust | **not supported** — Qwen-only; argmax-specific for 5/7 (§F) |

---

## 7. Status

E1–E4 complete; E5 `wu` and the reproduction check complete; E5 `seeds`/`m2sens`
optional and outstanding. No result is invalidated by any caveat in
`LIMITATIONS.md`. The remaining work is the write-up.
