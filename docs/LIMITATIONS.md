# Limitations — catalogue (draft for the paper)

> Drafted by the assistant for the researcher to audit. It collects every caveat
> surfaced during the confirmatory run, each with its **evidence + magnitude**,
> **how the design addresses it**, and **where it appears**. Measurement/record
> only — it states the caveats, it does not argue what the results *mean*.
>
> Cross-references: `docs/AMENDMENT_2026-07-13.md` (status of each run),
> `scripts/check_e2_repro.py` (the reproduction check), the E5 notebook (the
> robustness variants that formally quantify several of these).

---

## A. E2 grade assignment is not bit-reproducible run-to-run

**What.** Greedy decoding near the success/failure decision boundary flips a small
number of marginal samples between runs, changing which samples become matched
pairs.

**Evidence (2026-07-17, re-run vs the pre-fix artifacts).** 6/7 models drift by
**~1–2 % of pairs**; the resulting **sample-level bucket-rate drift is ≤ 2.9 pp**:

| model | bucket drift | pairs old→new |
|---|---|---|
| phi3.5 | 0.0 pp | 695 → 695 (exact) |
| mistral | 0.5 pp | 967 → 956 |
| olmo2 | 0.6 pp | 843 → 839 |
| llama3.1 | 0.6 pp | 714 → 701 |
| qwen2.5-3b | 1.2 pp | 454 → 450 |
| gemma2 | 2.6 pp | 614 → 625 |
| qwen2.5-7b | 2.9 pp | 115 → 113 |

**Cause.** Batch-composition padding (the adaptive OOM batch-halving in the grading
pass changes left-padding) **and** cross-session GPU kernel nondeterminism
(non-associative float reductions differ across GPU architectures). phi reproduces
exactly because both its runs were on the same 80 GB GPU.

**How addressed.** E3/E4 consume E2's recorded `pairs_by_cell`, so causality is
measured on exactly the sample E2 classified — **self-consistency holds for all 7
models** (`pairs_by_cell == pairs_used`), independent of run-to-run
reproducibility. The pre-registered **E5 seed-repeats** variant is the formal
robustness check (mean ± range, R_self). `check_e2_repro.py` verifies
self-consistency (exit 0/3/2).

**Materiality.** Immaterial to the conclusions: the three-regime carriers drift
≤ 1.2 pp; the largest drift (2.9 pp) is on the smallest-*n* model (§B). See
amendment §I.

---

## B. Statistical power varies sharply across the panel (qwen2.5-7b)

**What.** Matched pairs are scarce for models that resist the `shell_share`
difficulty, because few cells land in the breaking band.

**Evidence.** `pairs_used`: mistral 956, olmo2 839, llama 701, phi 695,
qwen2.5-3b 450, gemma2 625, **qwen2.5-7b 113** — an 8× spread.

**Consequence.** qwen2.5-7b's causal estimates rest on the smallest sample, drift
the most (§A, 2.9 pp), and its break effect only meets the margin at the highest
(exploratory) k. It is the **weakest case**; its "redundant-causal" verdict should
be read as suggestive.

**How addressed.** `pair_min` per-cell gate; BH correction across
{models × directions}; *n* reported per model.

---

## C. k = 20, 30 are exploratory (post-hoc), not pre-registered

**What.** The pre-registered sweep is k ∈ {1, 5, 10}. k ∈ {20, 30} was added
*after* seeing the curves had not saturated.

**Consequence.** The pre-registered result (k ≤ 10) is **3/7 causal** on the break
direction (olmo2, phi, qwen2.5-3b). The extension adds mistral and qwen2.5-7b
(break effect first meets margin at k = 30) — this "redundant-causal" verdict is
**exploratory** and labelled so.

**How addressed.** Labelled exploratory in `configs/e3.yaml`, the notebook filename
(`10*_EXPLORATORY` / `_ksweep_`), amendment §H, and to be labelled so in the paper.

---

## D. Head-set indeterminacy from detector-score ties

**What.** The detector score saturates at 1.0; where the top-k cut splits a block
of heads tied at the ceiling, the patched set is chosen by `(layer, head)` sort
order, not by evidence.

**Evidence.** gemma2 has **13 heads tied at 1.000**, so its k ∈ {1, 5, 10} sets are
arbitrary members of a 13-way tie; olmo2 has a milder tie at k = 10.
`detect.boundary_tie` flags it; E3 records `head_set_ties` per k.

**Consequence.** gemma2's low-k causal nulls are **uninterpretable**; gemma2 is
read only from k ≥ 20 (past the tie). The earlier "gemma2's retrieval heads are not
causal" reading was **withdrawn**.

**How addressed.** Amendment §J; the extended k-sweep covers the whole tied block.

---

## E. The M2 distractor-mass floor is permissive (and deliberately frozen)

**What.** `signature_rules.m2_capture.min_distractor_mass = 0.10` admits **95.6 %**
of all `distractor_hit` failures as M2.

**Evidence.** Pooled `distractor_mass` quantiles over distractor_hit failures
(*n* = 3619): p5 = 0.103, p25 = 0.154, p50 = 0.193.

**Consequence.** The M2 rate is floor-sensitive; a higher floor lowers it.

**How addressed.** The floor is **kept frozen** at the pre-registered 0.10 —
changing it after seeing the bucket rates would be HARKing (amendment §F). An
alternative floor (0.15) is reported as the **E5 `m2sens` sensitivity variant**,
side by side with the primary, and the permissiveness is stated as this limitation.
Enforced in code: `e2_signatures.py --m2-min-distractor-mass` refuses to run
without `--tag`.

---

## F. The M2 signature is argmax-head-specific for 5/7 models (detector-dependent)

**What.** The head set is Part-2's **argmax** (attention-based) detector at seed 42;
Part-3 never re-detects. The **E5 `wu` variant** re-runs E2 on the Wu/copy-score
head list to test whether the mechanism survives a different definition of
"retrieval head".

**Result (E5 `wu`, all 7 models — sample-level dominant bucket, argmax → copy).**

| model | argmax M2 | copy M2 | copy dominant | M2 detector-robust? |
|---|---|---|---|---|
| qwen2.5-3b | 0.73 | 0.66 | m2_capture | **yes** |
| qwen2.5-7b | 0.62 | 0.56 | m2_capture | **yes** |
| mistral | 0.84 | 0.20 | residual (0.40) | no |
| olmo2 | 0.62 | 0.13 | residual (0.39) | no |
| phi3.5 | 0.57 | 0.26 | residual (0.38) | no |
| llama3.1 | 0.62 | ~0 | m1_silence (1.00) | no |
| gemma2 | 0.14 | 0.04 | m1_silence (0.94) | no |

**Consequence.** The distractor-capture (M2) signature is **detector-robust only for
the Qwen family** (2/7). For the other five it is **specific to the argmax
(attention-based) heads**: the copy-score heads instead fail by **residual
attenuation** (mistral, olmo2, phi3.5) or **silence** (llama3.1, gemma2). So "when a
model fails, its retrieval heads show distractor-capture" is a property of *how the
retrieval head is defined*, for most of the panel — it does not generalise across
detector choices.

**How this bounds the claims.** The **causal** result (E3/E4) is measured on argmax
heads, where mechanism (M2) and causality co-occur, so it is internally coherent and
**unaffected**. What the `wu` check qualifies is the *mechanism* half: the M2 label
belongs to the argmax head population, not to "retrieval heads" defined arbitrarily.
This heterogeneity (attention-argmax heads capture distractors; token-copy heads
attenuate or fall silent, except in Qwen) is itself reportable, but the
detector-independence robustness claim is **not** supported and must not be made.

---

## G. Scope and generalization

* **Single headline seed (42).** Probe instantiation is seed-dependent; the E5
  seed-repeats variant (42, 43, 44) is the reliability check.
* **One difficulty manipulation.** `shell_share` (the distractor reuses the
  needle's ADJ or NOUN) is what breaks the strong models in-window; other
  distractor constructions are untested.
* **Repair is the weaker direction.** The repair flip (fixing a failure by
  injecting a success donor) is sparse and inconsistent across models, and *falls*
  at high k for some (e.g. qwen2.5-3b repair 0.33 at k=10 → 0.04 at k=30, plausibly
  over-injection). The **break** direction is the robust causal signal and should
  lead.
* **Positional-OOD is a distinct failure mode.** Failures *beyond* a model's native
  context window are positional-OOD, not distractor-competition; they are excluded
  by the in-window/beyond-window split. Empirically **zero** beyond-window cells
  entered the breaking band, so the confound did not materialise here, but the
  guard is retained.
* **Fixed chat-template date.** Probes stamp a fixed date so templates that inject
  "today" stay deterministic; date-conditional behaviour is not probed.
* **Resource / attention-backend caveat (not scientific).** gemma2 and phi3.5 are
  forced to **eager** attention (softcap correctness / no sdpa kernel in the pinned
  transformers) and need an **A100 80 GB** — eager materialises the full L×L
  attention matrix, which OOMs a 40 GB card at the panel's long contexts. The one
  documented OOM fallback (16384 → 12288) does not rescue eager at 40 GB.

---

## H. What is NOT a limitation (guarded and verified)

Recorded so these are not re-raised:

* **Capture correctness** — the manual attention row matches the model's own eager
  `output_attentions` to ≤ 5.66e-07 (fp32, 7/7), covering gemma2 softcap, olmo2
  QK-norm, phi3 partial rotary. Persisted as `gate_eager_reference.json`.
* **Self-patch** — patching a recipient with its own donor is a token-identical
  no-op for every model (`self_patch_ok = true` throughout).
* **No-patch determinism** — the no-patch control flip is ~0.
* **E2 ↔ E3 same sample** — guaranteed by construction via `pairs_by_cell`
  (self-consistency PASS 7/7), not by run-to-run reproducibility.
* **Control subtraction** — a causal `effect` requires the break flip to beat the
  random-control flip by the pre-registered margin (20 pp) *and* survive BH, so the
  high-k effects are specific, not "patching many heads breaks anything."
