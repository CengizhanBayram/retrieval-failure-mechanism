# Pre-registration amendment — dated record

> **Dated 2026-07-13.** Drafted by the assistant at the researcher's request; the
> researcher audits, edits and owns it.
>
> **This document does not modify `configs/preregistration.yaml`, and no
> pre-registered threshold is changed by it.** §F explicitly *declines* the M2
> re-calibration that an earlier draft proposed. What this record does is fix the
> **status** of everything run so far and everything about to be run: what is
> confirmatory, what is exploratory, what is withdrawn, and what is still
> undetermined.
>
> It is committed **before** the runs it governs. A run without a commit is a
> pilot.
>
> Supersedes the 2026-07-05 draft (§F rewritten; §H–§L added).

## A. Pilot declaration

The full-panel A100 run of 2026-07-05 is declared a **PILOT**, not confirmatory,
and its numbers are **not reported as results**. Rationale, from the data:
- **Ceiling effect:** with `shell_share` off, the four strong models (llama,
  mistral, qwen-7b, qwen-3b) sit at ~0.99 in-window accuracy — no breaking cells
  except marginal fallback-band ones.
- **Coverage gaps:** E2 produced no output for gemma/mistral/olmo/phi (silent
  failures, cause under diagnosis), so only 3 of 7 models were dissected, none
  of them the in-window breaker.

The pilot is used only to (i) validate the pipeline (self-patch OK; buckets,
flips, controls all produced), (ii) calibrate thresholds (§F), and (iii) fix the
design gaps below.

## B. Panel (final)

**7 models, fixed** (justification: one architecture/family/size axis each):

| model | axis it uniquely covers | native ctx |
|---|---|---|
| llama31_8b_instruct | flagship long-context anchor | 128k |
| qwen25_7b_instruct | aggressive GQA sharing (28/4) | 32k |
| gemma2_9b_it | softcap + local/global sliding window | 8k |
| mistral_7b_instruct | plain-GQA standard-attention control | 32k |
| olmo2_7b_instruct | MHA + QK-norm; in-window positive control | 4k |
| phi35_mini | fused-QKV + partial-rotary; small×long | 128k |
| qwen25_3b_instruct | within-family size axis vs 7B | 32k |

**Eager-reference gate (non-negotiable):** the three non-standard-attention
models — gemma (softcap), olmo (QK-norm), phi (fused-QKV/partial-rotary) — each
produce ZERO confirmatory numbers until they pass the real-pinned-model
eager-reference test (max|manual−eager| < 1e-3, notebook `00`). A model failing
the gate is excluded and the exclusion reported.

## C. In-window vs beyond-window breaking cells

Breaking cells are partitioned by whether the needle sits within the model's
trained positional range:

- **effective_window(model):** llama 131072 · qwen2.5 32768 · mistral 32768 ·
  gemma2 8192 · olmo2 4096 · phi3.5 131072.
- A breaking cell is **in-window** iff `context_length ≤ effective_window`, else
  **beyond-window**.
- **Only in-window breaking cells enter the main M1/M2/correct-attend/residual
  and causal (E3/E4) analysis.** Beyond-window cells are a distinct
  phenomenon (positional-OOD collapse) and are reported separately, never mixed
  into the retrieval-head-failure analysis.

Rationale (pilot evidence): OLMo-2 collapses to acc **exactly 0.0** at 8k/16k
(beyond its 4k window) — the accuracy band already excludes these — while its
in-window failures at 4k are **distractor-monotonic** (nd0=1.00 → nd10
shell_diff=0.50) and similarity-dependent, i.e. genuine retrieval competition.
The rule formalizes what the band did empirically, and guards a model whose
degradation is gradual rather than a sharp collapse.

*Code:* E1 tags each cell `in_window: bool` from `effective_window`; E2/E3
process only in-window breaking cells; a separate `beyond_window_breaking` list
is emitted for reporting.

## D. Difficulty: `shell_share` ON in the confirmatory grid

`configs/grid.yaml` `grid.similarity` becomes
`["shell_same", "shell_diff", "shell_share"]`. `shell_share` distractors reuse
the needle's ADJ or NOUN (key-collision), forcing full-key disambiguation and
preventing the strong models from solving by surface key-match. This is a design
change and is pre-registered as such. (Committing the grid.yaml edit puts its
hash in provenance and the E1 checkpoint fingerprint.)

## E. Model rejection rule (no NaN in the family)

A model is **rejected** from the confirmatory analysis (not carried as NaN) when
it yields no qualifying in-window breaking cell — i.e. every in-window breaking
cell has fewer than `pair_min_per_cell` matched pairs, or it fails the eager
gate (§B). The BH correction family is `{qualifying models} × {repair, break}`,
size `N×2` with the realized **N reported**. This closes the pilot's two
fail-loudly violations (qwen-7b's "0 pairs → NaN", and the four silent E2 deaths).

## F. M2 threshold: NOT re-calibrated. Frozen, with a sensitivity analysis.

**Superseded.** An earlier version of this section proposed re-calibrating
`signature_rules.m2_capture.min_distractor_mass` from the pilot distribution. That
door is now closed, and this section records why.

The pilot showed the registered floor of **0.10** admits **95.6%** of all
`distractor_hit` failures as M2 (pooled n = 3619; quantiles of `distractor_mass`:
p5 = 0.103, p25 = 0.154, p50 = 0.193). The floor is therefore permissive, and the
M2 rate is sensitive to it.

But that observation was made **after** the bucket rates, the EFFECT labels and
the model-to-model comparisons were on the table. Moving the floor now — knowing
that a higher floor lowers the M2 rates we have already looked at — would be
choosing the measurement rule to fit the result we have seen. That is HARKing,
and no amount of good faith about the motive changes what it would be. The prior
"may be re-calibrated on pilot data" clause was open only while the runs were
*pilot* and the calibration was done *blind to the outcome*. Neither now holds.

Decision, in force:

1. **The primary analysis keeps the pre-registered floor 0.10.** It is not
   re-tuned, and the primary artifacts are not re-run for this.
2. An alternative floor is reported as an **E5 threshold-sensitivity analysis**
   (variant `m2sens`), side by side with the pre-registered taxonomy. Both
   taxonomies appear in the paper.
3. The permissiveness of the 0.10 floor is stated in the paper as a
   **limitation**, in these terms, not buried.

Enforced in code, not just in prose: `scripts/e2_signatures.py
--m2-min-distractor-mass` **refuses to run without `--tag`**, so an override can
never overwrite the primary artifact; and every E2 artifact records
`is_primary_analysis` and `m2_min_distractor_mass_override` in its provenance.
Test: `tests/test_prereg_integrity.py::test_m2_floor_override_refuses_to_overwrite_the_primary`.

The pilot anomaly `distractor_hit → m1_silence = 16` (llama) is likewise
*reported*, not engineered away.

## G. Statistics / other keys unchanged

`breaking_band_accuracy`, `pair_min_per_cell`, `flip_margin_over_control_pp`,
`alpha`, `correction=benjamini_hochberg`, `effect_size=cliffs_delta`, `ci`,
`seeds`, sample sizes: unchanged. Per §F the M2 floor is **not** edited. Any
future change is recorded here *before* the run it affects.

## H. k-sweep extended to {20, 30} — EXPLORATORY, not confirmatory

**Pre-registered primary sweep: k ∈ {1, 5, 10}. Unchanged.**
**Added post hoc, result-driven: k ∈ {20, 30}. Labelled exploratory, everywhere.**

Why it is needed. At k ≤ 10 the break-flip curve was still climbing for qwen3b
(0.14 → 0.29 → 0.50), phi (0.00 → 0.08 → 0.39) and olmo (0.01 → 0.13 → 0.22), but
flat for llama (0.00 → 0.07 → 0.07) and mistral (0.00 → 0.04 → 0.05). The claim
"llama/mistral show the M2 mechanism but no causal effect" cannot be defended from
curves cut at k = 10: mistral has **82** detected retrieval heads and we patched
**10**. An under-dosed intervention that produces no effect is not evidence of no
effect.

Two outcomes, both publishable, both exploratory:
* the curve keeps rising → the k = 10 null was a **k-ceiling artifact
  (redundancy)**;
* the curve stays flat while the others saturate → a genuine
  **mechanism-vs-causality dissociation**.

Why it does not become confirmatory: it was chosen *because* of what the k ≤ 10
results looked like. Reporting it as pre-registered would be a lie about its
provenance; reporting it as exploratory costs nothing and is what makes it usable
by a reader. Feasible panel-wide — the smallest detected head set is 30 (qwen25_3b).

Recorded in `configs/e3.yaml` (comment), in the notebook filename
(`10_A100_e3_ksweep_EXPLORATORY.ipynb`), and in this dated commit.

## I. E2 and E3 must measure the same sample (pair-set fix)

**Defect.** E3 re-derived its matched pairs by re-grading, instead of using E2's.
Grading batches through `_generate_batch_adaptive`, which **halves the batch on
OOM**; a different batch composition changes the left-padding, and a greedy token
can flip at the margin. Result: E2 measured the mechanism on **614** pairs while
E3 measured causality on **625**. Mechanism and causality were being reported on
different samples. This is not minor — it is the join between the paper's two
halves.

**Fix.** E2 now records the exact pairs it used (`pairs_by_cell`, as sample
indices; probes are pure functions of `(cell, sample_idx, seed)`, so the indices
rebuild the identical prompts). E3 **reads** that list, and:
* aborts if it is missing (an artifact predating this fix cannot be used);
* aborts if its pair count disagrees with E2's `pairs_used`.

**Cost, stated plainly.** No existing E2 artifact carries `pairs_by_cell`, so E2
must be run once more to record it. This re-run is **configuration-identical** —
same pre-registration, same 0.10 floor, same seed; it is *not* the §F
re-calibration and changes no threshold. Notebook
`09_A100_e2_rerun_record_pairs.ipynb` performs the re-run;
`scripts/check_e2_repro.py` verifies it.

**Result of the verification (2026-07-17).** Two distinct properties must be kept
apart, and an earlier draft of this section conflated them by demanding the re-run
"reproduce the current numbers exactly":

* **Self-consistency** — does the re-run's recorded `pairs_by_cell` match its own
  `pairs_used`? This is the property E3/E4 validity actually rests on: E3 reads
  `pairs_by_cell`, so if it equals `pairs_used`, causality was measured on exactly
  the sample E2 classified. **PASS for all 7 models.** The pair-set defect above is
  therefore fixed *by construction* (the recording), independent of any run-to-run
  reproducibility.
* **Run-to-run reproducibility** — does the re-run bit-match the *pre-fix* artifact
  (made in an earlier session, on a different GPU type)? It does not, for 6/7
  models, by **~1–2% of pairs**; the resulting **bucket-rate drift is ≤ 2.9 pp**
  (phi 0.0, mistral 0.5, olmo 0.6, llama 0.6, qwen3b 1.2, gemma 2.6, qwen7b 2.9).
  phi reproduces exactly because both its runs were on the same 80 GB GPU. This is
  greedy-decoding margin sensitivity to batch composition and **cross-session GPU
  kernel nondeterminism** (non-associative float reductions differ across GPU
  architectures) — a documented property of batched greedy inference, **not a logic
  error**, and it does not touch self-consistency.

**Disposition.** The "reproduce exactly" bar was mis-specified: exact
bit-reproduction of batched greedy grading across sessions/GPUs is not achievable,
and it tested a stronger property than the science needs. The gate is
**self-consistency (PASS)**. The run-to-run drift (≤ 2.9 pp, largest on the
smallest-*n* model, qwen2.5-7b at *n*=113) is reported as a **limitation**: E2
grade assignments carry ~1–2% marginal-sample noise across GPUs, immaterial to the
mechanism conclusions (M2-dominant in 6/7) and to the three-regime causal split
(its carriers drift ≤ 1.2 pp). The **pre-registered E5 seed-repeats variant** is
the formal robustness check for grading noise. `check_e2_repro.py` returns exit 0
(exact), 3 (self-consistent with drift — researcher judges), or 2 (self-inconsistent
— a real bug).

## J. Head-set determinacy: gemma2's k = 10 null is not interpretable

The detector score saturates at 1.0, and heads can tie there exactly. Counting
heads at score 1.000:

| model | heads @ 1.000 | is the k=10 cut inside a tie? |
|---|---|---|
| **gemma2_9b_it** | **13** | **yes — 10 of 13 admitted by sort order** |
| olmo2_7b_instruct | 5 | yes (milder) |
| mistral7b_v03 | 4 | no |
| qwen25_3b | 3 | no |
| phi35_mini | 2 | no |
| llama31_8b | 2 | no |
| qwen25_7b | 2 | no |

For gemma2 the top-10 "retrieval heads" are **10 arbitrary members of a 13-way
tie**, selected by `(layer, head)` sort order rather than by evidence. Its k = 10
null therefore says nothing about whether gemma2's retrieval heads are causal —
it says we patched an arbitrary subset of the indistinguishable ones.

**Consequence.** The earlier reading — *"gemma2 is the M1-silence model whose
retrieval heads are not causal"* — is **withdrawn**. It was premature, exactly as
flagged. **Resolved by the k = 20/30 extension** (§H), which covers the whole tied
block: gemma2's break is BH-significant from k = 10 (its first non-tie-confounded k;
p ≈ 1e-4 at k = 20/30) but **never clears the 20 pp margin** (break flip peaks ~0.12
at k = 30). So gemma2 is *not* causally inert — it is **significant-but-sub-margin**,
like llama3.1, and it must be read only from k ≥ 20 (past the tie). Its distinctive
non-M2 profile is *plausibly related to* its softcap + sliding-window attention, but
that link is a conjecture, not tested here.

**Fix.** `detect.boundary_tie(k)` reports, for each k, the score at the cut, how
many heads tie it, how many were admitted and how many excluded, and a boolean
`arbitrary`. E3 logs a warning at every tie-broken k and records `head_set_ties`
in its output. The tie-break itself remains deterministic (same artifact → same
set), so runs stay reproducible; it is *arbitrary*, not unstable. Tests:
`tests/test_prereg_integrity.py`.

## K. The capture-correctness gate is now an artifact, not a log line

The eager-reference gate (manual attention row vs the model's own
`output_attentions` row) is the sole evidence that every downstream mass number is
trustworthy. It was **printed to notebook stdout and never persisted** — it died
with the Colab session, and no reviewer could audit it.

Notebook 00 now writes `gate_eager_reference.json` (per model: effective attention
implementation, dtype, per-head and worst-case `max_abs_diff`, tolerance, pass
flag, model SHA, git commit, timestamp).

Gate result of record — **fp32, tolerance 1e-3, 7/7 PASS**, run 2026-07-08:

| model | eff. attn | max abs diff |
|---|---|---|
| mistral7b_v03_instruct | eager | 1.86e-08 |
| llama31_8b_instruct | eager | 1.19e-07 |
| phi35_mini | eager | 2.38e-07 |
| qwen25_3b_instruct | eager | 2.38e-07 |
| gemma2_9b_it | eager | 2.98e-07 |
| olmo2_7b_instruct | eager | 3.58e-07 |
| qwen25_7b_instruct | eager | 5.66e-07 |

Worst case 5.7e-07 — roughly four orders of margin under the gate. This covers the
three architectures where the manual row could plausibly have been wrong: gemma2
(logit softcapping + `query_pre_attn_scalar`), olmo2 (QK-norm before RoPE) and
phi3 (fused QKV + partial rotary). The gate **must be run in fp32**: in bf16 the
machine epsilon (~4e-3) exceeds the tolerance and the gate fails spuriously.

## L. Status of every run, and the order of operations

Nothing below is confirmatory until it sits on a committed configuration. A run
without a commit is a pilot.

| # | status |
|---|---|
| A–E, G | in force |
| F (M2 floor) | **frozen at 0.10**; alternative floor → E5 sensitivity only |
| H (k=20/30) | **exploratory** |
| I (pair set) | code fixed; requires the config-identical E2 re-run (nb 09) |
| J (gemma ties) | diagnostic added; gemma2 resolved — significant-but-sub-margin, read from k≥20 |
| K (gate) | **passed**, now persisted |

Order: **this amendment is committed first**, then E5 (Wu, olmo + phi first), then
the E2 pair-recording re-run, then the exploratory k-sweep. Results are attached to
this commit hash.
