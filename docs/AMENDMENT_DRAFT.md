# Pre-registration amendment — DRAFT (for researcher audit)

> **Status: DRAFT written by the assistant at the researcher's request; the
> researcher audits, edits, and owns it.** This document does NOT modify
> `configs/preregistration.yaml`; it specifies the changes for the researcher to
> incorporate. Dated 2026-07-05. Motivated by the pilot run (`rfm_results/`).

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

## F. M2 threshold calibration (record)

`signature_rules.m2_capture.min_distractor_mass` is re-calibrated from the pilot
distribution, now that E2 emits `failure_sample_masses` (per failure sample:
sample-level needle/distractor mass, bucket, behavioral label). Procedure:
1. Pool the `distractor_hit`-behavioral failure samples across pilot models.
2. Inspect their `distractor_mass` distribution (histogram / percentiles).
3. Set `min_distractor_mass` so that behaviorally-distractor_hit failures with a
   genuinely dominant distractor are captured as M2, while ambient mass is not —
   e.g. at the distribution's lower mode, **value chosen by the researcher from
   the data, not defaulted here.**
4. Record the chosen value and the distribution it was read from in the
   pre-registration comment (audit trail).

Specifically resolves the pilot anomaly `distractor_hit → m1_silence = 16`
(llama): the amendment requires checking whether those samples' `distractor_mass`
sit just under the current 0.10 floor (0.05–0.10 band); if so, lower the floor.

## G. Statistics / other keys unchanged

`breaking_band_accuracy`, `pair_min_per_cell`, `flip_margin_over_control_pp`,
`alpha`, `correction=benjamini_hochberg`, `effect_size=cliffs_delta`, `ci`,
`seeds`, sample sizes: unchanged from the current pre-registration unless the
§F calibration or §E rejection rule requires an edit. Any change is recorded here
before the confirmatory run.
