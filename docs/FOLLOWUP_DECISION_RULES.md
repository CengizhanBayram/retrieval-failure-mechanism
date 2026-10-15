# Follow-up experiments A–F: pre-specified decision rules

Fixed on 2026-10-15, before any of these runs. The commit that adds this file is the record of the date. Every rule below is implemented in `scripts/followup_analysis.py`, which reads only the run outputs and applies the rules as written. Changing a threshold after seeing results means changing this file in a later, dated commit, and the manuscript must report the rule that was used.

All six experiments are **exploratory follow-ups** to the merged manuscript. None of them replaces the pre-specified E1–E5 analysis, with one exception: experiment E corrects a measurement error (see E). Experiment F was added the same day, after a review of the manuscript and still before any run.

Shared conventions:

- **Effect.** Flip rate of the treated heads minus flip rate of an equal number of random non-retrieval heads, computed on the same pairs. A *break* is a success that becomes not-correct. A *repair* is a failure that becomes correct. Break is primary, as in E3.
- **Statistics.**
  - 95% percentile bootstrap intervals use 10,000 resamples.
  - p-values come from a two-sided paired sign-flip permutation test with 10,000 permutations and the +1 correction.
  - Benjamini–Hochberg correction is applied within each model and experiment, over the comparisons named in the rule.
  - Seeds are fixed in the scripts and recorded in every output.
- **Margin.** 0.20 (20 percentage points), the same margin as the E3 decision. The equivalence bound for "preserved" is ±0.10 (half the margin).
- **Pairs.** Pairs are always E2's recorded success–failure pairs (`pairs_by_cell`). When a budget is set, pairs are subsampled round-robin over cells (`_common.select_pairs_round_robin`). The selection is deterministic and is recorded in the output.

---

## E — Gemma-2 capture correction (do this first)

**What went wrong.** `capture._row_for_head` indexed the attention row by cache position. Once a prompt is longer than Gemma-2's 4,096-token local window, HF's `HybridCache` keeps only the last 4,096 keys of each local layer. As a result:

- In the 4k cells, the rows of the two local-layer heads among the top five (L18_H3, L24_H0) were shifted by T − 4,096 tokens.
- In the 8k cells with a late needle, those rows were entirely NaN and their needle mass came out as 0.

219 of the 625 Gemma-2 failures (35%) had a NaN sample-level distractor mass and could not be classified as capture. The bug was fixed on 2026-10-15. The fix is covered by regression tests that compare the capture with the model's eager rows beyond the window. No other model has a window shorter than its probes, so no other model is affected. E3 patching does not read attention rows and is unaffected.

**Runs.**

1. `check_capture_long.py --model gemma2_9b_it` must pass (|Δmass| ≤ 0.02 on real 8k probes, all five heads).
2. `e2_signatures.py --model gemma2_9b_it --tag capfix --save-grades`.
3. `e2_signatures.py --model gemma2_9b_it --detector copy --tag wu_capfix`.

**Rules.**

- **E-1.** The corrected values *replace* the old ones everywhere in the manuscript. This covers:
  - the capture, silence, correct-attend and residual shares;
  - the per-head table and Cliff's δ;
  - the Wu-detector shares;
  - the capture-share versus break-effect correlations (Sections 6 and 7);
  - Figures `buckets` and `synthesis`.

  An appendix note states the error, which cells and heads it affected, and the old values.
- **E-2.** Pair-set identity. If the corrected run's `pairs_by_cell` equals the old one, E3's Gemma-2 results stand unchanged. If it differs, the share of identical pairs is reported, and E3's Gemma-2 numbers are labelled as measured on the earlier pair set.

---

## F — Specificity of the signature and task transfer of the lists (`e9_answer_rows.py`)

Experiment F was added on 2026-10-15 during the manuscript review, before any run. It is cheap: one prompt forward per probe, with no generation. It answers two reviewer-level questions about Part II.

**F-1 — Is "capture" specific to failures?** E2 applies the capture predicate to failing rows only:

- strongest distractor ≥ 0.10;
- distractor ≥ needle + 0.05.

F-1 applies the same predicate, at the sample level, to the five top argmax heads (window-limited heads excluded, as in E2). It does this for both members of up to 300 recorded pairs per model.

| Quantity | Definition |
|---|---|
| r_f | capture rate among failures |
| r_s | capture rate among the paired successes |
| D_F1 | paired difference r_f − r_s, with a 95% bootstrap CI over pairs |
| p | exact McNemar test on the discordant pairs |

BH correction runs across the seven models.

| Verdict | Condition |
|---|---|
| **failure-specific** | D_F1 ≥ 0.20, CI lower bound > 0, q < 0.05 |
| **not specific** | CI upper bound < 0.10 |
| **partially specific** | otherwise |

F-1 also reports the paired change in distractor mass and in needle mass (failure minus success). The verdict decides the wording of Section 6:
- **failure-specific**: capture may be called a failure signature.
- **otherwise**: the text calls it the attention pattern of the inspected heads at failure, and the failure-specific component is the paired needle-mass drop (Cliff's δ).

**F-2 — Do the Part I lists transfer to the Part II task?** The argmax detector is re-run on the Part II task. Each head's score is the fraction of probes whose answer-step argmax falls on the needle value. The probes come from held-out cells of each model:
- distractors present;
- accuracy ≥ 0.9;
- not breaking;
- up to 200 probes;
- disjoint from the E2/E3 pairs.

Quantities: Jaccard between the Part I set and the task set (τ = 0.1); overlap of the two top-10 rankings; Spearman correlation of the two score matrices.

| Verdict | Condition |
|---|---|
| **list transfers** | top-10 overlap ≥ 7 and Spearman ≥ 0.5 |
| **does not transfer** | top-10 overlap ≤ 3 |
| **partial** | otherwise |

F-2 also reports how many of the Part I top-5 heads have a task score below 0.1 (inactive on the task).

**F-3 — Patching the task list (conditional).** E3 is run with the task list on the same pairs:

`e3 --heads-from <model>_taskII --allow-other-heads --tag taskheads --k-list 5 10 20 --max-pairs 300`

with `RFM_EXTRA_DETECTION_DIRS=<results>/detection_taskII`.

It is run for:
- `llama31_8b_instruct` in every case, because it carries the Section 7 dissociation;
- every model whose F-2 verdict is not "list transfers".

| Outcome | Condition |
|---|---|
| **the null was a list-transfer effect** | the task list reaches a break effect ≥ 0.20 (q < 0.05) at some budget where the Part I list did not |
| **the dissociation holds with task-specific heads** | the task list also stays below 0.20 at k = 20 |

---

## A — Matched ablation with scope decomposition (`e6_matched_ablation.py`)

**Question.** Section 7 explains Llama-3.1's dissociation (knockout necessity 0.76, patching break effect 8.3 points) by the intervention *scope*. The knockout removes the heads at every position. Answer-step patching leaves their earlier contributions in place. A tests this directly. It zero-ablates the same head set on the same E2 probes, in bf16 and with E3's backend, under four scopes: `all` (the knockout operation), `context` (prompt positions 0..T−2), `answer` (the last prompt position, E3's scope) and `ansdec` (the last prompt position plus every decoding step).

**Models.** `llama31_8b_instruct` (primary), `qwen25_3b_instruct`, `mistral_7b_instruct` and `gemma2_9b_it`. These are one model from each regime: none, concentrated and distributed.

**Design.**

- 120 pairs per model.
- Head set: the complete detected argmax set (the knockout list).
- Control: one random draw per pair, reused across the four scopes.

**Quantities.**

- E_s is the break effect under scope s.
- x_s is the per-pair treat-minus-control break indicator.
- D = mean(x_context − x_answer).

BH correction runs over the four scopes.

**Rules.**

| Rule | Condition | Verdict |
|---|---|---|
| A-1 (knockout transfers to the Part II probes) | E_all ≥ 0.20 and q < 0.05 | **replicated** |
| | upper 95% bound of E_all < 0.20 | **not replicated** |
| | otherwise | inconclusive |
| A-2 (localisation) | D ≥ 0.20 and 95% CI of D > 0 | **context-carried** |
| | D ≤ −0.20 and CI < 0 | **answer-carried** |
| | otherwise | not localised |
| A-3 (Section 7's scope explanation, Llama-3.1 only) | A-1 replicated and A-2 context-carried | **supported** |
| | A-1 not replicated, or A-2 answer-carried | **contradicted** |
| | otherwise | inconclusive |

If A-3 is contradicted, Section 7's last paragraph ("the intervention scopes differ, but that difference is exactly what the dissociation consists of") must be rewritten.

**A-4 (descriptive).** E_answer (zero ablation at the answer step) is reported next to E3's break effect for the complete set, for every model.

**A-5 (descriptive).** For each scope, the grades of the treated break flips: distractor hit, other wrong or empty. This separates two cases:
- ablation turns retrieval into confusion: the model outputs a distractor value;
- ablation turns retrieval into omission: the model outputs no code, or an unrelated one.

---

## B — Part II under NF4 (`e7_quant_cells.py` → E2 → E3)

**Question.** Part I shows that retrieval-head identity survives NF4. B asks whether the failure *mechanism* and the *causal profile* of the same heads survive it too. It reruns E2 and E3 on the NF4 model. The cells are the bf16 parent's breaking cells, the probes are byte-identical (same tokenizer revision, parent's probe key), and the head list is the parent's.

**Models.**

| Parent | NF4 model |
|---|---|
| `qwen25_7b_instruct` | `qwen25_7b_instruct_bnb4` |
| `llama31_8b_instruct` | `llama31_8b_instruct_bnb4` |
| `gemma2_9b_it` | `gemma2_9b_it_bnb4` (only after E; compared with the *corrected* parent E2) |

**Configuration.**

- NF4 uses double quantization with **bf16 compute**, so the weights are the only difference from the bf16 reference. Part I used fp16 compute; the manuscript states this.
- Gates:
  - `e7` row gate: |manual − eager| ≤ 0.02 on a real prompt prefix.
  - `check_capture_long` for Gemma-2.
  - E3's self-patch gate and never-op gate.

**Runs (primary).**

- `e2 --heads-from <parent> --tag parentheads --save-grades`
- `e3 --heads-from <parent> --e2-tag parentheads --tag parentheads --k-list 10 20 --max-pairs 250`

The NF4 model's own detected list (`--tag ownheads`) is a secondary, optional check.

**Rules.**

| Rule | Measure | Verdict |
|---|---|---|
| B-1 (behaviour) | Pooled accuracy over the parent's breaking cells, NF4 (E2 `grades_by_cell`) minus bf16 (E1, same probes). 95% cluster bootstrap over cells. | **changed** if the CI excludes 0; descriptive otherwise |
| B-2 (signature) | Δ = NF4 − bf16 sample-level capture share among failures; cluster bootstrap over cells. | **preserved** if the CI ⊂ [−0.10, 0.10]; **changed** if the CI excludes 0 and \|Δ\| ≥ 0.10; inconclusive otherwise |
| B-3 (causal; k = 10 primary, k = 20 secondary, each direction) | Δ = NF4 effect − bf16 effect. The bf16 E3 runs stored rates and intervals but not per-pair indicators, so the 95% CI uses a normal approximation: each effect's SE is taken from its two stored rate intervals, treating treat and control as independent. This is conservative, because it ignores their positive within-pair correlation, and the same formula is applied to both precisions. | Same three-way rule as B-2 |

B-2 also reports the dominant bucket of each model. B-3 also reports the binary E3 decision (margin 0.20 and BH) under both precisions.

**B overall, per model.**

- **Mechanism preserved under NF4**: B-2 preserved, and B-3 preserved at k = 10 in both directions.
- **Changed**: any of these comparisons changed.
- **Inconclusive**: otherwise.

The manuscript reports the verdict next to Part I's identity verdict for the same ring.

---

## C — Attention-pattern transplant (`e8_pattern_transplant.py`)

**Question.** In E3 a transplanted head output carries both *where* the donor attended and *what* it read. C separates the two at the answer step, on the same pairs, with three variants:

- `activation` = the donor's z (E3);
- `pattern` = the donor's attention row applied to the recipient's values;
- `content` = the recipient's row applied to the donor's values.

**Models.**

- Primary: `qwen25_3b_instruct` (concentrated, five heads suffice), `olmo2_7b_instruct` (k = 10 break effect of 20.5 points) and `llama31_8b_instruct` (no patching effect).
- Optional: `phi35_mini`, which has the largest k = 10 effect but needs an 80 GB A100 because of its eager attention at 12k tokens.
- Gemma-2 is excluded, because its 4,096-token window truncates the caches.

**Design.**

- 120 pairs.
- Budgets: k = 10 and the complete detected set.
- Gate: the self-transplant must leave ≥ 90% of generations identical.

BH correction runs over variants × budgets × directions within each model. The primary comparison is break at the complete set.

**Rules.**

| Rule | Condition | Verdict |
|---|---|---|
| C-1 (the attention pattern alone is causally sufficient at the answer step) | E_pattern ≥ 0.20 and q < 0.05 | **sufficient** |
| | upper bound < 0.20 | **not sufficient** |
| | otherwise | inconclusive |
| C-2 (decomposition; only when E_activation ≥ 0.10) | E_pattern ≥ 0.5·E_activation and E_content < 0.5·E_activation | **pattern-carried** |
| | the reverse | **content-carried** |
| | both ≥ 0.5·E_activation | **both** |
| | neither | **neither** |

**C-3 (descriptive).** Among pattern-transplant break flips, the share whose output is one of the *recipient's own* distractor values. This is direct evidence that redirecting attention to a distractor produces capture behaviour.

The limitation paragraph on content transport (Section 9, "Patching design") is replaced by the measured C-2 decomposition.

---

## G — Layer-matched control (optional; `e3_causal.py --control-mode layer`)

**Question.** E3's control draws non-retrieval heads from any layer. Retrieval heads concentrate in the middle and upper layers, so a large effect against that control could belong to *the layers* rather than to *the listed heads*. G repeats E3 at k = 5 and 10 with a stricter control: for every treated head, one non-retrieval head from the same layer (the nearest layer when that layer has no free head).

**Models.** The three models with a pre-specified-budget break effect: `qwen25_3b_instruct`, `olmo2_7b_instruct` and `phi35_mini`, on the same hardware as their original E3 runs (A100 40 GB). G uses 300 pairs, round-robin over cells, with `--tag layerctrl`.

**Rule.** Let E_L be the effect against the layer-matched control (BH over models × budgets × directions).

| Verdict | Condition |
|---|---|
| **the listed heads carry the effect** | E_L ≥ 0.20 and q < 0.05 at the budget where the random-control effect met the decision |
| **a layer effect** | E_L < 0.10 there |
| **partly layer** | otherwise |

---

## D — Haystack source of the stored head sets (`scripts/rhp_haystack_check.py`, run against the retrieval-head-profile clone)

**Question.** The stored profiles do not record whether the detector's haystack was PG-19 or the 50-sentence fallback pool. D answers this and measures how much head identity depends on the haystack.

**Runs.**

- **D-fallback**: `qwen25_7b_instruct` and `llama31_8b_instruct`, with the fallback pool forced.
- **D-wikitext**: WikiText-103 (5,000 sentences, the loader's sentence filter) on the five core models plus the seven Part II models.

Detection settings are identical to the profile run: 4,096 tokens, 200 argmax probes plus 100 copy probes, seed 42 and the panel's loading.

**Rules.**

| Rule | Condition | Verdict |
|---|---|---|
| D-1 (which pool the stored sets came from) | For both models, argmax-score Spearman ρ ≥ 0.99 and max \|Δscore\| ≤ 0.03 against the stored matrix (identical probes reproduce the scores up to GPU noise; seed-to-seed resampling gives ρ = 0.93–0.95) | **fallback pool** |
| | For both models, ρ ≤ 0.97 | **not the fallback pool** |
| | otherwise | inconclusive |
| D-2 (haystack dependence, per model and detector) | r = J(stored, WikiText) / R_self (R_self as in the lineage rules: measured for the core models, panel median otherwise); r ≥ 0.8 | **haystack-robust** |
| | r < 0.5 | **haystack-dependent** |
| | otherwise | partial |

D-2 also reports the overlap of the ten highest-ranked argmax heads, which is what Part II uses.

The haystack limitation paragraph (Section 9) and the detection description (Section 3.2) are rewritten with the measured D-1 and D-2 results.
