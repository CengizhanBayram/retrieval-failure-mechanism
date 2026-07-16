# retrieval-failure-mechanism (Part 3)

When a model **fails** at long-context retrieval, what is the mechanistic
signature at the retrieval-head level — **silence (M1)**, **distractor-capture
(M2)**, **correct-attend-but-downstream-loss**, or **residual attenuation** — and
is that signature **causal** (verified by bidirectional activation patching)?

This repo is an inference-only experiment suite built **on top of** two prior
repos, which it imports and never edits:

- **Part 1** — *Does RoPE Prevent or Degrade Retrieval Heads?* — provides
  `src/` (model loader, activation patching, statistics).
- **Part 2** — *retrieval-head-profile* (`rhp/`) — provides the **pinned model
  registry** `configs/panel.yaml` (commit SHAs + `rope_theta`) and the
  **detection artifacts** (`datas/results/profile/{model}_seed{seed}.json`,
  the ranked retrieval-head lists).

Everything measured here is gated on a **researcher-authored pre-registration**
and every reported number carries a **provenance record**. No module in this
package interprets a result — **measurement only**. What the numbers *mean* is
argued in the paper, never in code, comments, or this README.

---

## Model panel (7 instruct models, 6 families, GQA + MHA)

| task key | family | attention | θ (rope) | min GPU |
|---|---|---|---|---|
| `llama31_8b_instruct` | llama | GQA 32/8 | 500k | A100 40 GB |
| `qwen25_7b_instruct` | qwen | GQA 28/4 | 1M | A100 40 GB |
| `gemma2_9b_it` | gemma | GQA 16/8 + softcap, local/global | 10k | **A100 80 GB** |
| `mistral_7b_instruct` | mistral | GQA 32/8 | 1M | A100 40 GB |
| `olmo2_7b_instruct` | olmo | **MHA 32/32 + QK-norm** | 500k | A100 40 GB |
| `phi35_mini` | phi | **MHA 32/32, fused qkv** | 10k | **A100 80 GB** |
| `qwen25_3b_instruct` | qwen | GQA 16/2 (size axis) | 1M | A100 40 GB |

**Gemma-2 and Phi-3 need an A100 80 GB.** Both are forced to *eager* attention
(Gemma-2 for softcap correctness, Phi-3 because the pinned transformers has no
Phi-3 sdpa kernel), and eager attention materialises the full `L×L` attention
matrix inside the model's own forward — at the panel's long contexts that single
tensor is ~18 GB, which OOMs a 40 GB card. The five sdpa models fit on 40 GB. The
shipped notebooks are all **A100-tagged** because a mixed-panel run must satisfy
the largest requirement.

Every model already has a Part-2 detection artifact, so Part 3 never re-detects.
`capture.py` is family-aware: standard GQA, Gemma-2 (query-scale + softcap +
sliding window), OLMo-2 (QK-norm before RoPE), and Phi-3 (fused-qkv split +
partial rotary) — each validated by the eager-reference gate (< 1e-3, fp32).

---

## Repository layout

```
src/failure_mech/
  prereg.py      # the pre-registration gate (§1.2)
  panel.py       # pinned loading (bf16, explicit attn backend), GQA mapping
  probes.py      # procedural probe construction + token-skeleton alignment (§4.1)
  spans.py       # token-span bookkeeping + decode-verify (§4.2)
  capture.py     # manual attention-row capture (§4.3)  ← recompute-RoPE, Gemma softcap
  patching.py    # z_h donor capture + patching + self-patch no-op (§4.4)
  grading.py     # answer grading + 4-way behavioral taxonomy (§4.5)
  classify.py    # four-bucket mechanistic classifier from prereg rules (§7)
  detect.py      # detection-artifact reader, top-k ranking, head-set tie diagnostic
  stats.py       # paired permutation, BH, Cliff's delta, bootstrap, Wilson (§4.6)
  provenance.py  # the provenance stamp reused everywhere (§1.6)
configs/
  grid.yaml decoding.yaml paths.yaml e3.yaml
  preregistration.yaml            # RESEARCHER-AUTHORED (committed by the researcher)
  preregistration.example.yaml    # schema-shaped example (not used by any run)
scripts/
  e1_breaking_surface.py e2_signatures.py e3_causal.py e4_families.py e5_robustness.py
  check_e2_repro.py               # offline reproduction check for the E2 re-run (§I)
  _common.py                      # config load, checkpoints, freshness, time-guard
notebooks/
  00.._A100_* … 11_A100_*         # Colab notebooks (GPU tag in the filename)
  build_notebooks.py              # the generator — edit here, never the .ipynb
tests/    pytest guardrails
docs/     PREREGISTRATION_SCHEMA.md, AMENDMENT_2026-07-13.md
results/  gitignored except manifests
```

Result JSONs are **machine-produced and never committed** — `.gitignore` blocks
the `e1_/e2_/e3_/e4_/e5_` artifacts and `*.pre_pairfix` in any directory, so a
results folder downloaded into the repo tree cannot leak into git.

---

## Setup

```bash
pip install -r requirements.txt          # install a CUDA-matched torch wheel first (see file)
# tell Part 3 where the two prior repos live (or rely on sibling autodiscovery):
export RHP_PART1_REPO=/path/to/Does-RoPE-...-Model-Families
export RHP_PART2_REPO=/path/to/retrieval-head-profile
# optional: point the scripts at a results directory (Colab uses Drive)
export RFM_RESULTS_DIR=/path/to/rfm_results
```

The GPU experiments run on **Google Colab** (A100) via the notebooks; a CPU-only
machine runs the test suite, the `--dry-run` planning, and `check_e2_repro.py`.

### The pre-registration gate (required)

`configs/preregistration.yaml` is the scientific pre-registration the causal
claim rests on. It is **researcher-authored** and committed by the researcher, per
**[docs/PREREGISTRATION_SCHEMA.md](docs/PREREGISTRATION_SCHEMA.md)**. This codebase
**never creates, defaults, or edits it** — if it is missing or a key is absent,
every script fails loudly naming the exact missing key:

```
PreregError: Pre-registration file not found: configs/preregistration.yaml ...
```

Thresholds live only here (the four-bucket predicates, `pair_min`, the breaking
band, the M2 `min_distractor_mass` floor, `flip_margin_over_control_pp`, `alpha`,
seeds). No threshold is hard-coded in any script.

---

## Reproduction (exact commands)

Order matters: **E2 needs E1, E3 needs E2** (E3 reads the exact matched pairs E2
recorded), **E4/E5 need E2/E3**.

```bash
# E1 — breaking surface (per model). Two-stage, resumable, Wilson CIs.
python scripts/e1_breaking_surface.py --model llama31_8b_instruct --stage auto
python scripts/e1_breaking_surface.py --model llama31_8b_instruct --dry-run   # plan + schema, no GPU

# E2 — signatures (matched pairs, capture, four-bucket classify, cross-tab).
#      Records pairs_by_cell so E3 measures causality on the SAME sample.
python scripts/e2_signatures.py --model llama31_8b_instruct

# E3 — bidirectional causal patching (repair/break/random/no-patch/self-patch).
#      Checkpointed per k; --max-hours stops cleanly between k values.
python scripts/e3_causal.py --model llama31_8b_instruct --max-hours 20
python scripts/e3_causal.py --model llama31_8b_instruct --fresh        # ignore checkpoints

# E4 — cross-family table + authoritative BH across {N models × 2 directions}
python scripts/e4_families.py --models llama31_8b_instruct gemma2_9b_it \
    mistral_7b_instruct qwen25_7b_instruct olmo2_7b_instruct phi35_mini qwen25_3b_instruct

# E5 — robustness. Each variant is a full E2 re-run written to a TAGGED artifact;
#      the pre-registered primary is never overwritten. Pick variants to stay tractable.
python scripts/e5_robustness.py --model olmo2_7b_instruct --variants wu
python scripts/e5_robustness.py --model olmo2_7b_instruct --variants wu,m2sens --m2-alt-floor 0.15

# Reproduction check for the E2 re-run (pure JSON, no GPU): compares each
# e2_signatures_{model}.json against its .pre_pairfix backup.
python scripts/check_e2_repro.py
```

### On Colab

Notebooks carry the required runtime in the filename (all **A100**). Base panel,
in order: `00_A100_setup_and_guardrails` → `01_A100_e1_breaking_surface` →
`02_A100_e2_signatures` → `03_A100_e3_causal` → `04_A100_e4_e5_analysis` →
`05_A100_analyze_breaking_models` (re-runs E2/E3/E4 restricted to the breaking
cells).

The later notebooks implement the amendment workflow and supersede the base E1/E2/E3
for the confirmatory run:

| notebook | purpose |
|---|---|
| `06_A100_shell_share_e1_e2` | E1+E2 with `shell_share` enabled (the confirmatory difficulty) |
| `07_A100_e3_causal_e4_families` | E3 + E4 on the shell_share panel |
| `08_A100_e3_backfill_e4` | E3 backfill for models a session couldn't finish |
| `09_A100_e2_rerun_record_pairs` | E2 re-run that records `pairs_by_cell` (config-identical; prints a reproduction check) |
| `10_A100_e3_ksweep_EXPLORATORY` | extended k-sweep (k = 1,5,10,**20,30**) — **exploratory**, see §H below |
| `11_A100_e5_robustness` | E5 variants (wu / ksens / steps3 / seeds / m2sens) |

Each notebook is **run-all** capable and **resume-safe to Drive**: a model is
skipped only when its artifact is *current* (its recorded config hashes still
match the configs on disk) — not merely because an output file exists. Changing
`grid.yaml` or `e3.yaml` therefore invalidates exactly the artifacts it should,
and a resumed run recomputes only those. An adaptive **23 h guard** stops before a
model that cannot finish under Colab's 24 h limit; a killed session is resumed by
re-running the notebook.

### Expected runtime (order of magnitude)

Assumptions: single A100, bf16, the pinned stack, `shell_share` grid, and the
pre-registered sample sizes. These are **nominal** — trust the guard, not these
figures.

| Experiment | Per model (nominal) | Notes |
|---|---|---|
| E1 | ~2–6 h | dominated by 16k-context generation; token-budgeted batching |
| E2 | ~2–5 h | 2 forwards/sample over breaking-cell pairs |
| E3 (k = 1,5,10) | ~4–8 h | 5 patch runs × 3 k per pair |
| E3 (k = 1,5,10,20,30) | ~8–16 h | the exploratory sweep; **checkpointed per k**, so it spans sessions |
| E4 | minutes | reads JSON + AutoConfig (no weights) |
| E5 | ~4 h / variant | one full E2 re-run per variant per model |

---

## Output schema (field-by-field)

Every JSON has a top-level `provenance` block (see PROVENANCE). Experiment payloads:

**`e1_surface_{model}.json` → `cells[<cell_hash>]`**

| field | meaning |
|---|---|
| `axes` | `{model_key, context_length, needle_position, n_distractors, similarity}` |
| `cell_hash` | 16-hex id (deterministic from axes) |
| `n_stage1` / `n_total` | stage-1 count / total graded samples |
| `accuracy` | fraction with `grade == correct` |
| `wilson_ci95` | `{lo, hi}` Wilson score interval on accuracy |
| `behavioral_grade_counts` | `{correct, distractor_hit, other_wrong, empty}` |
| `oom_fallback_applied` | whether the §5 context OOM fallback (16384→12288) fired |
| `effective_context_length` | context actually used (= axes context unless OOM fallback) |

**`e1_breaking_cells_{model}.json`**: `{breaking_cells: [cell_hash…], fallback_applied}`.

**`e2_signatures_{model}.json`**: `per_head` (per-head success/failure needle
means, paired-permutation p, Cliff's delta, mean-diff CI, `bucket_counts`,
`bh_rejected`), `sample_level.{bucket_counts, bucket_rates}`,
`crosstab_behavior_x_mechanism`, `reference_stats`, `failure_sample_masses`,
`pairs_used`, `cells_used`, and **`pairs_by_cell`** (`cell_hash → [[success_idx,
failure_idx]…]` — the exact matched pairs, so E3 measures causality on the sample
E2 measured the mechanism on). Provenance carries `is_primary_analysis` and
`m2_min_distractor_mass_override` so a tagged sensitivity variant can never be
mistaken for the primary.

**`e3_causal_{model}.json`**: `k[<k>]` = `repair/break/random_control/no_patch`
`{flip_rate, ci}`, `self_patch_ok`, `n_pairs`, `p_values`, `margins`,
`bh_rejected` (within-model provisional; authoritative BH in E4),
`random_control_seeds`; plus **`head_set_ties[<k>]`** (per-k head-set determinacy:
`cut_score`, `n_tied_at_cut`, `n_tied_inside_k`, `n_tied_excluded`, `arbitrary`).
Provenance carries `pairs_source` (`e2.pairs_by_cell`), `k_requested`,
`k_completed`, and **`complete`** — an interrupted k-sweep records `complete:false`
and cannot read as a finished one.

**`e4_families.json`**: `table[model]` (E2 bucket rates, E3 flip rates, Gemma
local/global head annotation) and `causal_decision[k][model][direction]`
= `{p, bh_rejected, margin_met, effect}` where `effect = bh_rejected AND margin_met`.

**`e5_robustness_{model}.json`**: `primary` (the pre-registered bucket rates,
carried alongside every variant), `headline_variants` (wu / ksens / steps3 /
m2sens), `seed_repeats.per_bucket` (mean/min/max/range), `R_self`, and `m2_floor`
(`{preregistered, alternative}`).

---

## The rules the researcher flagged (documented, not silent)

**OOM fallback (§5).** The *only* permitted fallback: if a run OOMs at context
16384 it retries at 12288, logs loudly, and sets `fallback_applied` in the
output. It is reserved for genuine context-driven OOM — E1 batching is
**token-budgeted** (`max_tokens_per_batch`), so a fixed batch never trips it at
long context. (This does not rescue Gemma-2/Phi-3 eager attention on a 40 GB card:
the full `L×L` matrix at 12288 is still ~18 GB — those two models need 80 GB.)

**Gemma-2 runs in eager, by design (§4.3, §8).** Gemma-2 is *not* plain
`softmax(qKᵀ/√d)`: it scales q by `1/√query_pre_attn_scalar`, softcaps the
attention logits (`cap·tanh(logits/cap)`) before softmax, and alternates
sliding-window (local) and global layers. Under **sdpa, transformers silently
drops softcapping** (it does *not* auto-promote to eager), which would make
Gemma-2's generation logits subtly wrong — so `panel.load_model` **forces eager
for the gemma family**. `capture.py` also reads the scale + softcap from config
and masks the sliding window; the eager-reference gate is the arbiter and passes
for Gemma-2 only when all three are applied. Phi-3 is likewise forced to eager
(no sdpa kernel in the pinned transformers).

**Gemma-2 window rule (§8).** A **local** retrieval head whose needle lies beyond
its sliding window at the answer step is marked `window_limited` and **excluded
from silence statistics** — window truncation is never counted as M1. E4
annotates each Gemma retrieval head local|global from its config.

**Recompute, don't read, post-RoPE q (§4.3).** Under sdpa you cannot read the
post-RoPE query (RoPE is applied inside the attention forward). `capture.py`
hooks `q_proj` (or the fused `qkv_proj`) for the pre-RoPE q, applies `q_norm`
when present (OLMo-2), and re-applies the model's own rotary (partial-aware for
Phi-3) with the answer step's `position_ids`; K (post-RoPE) is read from the KV
cache via the GQA map `kv = query_head // (n_query_heads // n_kv_heads)`.

**E2 and E3 measure the same sample (§I).** E3 must not re-derive its pairs by
re-grading: the grading pass halves the batch on OOM, which changes the
left-padding, and a greedy token can flip at the margin. E2 therefore **records**
its matched pairs (`pairs_by_cell`) and E3 **reads** them, aborting if the list is
missing or its count disagrees with E2's `pairs_used`. `check_e2_repro.py`
verifies the config-identical E2 re-run reproduced the pre-fix numbers.

**Head-set determinacy (§J).** The detector score saturates at 1.0, so several
heads can tie there exactly. Where the top-k cut splits a tied block, the patched
set is chosen by `(layer, head)` sort order, not by evidence, and a null causal
result at that k is not interpretable as "these heads are not causal".
`detect.boundary_tie(k)` reports the condition; E3 logs a warning at every
tie-broken k and records `head_set_ties`. The tie-break is deterministic (same
artifact → same set), so runs stay reproducible — it is *arbitrary*, not unstable.

### Design decisions

- **Chat template on by default** (`grid.yaml probe.use_chat_template`). All 7
  panel models are instruct-tuned; probes are wrapped in each model's chat
  template with a **fixed date** so templates that stamp "today" (Llama-3) stay
  deterministic. The wrapper is constant per cell, so token-skeleton alignment is
  preserved. Set false for base models.
- **0-distractor breaking cells are analyzed** in E2/E3 (M2 simply never fires;
  M1/correct-attend/residual still classify) — the cleanest silence-vs-downstream
  signal.
- **E3 flips are defined relative to the unpadded no-patch baseline**, computed
  with the same single-sequence path as the patched run, so a flip is
  apples-to-apples and independent of the batched pairing path. Self-patch is
  checked **once per model**. Donors are captured **once per probe over the union
  of every head set** it donates through and sliced per k — proven bitwise-identical
  to per-k capture by `tests/test_donor_slicing.py`.
- **The M2 floor is frozen (§F).** The pre-registered `min_distractor_mass` is the
  primary analysis; an alternative floor is reportable only as the tagged E5
  `m2sens` sensitivity variant, never as an overwrite. `e2_signatures.py
  --m2-min-distractor-mass` refuses to run without `--tag`.
- **E1 and E2/E3 tokenise identically**: every consumer uses the probe's
  canonical `input_ids`; batched (left-padded) and single-sequence greedy are
  checked token-equal by a test.

---

## PROVENANCE — the audit chain

Every results JSON embeds a `provenance` block (built once in `provenance.py`,
reused everywhere) so any number traces back to exactly what produced it:

```
provenance:
  script:        scripts/e2_signatures.py            # which script
  code:          {commit, branch, dirty, repo_root}  # exact code commit + clean/dirty
  config_hashes: {<path>: <sha256>, …}               # byte-hash of every config that fed it
  model_key / model_sha:  qwen25_7b_instruct / <pinned 40-hex SHA>   # exact weights
  seeds:         [42]                                 # the seed(s)
  timestamp_utc / platform / packages                # when + on what stack
  extra:         {detection_seed, detector, effective_attn, …}
```

**To audit a number**: read its file's `provenance`. `code.commit` +
`config_hashes` reproduce the exact code and configs; `model_sha` + `seeds`
reproduce the weights and probe instantiation; `extra.detection_seed` names the
detection artifact consumed. Re-running that script at that commit, with those
config bytes, that model SHA and that seed reproduces the number (up to documented
GPU kernel nondeterminism).

The same `config_hashes` drive **resume freshness**: `_common.artifact_is_current`
compares an artifact's recorded hashes against the configs on disk, so resume asks
"was this produced under today's configs?" rather than "does a file exist?".

**Capture-correctness gate.** Notebook 00 writes `gate_eager_reference.json`: per
model, the max absolute difference between the manual attention row and the
model's own `output_attentions` row (fp32, tolerance 1e-3), with pass flag, model
SHA, and commit. Persisted rather than printed, so the correctness claim behind
every mass number is auditable after the session ends.

**Determinism (§1.7).** python/numpy/torch are seeded per run; probe construction
is a pure function of `(cell, sample_idx, seed)`. Greedy decoding is deterministic.
The only residual nondeterminism is kernel-level GPU floating-point; it does not
change token-skeleton alignment, grading, or the pre-registered decision rules.

---

## The amendment (2026-07-13)

`docs/AMENDMENT_2026-07-13.md` is the dated record fixing the *status* of every
run. It changes **no pre-registered key**. In force:

- **§F** — the M2 `min_distractor_mass` floor stays frozen at its pre-registered
  value; an alternative is an E5 sensitivity variant only (re-tuning it after
  seeing results would be HARKing).
- **§H** — the k-sweep extension to **k ∈ {20, 30}** is **exploratory / post-hoc**
  (the pre-registered sweep is k ∈ {1, 5, 10}); it is labelled so in the config,
  the notebook filename, and the paper.
- **§I** — the E2/E3 pair-set fix (`pairs_by_cell`), requiring one
  config-identical E2 re-run that must reproduce the pre-fix numbers.
- **§J** — the head-set tie diagnostic; any causal reading at a tie-broken k is
  withheld pending the extended sweep.
- **§K** — the eager gate is persisted as an artifact.

---

## Tests

```bash
python -m pytest tests/ -q
```

Guardrails: the pre-registration red-test (missing file + each missing key), the
four-bucket total partition, BH/permutation known answers, grading taxonomy,
probe **determinism + token-skeleton alignment** + span decode-verify, the GQA
mapping, the **manual-row-vs-eager-reference** check (< 1e-3, incl. Gemma-2
softcap), the **self-patch no-op** (token-identical generation), the
**donor-slice bitwise-identity** (`test_donor_slicing.py`), and the
**pre-registration integrity** guards (`test_prereg_integrity.py`: the untagged
M2-floor override is refused; the head-set tie diagnostic; the deterministic
tie-break). Tiny-model tests skip cleanly offline; the real seven-model
eager-reference gate is the first cell block of notebook `00`.

---

## References

_To be added._
