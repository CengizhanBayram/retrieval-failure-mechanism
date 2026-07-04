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

### Model panel (7 instruct models, 6 families, GQA + MHA)

| task key | family | attention | θ (rope) | GPU |
|---|---|---|---|---|
| `llama31_8b_instruct` | llama | GQA 32/8 | 500k | L4 |
| `qwen25_7b_instruct` | qwen | GQA 28/4 | 1M | L4 |
| `gemma2_9b_it` | gemma | GQA 16/8 + softcap, local/global | 10k | A100 |
| `mistral_7b_instruct` | mistral | GQA 32/8 | 1M | L4 |
| `olmo2_7b_instruct` | olmo | **MHA 32/32 + QK-norm** | 500k | L4 |
| `phi35_mini` | phi | **MHA 32/32, fused qkv** | 10k | L4 |
| `qwen25_3b_instruct` | qwen | GQA 16/2 (size axis) | 1M | L4 |

Every model already has a Part-2 detection artifact, so Part 3 never re-detects.
`capture.py` is family-aware: standard GQA, Gemma-2 (query-scale + softcap +
sliding window), OLMo-2 (QK-norm before RoPE), and Phi-3 (fused-qkv split +
partial rotary) — each validated by the eager-reference test (< 1e-3).

Everything measured here is gated on a **researcher-authored pre-registration**
and every reported number carries a **provenance record**. No module in this
package interprets a result — measurement only.

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
  detect.py      # detection-artifact reader + canonical top-k ranking
  stats.py       # paired permutation, BH, Cliff's delta, bootstrap, Wilson (§4.6)
  provenance.py  # the provenance stamp reused everywhere (§1.6)
configs/
  grid.yaml decoding.yaml paths.yaml e3.yaml
  #  preregistration.yaml  ← RESEARCHER-AUTHORED, never in this repo by default
scripts/  e1_breaking_surface.py e2_signatures.py e3_causal.py e4_families.py e5_robustness.py
notebooks/  00..04 Colab notebooks + build_notebooks.py
tests/    pytest guardrails
docs/     PREREGISTRATION_SCHEMA.md
results/  gitignored except manifests
```

---

## Setup

```bash
pip install -r requirements.txt          # install a CUDA-matched torch wheel first (see file)
# tell Part 3 where the two prior repos live (or rely on sibling autodiscovery):
export RHP_PART1_REPO=/path/to/Does-RoPE-...-Model-Families
export RHP_PART2_REPO=/path/to/retrieval-head-profile
```

The GPU experiments run on **Google Colab** (A100/L4) via the notebooks; a
CPU-only machine runs the test suite and `--dry-run` planning.

### The pre-registration gate (required)

Author `configs/preregistration.yaml` per **[docs/PREREGISTRATION_SCHEMA.md](docs/PREREGISTRATION_SCHEMA.md)**
and commit it. Without it, every script fails loudly naming the missing key:

```
PreregError: Pre-registration file not found: configs/preregistration.yaml ...
```

This codebase will not create or default that file — it is the scientific
pre-registration the causal claim rests on.

---

## Reproduction (exact commands)

Order matters: E2 needs E1, E3 needs E2, E4/E5 need E2/E3.

```bash
# E1 — breaking surface (per model). Two-stage, resumable, Wilson CIs.
python scripts/e1_breaking_surface.py --model llama31_8b_instruct --stage auto
python scripts/e1_breaking_surface.py --model llama31_8b_instruct --dry-run   # plan + schema, no GPU

# E2 — signatures (matched pairs, capture, four-bucket classify, cross-tab)
python scripts/e2_signatures.py --model llama31_8b_instruct

# E3 — bidirectional causal patching (repair/break/random/no-patch/self-patch)
python scripts/e3_causal.py --model llama31_8b_instruct

# E4 — cross-family table + authoritative BH across {N models × 2 directions}
# (default --models is the full 7-model panel; listed explicitly here)
python scripts/e4_families.py --models llama31_8b_instruct gemma2_9b_it \
    mistral_7b_instruct qwen25_7b_instruct olmo2_7b_instruct phi35_mini qwen25_3b_instruct

# E5 — robustness (Wu head list, sensitivity k, seed repeats, answer_steps=3)
python scripts/e5_robustness.py --model llama31_8b_instruct
```

On Colab, run the notebooks in order: `00_setup_and_guardrails` →
`01_e1_breaking_surface` → `02_e2_signatures` → `03_e3_causal` →
`04_e4_e5_analysis`. Each is **run-all** capable, **resume-safe** to Drive
(finished models/cells are skipped), and guarded by an adaptive 23 h cap so a
free/Pro session finishes inside Colab's 24 h limit; a killed session is
resumed simply by re-running the notebook.

### Expected runtime (order of magnitude)

Assumptions: single 40–80 GB GPU, bf16, the pinned stack, the default grid, and
pre-registered sample sizes of a few dozen (stage-1) / ~120 (stage-2). These are
**nominal** — the notebooks' `time_guard` won't start a model that can't finish
under the cap, so trust the guard, not these figures.

| Experiment | Per model (nominal) | Notes |
|---|---|---|
| E1 | ~2–6 h | dominated by 16k-context generation; token-budgeted batching |
| E2 | ~2–5 h | 2 forwards/sample over breaking-cell pairs |
| E3 | ~4–8 h | 5 patch runs × 3 k per pair |
| E4 | minutes | reads JSON + AutoConfig (no weights) |
| E5 | ~6–10 h | several E2/E3 variants |

---

## Output schema (field-by-field)

Every JSON has a top-level `provenance` block (see below). Experiment payloads:

**`e1_surface_{model}.json` → `cells[<cell_hash>]`**

| field | meaning |
|---|---|
| `axes` | `{model_key, context_length, needle_position, n_distractors, similarity}` |
| `cell_hash` | 16-hex id (deterministic from axes) |
| `n_stage1` / `n_total` | stage-1 count / total graded samples |
| `accuracy` | fraction with `grade == correct` |
| `wilson_ci95` | `{lo, hi}` Wilson score interval on accuracy |
| `behavioral_grade_counts` | `{correct, distractor_hit, other_wrong, empty}` |
| `fallback_applied` | whether the widened breaking band was used |

**`e1_breaking_cells_{model}.json`**: `{breaking_cells: [cell_hash…], fallback_applied}`.

**`e2_signatures_{model}.json`**: `per_head` (per-head success/failure needle
means, paired-permutation p, Cliff's delta, mean-diff CI, `bucket_counts`,
`bh_rejected`), `sample_level.{bucket_counts, bucket_rates}`,
`crosstab_behavior_x_mechanism`, `reference_stats`, `pairs_used`, `cells_used`.

**`e3_causal_{model}.json` → `k[<k>]`**: `repair/break/random_control/no_patch`
`{flip_rate, ci}`, `self_patch_ok`, `n_pairs`, `p_values`, `margins`,
`bh_rejected` (within-model provisional; authoritative BH in E4).

**`e4_families.json`**: `table[model]` (E2 bucket rates, E3 flip rates, Gemma
local/global head annotation) and `causal_decision[k][model][direction]`
= `{p, bh_rejected, margin_met, effect}` where `effect = bh_rejected AND margin_met`.

**`e5_robustness_{model}.json`**: `headline_variants` (wu / ksens / steps3),
`seed_repeats.per_bucket` (mean/min/max/range), `R_self`.

---

## The rules the researcher flagged (documented, not silent)

**OOM fallback (§5).** The *only* permitted fallback: if a run OOMs at context
16384 it retries at 12288, logs loudly, and sets `fallback_applied` in the
output. It is reserved for genuine context-driven OOM — E1 batching is
**token-budgeted** (`max_tokens_per_batch`), so a fixed batch never trips it at
long context.

**Gemma-2 capture runs in eager, by design (§4.3, §8).** Gemma-2 is *not* plain
`softmax(qKᵀ/√d)`: it scales q by `1/√query_pre_attn_scalar`, softcaps the
attention logits (`cap·tanh(logits/cap)`) before softmax, and alternates
sliding-window (local) and global layers. HF forces Gemma-2 to eager because of
softcapping, so the `capture: sdpa` request effectively becomes eager for
Gemma-2. `capture.py` reads the scale + softcap from config and masks the
sliding window; the eager-reference test (< 1e-3) is the arbiter and passes for
Gemma-2 only when all three are applied.

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

### Design decisions

- **Chat template on by default** (`grid.yaml probe.use_chat_template`). All 7
  panel models are instruct-tuned; probes are wrapped in each model's chat
  template with a **fixed date** so templates that stamp "today" (Llama-3) stay
  deterministic. The wrapper is constant per cell, so token-skeleton alignment
  is preserved. Set false for base models.
- **0-distractor breaking cells are analyzed** in E2/E3 (M2 simply never fires;
  M1/correct-attend/residual still classify) — the cleanest silence-vs-downstream
  signal.
- **E3 flips are defined relative to the unpadded no-patch baseline**, computed
  with the same single-sequence path as the patched run, so a flip is
  apples-to-apples and independent of the batched pairing path. Self-patch is
  checked **once per model** (not per pair × k). Per-pair random-control seeds
  are recorded in `e3_causal_{model}.json` (`k[*].random_control_seeds`).
- **Sliding-window masking is general** (any model with an active
  `sliding_window`), not Gemma-only; for this panel only Gemma-2's 4096 window
  is below the studied contexts.
- **E1 and E2/E3 tokenise identically**: every consumer uses the probe's
  canonical `input_ids` (chat template + specials applied once by the factory);
  batched (left-padded) and single-sequence greedy are checked token-equal by a
  test.

---

## PROVENANCE — the audit chain

Every results JSON embeds a `provenance` block (built once in
`provenance.py`, reused everywhere) so any number traces back to exactly what
produced it:

```
provenance:
  script:        scripts/e2_signatures.py        # which script
  code:          {commit, branch, dirty, repo_root}   # exact code commit + clean/dirty
  config_hashes: {<path>: <sha256>, …}           # byte-hash of every config that fed it
  model_key / model_sha:  qwen25_7b_instruct / <pinned 40-hex SHA>   # exact weights
  seeds:         [42]                            # the seed(s)
  timestamp_utc / platform / packages           # when + on what stack
  extra:         {detection_seed, detector, effective_attn, …}
```

**To audit a number** (e.g. an E2 bucket rate): read its file's `provenance`.
`code.commit` + `config_hashes` reproduce the exact code and configs;
`model_sha` + `seeds` reproduce the exact weights and probe instantiation;
`extra.detection_seed` names which detection artifact was consumed. Re-running
that script at that commit, with those config bytes, that model SHA and that
seed reproduces the number (up to documented GPU kernel nondeterminism).

**Determinism (§1.7).** python/numpy/torch are seeded per run; probe
construction is a pure function of `(cell, sample_idx, seed)` — same inputs,
same bytes. Greedy decoding is deterministic. The only residual nondeterminism
is kernel-level GPU floating-point (non-associative reductions across CUDA
launches); it does not change token-skeleton alignment, grading, or the
pre-registered decision rules.

---

## Tests

```bash
python -m pytest tests/ -q
```

Guardrails: the pre-registration red-test (missing file + each missing key), the
four-bucket total partition, BH/permutation known answers, grading taxonomy,
probe **determinism + token-skeleton alignment** + span decode-verify, the GQA
mapping, the **manual-row-vs-eager-reference** check (< 1e-3, incl. Gemma-2
softcap), and the **self-patch no-op** (token-identical generation). Tiny-model
tests skip cleanly offline; the real four-model eager-reference check is the
first cell block of notebook `00`.
