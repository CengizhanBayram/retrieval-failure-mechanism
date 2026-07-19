# retrieval-failure-mechanism (Part 3)

**When a long-context language model *fails* to retrieve a fact, what breaks at the
retrieval-head level — and is that broken thing the *cause* of the failure, or just
a visible correlate?**

Retrieval-head *profiling* work (which head attends the needle, how strongly) is
correlational. This suite adds the missing half — **causal verification by
bidirectional activation patching** — across a 7-model panel, and finds that the
visible mechanism and the causal weight **come apart**.

> **Bottom line.** Distractor-capture (**M2** — the retrieval head attends the wrong,
> distractor token) is the common failure *mechanism* (6 of 7 models). But the
> *causal weight* of the retrieval heads — how much you must patch before retrieval
> actually breaks — is **graded by model and uncorrelated with the mechanism
> strength**: from a handful of heads sufficing (qwen2.5-3b) through
> many-heads-redundant (mistral, gemma2) to **significant-but-below-threshold even
> when the entire detected head set is patched** (llama3.1). The M2 rate does not
> predict the causal architecture. That gap — *visible signature vs graded causal
> weight* — is exactly what profiling cannot see. The full numbers are in
> [`docs/FINDINGS.md`](docs/FINDINGS.md); the paper argues significance.

This repo is **inference-only** and built **on top of** two prior repos, which it
imports and never edits:

- **Part 1** — *Does RoPE Prevent or Degrade Retrieval Heads?* — provides `src/`
  (model loader, activation patching, statistics).
- **Part 2** — *retrieval-head-profile* (`rhp/`) — provides the **pinned model
  registry** (`configs/panel.yaml`: commit SHAs + `rope_theta`) and the **detection
  artifacts** (`datas/results/profile/{model}_seed{seed}.json`, the ranked
  retrieval-head lists). Part 3 **never re-detects** — it consumes these.

Everything is gated on a **researcher-authored pre-registration**, every number
carries a **provenance record**, and no module in the package *interprets* a result:
the code measures, the docs and the paper interpret.

---

## 1. The question, precisely

A **probe** places a needle fact (an access code bound to an adjective+noun object)
in a long filler context, optionally alongside **distractors** (same-shell objects
with different codes), and asks for the code. When the model answers wrong, the
failure is classified — at the model's own **retrieval heads** — into one of four
mechanistic buckets:

| bucket | meaning |
|---|---|
| **M1 · silence** | the head barely attends the needle (low needle mass) and nothing else |
| **M2 · distractor-capture** | the head attends a **distractor** instead of the needle |
| **correct-attend** | the head *does* attend the needle, yet the answer is wrong (loss is downstream) |
| **residual** | diffuse attenuation — no dominant target |

Profiling stops at "which bucket." The causal question is: **if you overwrite those
heads' activations with a donor's (from a paired success/failure), does the
retrieval outcome flip?** A mechanism that is a *correlate* will not; a mechanism
that is the *bottleneck* will.

---

## 2. Model panel (7 instruct models, 6 families, GQA + MHA)

| task key | family | attention | θ (rope) | min GPU |
|---|---|---|---|---|
| `llama31_8b_instruct` | llama | GQA 32/8 | 500k | A100 40 GB |
| `qwen25_7b_instruct` | qwen | GQA 28/4 | 1M | A100 40 GB |
| `gemma2_9b_it` | gemma | GQA 16/8 + softcap, local/global | 10k | **A100 80 GB** |
| `mistral_7b_instruct` | mistral | GQA 32/8 | 1M | A100 40 GB |
| `olmo2_7b_instruct` | olmo | **MHA 32/32 + QK-norm** | 500k | A100 40 GB |
| `phi35_mini` | phi | **MHA 32/32, fused qkv** | 10k | **A100 80 GB** |
| `qwen25_3b_instruct` | qwen | GQA 16/2 (size axis) | 1M | A100 40 GB |

**Why these seven.** Six architecture families span the mechanistic axes that could
plausibly change the answer: **GQA vs MHA** (how value/key heads are shared),
**QK-norm** (olmo2), **fused QKV + partial rotary** (phi3), **logit softcap + a
sliding/global window mix** (gemma2), and a within-family **size axis**
(qwen2.5 3b vs 7b). All are instruct-tuned, so probes are wrapped in each model's
chat template.

**Why gemma2 and phi3 need 80 GB.** Both are forced to **eager** attention (gemma2
because sdpa silently drops its logit-softcapping; phi3 because the pinned
transformers has no phi3 sdpa kernel). Eager materialises the full `L×L` attention
matrix inside the model's own forward — ~18 GB at the panel's long contexts, which
OOMs a 40 GB card. The five sdpa models fit on 40 GB. All shipped notebooks are
A100-tagged because a mixed-panel run must satisfy the largest requirement.

`capture.py` is family-aware (standard GQA, gemma2 query-scale+softcap+sliding
window, olmo2 QK-norm-before-RoPE, phi3 fused-qkv split + partial rotary) and every
family is validated by the **eager-reference gate** below (< 1e-3, fp32).

---

## 3. The experiment pipeline — what each step measures and *why*

The pipeline is a funnel: find *where* models fail → *how* they fail → whether that
is *causal* → aggregate the decision → check it is *robust* → (follow-ups) localise
the bottleneck. Order matters: **E2 needs E1, E3 needs E2, E4/E5 need E2/E3.**

### E1 — Breaking surface  ·  `scripts/e1_breaking_surface.py`
**Measures:** for each grid cell (context length × needle position × distractor
count × distractor similarity), the greedy-decoding accuracy, with Wilson CIs; a
cell is **breaking** if its accuracy lands in the pre-registered band (not ceiling,
not total collapse).
**Why:** the mechanism is only interesting on *genuine* failures. E1 finds them and
excludes both the trivial (ceiling) and the confounded (total positional-OOD
collapse *beyond* a model's context window — the **in-window / beyond-window** rule).
The confirmatory grid enables the **`shell_share`** difficulty (a distractor reuses
the needle's adjective *or* noun), which is what breaks the strong models *in-window*.
Two-stage and resumable; records `oom_fallback_applied` for the one permitted
16384→12288 fallback.

### E2 — Signatures  ·  `scripts/e2_signatures.py`
**Measures:** within breaking cells, forms matched **success ↔ failure pairs**
(identical token skeleton), computes the manual attention rows for the target heads,
and classifies each failure into the four buckets; emits per-head paired statistics
(permutation p, Cliff's δ, BH) and the sample-level bucket rates.
**Why:** this is the mechanism half — "how do they fail." It also **records
`pairs_by_cell`** (the exact matched-pair sample indices) so E3 measures causality on
*exactly* the sample E2 measured the mechanism on (see §I of the amendment — this
fixed a real 614-vs-625 pair-set drift).

### E3 — Causality  ·  `scripts/e3_causal.py`
**Measures:** over E2's pairs, for each head-set size *k*, five runs per pair —
**repair** (failure ← success donor), **break** (success ← failure donor),
**random-control** (same *k*, non-retrieval heads), **no-patch rerun** (flip base
rate), and a **self-patch no-op** (recipient ← its own activation, which *must* be
token-identical). It records, per direction, the flip rate, the control flip, the
margin (flip − control), the permutation p, and BH.
**Why:** this is the causal half. **`break`** asks "does removing the retrieval-head
signal break a working retrieval?"; **`repair`** asks "does injecting it fix a broken
one?". The **random control** and the pre-registered **20 pp margin** guard against
the trivial "patching many heads disrupts anything" — an effect must beat the control
by ≥ 20 pp *and* survive BH.
Checkpointed per *k* (a killed session loses at most one *k*); `--max-hours` stops
cleanly on a *k* boundary; donors are captured once per probe over the union of head
sets and sliced per *k* (proven bitwise-identical, `tests/test_donor_slicing.py`).

### E4 — Family table + authoritative decision  ·  `scripts/e4_families.py`
**Measures:** aggregates E2 + E3 across the panel and computes the **authoritative
BH** across {models × directions} per *k*; the causal decision is
`effect = bh_rejected AND margin_met`. Loads no weights (config + JSON only) — runs
on CPU in minutes.
**Why:** multiple-comparison control must be done once, across the whole family, not
per model. E4 is where a raw flip rate becomes a pre-registered *decision*.

### E5 — Robustness  ·  `scripts/e5_robustness.py`
**Measures:** re-runs E2 under perturbations, each a **tagged** artifact that never
overwrites the primary: **`wu`** (the Wu/copy-head detector instead of argmax),
**`ksens`** (a different head count), **`steps3`** (averaging masses over 3 answer
steps), **`seeds`** (seed repeats → mean ± range, `R_self`), **`m2sens`** (an
alternative M2 floor, side-by-side with the frozen one).
**Why:** to show the headline is not an artifact of the detector, the head count, the
answer step, the seed, or the M2 threshold. `wu` is the primary check (does the
mechanism survive a *different definition* of "retrieval head"?).

### Follow-up A — Full-set patch (amendment §H)  ·  `--extra-k`, notebook 13
**Why:** llama3.1 and gemma2 are BH-significant on `break` but stay below the 20 pp
margin through k=10. A reviewer can object: *maybe that null is under-dosing — you
only patched 10 of their 39 / 77 detected heads.* This patches the **entire detected
set** (`--extra-k 39` / `77`) on the CLI, without touching the config or the completed
z_h checkpoints. **Result:** llama3.1 stays sub-margin at 39/39 (+0.083) → the
dissociation is *not* under-dosing; gemma2 clears the margin at 77/77 (+0.438) → it
*was* under-dosed and is redundant-causal, reclassified out of the dissociation bin.

### Follow-up B — Value / MLP patch site (amendment §M)  ·  `--site`, notebook 14
**Why:** since llama3.1's head *outputs* (`z_h`) are not the bottleneck even at the
full set, *where* is it? The patch site is moved downstream: **`v`** (the value
vectors — a **KV-cache swap** at the needle/distractor context positions, which is
what retrieval actually reads) and **`mlp`** (the layer's whole MLP output).
**Result:** the value swap is **confounded by content transport** (swapping the
needle's value literally swaps the answer → break-only 1.000/0.000) and is reported
as a *caution*, not circuit evidence; **mlp** is bidirectional on the causal model
(qwen2.5-3b) but **null for llama3.1 over all 12 of its head-layers**. So llama3.1's
failure is not localised to the retrieval-head circuit at either *interpretable* site
— the dissociation holds across two intervention points.

---

## 4. Headline results (summary; full record in `docs/FINDINGS.md`)

**Mechanism (E2, argmax heads).** M2-dominant in 6/7: mistral 0.84, qwen3b 0.73,
llama/qwen7b/olmo 0.62, phi 0.57; gemma2 is the outlier (residual/M1, M2 only 0.14).

**Causal weight (E3/E4, `break` direction).** Every model's break is BH-significant
by k=5–10 (**no model is causally inert**); what varies is *how much* you must patch
to clear the 20 pp margin:

| regime | models | reading |
|---|---|---|
| **causal** (few heads) | qwen2.5-3b (k=5), phi3.5 & olmo2 (k=10) | pre-registered |
| **redundant-causal** (many heads) | mistral & qwen2.5-7b (k=30), gemma2 (full 77-set) | exploratory (k>10) |
| **dissociation** (sub-margin at the full set) | **llama3.1** | the unique case |

**The central claim:** the M2 rate is **uncorrelated** with the causal architecture —
the lowest-M2 model (gemma2, 0.14) and the highest-M2 model (mistral, 0.84) are *both*
redundant; the mid-M2 llama3.1 (0.62) is the lone dissociation.

**Robustness (E5).** Detector independence (`wu`) is **model-specific**: M2 survives
the copy detector only for the Qwen family; for the other five it is argmax-head
specific (copy heads show residual/silence). This is itself a finding, and the
"detector-independent" claim is **not** made. Reproducibility drift between runs is
≤ 2.9 pp on bucket rates (cross-session GPU nondeterminism; self-consistency of the
E2↔E3 pair set holds 7/7).

**Everything from k > 10, the full-set patch, and the value/MLP sites is labelled
EXPLORATORY**; the pre-registered result is k ≤ 10.

---

## 5. Methodology & integrity guards

This suite is built so a reviewer can trust each number without re-running it.

- **Pre-registration gate.** `configs/preregistration.yaml` is researcher-authored
  and holds *every* threshold (the four-bucket predicates, `pair_min`, the breaking
  band, the M2 floor, the 20 pp margin, `alpha`, seeds). The code **never creates,
  defaults, or edits it**; a missing file or key fails loudly, naming the key. No
  threshold is hard-coded in any script.
- **Frozen thresholds, honest labels.** Thresholds are not re-tuned after seeing
  results (that would be HARKing). An alternative M2 floor is an E5 *sensitivity
  variant*, not an overwrite (`--m2-min-distractor-mass` refuses to run without
  `--tag`). Any post-hoc extension (k=20/30, the full set, the sites) is labelled
  **exploratory** in the config, the notebook filename, the amendment, and the paper.
- **Provenance on every JSON.** Producing script, exact git commit (+ dirty flag),
  byte-hash of every config, model key + pinned SHA, seeds, timestamp, platform,
  packages. Any number is auditable back to what produced it. The same hashes drive
  **resume freshness** (a run is skipped only if its recorded config hashes still
  match the configs on disk — not merely because an output file exists).
- **Self-patch no-op** (recipient ← own activation → token-identical) **and a
  never-op detector** (a *foreign* donor must actually change the target tensor) are
  enforced per site; the run aborts on failure. These caught the first value-site
  implementation, which was structurally null (§M).
- **Control + margin.** A causal `effect` requires beating the random control by
  ≥ 20 pp *and* BH significance — so a high flip at large *k* is not mistaken for a
  specific effect.
- **Capture-correctness gate.** The manual attention row is validated against the
  model's own eager `output_attentions` to ≤ 5.66e-07 (fp32, 7/7), persisted as
  `gate_eager_reference.json`. Covers gemma2 softcap, olmo2 QK-norm, phi3 partial
  rotary.
- **Determinism.** Probes are pure functions of `(cell, sample_idx, seed)`; greedy
  decoding is deterministic; the only residual nondeterminism is GPU-kernel
  floating-point, which does not change token-skeleton alignment or the decision
  rules. E3 measures causality on **exactly** E2's recorded pairs (`pairs_by_cell`).
- **Memory rule.** Never `output_attentions=True` in an experiment, never materialise
  a full `L×L` matrix — attention rows are computed manually for the target heads
  only. (The gate above is the *only* place `output_attentions` runs, in tests.)

---

## 6. Repository layout

```
src/failure_mech/
  prereg.py      # the pre-registration gate (§1.2)
  panel.py       # pinned loading (bf16, explicit attn backend), GQA mapping
  probes.py      # procedural probe construction + token-skeleton alignment (§4.1)
  spans.py       # token-span bookkeeping + decode-verify (§4.2)
  capture.py     # manual attention-row capture (§4.3): recompute-RoPE, family-aware
  patching.py    # donor capture + patching at 3 sites (z_h | v | mlp) + self-patch/never-op
  grading.py     # answer grading + 4-way behavioral taxonomy (§4.5)
  classify.py    # four-bucket mechanistic classifier from prereg rules (§7)
  detect.py      # detection-artifact reader, top-k ranking, head-set tie diagnostic
  stats.py       # paired permutation, BH, Cliff's delta, bootstrap, Wilson (§4.6)
  provenance.py  # the provenance stamp reused everywhere (§1.6)
configs/
  grid.yaml decoding.yaml paths.yaml e3.yaml
  preregistration.yaml            # RESEARCHER-AUTHORED (thresholds live only here)
  preregistration.example.yaml    # schema-shaped example (not used by any run)
scripts/
  e1_breaking_surface.py e2_signatures.py e3_causal.py e4_families.py e5_robustness.py
  check_e2_repro.py               # offline reproduction check (no GPU)
  _common.py                      # config load, checkpoints, freshness, time-guard
notebooks/
  build_notebooks.py              # the GENERATOR — edit here, never the .ipynb
  00_A100_setup_and_guardrails … 14_A100_e3_patch_sites   # GPU/CPU tag in each name
tests/                            # 80 pytest guardrails
docs/
  FINDINGS.md                     # the full results record
  LIMITATIONS.md                  # every caveat, with evidence + mitigation
  AMENDMENT_2026-07-13.md         # the dated amendment (status of every run)
  PREREGISTRATION_SCHEMA.md       # the prereg schema
rfm_results/                      # the published result JSONs (see §8)
```

Notebooks are **generated** from `notebooks/build_notebooks.py` — edit the generator
and regenerate; never hand-edit the `.ipynb`. Each filename carries the runtime it
needs (`_A100_` / `_CPU_`).

---

## 7. Reproduce

### Local (tests, planning, offline checks — no GPU)
```bash
pip install -r requirements.txt          # install a CUDA-matched torch wheel first (see file)
export RHP_PART1_REPO=/path/to/Does-RoPE-...-Model-Families
export RHP_PART2_REPO=/path/to/retrieval-head-profile
export RFM_RESULTS_DIR=/path/to/rfm_results

python -m pytest -q                                        # 80 guardrails
python scripts/e1_breaking_surface.py --model llama31_8b_instruct --dry-run   # plan, no GPU
python scripts/check_e2_repro.py                           # E2 re-run reproduction check
```

### The pipeline (per model; A100 via the notebooks)
```bash
python scripts/e1_breaking_surface.py --model <m> --stage auto
python scripts/e2_signatures.py       --model <m>                 # records pairs_by_cell
python scripts/e3_causal.py           --model <m> --max-hours 20  # checkpointed per k
python scripts/e4_families.py --models <all 7>                    # authoritative BH
python scripts/e5_robustness.py       --model <m> --variants wu
# follow-ups (exploratory):
python scripts/e3_causal.py --model llama31_8b_instruct --extra-k 39            # full set
python scripts/e3_causal.py --model llama31_8b_instruct --site v   --tag site-v --k-list 10 30 --extra-k 39
python scripts/e3_causal.py --model llama31_8b_instruct --site mlp --tag site-mlp --k-list 10 30 --extra-k 39
```

### On Colab (the intended path)
Run the notebooks in order; each is run-all capable, **resume-safe to Drive** (a
model is skipped only when its artifact is *current*), and guarded by a 23 h cap so a
free/Pro session finishes inside Colab's 24 h limit. Key notebooks:

| notebook | GPU | does |
|---|---|---|
| `00_A100_setup_and_guardrails` | A100 | clone the 3 repos, wire paths, **pre-reg gate + eager-reference gate** (fp32, 7/7) |
| `06_A100_shell_share_e1_e2` | A100 | E1 + E2 on the confirmatory `shell_share` grid |
| `07_A100_e3_causal_e4_families` | A100 | E3 + E4 |
| `09_A100_e2_rerun_record_pairs` | A100 | E2 re-run that records `pairs_by_cell` (+ reproduction check) |
| `10_/10a/10b/10c_A100_e3_ksweep_EXPLORATORY` | A100 | extended k-sweep, single or 3-way parallel split |
| `11_A100_e5_robustness` | A100 | E5 variants |
| `12_CPU_repro_check` | **CPU** | the E2 reproduction check |
| `13_A100_e3_fullset_llama_gemma` | A100 80 GB | the decisive full-set patch |
| `14_A100_e3_patch_sites` | A100 | the value/MLP site follow-up |

---

## 8. Published results (`rfm_results/`)

The result JSONs are committed so the numbers can be read without a GPU. Heavy or
internal artifacts are **not** published (they are regenerated): the per-cell
`checkpoints/`, the donor `*.npz` stores, and the `*.pre_pairfix` E2 backups.

| file | what |
|---|---|
| `e1_surface_{model}.json`, `e1_breaking_cells_{model}.json` | the breaking surface |
| `e2_signatures_{model}.json` | mechanism buckets + `pairs_by_cell` |
| `e2_signatures_{model}_{wu,ksens,steps3,seed*}.json` | E5 tagged variants |
| `e3_causal_{model}.json` | the causal k-sweep (incl. `--extra-k` full sets) |
| `e3_causal_{model}_site-{v,mlp}.json` | the value/MLP site follow-up |
| `e4_families.json` | the family table + authoritative `causal_decision` |
| `e5_robustness_{model}.json` | the robustness aggregate |
| `gate_eager_reference.json` | the capture-correctness gate (7/7 pass) |

Every file carries its `provenance` block; the output schema is documented per file
in [`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## 9. Documentation map

| doc | read it for |
|---|---|
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | the full results — every table, exploratory labels, confirmatory-vs-exploratory split |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | every caveat, with evidence, magnitude, and how the design addresses it |
| [`docs/AMENDMENT_2026-07-13.md`](docs/AMENDMENT_2026-07-13.md) | the dated amendment — the *status* of every run (confirmatory / exploratory / withdrawn), committed before the runs it governs |
| [`docs/PREREGISTRATION_SCHEMA.md`](docs/PREREGISTRATION_SCHEMA.md) | the pre-registration schema |

---

## 10. Tests

```bash
python -m pytest -q          # 80 passing
```

Guardrails: the pre-registration red-test (missing file + each missing key), the
four-bucket total partition, BH/permutation known answers, grading taxonomy, probe
determinism + token-skeleton alignment + span decode-verify, the GQA mapping, the
**manual-row-vs-eager-reference** check (< 1e-3, incl. gemma2 softcap), the
**self-patch no-op** and **never-op** detector for all three sites, the
**donor-slice bitwise-identity**, and the **pre-registration integrity** guards.
Tiny-model tests skip cleanly offline; the real seven-model eager-reference gate is
the first cell block of notebook `00`.
