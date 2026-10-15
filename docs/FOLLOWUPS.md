# Follow-up experiments for the merged manuscript (A–G)

All follow-ups, including D, run from this repository. These runs connect the two halves of the merged paper (Part I: identity and stability of retrieval heads; Part II: the failure mechanism and its causal tests). They also correct one measurement error. The decision rules were fixed before any run: see [`FOLLOWUP_DECISION_RULES.md`](FOLLOWUP_DECISION_RULES.md). All runs are exploratory, except E, which is a correction.

| | What it answers | Script(s) | Notebook | GPU | Est. time |
|---|---|---|---|---|---|
| **E** | Gemma-2 attention capture was mis-indexed beyond its 4k window: corrected signature | `check_capture_long.py`, `e2_signatures.py --tag capfix` | 15 | A100 40 GB | 2.5–3 h |
| **F** | Is "capture" specific to failures (same rule applied to the paired successes)? Do the Part I lists transfer to the Part II task (re-detection on held-out Part II cells)? Conditional F-3: patch the task list | `e9_answer_rows.py` (+ `e3 --heads-from <m>_taskII`) | 16 | A100 40 GB | ~1.5 h (7 models) + F-3 ~2.5 h |
| **A** | Does zero-ablation of the knockout head set break the Part II probes, and at which positions (scope)? Tests Section 7's explanation of Llama-3.1 | `e6_matched_ablation.py` | 17 | A100 40 GB | 3–4.5 h (4 models) |
| **B** | Does the failure mechanism (E2) and the causal profile (E3) survive NF4, where Part I shows identity survives? | `e7_quant_cells.py` → `e2 --heads-from` → `e3 --max-pairs` | 19 | L4 is enough | 8–11 h (3 models) |
| **C** | Is the patching effect carried by *where* heads attend or by *what* they read? | `e8_pattern_transplant.py` | 18 | A100 40 GB | 3–4.5 h (3 models) |
| **D** | Which haystack produced the stored head sets, and how haystack-dependent is head identity? | `rhp_haystack_check.py` (runs the detector of the retrieval-head-profile clone) | 21 | L4 | 4–5 h (14 runs) |
| **G** (optional) | Does E3's break effect belong to the listed heads or to their layers (same-layer control)? | `e3_causal.py --control-mode layer --tag layerctrl` | 18 (`RUN_G`) | A100 40 GB | ~1 h per model (3 models) |
| — | Apply the rules, write the report | `followup_analysis.py` | 20 (CPU) | — | 2 min |

**Run order.** Run E first, because B's Gemma-2 comparison needs the corrected parent E2. Then run F, which is cheap and decides whether F-3 is needed and how Section 6 is worded. A, C and D are independent and can run in parallel sessions. G is optional and can run whenever an A100 is free.

## What changed in the code

- **`src/failure_mech/capture.py`** (fix): `_row_for_head` aligns the row to absolute positions. A window-truncated `HybridCache` (Gemma-2 local layers once T > 4,096) now gets left-padded with −inf instead of being read from position 0. Regression tests check this against the model's eager rows below the window, between one and two windows, and beyond two windows.
- **`src/failure_mech/panel.py`**: `quant: bnb4` panel entries load as NF4 with double quantization and bf16 compute. Loading asserts that real `Linear4bit` modules are present, and the setting is recorded in provenance. AWQ and GPTQ are refused explicitly.
- **`scripts/e2_signatures.py`**: new options `--heads-from`, `--probe-key` and `--save-grades`. `e3_causal.py`: new options `--heads-from`, `--probe-key`, `--e2-tag`, `--max-pairs`, `--allow-other-heads` and `--control-mode {random,layer}` (G). All defaults reproduce the original runs. The variants require `--tag`, get their own checkpoint directory and fingerprint, and are refused when the head list differs from the one their E2 used. E3 now also stores per-pair indicators.
- **`scripts/rhp_haystack_check.py`** (new, D): the haystack check of the Part I detector. It runs against the retrieval-head-profile clone given by `--rhp-repo` or `RHP_PART2_REPO` and loads that repository's helpers by file path, so it cannot pick up this repository's `scripts/_common.py`. Nothing in the clone is written.
- **`src/failure_mech/detect.py`**: optional `RFM_EXTRA_DETECTION_DIRS` lookup, used only by F-3 for the `<model>_taskII` lists. Unset, it behaves as before.
- **`src/failure_mech/followups.py`** (new): contains
  - the all-head answer-step summaries (F);
  - the scoped ablation mask;
  - answer-step rows and values for the transplant;
  - the NF4 row gate;
  - low-memory eager reference rows at full prompt length.
- **`tests/followups/`** (new): offline tests on tiny models built from config, with a locally trained tokenizer and no downloads:
  - mask scopes, including equality with weight surgery;
  - manual rows vs eager rows for Llama, Qwen2, OLMo-2 and Gemma-2;
  - transplant vectors;
  - the Gemma-2 window regression;
  - an end-to-end run of every follow-up script.

Run them with `python -m pytest tests/followups -q`.

## Outputs (all in `RFM_RESULTS_DIR`)

- **E:**
  - `capture_long_check_<model>.json`
  - `e2_signatures_gemma2_9b_it_capfix.json`
  - `e2_signatures_gemma2_9b_it_wu_capfix.json`

  The manuscript's analysis code (`code/common.py`) uses the `_capfix` files automatically when they exist.
- **F:**
  - `e9_answer_rows_<model>.json` and `.npz`
  - `detection_taskII/<model>_taskII_seed42.json`
  - for F-3: `e3_causal_<model>_taskheads.json`
- **A:** `e6_matched_ablation_<model>.json`
- **B:**
  - `e7_quant_cells_<quant>.json`
  - `e1_*_<quant>.json` (the parent's cells, `probe_key` = parent)
  - `e2_signatures_<quant>_parentheads.json`
  - `e3_causal_<quant>_parentheads.json`
- **C:** `e8_pattern_transplant_<model>.json`
- **G:** `e3_causal_<model>_layerctrl.json`
- **D:** `rhprofile_results_followup/robustness/D_haystack/{fallback,wikitext}/<model>_seed42.json`, plus `summary.json` (a separate Drive folder; the stored Part I profiles are only read).
- **Report:** `followup_summary.json` and `followup_report.md`.
