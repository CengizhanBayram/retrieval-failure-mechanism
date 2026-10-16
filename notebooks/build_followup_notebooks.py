"""
Generate the Colab notebooks for the merged-paper follow-up experiments
(docs/FOLLOWUP_DECISION_RULES.md):

    15_A100_followup_E_gemma_capfix.ipynb        E  Gemma-2 capture correction (run FIRST)
    16_A100_followup_F_specificity_transfer.ipynb F  signature specificity + task transfer (+ F-3)
    17_A100_followup_A_matched_ablation.ipynb    A  matched ablation, scope decomposition
    18_A100_followup_C_transplant.ipynb          C  attention-pattern transplant (+ optional G, layer-matched control)
    19_GPU_followup_B_nf4.ipynb                  B  Part II under NF4 (needs E for Gemma-2)
    20_CPU_followup_analysis.ipynb               -  apply the decision rules, write the report
    21_GPU_followup_D_haystack.ipynb             D  haystack source of the Part I head sets

Every follow-up runs from this repository. D measures the Part I detector of the
retrieval-head-profile clone that the setup cells already make (/content/rope-part2);
nothing in that repository is modified.

Run:  python notebooks/build_followup_notebooks.py
This generator only writes the seven files above; it never touches 00-14.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from build_notebooks import (  # noqa: E402  canonical helpers + setup cells
    md, code, notebook, SETUP_GPU_DRIVE, SETUP_PIP, SETUP_CLONE, SETUP_PATHS, HARD_CAP_H,
)

INTRO_COMMON = (
    "**Before the first run:** these notebooks clone the repos from GitHub, so the "
    "follow-up code (scripts `e6`-`e9`, `check_capture_long.py`, `rhp_haystack_check.py`, "
    "`followup_analysis.py`, `src/failure_mech/followups.py`, the capture fix and "
    "`docs/FOLLOWUP_DECISION_RULES.md`) must be committed and pushed first. "
    "Results go to the same Drive folder as E1-E5 (`rfm_results`), next to the "
    "artifacts they build on; every script is resume-safe."
)

CONFIG_GATE = code(r"""
# Cell 4 - analysis-config gate. The follow-ups reuse the thresholds of the
# pre-specified E1-E5 analysis (configs/preregistration.yaml) and add the rules in
# docs/FOLLOWUP_DECISION_RULES.md; both files must be present in the clone.
import os, sys
sys.path.insert(0, '/content/rope-part3/src')
from failure_mech import prereg as PR
cfg = PR.load_prereg('/content/rope-part3/configs/preregistration.yaml')
rules = '/content/rope-part3/docs/FOLLOWUP_DECISION_RULES.md'
assert os.path.exists(rules), 'docs/FOLLOWUP_DECISION_RULES.md missing - push the follow-up commit first'
for f in ['scripts/e6_matched_ablation.py', 'scripts/e7_quant_cells.py', 'scripts/e8_pattern_transplant.py',
          'scripts/e9_answer_rows.py', 'scripts/check_capture_long.py', 'scripts/followup_analysis.py',
          'scripts/rhp_haystack_check.py', 'src/failure_mech/followups.py']:
    assert os.path.exists('/content/rope-part3/' + f), f + ' missing - push the follow-up commit first'
print('Analysis config OK; follow-up rules present:', rules)
""")

TESTS = code(r"""
# Cell 5 - CPU tests of the follow-up machinery on tiny LOCAL models (no download,
# ~1 min): scopes of the ablation mask, manual rows vs eager rows (incl. Gemma-2
# beyond its window - the regression test of the capture fix), transplant vectors,
# and an end-to-end run of every follow-up script.
import subprocess
subprocess.run(['pip', 'install', '-q', 'pytest'], check=False)
run(['-m', 'pytest', 'tests/followups', '-q'])
""")


def setup_cells(title: str, body: str) -> list[dict]:
    return [md(f"# {title}\n{body}\n\n{INTRO_COMMON}"),
            md("### Setup - run cells 0-5 once per session\nPaste your `HF_TOKEN` "
               "(gated Llama/Gemma) in Cell 2."),
            SETUP_GPU_DRIVE, SETUP_PIP, SETUP_CLONE, SETUP_PATHS, CONFIG_GATE, TESTS]


def model_loop(models_expr: str, argv_expr: str, skip_expr: str, first_est_h: float,
               title: str, note: str = "") -> list[dict]:
    """Resume-safe, 24 h-guarded loop. ``argv_expr`` may use ``key`` and
    ``budget_h`` (hours left for this model under the cap)."""
    return [md(f"## {title}\n{note}"), code(r"""
import os, time, gc, torch
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
MODELS = %(models)s
start = time.time(); model_times = []
for key in MODELS:
    if %(skip)s:
        print(key, '-> complete output exists, skip'); continue
    ok, elapsed_h, est_h = C.time_guard(start, model_times, first_est_h=%(est).1f)
    if not ok:
        print(f'STOP before {key}: {elapsed_h:.1f}h + est {est_h:.1f}h > %(cap)d h. Re-run to resume.'); break
    budget_h = max(0.5, %(cap)d - elapsed_h - 0.5)
    t0 = time.time()
    try:
        run(%(argv)s)
        model_times.append((time.time() - t0) / 3600.0)
        print(key, 'done in', round(model_times[-1], 2), 'h')
    except Exception as e:
        print(f'{key} FAILED: {type(e).__name__}: {str(e)[:300]} -- continuing.')
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
""" % {"models": models_expr, "argv": argv_expr, "skip": skip_expr, "est": first_est_h,
       "cap": HARD_CAP_H})]


COMPLETE_JSON = r"""
def complete(path):
    import json, os
    if not os.path.exists(path):
        return False
    try:
        return bool(json.load(open(path))['provenance']['extra'].get('complete', True))
    except Exception:
        return False
"""

# Self-contained: it does not need the setup cells, so it still works after a
# runtime restart or disconnect (the earlier ``run([...])`` version raised NameError).
ANALYSIS_SRC = r"""
# Apply the pre-specified rules to everything that has finished so far.
# Self-contained: needs no setup cell, so it also works after a restart/disconnect.
import os, subprocess, sys
if not os.path.isdir('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
os.environ['RFM_RESULTS_DIR'] = '/content/drive/MyDrive/rfm_results'
D_SUMMARY = '/content/drive/MyDrive/rhprofile_results_followup/robustness/D_haystack/summary.json'
if not os.path.isdir('/content/rope-part3'):
    subprocess.run(['git', 'clone', 'https://github.com/CengizhanBayram/retrieval-failure-mechanism.git',
                    '/content/rope-part3'], check=True)
else:
    subprocess.run(['git', '-C', '/content/rope-part3', 'pull'], check=False)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy', 'scipy', 'pyyaml'], check=False)
args = [sys.executable, 'scripts/followup_analysis.py']
if os.path.exists(D_SUMMARY):
    args += ['--haystack-summary', D_SUMMARY]
r = subprocess.run(args, cwd='/content/rope-part3', capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr)
    raise RuntimeError(f'followup_analysis.py failed ({r.returncode}); see the error above')
"""

ANALYSIS = code(ANALYSIS_SRC)


# ---------------------------------------------------------------------------
# 15 - E: Gemma-2 capture correction
# ---------------------------------------------------------------------------

def nb_15_E() -> dict:
    cells = setup_cells(
        "15 · Follow-up E - Gemma-2 capture correction (A100 40 GB, ~2.5 h)",
        "Fixes an indexing error in the Part II attention capture: once a prompt is longer than "
        "Gemma-2's 4,096-token local window, HF's `HybridCache` keeps only the last 4,096 keys of "
        "a local layer, and `capture._row_for_head` read that cache as if it started at position 0. "
        "Two of Gemma-2's five inspected heads (L18_H3, L24_H0) sit in local layers; 219 of its 625 "
        "failures had a NaN distractor mass and could not be classified as capture. The fix is in "
        "the code; this notebook (1) verifies the corrected capture against the model's own eager "
        "attention on real 8k probes and (2) re-runs Gemma-2's E2 (primary heads and Wu heads). "
        "No other model has a window shorter than its probes. **Run this first (notebooks 16 and 19 build on it).**")
    cells += [
        md("## 1. Gate: corrected capture vs eager attention on real Gemma-2 probes\n"
           "Must print `PASSED`. (With the old code this check fails on the local-layer heads.)"),
        code("run(['scripts/check_capture_long.py', '--model', 'gemma2_9b_it', '--n-cells', '4'])"),
        md("## 2 (recommended, ~3 min/model). The same check on the other six models\n"
           "Strengthens the methods statement from short test prompts to real breaking-cell probes "
           "(cells up to 8k tokens; the eager reference materialises one layer at a time)."),
        code(r"""
for m in ['llama31_8b_instruct', 'qwen25_7b_instruct', 'mistral_7b_instruct', 'olmo2_7b_instruct',
          'phi35_mini', 'qwen25_3b_instruct']:
    try:
        run(['scripts/check_capture_long.py', '--model', m, '--n-cells', '3'])
    except Exception as e:
        print(m, 'check failed or skipped:', str(e)[:200])
"""),
        md("## 3. E2 re-run, primary (argmax) heads - writes `e2_signatures_gemma2_9b_it_capfix.json`"),
        code(COMPLETE_JSON + r"""
import os
out = os.path.join(os.environ['RFM_RESULTS_DIR'], 'e2_signatures_gemma2_9b_it_capfix.json')
if os.path.exists(out):
    print('exists, skip:', out)
else:
    run(['scripts/e2_signatures.py', '--model', 'gemma2_9b_it', '--tag', 'capfix', '--save-grades'])
"""),
        md("## 4. E2 re-run, copy-score (Wu) heads - writes `..._wu_capfix.json`"),
        code(r"""
import os
out = os.path.join(os.environ['RFM_RESULTS_DIR'], 'e2_signatures_gemma2_9b_it_wu_capfix.json')
if os.path.exists(out):
    print('exists, skip:', out)
else:
    run(['scripts/e2_signatures.py', '--model', 'gemma2_9b_it', '--detector', 'copy', '--tag', 'wu_capfix'])
"""),
        md("## 5. Corrected vs old numbers (rule E-1/E-2)"),
        ANALYSIS,
    ]
    return notebook(cells)


# ---------------------------------------------------------------------------
# 16 - F: signature specificity + task transfer (+ conditional F-3)
# ---------------------------------------------------------------------------

F_MODELS = ["qwen25_7b_instruct", "qwen25_3b_instruct", "olmo2_7b_instruct", "mistral_7b_instruct",
            "llama31_8b_instruct", "gemma2_9b_it", "phi35_mini"]


def nb_16_F() -> dict:
    cells = setup_cells(
        "16 · Follow-up F - is the signature specific to failures, and do the lists transfer to the "
        "Part II task? (A100 40 GB, ~1.5 h; + F-3 ~2.5 h)",
        "One prompt forward per probe, no generation. **F-1** applies E2's capture predicate to the "
        "paired SUCCESSES as well (base rate) - only the excess at failure is a failure signature. "
        "**F-2** re-runs the argmax detector on the Part II task itself, on held-out cells the model "
        "solves (accuracy >= 0.9), and compares that list with the Part I list Part II used. "
        "**F-3** (conditional, pre-specified): patch the task list with E3 for Llama-3.1 and for every "
        "model whose list does not transfer. Run after notebook 15 (Gemma-2 needs the capture fix, "
        "which is in the code). Phi-3.5 runs last; on a 40 GB A100 its eager attention at 12k tokens "
        "may run out of memory, in which case it is skipped.")
    cells += model_loop(
        json.dumps(F_MODELS),
        "['scripts/e9_answer_rows.py', '--model', key]",
        "os.path.exists(os.path.join(RESULTS_DIR, f'e9_answer_rows_{key}.json'))",
        0.4, "F-1 and F-2 for the seven models")
    cells += [
        md("## F-3 plan (pre-specified trigger)\nLlama-3.1 always; plus every model whose F-2 verdict "
           "is not 'list transfers' (top-10 overlap >= 7 and score Spearman >= 0.5)."),
        code(r"""
import json, os
R = os.environ['RFM_RESULTS_DIR']
F3 = []
for m in %(models)s:
    p = f'{R}/e9_answer_rows_{m}.json'
    if not os.path.exists(p):
        continue
    f2 = json.load(open(p))['F2_task_transfer']
    top10, rho = f2['top_k_overlap'].get('10'), f2['score_spearman']
    transfers = top10 is not None and top10 >= 7 and rho is not None and rho >= 0.5
    print(f"{m:22s} top-10 overlap {top10}  Spearman {rho:.2f}  task heads {f2['n_task']}  "
          f"-> {'transfers' if transfers else 'F-3'}")
    if m == 'llama31_8b_instruct' or not transfers:
        F3.append((m, f2['n_task']))
print('F-3 runs:', F3)
""" % {"models": json.dumps(F_MODELS)}),
        md("## F-3 run (E3 with the Part II-task list on the same pairs)"),
        code(COMPLETE_JSON + r"""
import os, time
R = os.environ['RFM_RESULTS_DIR']
os.environ['RFM_EXTRA_DETECTION_DIRS'] = os.path.join(R, 'detection_taskII')
start = time.time()
for m, n_task in F3:
    ks = [k for k in (5, 10, 20) if k <= n_task] or [n_task]
    if complete(f'{R}/e3_causal_{m}_taskheads.json'):
        print(m, 'done, skip'); continue
    budget = max(0.5, %(cap)d - (time.time() - start) / 3600 - 0.5)
    try:
        run(['scripts/e3_causal.py', '--model', m, '--heads-from', f'{m}_taskII', '--allow-other-heads',
             '--tag', 'taskheads', '--k-list', *map(str, ks), '--max-pairs', '300',
             '--max-hours', str(budget)])
    except Exception as e:
        print(m, 'F-3 FAILED:', str(e)[:300])
""" % {"cap": HARD_CAP_H}),
        md("## Results under the pre-specified rules"),
        ANALYSIS,
    ]
    return notebook(cells)


# ---------------------------------------------------------------------------
# 17 - A: matched ablation
# ---------------------------------------------------------------------------

A_MODELS = ["llama31_8b_instruct", "qwen25_3b_instruct", "mistral_7b_instruct", "gemma2_9b_it"]


def nb_17_A() -> dict:
    cells = setup_cells(
        "17 · Follow-up A - matched ablation with scope decomposition (A100 40 GB, ~4-5 h)",
        "Zero-ablates the complete detected head set (the Part I knockout list) on the SAME E2 "
        "success/failure probes, in bf16 with E3's backend, under four scopes: `all` (= knockout), "
        "`context` (prompt positions before the answer step), `answer` (= E3's scope) and `ansdec` "
        "(answer step + decoding). Tests Section 7's explanation of Llama-3.1's dissociation "
        "(rule A-3). 120 pairs per model; each pair is checkpointed.")
    cells += model_loop(
        json.dumps(A_MODELS),
        "['scripts/e6_matched_ablation.py', '--model', key, '--max-pairs', '120', '--sets', 'full', "
        "'--max-hours', str(budget_h)]",
        "complete(os.path.join(RESULTS_DIR, f'e6_matched_ablation_{key}.json'))",
        2.0, "Run A (Llama-3.1 first: it carries rule A-3)",
        "Optional extension (separate tag, ~+40%): add `'--sets', 'full', 'top10', '--tag', 'top10'`.")
    cells[-1]["source"] = [COMPLETE_JSON.strip("\n") + "\n\n"] + cells[-1]["source"]
    cells += [md("## Results under the pre-specified rules"), ANALYSIS]
    return notebook(cells)


# ---------------------------------------------------------------------------
# 19 - B: NF4
# ---------------------------------------------------------------------------

def nb_19_B() -> dict:
    cells = setup_cells(
        "19 · Follow-up B - Part II under NF4 (L4 24 GB is enough; A100 faster; ~2-5 h per model)",
        "Re-runs E2 and E3 on the bitsandbytes NF4 version of three Part II models, on the bf16 "
        "parent's breaking cells with byte-identical probes and the parent's head list, so that "
        "Part I's identity result (heads survive NF4) meets Part II's mechanism. NF4 uses double "
        "quantization and bf16 compute (the only difference from the bf16 reference is the weights). "
        "Gemma-2 runs only after notebook 15 (it needs the corrected parent E2).")
    cells += [
        md("## Configuration"),
        code(r"""
PAIRS = [('qwen25_7b_instruct', 'qwen25_7b_instruct_bnb4'),     # cheapest first (113 pairs)
         ('llama31_8b_instruct', 'llama31_8b_instruct_bnb4'),
         ('gemma2_9b_it', 'gemma2_9b_it_bnb4')]
E3_K = ['10', '20']          # 10 = pre-specified maximum budget (primary), 20 = secondary
E3_MAX_PAIRS = '250'         # deterministic round-robin subsample of E2's pairs
RUN_OWNHEADS = False         # secondary: the NF4 model's own detected head list (E2 only)
"""),
        md("## Run B (resume-safe; each stage skips when its output exists)"),
        code(COMPLETE_JSON + r"""
import os, time, gc, torch
R = os.environ['RFM_RESULTS_DIR']
start = time.time(); times = []
for parent, quant in PAIRS:
    if parent == 'gemma2_9b_it' and not os.path.exists(f'{R}/e2_signatures_gemma2_9b_it_capfix.json'):
        print('gemma2: run notebook 15 (experiment E) first -> skip'); continue
    ok, el, est = C.time_guard(start, times, first_est_h=4.0)
    if not ok:
        print('time cap reached; re-run to resume'); break
    t0 = time.time()
    try:
        if not os.path.exists(f'{R}/e1_surface_{quant}.json'):
            run(['scripts/e7_quant_cells.py', '--model', quant, '--parent', parent])
        if parent == 'gemma2_9b_it' and not os.path.exists(f'{R}/capture_long_check_{quant}.json'):
            run(['scripts/check_capture_long.py', '--model', quant, '--heads-from', parent])
        if not os.path.exists(f'{R}/e2_signatures_{quant}_parentheads.json'):
            run(['scripts/e2_signatures.py', '--model', quant, '--heads-from', parent,
                 '--tag', 'parentheads', '--save-grades'])
        if not complete(f'{R}/e3_causal_{quant}_parentheads.json'):
            budget = max(0.5, %(cap)d - (time.time() - start) / 3600 - 0.5)
            run(['scripts/e3_causal.py', '--model', quant, '--heads-from', parent,
                 '--e2-tag', 'parentheads', '--tag', 'parentheads', '--k-list', *E3_K,
                 '--max-pairs', E3_MAX_PAIRS, '--max-hours', str(budget)])
        if RUN_OWNHEADS and not os.path.exists(f'{R}/e2_signatures_{quant}_ownheads.json'):
            run(['scripts/e2_signatures.py', '--model', quant, '--tag', 'ownheads', '--save-grades'])
        times.append((time.time() - t0) / 3600)
    except Exception as e:
        print(quant, 'FAILED:', type(e).__name__, str(e)[:300], '-- continuing')
    finally:
        gc.collect(); torch.cuda.empty_cache()
""" % {"cap": HARD_CAP_H}),
        md("## Results under the pre-specified rules"),
        ANALYSIS,
    ]
    return notebook(cells)


# ---------------------------------------------------------------------------
# 18 - C: transplant
# ---------------------------------------------------------------------------

C_MODELS = ["qwen25_3b_instruct", "olmo2_7b_instruct", "llama31_8b_instruct"]


def nb_18_C() -> dict:
    cells = setup_cells(
        "18 · Follow-up C - attention-pattern transplant (+ optional G) (A100 40 GB, ~1.5 h per model)",
        "Separates WHERE a head attended from WHAT it read at the answer step: `activation` "
        "(the donor's z, = E3), `pattern` (donor's attention row x recipient's values) and "
        "`content` (recipient's row x donor's values), on the same pairs, at k = 10 and the "
        "complete detected set. A self-transplant gate (>= 90% identical generations) runs "
        "first. Gemma-2 is excluded (4,096-token window). Phi-3.5 is optional and needs an "
        "80 GB A100 (eager attention at 12k tokens).")
    cells += model_loop(
        json.dumps(C_MODELS),
        "['scripts/e8_pattern_transplant.py', '--model', key, '--max-pairs', '120', '--k-list', '10', "
        "'--max-hours', str(budget_h)]",
        "complete(os.path.join(RESULTS_DIR, f'e8_pattern_transplant_{key}.json'))",
        1.5, "Run C", "Optional on an 80 GB A100: add `'phi35_mini'` to `MODELS`.")
    cells[-1]["source"] = [COMPLETE_JSON.strip("\n") + "\n\n"] + cells[-1]["source"]
    cells += [
        md("## Optional G - layer-matched control (~1 h per model)\n"
           "Repeats E3 at k = 5 and 10 with a stricter control: for every patched head, a "
           "non-retrieval head from the same layer (nearest layer if none is free). Tests whether "
           "the break effect belongs to the listed heads or to the layers they sit in (rule G). "
           "Set `RUN_G = True` to run it."),
        code(COMPLETE_JSON + r"""
RUN_G = False
G_MODELS = ['qwen25_3b_instruct', 'olmo2_7b_instruct', 'phi35_mini']
if RUN_G:
    import os, time
    R = os.environ['RFM_RESULTS_DIR']
    start = time.time()
    for m in G_MODELS:
        if complete(f'{R}/e3_causal_{m}_layerctrl.json'):
            print(m, 'done, skip'); continue
        budget = max(0.5, %(cap)d - (time.time() - start) / 3600 - 0.5)
        try:
            run(['scripts/e3_causal.py', '--model', m, '--control-mode', 'layer', '--tag', 'layerctrl',
                 '--k-list', '5', '10', '--max-pairs', '300', '--max-hours', str(budget)])
        except Exception as e:
            print(m, 'G FAILED:', str(e)[:300])
""" % {"cap": HARD_CAP_H}),
        md("## Results under the pre-specified rules"), ANALYSIS]
    return notebook(cells)


# ---------------------------------------------------------------------------
# 20 - analysis (CPU)
# ---------------------------------------------------------------------------

def nb_20_analysis() -> dict:
    cells = [
        md("# 20 · Follow-up analysis (CPU, ~2 min)\nApplies `docs/FOLLOWUP_DECISION_RULES.md` to "
           "every finished follow-up output (A, B, C, E, F, G from `rfm_results`; D from the "
           "`rhprofile_results_followup` folder written by notebook 21) and writes `followup_summary.json` and "
           "`followup_report.md` next to the results. Re-run whenever new results arrive."),
        ANALYSIS,
    ]
    return notebook(cells, gpu=False)


# ---------------------------------------------------------------------------
# 21 - D: haystack source (Part I detector, run from this repository)
# ---------------------------------------------------------------------------

def nb_21_D() -> dict:
    cells = setup_cells(
        "21 · Follow-up D - haystack source of the Part I head sets (L4 24 GB is enough; ~4-5 h)",
        "The detector's corpus loader requests PG-19 through a dataset script that the recorded "
        "`datasets` version no longer runs, so the stored Part I profiles were most probably "
        "detected on the loader's 50-sentence fallback pool. D settles this with the profile's own "
        "detectors and settings (4,096 tokens, 200 + 100 probes, seed 42, 8-bit loading). "
        "**D-1** forces the fallback pool for two models: if the stored sets came from it, the "
        "scores reproduce (Spearman >= 0.99, max |Δscore| <= 0.03). **D-2** re-detects 12 models "
        "on WikiText-103 and reads J against R_self. The script lives in this repository and runs "
        "the detector of the retrieval-head-profile clone (`/content/rope-part2`); results go to "
        "a separate Drive folder (`rhprofile_results_followup`), nothing is overwritten.")
    cells += [
        md("## Helper"),
        code(r"""
import os, subprocess
subprocess.run(['pip', 'install', '-q', 'datasets'], check=False)   # WikiText stream (no-op if present)
D_OUT = '/content/drive/MyDrive/rhprofile_results_followup/robustness'
STORED = '/content/rope-part2/datas/results'                        # stored Part I profiles (read-only)
os.makedirs(D_OUT, exist_ok=True)
def run_d(*args):
    run(['scripts/rhp_haystack_check.py', '--rhp-repo', '/content/rope-part2',
         '--part1-repo', '/content/rope-part1', '--results-dir', STORED, '--out-dir', D_OUT, *args])
print('ready; D outputs ->', D_OUT)
"""),
        md("## D-1. Fallback pool forced (2 models, ~40 min)"),
        code("run_d('--models', 'qwen25_7b_instruct', 'llama31_8b_instruct', '--haystack', 'fallback')"),
        md("## D-2. WikiText-103 (12 models, ~4 h)\nFive core models (they have a test-retest "
           "R_self) and the seven Part II models. Re-run the cell to resume; finished models are skipped."),
        code("run_d('--models', 'core', 'partII', '--haystack', 'wikitext', '--time-budget-hours', '22')"),
        md("## Summary (`rhprofile_results_followup/robustness/D_haystack/summary.json`)"),
        code(r"""
import json
run_d('--summarize')
s = json.load(open(os.path.join(D_OUT, 'D_haystack', 'summary.json')))
for m, row in s['models'].items():
    for hs, e in row.items():
        a, c = e.get('argmax', {}), e.get('copy', {})
        print(f"{m:24s} {hs:9s} argmax J={a.get('jaccard', float('nan')):.3f} "
              f"rho={a.get('spearman_scores', float('nan')):.3f} J/R={a.get('J_over_Rself', float('nan')):.2f} | "
              f"copy J={c.get('jaccard', float('nan')):.3f} J/R={c.get('J_over_Rself', float('nan')):.2f}")
"""),
        md("Then run notebook 20; it reads this summary and applies rules D-1/D-2."),
    ]
    return notebook(cells)


def main():
    outputs = {
        "15_A100_followup_E_gemma_capfix.ipynb": nb_15_E(),
        "16_A100_followup_F_specificity_transfer.ipynb": nb_16_F(),
        "17_A100_followup_A_matched_ablation.ipynb": nb_17_A(),
        "18_A100_followup_C_transplant.ipynb": nb_18_C(),
        "19_GPU_followup_B_nf4.ipynb": nb_19_B(),
        "20_CPU_followup_analysis.ipynb": nb_20_analysis(),
        "21_GPU_followup_D_haystack.ipynb": nb_21_D(),
    }
    # remove superseded follow-up notebooks from the first numbering
    for stale in ("16_A100_followup_A_matched_ablation.ipynb", "17_A100_followup_B_nf4.ipynb",
                  "19_CPU_followup_analysis.ipynb"):
        p = NB_DIR / stale
        if p.exists() and stale not in outputs:
            p.unlink()
            print("removed", stale)
    for name, nb in outputs.items():
        path = NB_DIR / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
