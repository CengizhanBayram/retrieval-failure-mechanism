"""
Generate the Colab notebooks for retrieval-failure-mechanism (Part 3).

Run from anywhere:
    python notebooks/build_notebooks.py

Mirrors the Part-2 notebook design (self-contained tasks, 24 h-safe via an
adaptive guard, resume-safe to Google Drive, reuse the tested helpers rather
than re-implementing). Each notebook clones THREE repos — Part-1 (inherited
``src/``), Part-2 (``rhp/`` + pinned ``configs/panel.yaml`` + detection
artifacts) and Part-3 (this repo) — wires the paths, and runs the experiment
scripts with ``RFM_RESULTS_DIR`` pointing at Drive.

The pre-registration gate is honored: notebook 00 CHECKS for the researcher-
authored ``configs/preregistration.yaml`` and refuses to proceed without it
(this generator never writes or defaults that file, task §1.2).
"""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent

# Nominal per-model wall-clock estimates (A100/L4, bf16). The real guarantee is
# the adaptive time_guard (23 h cap) inside every model loop, which won't start a
# model that cannot finish in time. Tune after the first session.
HARD_CAP_H = 23


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(text)}


def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    lines = text.split("\n")
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def notebook(cells: list[dict], gpu: bool = True) -> dict:
    meta = {
        "accelerator": "GPU" if gpu else "None",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


# ---------------------------------------------------------------------------
# Shared setup cells
# ---------------------------------------------------------------------------

SETUP_GPU_DRIVE = code(r"""
# Cell 0 — GPU check + Google Drive + results dir on Drive
import subprocess, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # less fragmentation
print(subprocess.check_output('nvidia-smi', shell=True).decode())

USE_DRIVE = True   # keep True so results survive a disconnect and resume
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RESULTS_DIR = '/content/drive/MyDrive/rfm_results'
else:
    RESULTS_DIR = '/content/rfm_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ['RFM_RESULTS_DIR'] = RESULTS_DIR    # scripts honor this override
print('Results dir:', RESULTS_DIR)
""")

SETUP_PIP = code(r"""
%%bash
# Cell 1 — dependencies. Pin transformers to match the Part-1/Part-2 artifacts
# so a captured/patched value is bit-compatible with the detection artifacts.
pip install -q transformers==4.47.0 accelerate==1.13.0 bitsandbytes==0.49.2
pip install -q "numpy==2.0.2" "scipy==1.16.3" pyyaml huggingface_hub sentencepiece
echo 'Install complete.'
""")

SETUP_CLONE = code(r"""
# Cell 2 — tokens + clone THREE repos
#   Part 1: inherited src/ (model_loader, activation_patching, stats_utils).
#   Part 2: rhp/, configs/panel.yaml (PINNED SHAs), detection artifacts.
#   Part 3: this repo (failure_mech/, scripts/, configs/).
import os, subprocess

GITHUB_TOKEN = ""          # ghp_...  (only for private repos)
HF_TOKEN     = ""          # hf_...   (needed for gated models: Llama/Gemma)
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN

PART1 = dict(owner='CengizhanBayram',
             name='Does-RoPE-Prevent-or-Degrade-Retrieval-Heads-A-Mechanistic-Analysis-Across-Model-Families',
             dir='/content/rope-part1')
PART2 = dict(owner='CengizhanBayram', name='retrieval-head-profile', dir='/content/rope-part2')
PART3 = dict(owner='CengizhanBayram', name='retrieval-failure-mechanism', dir='/content/rope-part3')

def clone(repo):
    tok = GITHUB_TOKEN
    pub  = f"https://github.com/{repo['owner']}/{repo['name']}.git"
    auth = f"https://x-access-token:{tok}@github.com/{repo['owner']}/{repo['name']}.git" if tok else pub
    if not os.path.isdir(repo['dir']):
        r = subprocess.run(['git', 'clone', auth, repo['dir']], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).replace(tok or '___', '***'))
        if tok:
            subprocess.run(['git', '-C', repo['dir'], 'remote', 'set-url', 'origin', pub])
    else:
        subprocess.run(['git', '-C', repo['dir'], 'pull'], capture_output=True, text=True)
    print('ready:', repo['dir'])

for r in (PART1, PART2, PART3):
    clone(r)
""")

SETUP_PATHS = code(r"""
# Cell 3 — env wiring + HF login. Part-3 panel.py resolves the sibling repos from
# these env vars; RFM_RESULTS_DIR redirects all outputs to Drive.
import os, sys, subprocess
os.environ['RHP_PART1_REPO'] = '/content/rope-part1'
os.environ['RHP_PART2_REPO'] = '/content/rope-part2'
PART3 = '/content/rope-part3'
sys.path.insert(0, PART3 + '/src')
sys.path.insert(0, PART3 + '/scripts')

if os.environ.get('HF_TOKEN'):
    try:
        from huggingface_hub import login
        login(os.environ['HF_TOKEN'])
    except Exception as e:
        print('HF login skipped:', e)

def run(argv):
    '''Run a Part-3 script as a subprocess in the Part-3 dir, streaming output.'''
    env = dict(os.environ)
    p = subprocess.Popen(['python'] + argv, cwd=PART3, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='')
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f'script failed ({p.returncode}): {argv}')

print('Setup OK. PART3 =', PART3, '| RESULTS_DIR =', os.environ['RFM_RESULTS_DIR'])
""")

PREREG_GATE = code(r"""
# Cell 4 — PRE-REGISTRATION GATE (task §1.2). This codebase NEVER creates or
# defaults configs/preregistration.yaml. The researcher must author + commit it
# to the Part-3 repo BEFORE any analysis. This cell only checks it and runs the
# gate; a missing file or key fails loudly, naming what is missing.
import sys
sys.path.insert(0, '/content/rope-part3/src')
from failure_mech import prereg as PR

prereg_path = '/content/rope-part3/configs/preregistration.yaml'
try:
    cfg = PR.load_prereg(prereg_path)
    print('Pre-registration OK. All', len(PR.REQUIRED_KEYS), 'required keys present.')
    print('  breaking band :', PR.get(cfg, 'breaking_band_accuracy'))
    print('  stage1 / stage2:', PR.get(cfg, 'sampling.stage1_n_per_cell'),
          '/', PR.get(cfg, 'sampling.stage2_topup_n_breaking_cells'))
    print('  mode precedence:', PR.get(cfg, 'signature_rules.mode_precedence'))
except PR.PreregError as e:
    print('PREREG GATE FAILED — no analysis may run until this is fixed:\n')
    print(e)
    print('\nAuthor configs/preregistration.yaml in your Part-3 repo with these keys:')
    for k in PR.REQUIRED_KEYS:
        print('  -', k)
    raise
""")


def setup_cells(extra_intro: str = "") -> list[dict]:
    intro = ("### Setup — run cells 0–4 once per session\n"
             "Mounts Drive, installs the pinned stack, clones Part-1/2/3, wires paths, "
             "and runs the **pre-registration gate**. Edit the repo owners and paste "
             "your `HF_TOKEN` (gated Llama/Gemma) in Cell 2 before running.")
    if extra_intro:
        intro += "\n\n" + extra_intro
    return [md(intro), SETUP_GPU_DRIVE, SETUP_PIP, SETUP_CLONE, SETUP_PATHS, PREREG_GATE]


MODELS = ["llama31_8b_instruct", "gemma2_9b_it", "mistral_7b_instruct",
          "qwen25_7b_instruct", "olmo2_7b_instruct", "phi35_mini",
          "qwen25_3b_instruct"]


# ---------------------------------------------------------------------------
# 00 — setup + guardrails (the researcher's first-run gate)
# ---------------------------------------------------------------------------

def nb_00() -> dict:
    cells = [md(
        "# 00 · Setup & Guardrails\n"
        "Validates the skeleton BEFORE any experiment burns GPU time. Runs the CPU "
        "guardrail tests (pre-registration red-test, four-bucket partition, stats, "
        "grading, probe skeleton alignment) and then the **eager-reference capture "
        "check on the four PINNED models** — the sole arbiter of the manual "
        "attention row (recompute-RoPE + Gemma-2 softcap/query-scale). Trust no "
        "capture number until this passes on all four (esp. Gemma-2).")]
    cells += setup_cells()
    cells.append(md("## Guardrail tests (CPU) — prereg gate, stats, classify, grading, probes"))
    cells.append(code(r"""
# Runs the tests that need no GPU (tiny-model tests are skipped if offline).
run(['-m', 'pytest', 'tests/test_prereg.py', 'tests/test_stats.py',
     'tests/test_grading.py', 'tests/test_classify.py', 'tests/test_probes.py', '-q'])
"""))
    cells.append(md("## Eager-reference capture check on the 7 PINNED models (§4.3, §10)\n"
                    "For each model: load at its pinned SHA in **eager** (Gemma-2 requires "
                    "eager for softcapping), then confirm the manual row matches the model's "
                    "own attention row to < 1e-3 for several detected retrieval heads. This "
                    "exercises every attention pipeline in the panel: standard GQA "
                    "(Llama/Qwen/Mistral), Gemma-2 softcap + query-scale + sliding window, "
                    "OLMo-2 QK-norm, and Phi-3 fused-qkv + partial rotary. **This is the "
                    "arbiter — trust no capture number until every model prints PASS.**"))
    cells.append(code(r"""
import numpy as np, torch
from failure_mech import panel as P, detect, capture as CAP
from scripts import _common as C

paths = C.load_paths_cfg(); panel_reg = P.load_panel(paths); P.ensure_reuse_on_path(paths)
TEXT = "The access code for the golden lantern is K7QW2Z. Remember it well."

for key in %(models)s:
    model, tok, mcfg = P.load_model(paths, panel_reg, key,
                                    attn_implementation='eager', dtype='bfloat16')
    ids = tok(TEXT, add_special_tokens=True)['input_ids']
    det = detect.load_detection(P.detection_dir(paths), P.resolve_model_key(paths, key),
                                int(paths['detection_artifacts']['seed']))
    heads = det.top_k_heads(3, detector='argmax')
    worst = 0.0
    for (l, h) in heads:
        ref = CAP.eager_reference_row(model, ids, l, h)
        # manual row via the library machinery
        with torch.no_grad(), CAP._QProjTap(model, {l}) as tap:
            pos = torch.arange(0, len(ids)).unsqueeze(0).to(next(model.parameters()).device)
            out = model(input_ids=torch.tensor([ids]).to(pos.device), position_ids=pos, use_cache=True)
            cos, sin = CAP._rope_cos_sin(model, pos)
            nH, nKV = P.head_counts(model); hd = tap.store[l].shape[-1]//nH
            ql = tap.store[l][:, -1, :].view(1, nH, hd)
            qr = CAP._apply_rope(ql, cos[:, -1, :], sin[:, -1, :])
            manual = CAP._row_for_head(model, l, h, qr, CAP._layer_keys(out.past_key_values, l),
                                       len(ids)-1, nH, nKV, P.attention_scale(model),
                                       P.attn_logit_softcap(model))
        worst = max(worst, float(np.max(np.abs(manual - ref))))
    status = 'PASS' if worst < 1e-3 else 'FAIL'
    print(f'{key:24s} eff_attn={mcfg["effective_attn"]:6s} max|manual-eager|={worst:.2e}  {status}')
    del model; torch.cuda.empty_cache()
""" % {"models": json.dumps(MODELS)}))
    return notebook(cells)


# ---------------------------------------------------------------------------
# Experiment-loop notebooks (E1..E5)
# ---------------------------------------------------------------------------

def _model_loop(title, subtitle, script_argv_tmpl, first_est_h, skip_check) -> list[dict]:
    """A resume-safe, 24 h-guarded model loop that shells out to a Part-3 script.
    ``skip_check`` is a python expression (given ``key``, ``RESULTS_DIR``) that is
    True when the model's output already exists."""
    return [
        md(f"## {title}\n{subtitle}"),
        code(r"""
import os, time
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
MODELS = %(models)s
start = time.time(); model_times = []
for key in MODELS:
    if %(skip)s:
        print(key, '-> output exists, skip'); continue
    ok, elapsed_h, est_h = C.time_guard(start, model_times, first_est_h=%(est).1f)
    if not ok:
        print(f'STOP before {key}: {elapsed_h:.1f}h + est {est_h:.1f}h > %(cap)d h cap. '
              f'Re-run to resume (finished models are skipped).'); break
    t0 = time.time()
    run(%(argv)s)
    model_times.append((time.time()-t0)/3600.0)
    print(key, 'done in', round(model_times[-1], 2), 'h')
""" % {"models": json.dumps(MODELS), "skip": skip_check, "est": first_est_h,
        "cap": HARD_CAP_H, "argv": script_argv_tmpl}),
    ]


def nb_e1() -> dict:
    cells = [md(
        "# 01 · E1 — Breaking Surface (§5)\n"
        "Sweeps the 140-cell grid per model (context x position x distractor "
        "config), grades greedy generations, and records the breaking cells. "
        "**Resume-safe**: E1 checkpoints every cell to Drive; a re-run skips "
        "finished cells. Two-stage (cheap pass, then top-up of breaking cells) is "
        "handled by the script (`--stage auto`).")]
    cells += setup_cells()
    cells += _model_loop(
        "E1 for all four models",
        "One model per iteration; the guard won't start a model that can't finish "
        "under the 23 h cap.",
        "['scripts/e1_breaking_surface.py', '--model', key, '--stage', 'auto']",
        first_est_h=5.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e1_breaking_cells_{key}.json')")
    return notebook(cells)


def nb_e2() -> dict:
    cells = [md(
        "# 02 · E2 — Mechanistic Signatures (§6)\n"
        "Within breaking cells, forms matched success/failure pairs (identical "
        "token skeleton), captures retrieval-head attention masses at the answer "
        "step, builds the success reference distribution, classifies failures into "
        "the four buckets, and cross-tabs behavior x mechanism. Needs E1 outputs.")]
    cells += setup_cells()
    cells += _model_loop(
        "E2 for all four models",
        "Capture runs sdpa (Gemma-2 auto-eager for softcapping).",
        "['scripts/e2_signatures.py', '--model', key]",
        first_est_h=4.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e2_signatures_{key}.json')")
    return notebook(cells)


def nb_e3() -> dict:
    cells = [md(
        "# 03 · E3 — Bidirectional Causal Patching (§7)\n"
        "Over E2's matched pairs, for each k in {1,5,10}: repair / break / random-"
        "control / no-patch / self-patch. Self-patch MUST be token-identical — the "
        "script ABORTS if not (§4.4). All four repair/break outcomes are reported "
        "neutrally. Needs E2 outputs.")]
    cells += setup_cells()
    cells += _model_loop(
        "E3 for all four models",
        "Heavier than E2 (5 runs/pair x 3 k). The guard protects the 24 h budget.",
        "['scripts/e3_causal.py', '--model', key]",
        first_est_h=6.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e3_causal_{key}.json')")
    return notebook(cells)


def nb_e4_e5() -> dict:
    cells = [md(
        "# 04 · E4 Families + E5 Robustness (§8, §9)\n"
        "E4 assembles the cross-family table and finalizes the authoritative BH "
        "across {4 models x 2 directions} for the causal decision (+ Gemma-2 "
        "local/global head annotation). E5 recomputes the E2 headlines on the Wu "
        "head list, at the sensitivity k, over seed repeats (mean +/- range), and "
        "with answer_steps=3. Needs E2/E3 outputs.")]
    cells += setup_cells()
    cells.append(md("## E4 — family comparison table + causal decision"))
    cells.append(code(
        "run(['scripts/e4_families.py', '--models'] + %s)" % json.dumps(MODELS)))
    cells += _model_loop(
        "E5 — robustness per model",
        "Each variant is a clean subprocess with the pinned model.",
        "['scripts/e5_robustness.py', '--model', key]",
        first_est_h=8.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e5_robustness_{key}.json')")
    cells.append(md("## Inspect the headline table"))
    cells.append(code(r"""
import json, os
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
with open(f'{RESULTS_DIR}/e4_families.json') as f:
    e4 = json.load(f)
print('causal_decision (effect = BH-rejected AND margin met):')
print(json.dumps(e4['causal_decision'], indent=2)[:2000])
"""))
    return notebook(cells)


def main():
    outputs = {
        "00_setup_and_guardrails.ipynb": nb_00(),
        "01_e1_breaking_surface.ipynb": nb_e1(),
        "02_e2_signatures.ipynb": nb_e2(),
        "03_e3_causal.ipynb": nb_e3(),
        "04_e4_e5_analysis.ipynb": nb_e4_e5(),
    }
    for name, nb in outputs.items():
        path = NB_DIR / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
