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
# NOTE: Colab often ships a newer transformers; the pin below downgrades it. If
# the version check in the next cell shows != 4.47.0, RESTART THE RUNTIME and
# re-run (a pre-imported transformers won't downgrade in-place). The capture code
# is version-robust either way, but 4.47.0 is what the artifacts were made with.
pip install -q "transformers==4.47.0" "accelerate==1.13.0" "bitsandbytes==0.49.2"
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
sys.path.insert(0, PART3 + '/src')       # failure_mech
sys.path.insert(0, PART3 + '/scripts')   # _common
import _common as C                       # shared helpers (time_guard, load_paths_cfg, ...)

if os.environ.get('HF_TOKEN'):
    try:
        from huggingface_hub import login
        login(os.environ['HF_TOKEN'])
    except Exception as e:
        print('HF login skipped:', e)

def run(argv):
    '''Run a Part-3 script as a subprocess in the Part-3 dir, streaming output.'''
    env = dict(os.environ)
    p = subprocess.Popen([sys.executable] + argv, cwd=PART3, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='')
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f'script failed ({p.returncode}): {argv}')

import transformers, torch
_vok = transformers.__version__.startswith('4.47')
print('Setup OK. C, run() ready. PART3 =', PART3, '| RESULTS_DIR =', os.environ['RFM_RESULTS_DIR'])
print(f'transformers={transformers.__version__} torch={torch.__version__}',
      '' if _vok else '  <-- NOT the pinned 4.47.0; RESTART RUNTIME after Cell 1 for a '
                       'provenance-clean run (code still works, recorded in provenance).')
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


# Idempotent shell_share enable. Shared by notebooks 06 (E1+E2) and 07 (E3+E4) —
# BOTH need it because E3 rebuilds the probes from grid.yaml, and a fresh clone
# without shell_share would produce different probes than E1/E2 used.
SHELL_SHARE_CELL = code(r"""
gp = '/content/rope-part3/configs/grid.yaml'
txt = open(gp).read()
OLD = 'similarity: ["shell_same", "shell_diff"]'          # the active (uncommented) grid line
NEW = 'similarity: ["shell_same", "shell_diff", "shell_share"]'
if OLD in txt:                                            # OLD matches only the active line, not the comment
    open(gp, 'w').write(txt.replace(OLD, NEW, 1))
    print('shell_share ENABLED in grid.similarity (commit grid.yaml for a provenance-clean run).')
elif NEW in txt.replace('#', ''):                         # already uncommented
    print('shell_share already enabled.')
else:
    print('Could not find the grid.similarity line to patch — inspect configs/grid.yaml manually.')
# sanity: show the active line
for ln in open(gp):
    s = ln.strip()
    if s.startswith('similarity:'):
        print('active grid.similarity ->', s); break
""")


# ---------------------------------------------------------------------------
# 00 — setup + guardrails (the researcher's first-run gate)
# ---------------------------------------------------------------------------

def nb_00() -> dict:
    cells = [md(
        "# 00 · Setup & Guardrails\n"
        "Validates the skeleton BEFORE any experiment burns GPU time. Runs the CPU "
        "guardrail tests (pre-registration red-test, four-bucket partition, stats, "
        "grading, probe skeleton alignment) and then the **eager-reference capture "
        "check on the seven PINNED models** — the sole arbiter of the manual "
        "attention row (recompute-RoPE + Gemma-2 softcap/query-scale + OLMo-2 "
        "QK-norm + Phi-3 fused-qkv). Trust no capture number until this passes.")]
    cells += setup_cells()
    cells.append(md("## Guardrail tests (CPU) — prereg gate, stats, classify, grading, probes"))
    cells.append(code(r"""
# Runs the tests that need no GPU (tiny-model tests are skipped if offline).
run(['-m', 'pytest', 'tests/test_prereg.py', 'tests/test_stats.py',
     'tests/test_grading.py', 'tests/test_classify.py', 'tests/test_probes.py', '-q'])
"""))
    cells.append(md("## Eager-reference capture check on the 7 PINNED models (§4.3, §10)\n"
                    "Confirms the manual attention row matches the model's own attention row, "
                    "exercising every pipeline: standard GQA (Llama/Qwen/Mistral), Gemma-2 "
                    "softcap+query-scale+window, OLMo-2 QK-norm, Phi-3 fused-qkv+partial-rotary.\n\n"
                    "**Loaded in fp32** on purpose: this isolates *is the formula right?* from "
                    "bf16 rounding. bf16 has ~2⁻⁸ ≈ 4e-3 precision, so a bf16 model's own "
                    "attention differs from an fp32 recompute by a few e-3 — that is NOT a bug "
                    "and is far below the mass thresholds (m1 floor 0.02, m2 min 0.10). The "
                    "gate is: **fp32 max|manual−eager| < 1e-3 for every model.** The reported "
                    "bf16 delta is informational."))
    cells.append(code(r"""
import numpy as np, torch, gc
from failure_mech import panel as P, detect, capture as CAP
# C is provided by the setup cell (import _common as C).

paths = C.load_paths_cfg(); panel_reg = P.load_panel(paths); P.ensure_reuse_on_path(paths)
TEXT = "The access code for the golden lantern is K7QW2Z. Remember it well."

def manual_row(model, ids, l, h):
    '''Reproduce capture's manual row via the SAME library helpers (family-aware:
    q_proj/qkv_proj, q_norm, partial rotary). Never re-implement it here.'''
    nH, nKV = P.head_counts(model)
    with torch.no_grad(), CAP._QProjTap(model, {l}) as tap:
        pos = torch.arange(0, len(ids)).unsqueeze(0).to(next(model.parameters()).device)
        out = model(input_ids=torch.tensor([ids]).to(pos.device), position_ids=pos, use_cache=True)
        cos, sin = tap.pos_emb if tap.pos_emb is not None else CAP._rope_cos_sin(model, pos)
        qr = CAP.query_rotated_last(model, l, tap.store[l], cos[:, -1, :], sin[:, -1, :],
                                    nH, P.head_dim(model))
        return CAP._row_for_head(model, l, h, qr, CAP._layer_keys(out.past_key_values, l),
                                 len(ids) - 1, nH, nKV, P.attention_scale(model),
                                 P.attn_logit_softcap(model))

GATE_TOL = 1e-3
gate = {}
for key in %(models)s:
    try:
        # fp32 for the math gate (bf16 rounding would mask correctness at ~e-3).
        model, tok, mcfg = P.load_model(paths, panel_reg, key,
                                        attn_implementation='eager', dtype='float32')
        ids = tok(TEXT, add_special_tokens=True)['input_ids']
        det = detect.load_detection(P.detection_dir(paths), P.resolve_model_key(paths, key),
                                    int(paths['detection_artifacts']['seed']))
        worst, per_head = 0.0, []
        for (l, h) in det.top_k_heads(3, detector='argmax'):
            ref = CAP.eager_reference_row(model, ids, l, h)
            d = float(np.max(np.abs(manual_row(model, ids, l, h) - ref)))
            per_head.append({'layer': int(l), 'head': int(h), 'max_abs_diff': d})
            worst = max(worst, d)
        status = 'PASS' if worst < GATE_TOL else 'FAIL  <-- investigate before trusting capture'
        gate[key] = {'effective_attn': mcfg['effective_attn'], 'dtype': 'float32',
                     'max_abs_diff': worst, 'tolerance': GATE_TOL,
                     'passed': bool(worst < GATE_TOL), 'per_head': per_head,
                     'model_sha': mcfg.get('revision')}
        print(f'{key:22s} eff_attn={mcfg["effective_attn"]:6s} max|manual-eager|={worst:.2e}  {status}')
        del model
    except Exception as e:
        gate[key] = {'error': f'{type(e).__name__}: {str(e)[:200]}', 'passed': False}
        print(f'{key:22s} SKIPPED/ERROR: {type(e).__name__}: {str(e)[:120]}')
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# PERSIST the gate. A capture-correctness claim that lives only in a notebook's
# stdout is not auditable: it dies with the Colab session. This artifact is the
# evidence that every downstream mass number is trustworthy.
import json as _json, subprocess, datetime
prov = {
    'script': 'notebooks/00_A100_setup_and_guardrails.ipynb',
    'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd='/content/rope-part3',
                                 capture_output=True, text=True).stdout.strip(),
    'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
    'tolerance': GATE_TOL,
    'dtype': 'float32',
    'definition': 'max |manual attention row - eager output_attentions row| over the '
                  'top-3 argmax heads, last prompt position',
}
n_pass = sum(1 for v in gate.values() if v.get('passed'))
C.write_json(f'{RESULTS_DIR}/gate_eager_reference.json',
             {'provenance': prov, 'n_pass': n_pass, 'n_models': len(gate), 'models': gate})
print(f'\nwrote {RESULTS_DIR}/gate_eager_reference.json  ({n_pass}/{len(gate)} PASS)')
if n_pass < len(gate):
    print('*** GATE NOT CLEAN — do not trust capture numbers until every model passes. ***')
""" % {"models": json.dumps(MODELS)}))
    return notebook(cells)


# ---------------------------------------------------------------------------
# Experiment-loop notebooks (E1..E5)
# ---------------------------------------------------------------------------

def _model_loop(title, subtitle, script_argv_tmpl, first_est_h, skip_check,
                fresh_arg=None, overwrite_default=False, models=None) -> list[dict]:
    """A resume-safe, 24 h-guarded model loop that shells out to a Part-3 script.
    ``skip_check`` is a python expression (given ``key``, ``RESULTS_DIR``) that is
    True when the model's output already exists. ``OVERWRITE`` in the generated
    cell recomputes + overwrites ALL models (ignore existing outputs);
    ``fresh_arg`` (e.g. '--fresh') is appended to the script call in that mode so
    it also ignores per-cell checkpoints (E1). ``overwrite_default`` sets the
    cell's initial ``OVERWRITE`` value. ``models`` overrides the model list
    (default: the full panel)."""
    # ``models`` is either a literal list (embedded as JSON) or the NAME of a
    # variable defined in an earlier cell (embedded verbatim as an expression).
    # json.dumps()-ing a name would emit MODELS = "E5_MODELS" -- a string, which
    # `for key in MODELS` then iterates CHARACTER BY CHARACTER. That is a silent,
    # expensive failure: the loop runs, every "model" (E, 5, _, M, O, D, S) fails
    # its lookup, each failure is caught and logged as a skipped variant, and the
    # notebook finishes "successfully" having computed nothing.
    if models is None:
        models_src = json.dumps(MODELS)
    elif isinstance(models, str):
        models_src = models
    else:
        models_src = json.dumps(list(models))
    fresh = json.dumps([fresh_arg] if fresh_arg else [])
    return [
        md(f"## {title}\n{subtitle}"),
        code(r"""
import os, time, gc, torch
OVERWRITE = %(ovr)s   # True = recompute + OVERWRITE every model; False = skip finished (resume)
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
MODELS = %(models)s
FRESH = %(fresh)s
# A bare string here would be iterated character by character, and every "model"
# would fail its lookup and be logged as a skipped variant — a loop that appears
# to run fine while computing nothing. Fail immediately instead.
if not isinstance(MODELS, (list, tuple)) or not all(isinstance(m, str) for m in MODELS):
    raise TypeError(f'MODELS must be a list of model keys, got {MODELS!r}')
start = time.time(); model_times = []
for key in MODELS:
    if (not OVERWRITE) and (%(skip)s):
        print(key, '-> output exists, skip'); continue
    ok, elapsed_h, est_h = C.time_guard(start, model_times, first_est_h=%(est).1f)
    if not ok:
        print(f'STOP before {key}: {elapsed_h:.1f}h + est {est_h:.1f}h > %(cap)d h cap. '
              f'Re-run to resume (finished models are skipped).'); break
    t0 = time.time()
    try:
        run(%(argv)s + (FRESH if OVERWRITE else []))
        model_times.append((time.time() - t0) / 3600.0)
        print(key, 'done in', round(model_times[-1], 2), 'h')
    except Exception as e:
        # one model failing (OOM, gated w/o token, ...) must not lose the others;
        # it has no output file, so a re-run retries just this model.
        print(f'{key} FAILED after {(time.time()-t0)/3600:.2f}h: '
              f'{type(e).__name__}: {str(e)[:200]}  -- continuing to next model.')
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
""" % {"models": models_src, "skip": skip_check, "est": first_est_h,
        "cap": HARD_CAP_H, "argv": script_argv_tmpl, "fresh": fresh,
        "ovr": "True" if overwrite_default else "False"}),
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
        "E1 for all seven models",
        "One model per iteration; the guard won't start a model that can't finish "
        "under the 23 h cap.",
        "['scripts/e1_breaking_surface.py', '--model', key, '--stage', 'auto']",
        first_est_h=5.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e1_breaking_cells_{key}.json')",
        fresh_arg="--fresh")
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
        "E2 for all seven models",
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
        "E3 for all seven models",
        "Heavier than E2 (5 runs/pair x 3 k). The guard protects the 24 h budget.",
        "['scripts/e3_causal.py', '--model', key]",
        first_est_h=6.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e3_causal_{key}.json')")
    return notebook(cells)


def nb_e4_e5() -> dict:
    cells = [md(
        "# 04 · E4 Families + E5 Robustness (§8, §9)\n"
        "E4 assembles the cross-family table and finalizes the authoritative BH "
        "across {7 models x 2 directions} for the causal decision (+ Gemma-2 "
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


# ---------------------------------------------------------------------------
# 05 — analyze the models that already broke (reuses existing E1 on Drive)
# ---------------------------------------------------------------------------

def nb_05_analyze_breaking() -> dict:
    cells = [md(
        "# 05 · Analyze the models that already broke\n"
        "Your first run showed **OLMo-2** breaking naturally (15 cells, mean acc "
        "0.43) and **Gemma-2** collapsing on long-context cells, but E2/E3 only "
        "completed for the near-ceiling models. This notebook runs **E2 -> E3 -> "
        "E4** for every model that is MISSING its output, **reusing the existing "
        "E1 results on Drive** (no E1 re-run, no config change). OLMo-2 is the "
        "one to watch.")]
    cells += setup_cells()
    cells.append(md("## Which models have breaking cells worth analyzing?"))
    cells.append(code(r"""
import os, json, glob
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
print(f'{"model":24s} {"breaking":>8s} {"E2 done":>8s}')
for f in sorted(glob.glob(f'{RESULTS_DIR}/e1_breaking_cells_*.json')):
    m = os.path.basename(f)[len('e1_breaking_cells_'):-5]
    nb = len(json.load(open(f))['breaking_cells'])
    e2 = os.path.exists(f'{RESULTS_DIR}/e2_signatures_{m}.json')
    print(f'{m:24s} {nb:8d} {str(e2):>8s}')
print('\n-> E2/E3 below run only for models MISSING their output. '
      'Delete an e2/e3 json to force a re-run.')
"""))
    cells += _model_loop(
        "E2 — signatures for the missing models",
        "Runs only where `e2_signatures_{model}.json` is absent (OLMo-2, Gemma-2, "
        "Mistral, Phi from your run). Near-ceiling models just produce 0 cells fast.",
        "['scripts/e2_signatures.py', '--model', key]",
        first_est_h=4.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e2_signatures_{key}.json')")
    cells += _model_loop(
        "E3 — causal patching for the missing models",
        "Runs only where `e3_causal_{model}.json` is absent.",
        "['scripts/e3_causal.py', '--model', key]",
        first_est_h=6.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e3_causal_{key}.json')")
    cells.append(md("## E4 — refresh the family table + causal decision"))
    cells.append(code("run(['scripts/e4_families.py', '--models'] + %s)" % json.dumps(MODELS)))
    cells.append(md("## Read the headline signal"))
    cells.append(code(r"""
import json, os
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
for f in sorted(__import__('glob').glob(f'{RESULTS_DIR}/e2_signatures_*.json')):
    if any(t in f for t in ('_wu','_ksens','_seed','_steps3')): continue
    d = json.load(open(f)); m = os.path.basename(f)[len('e2_signatures_'):-5]
    rates = {k: round(v,3) for k,v in d['sample_level']['bucket_rates'].items() if v>0}
    print(f'{m:24s} cells={len(d["cells_used"])} pairs={d["pairs_used"]}  buckets={rates}')
print()
e4 = json.load(open(f'{RESULTS_DIR}/e4_families.json'))
print('causal effects (effect = BH-rejected AND margin met):')
for k, per in e4.get('causal_decision', {}).items():
    for mdl, dirs in per.items():
        for d_, v in dirs.items():
            if v.get('effect'):
                print(f'  k={k} {mdl} {d_}: EFFECT (p={v["p"]}, margin_met={v["margin_met"]})')
print('  (none printed above = no pre-registered causal effect yet)')
"""))
    return notebook(cells)


# ---------------------------------------------------------------------------
# 06 — harder task: enable shell_share, clean re-run of the whole panel
# ---------------------------------------------------------------------------

def nb_06_shell_share_rerun() -> dict:
    cells = [md(
        "# 06 · Harder task (shell_share) — clean re-run: **E1 + E2**\n"
        "The strong models sit at ~0.99 accuracy on shell_same/shell_diff, so they "
        "produce no breaking cells. This notebook turns on the **key-overlap "
        "distractor** (`shell_share`) and re-runs **E1 -> E2** for the whole panel, "
        "**overwriting** any stale results.\n\n"
        "**E3 + E4 now live in notebook 07** — E3 is the heaviest experiment and was "
        "overrunning the Colab session when bundled here. Run 06, then 07.\n\n"
        "> **This is a design change** — note `shell_share` in your pre-registration, "
        "and ideally commit it to `grid.yaml` (so its hash is in provenance) rather "
        "than only patching it here. The grid grows from 140 to ~200 cells/model.")]
    cells += setup_cells()
    cells.append(md("## 1) Enable shell_share in the cloned grid.yaml"))
    cells.append(SHELL_SHARE_CELL)
    cells.append(md("## 2) Clear results for a clean re-run (config changed)\n"
                    "The old 140-cell results are incompatible with the new grid. This wipes "
                    "`RFM_RESULTS_DIR`. **Set `CONFIRM = True` to run it.**"))
    cells.append(code(r"""
import os, shutil
CONFIRM = False   # <-- set True to actually clear results
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
if CONFIRM:
    shutil.rmtree(RESULTS_DIR, ignore_errors=True); os.makedirs(RESULTS_DIR, exist_ok=True)
    print('cleared', RESULTS_DIR)
else:
    print('SKIPPED clear (CONFIRM=False). OVERWRITE=True below still recomputes every '
          'model, but clearing also removes orphaned files from older runs.')
"""))
    cells += _model_loop(
        "3) E1 — breaking surface (harder grid)",
        "~200 cells/model now. `OVERWRITE=True` + `--fresh` recompute and overwrite "
        "every cell — every model must log `200 cells`, none should say `skip`.",
        "['scripts/e1_breaking_surface.py', '--model', key, '--stage', 'auto']",
        first_est_h=6.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e1_breaking_cells_{key}.json')",
        fresh_arg="--fresh", overwrite_default=True)
    cells += _model_loop(
        "4) E2 — signatures",
        "Capture + four-bucket classification on the breaking cells.",
        "['scripts/e2_signatures.py', '--model', key]",
        first_est_h=4.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e2_signatures_{key}.json')",
        overwrite_default=True)
    cells.append(md("## 5) Check the run is clean, then continue in notebook 07\n"
                    "Every model must show `ncells=200` and `shell_share` in its sims — if any "
                    "shows 140, it was skipped and the panel is mixed."))
    cells.append(code(r"""
import json, glob, os, statistics
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
for f in sorted(glob.glob(f'{RESULTS_DIR}/e1_surface_*.json')):
    m = os.path.basename(f)[len('e1_surface_'):-5]
    cells_ = json.load(open(f))['cells']
    sims = sorted({c['axes']['similarity'] for c in cells_.values()})
    accs = [c['accuracy'] for c in cells_.values()]
    brk = sum(1 for a in accs if 0.2 <= a <= 0.8)
    ok = (len(cells_) == 200 and 'shell_share' in sims)
    print(f'{m:24s} ncells={len(cells_):3d} mean={statistics.mean(accs):.3f} '
          f'breaking={brk:3d} {"OK" if ok else "<-- STALE/MIXED"}')
print('\nIf all OK -> run notebook 07 (E3 causal + E4).')
"""))
    return notebook(cells)


# ---------------------------------------------------------------------------
# 07 — E3 causal + E4 (split out of 06: E3 is the heaviest experiment)
# ---------------------------------------------------------------------------

def nb_07_e3_e4() -> dict:
    cells = [md(
        "# 07 · E3 Causal Patching + E4 Families\n"
        "Split out of notebook 06 because **E3 is the heaviest experiment** "
        "(3 k-values x ~8 forwards per pair x hundreds of pairs) and 06 was "
        "overrunning the Colab session.\n\n"
        "**Run this AFTER 06 finishes E1+E2.** It reads E1/E2 outputs from Drive.\n\n"
        "> It re-enables `shell_share` in `grid.yaml` because **E3 rebuilds the probes** "
        "— a fresh clone without `shell_share` would generate different probes than "
        "E1/E2 did. (Committing `shell_share` to `grid.yaml` makes this a no-op.)")]
    cells += setup_cells()
    cells.append(md("## 1) Re-enable shell_share (must match the grid E1/E2 used)"))
    cells.append(SHELL_SHARE_CELL)
    cells.append(md("## 2) Sanity: E1/E2 outputs present and on the 200-cell grid?"))
    cells.append(code(r"""
import json, glob, os
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
bad = 0
for f in sorted(glob.glob(f'{RESULTS_DIR}/e1_surface_*.json')):
    m = os.path.basename(f)[len('e1_surface_'):-5]
    cells_ = json.load(open(f))['cells']
    sims = sorted({c['axes']['similarity'] for c in cells_.values()})
    e2 = os.path.exists(f'{RESULTS_DIR}/e2_signatures_{m}.json')
    ok = (len(cells_) == 200 and 'shell_share' in sims and e2)
    bad += (not ok)
    print(f'{m:24s} ncells={len(cells_):3d} e2={str(e2):5s} {"OK" if ok else "<-- fix in 06 first"}')
if bad:
    raise SystemExit('E1/E2 not clean for every model — re-run notebook 06 with OVERWRITE=True.')
print('\nAll clean. Proceeding to E3.')
"""))
    cells += _model_loop(
        "3) E3 — bidirectional causal patching",
        "The heavy one. Self-patch must stay token-identical (the script ABORTS "
        "otherwise). The 23 h guard stops before a model that can't finish; re-run "
        "with `OVERWRITE=False` to resume the remaining models.",
        "['scripts/e3_causal.py', '--model', key]",
        first_est_h=6.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e3_causal_{key}.json')",
        overwrite_default=True)
    cells.append(md("## 4) E4 — family table + authoritative causal decision"))
    cells.append(code("run(['scripts/e4_families.py', '--models'] + %s)" % json.dumps(MODELS)))
    cells.append(md("## 5) Headline"))
    cells.append(code(r"""
import json, os, glob
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
for f in sorted(glob.glob(f'{RESULTS_DIR}/e2_signatures_*.json')):
    if any(t in f for t in ('_wu','_ksens','_seed','_steps3')): continue
    d = json.load(open(f)); m = os.path.basename(f)[len('e2_signatures_'):-5]
    r = {k: round(v,3) for k,v in d['sample_level']['bucket_rates'].items() if v>0}
    print(f'{m:24s} cells={len(d["cells_used"]):2d} pairs={d["pairs_used"]:4d}  {r}')
print()
e4 = json.load(open(f'{RESULTS_DIR}/e4_families.json'))
print('causal effects (effect = BH-rejected AND margin met):')
hit = False
for k, per in e4.get('causal_decision', {}).items():
    for mdl, dirs in per.items():
        for d_, v in dirs.items():
            if v.get('effect'):
                hit = True; print(f'  k={k} {mdl} {d_}: EFFECT (p={v["p"]})')
if not hit:
    print('  none met the pre-registered criteria')
"""))
    return notebook(cells)


# ---------------------------------------------------------------------------
# 08 — E3 backfill for the models 07 couldn't finish in 24 h, then E4
# ---------------------------------------------------------------------------

# Edit this list to whichever models still have a stale/small-N E3 (compare each
# e3_causal_{m}.json 'n_pairs' to e2_signatures_{m}.json 'pairs_used'; a mismatch
# means E3 didn't finish for that model). From the clean shell_share run: gemma
# and qwen-3b were the two that overran.
E3_BACKFILL_MODELS = ["gemma2_9b_it", "qwen25_3b_instruct"]


def nb_08_e3_backfill() -> dict:
    cells = [md(
        "# 08 · E3 backfill (models 07 couldn't finish) + E4\n"
        "E3 over all 7 didn't fit in one 24 h session, so this notebook runs E3 for "
        f"**just the leftover models** ({', '.join(E3_BACKFILL_MODELS)}) and then "
        "re-runs **E4 over the full panel**.\n\n"
        "Edit `MODELS` in the E3 cell if a different set is outstanding — a model's "
        "E3 is stale when `e3_causal_{m}.json` `n_pairs` ≠ `e2_signatures_{m}.json` "
        "`pairs_used`. `OVERWRITE=True` overwrites the stale small-N E3 files.")]
    cells += setup_cells()
    cells.append(md("## 1) Re-enable shell_share (E3 rebuilds probes from grid.yaml)"))
    cells.append(SHELL_SHARE_CELL)
    cells.append(md("## 2) Which models still need E3? (n_pairs mismatch = stale)"))
    cells.append(code(r"""
import json, os
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
for m in %(all)s:
    try:
        e2 = json.load(open(f'{RESULTS_DIR}/e2_signatures_{m}.json'))['pairs_used']
    except Exception:
        print(f'{m:24s} NO E2 — run 06 first'); continue
    p = f'{RESULTS_DIR}/e3_causal_{m}.json'
    if not os.path.exists(p):
        print(f'{m:24s} E2_pairs={e2:4d}  E3 MISSING  <-- backfill'); continue
    n3 = (json.load(open(p))['k'].get('10') or {}).get('n_pairs')
    print(f'{m:24s} E2_pairs={e2:4d}  E3_n={n3}  {"STALE <-- backfill" if n3 != e2 else "ok"}')
""" % {"all": json.dumps(MODELS)}))
    cells += _model_loop(
        "3) E3 — backfill the leftover models",
        "Only the models listed here (edit `MODELS` if needed). `OVERWRITE=True` so "
        "a stale small-N E3 file gets overwritten. The 23 h guard still protects the "
        "session.",
        "['scripts/e3_causal.py', '--model', key]",
        first_est_h=6.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e3_causal_{key}.json')",
        overwrite_default=True, models=E3_BACKFILL_MODELS)
    cells.append(md("## 4) E4 over the FULL panel (now that every E3 is present)"))
    cells.append(code("run(['scripts/e4_families.py', '--models'] + %s)" % json.dumps(MODELS)))
    cells.append(md("## 5) Headline"))
    cells.append(code(r"""
import json, os, glob
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
e4 = json.load(open(f'{RESULTS_DIR}/e4_families.json'))
print('causal effects (effect = BH-rejected AND margin met):')
hit = False
for k, per in e4.get('causal_decision', {}).items():
    for mdl, dirs in per.items():
        for d_, v in dirs.items():
            if v.get('effect'):
                hit = True; print(f'  k={k} {mdl:22s} {d_:6s} EFFECT (p={v["p"]})')
if not hit:
    print('  none met the pre-registered criteria')
"""))
    return notebook(cells)


# ---------------------------------------------------------------------------
# 09 — E2 re-run that RECORDS its pair list (prerequisite for any new E3)
# ---------------------------------------------------------------------------

def nb_09_e2_pairs() -> dict:
    cells = [md(
        "# 09 · E2 re-run — record the pair list  ·  **needs A100**\n"
        "**Not a re-analysis. The pre-registration is untouched** (M2 floor stays at "
        "the registered 0.10). This run is byte-identical in configuration to the one "
        "that produced the current E2 artifacts; the only difference is that it now "
        "*writes down* which matched pairs it measured (`pairs_by_cell`).\n\n"
        "**Why it is needed.** E3 used to re-derive the pairs by re-grading. Grading "
        "batches through `_generate_batch_adaptive`, which halves the batch on OOM — a "
        "different batch composition changes the left-padding, and a greedy token can "
        "flip at the margin. So E2 measured the mechanism on **614** pairs while E3 "
        "measured causality on **625**. Mechanism and causality must be measured on "
        "the *same* sample. E3 now READS this list and aborts if it is missing.\n\n"
        "**Read the reproduction check in step 3.** The re-run should reproduce the "
        "old bucket rates exactly. If it does not, E2 is not reproducible run-to-run "
        "(the adaptive-batching path is the suspect) — that is a finding in itself, "
        "and it must be resolved before anything is called confirmatory.")]
    cells += setup_cells()
    cells.append(md(
        "## 1) Back up the pre-fix artifacts (idempotent — safe to re-run every session)\n"
        "This does **not** overwrite a backup that already exists. The `.pre_pairfix` "
        "files are the only surviving record of the pre-fix numbers, and the "
        "reproduction check in step 4 reads them **from disk** — so the check still "
        "works in a *later* session, after this one has been recycled."))
    cells.append(code(r"""
import json, os, glob, shutil
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']

def is_primary(f):
    return not any(t in f for t in ('_wu','_ksens','_seed','_steps3','_m2sens'))

for f in sorted(glob.glob(f'{RESULTS_DIR}/e2_signatures_*.json')):
    if not is_primary(f): continue
    bak = f + '.pre_pairfix'
    if not os.path.exists(bak):          # NEVER clobber an existing backup: a second
        shutil.copy(f, bak)              # session would back up the RE-RUN as if it
                                         # were the original and destroy the evidence.

print(f'{"model":22s} {"pairs":>6s}  pairs_by_cell?  backup')
for f in sorted(glob.glob(f'{RESULTS_DIR}/e2_signatures_*.json')):
    if not is_primary(f): continue
    m = os.path.basename(f)[len('e2_signatures_'):-5]
    d = json.load(open(f))
    print(f'{m:22s} {d["pairs_used"]:6d}  {str("pairs_by_cell" in d):14s} '
          f'{"yes" if os.path.exists(f + ".pre_pairfix") else "MISSING"}')

# A model is DONE only when its E2 artifact carries the pair list. Every model
# already has an e2_signatures_*.json from the pre-fix runs, so testing mere file
# existence would skip the entire panel and this notebook would do nothing.
def _pairs_recorded(key):
    p = f'{RESULTS_DIR}/e2_signatures_{key}.json'
    if not os.path.exists(p):
        return False
    with open(p) as fh:
        return 'pairs_by_cell' in json.load(fh)

print('\nalready done (pair list recorded):',
      [m for m in %(panel)s if _pairs_recorded(m)] or 'none')
""" % {"panel": json.dumps(MODELS)}))
    cells.append(md("## 2) shell_share must match the grid the current results used"))
    cells.append(SHELL_SHARE_CELL)
    cells.append(md(
        "## 3) E2 — same config, now recording `pairs_by_cell`\n"
        "**This does not fit in one 24 h Colab session** (llama alone is ~1.4 h; gemma "
        "is far slower on eager attention). It is built to be run over several "
        "sessions: leave `OVERWRITE = False` and re-run this notebook until every model "
        "reports `pairs_by_cell`. A model is skipped only if it **already has its pair "
        "list** — *not* merely because an `e2_signatures_*.json` exists, since every "
        "model has one of those from the pre-fix runs."))
    cells += _model_loop(
        "E2 per model (resumable)",
        "Identical pre-registration; the M2 floor is NOT changed here.",
        "['scripts/e2_signatures.py', '--model', key]",
        first_est_h=4.0,
        # Resume on the PAIR LIST, not on file existence — see the note above.
        skip_check="_pairs_recorded(key)",
        overwrite_default=False)
    cells.append(md(
        "## 4) Reproduction check — did the identical re-run give the identical numbers?\n"
        "Reads the `.pre_pairfix` backups from disk, so it is valid in any later "
        "session. Run it once every model has been re-run."))
    cells.append(code(r"""
import json, os, glob
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
print(f'{"model":22s} {"pairs before -> after":24s} {"bucket rates":11s} verdict')
drift, pending = [], []
for bak in sorted(glob.glob(f'{RESULTS_DIR}/e2_signatures_*.json.pre_pairfix')):
    cur = bak[:-len('.pre_pairfix')]
    m = os.path.basename(cur)[len('e2_signatures_'):-5]
    old, new = json.load(open(bak)), json.load(open(cur))
    if 'pairs_by_cell' not in new:
        pending.append(m)
        print(f'{m:22s} {"(not re-run yet)":24s}')
        continue
    same_pairs = new['pairs_used'] == old['pairs_used']
    nb_, ob_ = new['sample_level']['bucket_rates'], old['sample_level']['bucket_rates']
    same_rates = all(abs(nb_.get(k,0)-ob_.get(k,0)) < 1e-9 for k in set(nb_)|set(ob_))
    ok = same_pairs and same_rates
    if not ok: drift.append(m)
    print(f'{m:22s} {old["pairs_used"]:>8d} -> {new["pairs_used"]:<12d} '
          f'{"identical" if same_rates else "CHANGED":11s} '
          f'{"reproduced" if ok else "*** DRIFT ***"}')
print()
if pending:
    print('still to re-run:', pending, '-> re-run section 3 (OVERWRITE=False resumes).')
if drift:
    print('*** E2 did NOT reproduce for:', drift)
    print('*** The pipeline is not deterministic run-to-run. Do NOT treat anything as')
    print('*** confirmatory until the source of the drift is found. Report this.')
elif not pending:
    print('E2 reproduced exactly. The recorded pair list is now the single source of')
    print('truth, and E3 will measure causality on exactly these pairs.')
"""))
    return notebook(cells)


# ---------------------------------------------------------------------------
# 10 — EXPLORATORY extended k-sweep (redundancy vs dissociation) + E4
# ---------------------------------------------------------------------------

def nb_10_ksweep() -> dict:
    cells = [md(
        "# 10 · Extended k-sweep — **EXPLORATORY**  ·  needs A100\n"
        "> ### Status: post-hoc, result-driven extension. NOT pre-registered.\n"
        "> The pre-registered primary sweep is **k ∈ {1, 5, 10}**. k ∈ {20, 30} was "
        "added *after* seeing that the curves had not saturated. Whatever comes out — "
        "redundancy or dissociation — it is reported as **exploratory** in the paper, "
        "with this notebook and the dated amendment commit as its provenance. Labelling "
        "it honestly does not weaken the claim; it is what makes the claim usable.\n\n"
        "**What it resolves.** At k ≤ 10 the break-flip curve was still climbing for "
        "qwen3b (0.50), phi (0.39), olmo (0.22) but flat for llama (0.07) and mistral "
        "(0.05). \"llama/mistral have no causal effect\" is not defensible from curves "
        "cut at k=10: mistral has **82** retrieval heads and we patched 10 of them. "
        "That is an under-dosed drug, not a null.\n\n"
        "* curve keeps rising → the null was a **k-ceiling artifact (redundancy)**;\n"
        "* curve stays flat while others saturate → a **genuine dissociation**.\n\n"
        "**Also watch the tie diagnostic.** gemma2 has 13 heads tied at score 1.000, so "
        "its k=10 set was 10 arbitrary members of a 13-way tie — its k=10 null was never "
        "interpretable. k=20/30 covers the whole tied block.\n\n"
        "**Prerequisite: notebook 09.** E3 reads E2's recorded pair list and will abort "
        "without it.")]
    cells += setup_cells()
    cells.append(md("## 1) shell_share must match the grid E1/E2 used (E3 rebuilds probes)"))
    cells.append(SHELL_SHARE_CELL)
    cells.append(md("## 2) Preconditions: extended k_list live, and E2 recorded its pairs"))
    cells.append(code(r"""
import yaml, json, os, glob
k = yaml.safe_load(open('/content/rope-part3/configs/e3.yaml'))['e3']['k_list']
print('e3.k_list =', k)
assert 20 in k and 30 in k, 'pull the latest configs/e3.yaml (k_list must include 20, 30)'

RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
missing = []
for f in sorted(glob.glob(f'{RESULTS_DIR}/e2_signatures_*.json')):
    if any(t in f for t in ('_wu','_ksens','_seed','_steps3','_m2sens')): continue
    if 'pairs_by_cell' not in json.load(open(f)):
        missing.append(os.path.basename(f))
if missing:
    raise SystemExit('These E2 artifacts predate the pair-set fix: %s\n'
                     'Run notebook 09 first — otherwise E3 would measure causality on a '
                     'different sample than E2 measured the mechanism on.' % missing)
print('OK — extended sweep active, and every E2 artifact carries its pair list.')
"""))
    cells += _model_loop(
        "3) E3 — extended k-sweep (exploratory)",
        "Recomputes E3 across k = 1,5,10,20,30, on E2's recorded pairs.",
        "['scripts/e3_causal.py', '--model', key]",
        first_est_h=8.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e3_causal_{key}.json')",
        overwrite_default=True)
    cells.append(md("## 4) E4 — family table + causal decision"))
    cells.append(code("run(['scripts/e4_families.py', '--models'] + %s)" % json.dumps(MODELS)))
    cells.append(md("## 5) The saturation curves, with the head-set tie flag"))
    cells.append(code(r"""
import json, os
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']
KS = ['1','5','10','20','30']
print(f'{"model":22s} ' + ' '.join(f'k={k:<5s}' for k in KS) + ' tie-broken k  trend')
for m in %(models)s:
    p = f'{RESULTS_DIR}/e3_causal_{m}.json'
    if not os.path.exists(p): continue
    d = json.load(open(p))
    ks, ties = d['k'], d.get('head_set_ties', {})
    row, vals = [], []
    for k in KS:
        v = (ks.get(k) or {}).get('break', {}).get('flip_rate')
        row.append('  -  ' if v is None else f'{v:.3f}')
        if v is not None: vals.append(v)
    tie_ks = ','.join(k for k in KS if (ties.get(k) or {}).get('arbitrary')) or '-'
    trend = ''
    if len(vals) >= 2:
        trend = 'RISING' if vals[-1] - vals[-2] > 0.02 else 'FLAT'
    print(f'{m:22s} ' + ' '.join(f'{r:<7s}' for r in row) + f' {tie_ks:12s}  {trend}')
print()
print('RISING at the top end  => the k=10 null was a k-ceiling artifact (redundancy).')
print('FLAT while others rise => mechanism-vs-causality dissociation.')
print('tie-broken k           => at that k the head SET was decided by sort order, not')
print('                          score; a null there says nothing about the heads.')
print('\nEXPLORATORY — report as such.')
""" % {"models": json.dumps(MODELS)}))
    return notebook(cells)


# ---------------------------------------------------------------------------
# 11 — E5 robustness
# ---------------------------------------------------------------------------

def nb_11_e5() -> dict:
    cells = [md(
        "# 11 · E5 Robustness  ·  **needs A100**\n"
        "The §9 robustness checks on the clean panel. Every variant writes a **tagged** "
        "artifact and **never touches the primary one** — the pre-registered analysis "
        "stays exactly where it is, and each check is read *beside* it.\n\n"
        "| variant | what it answers | cost |\n|---|---|---|\n"
        "| **wu** | does the mechanism survive a **different head detector** (Wu/copy "
        "heads, not argmax)? **the insurance policy on the headline** | 1 E2 / model |\n"
        "| **m2sens** | how much of the M2 story is the **0.10 distractor-mass floor**? "
        "Reported side-by-side with the registered floor — *not* a re-tune | 1 E2 / model |\n"
        "| ksens | does it hold at the sensitivity k? | 1 E2 / model |\n"
        "| steps3 | does it hold averaging masses over 3 answer steps? | 1 E2 / model |\n"
        "| seeds | reliability: mean ± range over seed repeats (+ R_self) | 3 E2 / model |\n\n"
        "**First pass: `wu`, on olmo + phi.** Those carry the causal headline, so "
        "detector-independence there is what a reviewer will demand first.\n\n"
        "> **On m2sens.** The pilot showed the registered floor 0.10 admits ~95.6% of "
        "distractor_hit failures as M2. That observation was made *after* seeing the "
        "results, so moving the floor now and re-running the primary would be HARKing. "
        "The floor therefore stays at 0.10 for the primary analysis; the alternative is "
        "reported as a sensitivity artifact and the permissiveness becomes a stated "
        "limitation. `scripts/e2_signatures.py` enforces this: `--m2-min-distractor-mass` "
        "refuses to run without a `--tag`.")]
    cells += setup_cells()
    cells.append(md("## 1) shell_share must match the grid E1/E2 used"))
    cells.append(SHELL_SHARE_CELL)
    cells.append(md("## 2) Choose the variants and the models"))
    cells.append(code(r"""
# Comma-separated: wu, ksens, steps3, seeds, m2sens   (or 'all')
VARIANTS = 'wu'

# The alternative M2 floor, for the m2sens variant ONLY. The PRE-REGISTERED floor
# (0.10) remains the primary analysis and is not modified. Pooled pilot quantiles
# of distractor_mass over distractor_hit failures: p5=0.103 p25=0.154 p50=0.193.
M2_ALT_FLOOR = 0.15

# olmo + phi carry the causal headline -> check them first. Widen to the full panel
# once you have seen the per-model cost.
E5_MODELS = ['olmo2_7b_instruct', 'phi35_mini']
# E5_MODELS = %(panel)s   # <- the whole panel

print('E5 variants =', VARIANTS, '| models =', E5_MODELS, '| m2 alt floor =', M2_ALT_FLOOR)
""" % {"panel": json.dumps(MODELS)}))
    cells += _model_loop(
        "3) E5 — robustness per model",
        "Each variant re-runs E2 in a clean subprocess with the pinned model and "
        "writes a TAGGED artifact; the primary is never overwritten.",
        "['scripts/e5_robustness.py', '--model', key, '--variants', VARIANTS,"
        " '--m2-alt-floor', str(M2_ALT_FLOOR)]",
        first_est_h=5.0,
        skip_check="os.path.exists(f'{RESULTS_DIR}/e5_robustness_{key}.json')",
        overwrite_default=True,
        models="E5_MODELS")
    cells.append(md("## 4) Detector independence — does the mechanism survive the Wu head list?"))
    cells.append(code(r"""
import json, os
RESULTS_DIR = os.environ['RFM_RESULTS_DIR']

def buckets(p):
    if not os.path.exists(p): return '-'
    r = json.load(open(p))['sample_level']['bucket_rates']
    return ' '.join(f'{k[:4]}={v:.2f}' for k, v in r.items() if v > 0.01)

print(f'{"model":22s} {"PRIMARY (argmax, m2=0.10)":36s} {"Wu / copy heads":36s}')
for m in E5_MODELS:
    print(f'{m:22s} {buckets(f"{RESULTS_DIR}/e2_signatures_{m}.json"):36s} '
          f'{buckets(f"{RESULTS_DIR}/e2_signatures_{m}_wu.json"):36s}')
print('\nIf the Wu column reproduces the primary column, the mechanism is not an '
      'artifact of the detector choice.')
"""))
    cells.append(md(
        "## 5) M2 threshold sensitivity — side by side, not a replacement\n"
        "The left column is the analysis of record. The right column shows how much of "
        "it rests on the floor. Both go in the paper."))
    cells.append(code(r"""
left  = 'PRIMARY (m2 floor 0.10, prereg)'
right = 'SENSITIVITY (m2 floor ' + format(M2_ALT_FLOOR, '.2f') + ')'
print(f'{"model":22s} {left:36s} {right:36s}')
for m in E5_MODELS:
    print(f'{m:22s} {buckets(f"{RESULTS_DIR}/e2_signatures_{m}.json"):36s} '
          f'{buckets(f"{RESULTS_DIR}/e2_signatures_{m}_m2sens.json"):36s}')
print('\nA large M2 drop on the right does NOT mean the primary is wrong — it means the')
print('M2 rate is floor-sensitive, and that belongs in the paper as a limitation.')
"""))
    return notebook(cells)


def main():
    # Filenames carry the Colab runtime they need, so you never start a 20 h run
    # on the wrong GPU.
    #   A100  — every notebook that LOADS a model. gemma-2-9b is forced to eager
    #           attention (sdpa silently drops its logit-softcapping) and nb 00
    #           loads fp32 for the math gate, so 40 GB is the floor. L4/T4 will
    #           OOM.
    #   CPU   — pure-JSON analysis; no accelerator needed (works on any runtime).
    outputs = {
        "00_A100_setup_and_guardrails.ipynb": nb_00(),
        "01_A100_e1_breaking_surface.ipynb": nb_e1(),
        "02_A100_e2_signatures.ipynb": nb_e2(),
        "03_A100_e3_causal.ipynb": nb_e3(),
        "04_A100_e4_e5_analysis.ipynb": nb_e4_e5(),
        "05_A100_analyze_breaking_models.ipynb": nb_05_analyze_breaking(),
        "06_A100_shell_share_e1_e2.ipynb": nb_06_shell_share_rerun(),
        "07_A100_e3_causal_e4_families.ipynb": nb_07_e3_e4(),
        "08_A100_e3_backfill_e4.ipynb": nb_08_e3_backfill(),
        "09_A100_e2_rerun_record_pairs.ipynb": nb_09_e2_pairs(),
        "10_A100_e3_ksweep_EXPLORATORY.ipynb": nb_10_ksweep(),
        "11_A100_e5_robustness.ipynb": nb_11_e5(),
    }
    # Drop the pre-GPU-tag filenames so the folder never shows two copies.
    for stale in NB_DIR.glob("*.ipynb"):
        if stale.name not in outputs:
            stale.unlink()
            print("removed stale", stale.name)
    for name, nb in outputs.items():
        path = NB_DIR / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
