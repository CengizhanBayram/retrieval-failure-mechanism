"""End-to-end plumbing of the follow-up scripts on tiny offline models.

A fake panel (one 'bf16' tiny Llama + an emulated 4-bit copy of it), synthetic
detection artifacts and synthetic E1 files are set up in a temp results dir; the
grader is replaced by a deterministic pseudo-grader (random models never answer
correctly). Then the real script ``main()`` functions run in order:

    E2 (parent) -> E3 (parent) -> E6 -> E8
    E7 (cell transfer + gate) -> E2 (quant, transferred heads) -> E3 (quant, --max-pairs)

This checks argument handling, file naming, checkpoint isolation, the probe-key
and head-list plumbing and the gates - not any experimental result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import harness  # noqa: E402
from conftest import VALID_PREREG  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"
PARENT, QUANT = "tiny_llama", "tiny_llama_bnb4"


def _emulate_nf4(model):
    """Per-row absmax 4-bit rounding of every Linear weight (a stand-in for NF4)."""
    q = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in q.named_parameters():
            if p.dim() == 2 and "embed" not in name:
                s = p.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                p.copy_(torch.round(p / s * 7) / 7 * s)
    return q


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("followup_env")
    results = root / "results"
    det = root / "det"
    results.mkdir()
    model, tok = harness.tiny_model("llama", n_layers=3, attn="sdpa")
    qmodel = _emulate_nf4(model)
    harness.detection_artifact(model, det, PARENT, n_retrieval=6)
    # the quantized model's own detection differs by one head (as in Part I)
    art = json.loads((det / f"{PARENT}_seed42.json").read_text())
    art["argmax_heads"] = art["argmax_heads"][:-1] + [[0, 0]] if [0, 0] not in art["argmax_heads"] \
        else art["argmax_heads"][:-1] + [[2, 3]]
    (det / f"{QUANT}_seed42.json").write_text(json.dumps(art))
    prereg = root / "prereg.yaml"
    prereg.write_text(yaml.safe_dump(VALID_PREREG))
    panel = {"defaults": {}, "models": {PARENT: {"hf_id": "tiny/llama", "revision": SHA,
                                                 "family": "llama"}},
             "quant_models": {QUANT: {"hf_id": "tiny/llama", "revision": SHA,
                                      "family": "llama", "quant": "bnb4"}}}
    # synthetic parent E1: three breaking cells, 20 probes each
    from failure_mech.probes import CellSpec
    cells = [CellSpec(PARENT, 192, 0.5, 2, "shell_share"),
             CellSpec(PARENT, 256, 0.3, 5, "shell_share"),
             CellSpec(PARENT, 192, 0.7, 2, "shell_same")]
    surface = {c.cell_hash(): {"axes": c.axes(), "cell_hash": c.cell_hash(), "n_stage1": 20,
                               "n_total": 20, "accuracy": 0.5, "wilson_ci95": {"lo": 0.3, "hi": 0.7},
                               "behavioral_grade_counts": {"correct": 10, "distractor_hit": 10,
                                                           "other_wrong": 0, "empty": 0},
                               "oom_fallback_applied": False,
                               "effective_context_length": c.context_length} for c in cells}
    # two solved (non-breaking) cells with distractors: held-out probes for e9 (F-2)
    for c in [CellSpec(PARENT, 192, 0.3, 2, "shell_diff"), CellSpec(PARENT, 256, 0.7, 5, "shell_same")]:
        surface[c.cell_hash()] = {"axes": c.axes(), "cell_hash": c.cell_hash(), "n_stage1": 24,
                                  "n_total": 24, "accuracy": 0.95,
                                  "wilson_ci95": {"lo": 0.8, "hi": 0.99},
                                  "behavioral_grade_counts": {"correct": 23, "distractor_hit": 1,
                                                              "other_wrong": 0, "empty": 0},
                                  "oom_fallback_applied": False,
                                  "effective_context_length": c.context_length}
    (results / f"e1_surface_{PARENT}.json").write_text(json.dumps({"provenance": {}, "cells": surface}))
    (results / f"e1_breaking_cells_{PARENT}.json").write_text(json.dumps(
        {"provenance": {}, "breaking_cells": sorted(c.cell_hash() for c in cells),
         "fallback_applied": False}))
    return {"root": root, "results": results, "det": det, "prereg": str(prereg), "panel": panel,
            "models": {PARENT: (model, tok), QUANT: (qmodel, tok)}}


@pytest.fixture(autouse=True)
def patched(env, monkeypatch):
    from failure_mech import panel as P
    from failure_mech import grading

    monkeypatch.setenv("RFM_RESULTS_DIR", str(env["results"]))
    monkeypatch.setattr(P, "load_panel", lambda paths_cfg: env["panel"])
    monkeypatch.setattr(P, "ensure_reuse_on_path", lambda paths_cfg: (None, env["root"]))
    monkeypatch.setattr(P, "detection_dir", lambda paths_cfg: env["det"])

    def fake_load(paths_cfg, panel, key, *, attn_implementation="sdpa", dtype="bfloat16",
                  device_map="auto"):
        cfg = P.model_cfg(paths_cfg, panel, key)
        model, tok = env["models"][key]
        cfg.update({"effective_attn": "sdpa", "n_query_heads": 4, "n_kv_heads": 2,
                    "head_dim": P.head_dim(model)})
        if cfg.get("quant"):
            cfg["quantization"] = {"method": "bnb4", "emulated": True}
        return model, tok, cfg

    monkeypatch.setattr(P, "load_model", fake_load)
    real = grading.grade_generation

    def pseudo_grade(text, needle, distractors):
        g = real(text, needle, distractors)
        ok = int(hashlib.sha1((text + needle).encode()).hexdigest(), 16) % 2 == 0
        return type(g)(correct=ok, grade="correct" if ok else "distractor_hit")

    monkeypatch.setattr(grading, "grade_generation", pseudo_grade)
    yield


def _load(env, name):
    return json.loads((env["results"] / name).read_text())


def test_01_parent_e2_e3(env):
    import e2_signatures
    import e3_causal
    assert e2_signatures.main(["--model", PARENT, "--prereg", env["prereg"], "--save-grades"]) == 0
    e2 = _load(env, f"e2_signatures_{PARENT}.json")
    assert e2["pairs_used"] > 0 and set(e2["grades_by_cell"]) == set(
        _load(env, f"e1_breaking_cells_{PARENT}.json")["breaking_cells"])
    assert e2["provenance"]["extra"]["heads_from"] == PARENT
    assert e3_causal.main(["--model", PARENT, "--prereg", env["prereg"], "--k-list", "1", "2"]) == 0
    e3 = _load(env, f"e3_causal_{PARENT}.json")
    assert set(e3["k"]) == {"1", "2"} and e3["k"]["2"]["self_patch_ok"]
    assert e3["provenance"]["extra"]["pairs_selected"] == "all"


def test_02_variant_flags_require_tag(env):
    import e2_signatures
    import e3_causal
    with pytest.raises(SystemExit):
        e2_signatures.main(["--model", PARENT, "--prereg", env["prereg"], "--heads-from", QUANT])
    with pytest.raises(SystemExit):
        e3_causal.main(["--model", PARENT, "--prereg", env["prereg"], "--max-pairs", "3"])


def test_03_e6_matched_ablation(env):
    import e6_matched_ablation as e6
    assert e6.main(["--model", PARENT, "--prereg", env["prereg"], "--max-pairs", "4"]) == 0
    out = _load(env, f"e6_matched_ablation_{PARENT}.json")
    assert out["provenance"]["extra"]["n_pairs_done"] == 4
    assert set(out["summary"]["full"]) == {"all", "context", "answer", "ansdec"}
    assert set(out["summary"]["top10"]) == {"all", "answer"}
    rec = out["records"][0]
    assert len(rec["cond"]) == (4 + 2) * 4


def test_04_e8_pattern_transplant(env):
    import e8_pattern_transplant as e8
    assert e8.main(["--model", PARENT, "--prereg", env["prereg"], "--max-pairs", "3",
                    "--k-list", "2", "--gate-n", "3"]) == 0
    out = _load(env, f"e8_pattern_transplant_{PARENT}.json")
    gate = out["provenance"]["extra"]["gate"]
    assert gate["identical"] == gate["n"] == 3 and gate["max_rel_err"] < 1e-4
    assert set(out["summary"]) == {"2", "6"}            # budget + full detected set
    assert set(out["summary"]["2"]) == {"activation", "pattern", "content"}
    # the activation variant equals E3-style patching with the donor's own z_h:
    # with full-precision tiny models, its break/repair indicators are defined
    rec = out["records"][0]
    assert len(rec["cond"]) == 2 * 3 * 2 * 2          # k x variants x arms x directions


def test_05_quant_chain(env):
    import e7_quant_cells
    import e2_signatures
    import e3_causal
    assert e7_quant_cells.main(["--model", QUANT, "--parent", PARENT, "--prereg", env["prereg"],
                                "--gate-tokens", "64", "--gate-heads", "3"]) == 0
    rep = _load(env, f"e7_quant_cells_{QUANT}.json")
    assert rep["gate"]["passed"] and rep["probe_key"] == PARENT
    assert rep["head_lists"]["top_k"]["5"]["overlap"] >= 4
    surf = _load(env, f"e1_surface_{QUANT}.json")
    assert surf["probe_key"] == PARENT and surf["transferred_from"] == PARENT
    assert all(c["accuracy"] is None and c["parent_accuracy"] == 0.5 for c in surf["cells"].values())
    # probes are byte-identical to the parent's (hash recorded by the gate)
    from failure_mech.probes import ProbeFactory, CellSpec
    tok = env["models"][PARENT][1]
    grid = yaml.safe_load(open(REPO / "configs" / "grid.yaml"))
    fac = ProbeFactory(tok, grid, PARENT)
    h = sorted(surf["cells"])[0]
    ax = surf["cells"][h]["axes"]
    p = fac.build(CellSpec(ax["model_key"], ax["context_length"], ax["needle_position"],
                           ax["n_distractors"], ax["similarity"]), 0, 42)
    assert hashlib.sha1(",".join(map(str, p.input_ids)).encode()).hexdigest() == \
        rep["gate"]["probe_hashes"][h]["sha1_input_ids_sample0"]

    assert e2_signatures.main(["--model", QUANT, "--prereg", env["prereg"], "--heads-from", PARENT,
                               "--tag", "parentheads", "--save-grades"]) == 0
    e2q = _load(env, f"e2_signatures_{QUANT}_parentheads.json")
    ex = e2q["provenance"]["extra"]
    assert ex["heads_from"] == PARENT and ex["probe_key"] == PARENT
    assert ex["e1_cells_transferred_from"] == PARENT and ex["quantization"]["method"] == "bnb4"
    e2p = _load(env, f"e2_signatures_{PARENT}.json")
    assert set(e2q["per_head"]) == set(e2p["per_head"])   # same (transferred) heads

    # E3 refuses a head list that differs from the one E2 used
    with pytest.raises(SystemExit):
        e3_causal.main(["--model", QUANT, "--prereg", env["prereg"], "--e2-tag", "parentheads",
                        "--tag", "x", "--k-list", "1"])
    assert e3_causal.main(["--model", QUANT, "--prereg", env["prereg"], "--heads-from", PARENT,
                           "--e2-tag", "parentheads", "--tag", "parentheads", "--k-list", "1", "2",
                           "--max-pairs", "6"]) == 0
    e3q = _load(env, f"e3_causal_{QUANT}_parentheads.json")
    assert e3q["k"]["1"]["n_pairs"] == 6
    assert (env["results"] / "checkpoints" / f"e3_parentheads_{QUANT}").is_dir()
    sel = e3q["provenance"]["extra"]["pairs_selected"]
    assert sum(len(v) for v in sel.values()) == 6 and len(sel) == min(3, len(e2q["cells_used"]))


def test_06_capture_long_check(env):
    import check_capture_long
    assert check_capture_long.main(["--model", PARENT, "--prereg", env["prereg"],
                                    "--k-heads", "3", "--n-cells", "2"]) == 0
    out = _load(env, f"capture_long_check_{PARENT}.json")
    assert out["passed"] and out["max_abs_diff"] < 1e-5 and len(out["records"]) == 6


def test_07_followup_analysis(env):
    import followup_analysis as FA
    assert FA.main(["--results-dir", str(env["results"]), "--a-models", PARENT,
                    "--c-models", PARENT, "--f-models", PARENT,
                    "--b-pairs", f"{PARENT}:{QUANT}"]) == 0
    s = _load(env, "followup_summary.json")
    a = s["A"][PARENT]
    assert set(a["effects_break_full"]) == {"all", "context", "answer", "ansdec"}
    assert a["A1"] in ("replicated", "not replicated", "inconclusive")
    assert a["A3_scope_explanation"] in ("supported", "contradicted", "inconclusive")
    b = s["B"][QUANT]
    assert {"B1_behaviour", "B2_signature", "B3_causal"} <= set(b)
    assert b["B2_signature"]["verdict"] in ("preserved", "changed", "inconclusive")
    assert b["B_overall"] in ("mechanism preserved under NF4", "changed", "inconclusive", "incomplete")
    c = s["C"][PARENT]
    assert "C1_full" in c and "C2_full" in c
    assert s["D"]["status"].startswith("not run")
    assert "A5_flip_grades" in a
    f = s["F"][PARENT]
    if f.get("status") != "not run":          # e9 runs in test_08 (after this test)
        assert f["F1"]["verdict"] in ("failure-specific", "not specific", "partially specific")
    assert (env["results"] / "followup_report.md").exists()


def test_08_e9_specificity_and_task_transfer(env, monkeypatch):
    import e9_answer_rows
    import e3_causal
    assert e9_answer_rows.main(["--model", PARENT, "--prereg", env["prereg"], "--max-pairs", "4",
                                "--n-detect", "6"]) == 0
    out = _load(env, f"e9_answer_rows_{PARENT}.json")
    f1, f2 = out["F1_specificity"], out["F2_task_transfer"]
    assert f1["n_pairs"] == 4 and 0.0 <= f1["capture_rate_failure"] <= 1.0
    assert f2["n_probes"] == 6 and f2["n_cells"] == 2
    assert (env["results"] / f"e9_answer_rows_{PARENT}.npz").exists()
    art_path = env["results"] / "detection_taskII" / f"{PARENT}_taskII_seed42.json"
    art = json.loads(art_path.read_text())
    assert len(art["argmax_scores"]) == 3
    # F-3 plumbing: E3 patches the task list through the extra detection dir
    art["argmax_heads"] = [[0, 1], [1, 2], [2, 0], [2, 3]]
    art_path.write_text(json.dumps(art))
    monkeypatch.setenv("RFM_EXTRA_DETECTION_DIRS", str(art_path.parent))
    assert e3_causal.main(["--model", PARENT, "--prereg", env["prereg"],
                           "--heads-from", f"{PARENT}_taskII", "--allow-other-heads",
                           "--tag", "taskheads", "--k-list", "1", "2", "--max-pairs", "3"]) == 0
    e3t = _load(env, f"e3_causal_{PARENT}_taskheads.json")
    assert e3t["provenance"]["extra"]["heads_from"] == f"{PARENT}_taskII"
    assert e3t["k"]["2"]["n_pairs"] == 3


def test_09_followup_analysis_with_F(env):
    import followup_analysis as FA
    assert FA.main(["--results-dir", str(env["results"]), "--a-models", PARENT,
                    "--c-models", PARENT, "--f-models", PARENT,
                    "--b-pairs", f"{PARENT}:{QUANT}"]) == 0
    f = _load(env, "followup_summary.json")["F"][PARENT]
    assert f["F1"]["verdict"] in ("failure-specific", "not specific", "partially specific")
    assert f["F2"]["verdict"] in ("list transfers", "does not transfer", "partial", "not computable")
    assert f["F3"]["verdict"] in ("null was a list-transfer effect",
                                  "dissociation holds with task-specific heads", "inconclusive")


def test_10_layer_matched_control(env):
    import e3_causal
    from failure_mech import detect
    with pytest.raises(SystemExit):                      # a different control needs a tag
        e3_causal.main(["--model", PARENT, "--prereg", env["prereg"], "--control-mode", "layer"])
    assert e3_causal.main(["--model", PARENT, "--prereg", env["prereg"], "--control-mode", "layer",
                           "--tag", "layerctrl", "--k-list", "2", "--max-pairs", "3"]) == 0
    out = _load(env, f"e3_causal_{PARENT}_layerctrl.json")
    assert out["provenance"]["extra"]["control_mode"] == "layer"
    det = detect.load_detection(env["det"], PARENT, 42)
    model = env["models"][PARENT][0]
    treat = det.top_k_heads(4, detector="argmax")
    ctrl = e3_causal._control_heads_layer_matched(det, treat, 7, model)
    retr = det.retrieval_head_set("argmax")
    assert len(ctrl) == 4 and len(set(ctrl)) == 4 and not (set(ctrl) & retr)
    free = {l: sum(1 for (ll, _h) in det.non_retrieval_heads() if ll == l) for l in range(3)}
    need = {}
    for (lt, _h), (lc, _c) in zip(treat, ctrl):
        need[lt] = need.get(lt, 0) + 1
        if need[lt] <= free[lt]:                          # same layer while it has free heads
            assert lc == lt


def test_11_followup_analysis_with_G(env):
    import followup_analysis as FA
    assert FA.main(["--results-dir", str(env["results"]), "--a-models", PARENT,
                    "--c-models", PARENT, "--f-models", PARENT, "--g-models", PARENT, "missing_model",
                    "--b-pairs", f"{PARENT}:{QUANT}"]) == 0
    g = _load(env, "followup_summary.json")["G"]
    assert g["missing_model"]["status"] == "not run"
    e = g[PARENT]["by_k"]["2"]
    assert set(e) == {"break", "repair"}
    for d in ("break", "repair"):
        assert e[d]["n"] == 3 and 0.0 <= e[d]["q"] <= 1.0
        assert e[d]["random_ctrl_effect"] is not None          # original E3 ran k = 2 (test_01)
        assert e[d]["verdict"] in ("no random-control effect at this budget", "listed heads carry the effect",
                                   "layer effect", "partly layer")
    assert "## G - layer-matched control" in (env["results"] / "followup_report.md").read_text()
