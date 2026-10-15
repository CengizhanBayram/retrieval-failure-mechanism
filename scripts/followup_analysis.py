"""
Apply docs/FOLLOWUP_DECISION_RULES.md to the outputs of follow-up experiments
A-G and write ``followup_summary.json`` + ``followup_report.md``.

Reads result JSONs only (no model, no GPU). Every experiment is optional: a
missing output is reported as "not run" and the other experiments are still
evaluated, so the script can be re-run as results arrive.

Usage:
    python scripts/followup_analysis.py
    python scripts/followup_analysis.py --haystack-summary /path/to/D_haystack/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _common as C  # noqa: E402

from failure_mech import stats  # noqa: E402

MARGIN = 0.20
EQUIV = 0.10
LEVEL = 0.95
SEED = 42
N_BOOT = 10000
Z = 1.959963984540054

A_MODELS = ["llama31_8b_instruct", "qwen25_3b_instruct", "mistral_7b_instruct", "gemma2_9b_it"]
B_PAIRS = [("qwen25_7b_instruct", "qwen25_7b_instruct_bnb4"),
           ("llama31_8b_instruct", "llama31_8b_instruct_bnb4"),
           ("gemma2_9b_it", "gemma2_9b_it_bnb4")]
C_MODELS = ["qwen25_3b_instruct", "olmo2_7b_instruct", "llama31_8b_instruct", "phi35_mini"]
D_FALLBACK_MODELS = ["qwen25_7b_instruct", "llama31_8b_instruct"]
F_MODELS = ["qwen25_3b_instruct", "olmo2_7b_instruct", "phi35_mini", "qwen25_7b_instruct",
            "mistral_7b_instruct", "gemma2_9b_it", "llama31_8b_instruct"]
G_MODELS = ["qwen25_3b_instruct", "olmo2_7b_instruct", "phi35_mini"]
SCOPES = ["all", "context", "answer", "ansdec"]
BUCKETS = ["m2_capture", "m1_silence", "correct_attend", "residual"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean_ci(x, seed=SEED):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    ci = stats.bootstrap_ci(lambda a: float(a.mean()), x, n=N_BOOT, level=LEVEL, seed=seed)
    return {"est": float(x.mean()), "lo": float(ci["lo"]), "hi": float(ci["hi"]), "n": int(x.size)}


def perm_p(x, seed=SEED):
    x = list(map(float, x))
    return float(stats.paired_permutation(x, n_perm=N_BOOT, two_sided=True, seed=seed)["p"]) if x else float("nan")


def bh(pvals: dict) -> dict:
    keys = list(pvals)
    if not keys:
        return {}
    res = stats.bh_correct([pvals[k] for k in keys], 0.05)
    return {k: float(q) for k, q in zip(keys, res["qvalues"])}


def three_way(est, lo, hi, bound=EQUIV) -> str:
    """preserved / changed / inconclusive (rules B-2, B-3)."""
    if any(map(lambda v: v != v, (est, lo, hi))):
        return "not computable"
    if lo >= -bound and hi <= bound:
        return "preserved"
    if (lo > 0 or hi < 0) and abs(est) >= bound:
        return "changed"
    return "inconclusive"


def flip(base, cond, direction) -> int:
    if direction == "break":
        return int(bool(base) and not cond["correct"])
    return int((not base) and cond["correct"])


def cluster_boot(groups: list, stat, seed=SEED, n=N_BOOT):
    """Percentile bootstrap over clusters: ``groups`` is a list of per-cluster
    data; ``stat(list_of_groups) -> float``."""
    rng = np.random.default_rng(seed)
    m = len(groups)
    if m == 0:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        vals.append(stat([groups[i] for i in idx]))
    lo, hi = np.nanpercentile(vals, [100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# A - matched ablation
# ---------------------------------------------------------------------------

def analyze_A(rd: Path, models=None) -> dict:
    out = {}
    for m in (models or A_MODELS):
        d = _load(rd / f"e6_matched_ablation_{m}.json")
        if d is None:
            out[m] = {"status": "not run"}
            continue
        recs = d["records"]
        x = {}
        for sc in SCOPES:
            key_t, key_c = f"full|{sc}|treat|break", f"full|{sc}|ctrl|break"
            if not recs or key_t not in recs[0]["cond"]:
                continue
            x[sc] = [flip(r["base_s"], r["cond"][key_t], "break") -
                     flip(r["base_s"], r["cond"][key_c], "break") for r in recs]
        eff = {sc: {**mean_ci(v), "p": perm_p(v)} for sc, v in x.items()}
        q = bh({sc: e["p"] for sc, e in eff.items()})
        for sc in eff:
            eff[sc]["q"] = q[sc]
        res = {"n_pairs": len(recs), "complete": d["provenance"]["extra"].get("complete"),
               "effects_break_full": eff}
        # A-5: what the treated break flips output (distractor hit = confusion,
        # other_wrong / empty = omission)
        res["A5_flip_grades"] = {
            sc: dict(Counter(r["cond"][f"full|{sc}|treat|break"]["grade"] for r in recs
                             if flip(r["base_s"], r["cond"][f"full|{sc}|treat|break"], "break")))
            for sc in x}
        a1 = "not computable"
        if "all" in eff:
            e = eff["all"]
            a1 = ("replicated" if e["est"] >= MARGIN and e["q"] < 0.05 else
                  "not replicated" if e["hi"] < MARGIN else "inconclusive")
        a2 = "not computable"
        if "context" in x and "answer" in x:
            dd = [a - b for a, b in zip(x["context"], x["answer"])]
            dci = mean_ci(dd)
            res["D_context_minus_answer"] = dci
            a2 = ("context-carried" if dci["est"] >= MARGIN and dci["lo"] > 0 else
                  "answer-carried" if dci["est"] <= -MARGIN and dci["hi"] < 0 else "not localised")
        res["A1"], res["A2"] = a1, a2
        if m == (models or A_MODELS)[0]:      # the primary model (Llama-3.1 by default)
            res["A3_scope_explanation"] = (
                "supported" if (a1 == "replicated" and a2 == "context-carried") else
                "contradicted" if (a1 == "not replicated" or a2 == "answer-carried") else
                "inconclusive")
        # A-4: E3 break effect at the complete set (or the largest k run)
        e3 = _load(rd / f"e3_causal_{m}.json")
        n_full = d["provenance"]["extra"].get("head_set_sizes", {}).get("full")
        if e3:
            ks = sorted(int(k) for k in e3["k"])
            kk = n_full if (n_full is not None and n_full in ks) else ks[-1]
            res["A4_e3_break_effect"] = {
                "k": kk, "is_complete_set": kk == n_full,
                "effect": e3["k"][str(kk)]["margins"]["break"]["flip_minus_control"]}
        out[m] = res
    return out


# ---------------------------------------------------------------------------
# B - NF4
# ---------------------------------------------------------------------------

def _rate_se(ci: dict) -> float:
    return (float(ci["hi"]) - float(ci["lo"])) / (2 * Z)


def _effect_and_se(e3k: dict, direction: str) -> tuple[float, float]:
    eff = float(e3k["margins"][direction]["flip_minus_control"])
    se = math.sqrt(_rate_se(e3k[direction]["ci"]) ** 2 +
                   _rate_se(e3k["random_control"][direction]["ci"]) ** 2)
    return eff, se


def _bucket_groups(e2: dict) -> dict:
    g: dict = {}
    for f in e2.get("failure_sample_masses", []):
        g.setdefault(f["cell"], []).append(f["bucket"])
    return g


def _share(groups: list, bucket="m2_capture") -> float:
    allb = [b for g in groups for b in g]
    return sum(b == bucket for b in allb) / len(allb) if allb else float("nan")


def parent_e2(rd: Path, parent: str):
    """The parent's E2 - the corrected Gemma-2 artifact when present (rule E-1)."""
    fixed = rd / f"e2_signatures_{parent}_capfix.json"
    if fixed.exists():
        return _load(fixed), fixed.name
    p = rd / f"e2_signatures_{parent}.json"
    return _load(p), p.name


def analyze_B(rd: Path, tag: str = "parentheads", pairs=None) -> dict:
    out = {}
    for parent, quant in (pairs or B_PAIRS):
        e2q = _load(rd / f"e2_signatures_{quant}_{tag}.json")
        if e2q is None:
            out[quant] = {"status": "not run"}
            continue
        res = {"parent": parent}
        if parent == "gemma2_9b_it" and not (rd / f"e2_signatures_{parent}_capfix.json").exists():
            res["warning"] = ("parent Gemma-2 E2 has not been re-run with the capture fix "
                              "(experiment E); B-2 compares against the affected artifact")
        # ---- B-1 behaviour
        surf = _load(rd / f"e1_surface_{parent}.json")
        grades = e2q.get("grades_by_cell")
        if surf and grades:
            cells = [h for h in sorted(grades) if h in surf["cells"]]
            groups = []
            for h in cells:
                g = grades[h]
                n_b = int(surf["cells"][h]["n_total"])
                c_b = int(round(float(surf["cells"][h]["accuracy"]) * n_b))
                groups.append((sum(int(x[0]) for x in g), len(g), c_b, n_b))

            def dacc(gs):
                cq = sum(a for a, _, _, _ in gs); nq = sum(b for _, b, _, _ in gs)
                cb = sum(c for _, _, c, _ in gs); nb = sum(d for _, _, _, d in gs)
                return cq / nq - cb / nb if nq and nb else float("nan")

            est = dacc(groups)
            lo, hi = cluster_boot(groups, dacc)
            acc_q = sum(a for a, _, _, _ in groups) / max(1, sum(b for _, b, _, _ in groups))
            acc_b = sum(c for _, _, c, _ in groups) / max(1, sum(d for _, _, _, d in groups))
            fails_q = Counter(x[1] for h in cells for x in grades[h] if not int(x[0]))
            res["B1_behaviour"] = {
                "n_cells": len(cells), "acc_nf4": acc_q, "acc_bf16": acc_b,
                "delta": est, "lo": lo, "hi": hi,
                "verdict": "changed" if (lo > 0 or hi < 0) else "no detectable change",
                "nf4_failure_grades": dict(fails_q)}
        # ---- B-2 signature
        e2p, e2p_name = parent_e2(rd, parent)
        if e2p:
            gq, gp = _bucket_groups(e2q), _bucket_groups(e2p)
            lq, lp = list(gq.values()), list(gp.values())
            est = _share(lq) - _share(lp)
            rng = np.random.default_rng(SEED)
            boots = []
            for _ in range(N_BOOT):
                iq = rng.integers(0, len(lq), size=len(lq)) if lq else []
                ip = rng.integers(0, len(lp), size=len(lp)) if lp else []
                boots.append(_share([lq[i] for i in iq]) - _share([lp[i] for i in ip]))
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rates_q = {b: _share(lq, b) for b in BUCKETS}
            rates_p = {b: _share(lp, b) for b in BUCKETS}
            res["B2_signature"] = {
                "parent_artifact": e2p_name, "capture_nf4": rates_q["m2_capture"],
                "capture_bf16": rates_p["m2_capture"], "delta": est, "lo": float(lo), "hi": float(hi),
                "rates_nf4": rates_q, "rates_bf16": rates_p,
                "dominant_nf4": max(rates_q, key=lambda b: rates_q[b]),
                "dominant_bf16": max(rates_p, key=lambda b: rates_p[b]),
                "n_failures_nf4": sum(map(len, lq)), "n_failures_bf16": sum(map(len, lp)),
                "verdict": three_way(est, float(lo), float(hi))}
            # descriptive per-head Cliff's delta (B, descriptive)
            res["per_head_cliffs_delta"] = {
                h: {"nf4": e2q["per_head"][h]["cliffs_delta_success_vs_failure"],
                    "bf16": e2p["per_head"].get(h, {}).get("cliffs_delta_success_vs_failure")}
                for h in e2q.get("per_head", {})}
        # ---- B-3 causal
        e3q = _load(rd / f"e3_causal_{quant}_{tag}.json")
        e3p = _load(rd / f"e3_causal_{parent}.json")
        if e3q and e3p:
            b3 = {}
            for k in ("10", "20"):
                if k not in e3q["k"] or k not in e3p["k"]:
                    continue
                for direction in ("break", "repair"):
                    eq, sq = _effect_and_se(e3q["k"][k], direction)
                    ep, sp = _effect_and_se(e3p["k"][k], direction)
                    d = eq - ep
                    se = math.sqrt(sq ** 2 + sp ** 2)
                    lo, hi = d - Z * se, d + Z * se
                    b3[f"k{k}_{direction}"] = {
                        "effect_nf4": eq, "effect_bf16": ep, "delta": d, "lo": lo, "hi": hi,
                        "n_pairs_nf4": e3q["k"][k]["n_pairs"], "n_pairs_bf16": e3p["k"][k]["n_pairs"],
                        "decision_nf4": bool(e3q["k"][k]["margins"][direction]["meets_margin"]
                                             and e3q["k"][k]["bh_rejected"][direction]),
                        "decision_bf16": bool(e3p["k"][k]["margins"][direction]["meets_margin"]
                                              and e3p["k"][k]["bh_rejected"][direction]),
                        "verdict": three_way(d, lo, hi)}
            res["B3_causal"] = b3
        # ---- overall
        v2 = res.get("B2_signature", {}).get("verdict")
        v3 = [res.get("B3_causal", {}).get(f"k10_{d}", {}).get("verdict") for d in ("break", "repair")]
        allv = [v2] + v3
        if all(v == "preserved" for v in allv):
            res["B_overall"] = "mechanism preserved under NF4"
        elif any(v == "changed" for v in allv):
            res["B_overall"] = "changed"
        elif any(v is None for v in allv):
            res["B_overall"] = "incomplete"
        else:
            res["B_overall"] = "inconclusive"
        e7 = _load(rd / f"e7_quant_cells_{quant}.json")
        if e7:
            res["identity"] = e7.get("head_lists")
            res["gate"] = (e7.get("gate") or {}).get("passed")
        out[quant] = res
    return out


# ---------------------------------------------------------------------------
# C - pattern transplant
# ---------------------------------------------------------------------------

def analyze_C(rd: Path, models=None) -> dict:
    out = {}
    for m in (models or C_MODELS):
        d = _load(rd / f"e8_pattern_transplant_{m}.json")
        if d is None:
            out[m] = {"status": "not run"}
            continue
        recs = d["records"]
        ks = [int(k) for k in d["provenance"]["extra"]["k_list"]]
        variants = d["provenance"]["extra"]["variants"]
        kfull = max(ks)
        eff, pv = {}, {}
        for k in ks:
            for v in variants:
                for direction, base in (("break", "base_s"), ("repair", "base_f")):
                    kt, kc = f"{k}|{v}|treat|{direction}", f"{k}|{v}|ctrl|{direction}"
                    x = [flip(r[base], r["cond"][kt], direction) - flip(r[base], r["cond"][kc], direction)
                         for r in recs]
                    key = f"k{k}|{v}|{direction}"
                    eff[key] = {**mean_ci(x), "p": perm_p(x)}
                    pv[key] = eff[key]["p"]
        q = bh(pv)
        for key in eff:
            eff[key]["q"] = q[key]
        res = {"n_pairs": len(recs), "k_full": kfull, "effects": eff,
               "gate": d["provenance"]["extra"].get("gate"),
               "complete": d["provenance"]["extra"].get("complete")}
        for k in ks:
            ep = eff.get(f"k{k}|pattern|break")
            ea = eff.get(f"k{k}|activation|break")
            ec = eff.get(f"k{k}|content|break")
            tag = "full" if k == kfull else f"k{k}"
            if ep:
                res[f"C1_{tag}"] = ("sufficient" if ep["est"] >= MARGIN and ep["q"] < 0.05 else
                                    "not sufficient" if ep["hi"] < MARGIN else "inconclusive")
            if ep and ea and ec:
                if ea["est"] < 0.10:
                    res[f"C2_{tag}"] = "no activation effect to decompose"
                else:
                    hp, hc = ep["est"] >= 0.5 * ea["est"], ec["est"] >= 0.5 * ea["est"]
                    res[f"C2_{tag}"] = ("both" if hp and hc else "pattern-carried" if hp else
                                        "content-carried" if hc else "neither")
        # C-3: pattern-break flips that output the recipient's own distractor value
        grades = [r["cond"][f"{kfull}|pattern|treat|break"]["grade"] for r in recs
                  if flip(r["base_s"], r["cond"][f"{kfull}|pattern|treat|break"], "break")]
        res["C3_pattern_break_flip_grades"] = dict(Counter(grades))
        res["C3_distractor_share"] = (sum(g == "distractor_hit" for g in grades) / len(grades)
                                      if grades else float("nan"))
        out[m] = res
    return out


# ---------------------------------------------------------------------------
# G - layer-matched control (optional)
# ---------------------------------------------------------------------------

def analyze_G(rd: Path, models=None) -> dict:
    """Effect of the listed heads against a same-layer non-retrieval control
    (``e3_causal_<m>_layerctrl.json``), read at the budgets where the original
    random-control effect met the decision. BH over models x budgets x directions."""
    out, eff, pv = {}, {}, {}
    for m in (models or G_MODELS):
        g = _load(rd / f"e3_causal_{m}_layerctrl.json")
        if g is None:
            out[m] = {"status": "not run"}
            continue
        if (g["provenance"]["extra"].get("control_mode") or "random") != "layer":
            out[m] = {"status": "artifact is not a layer-matched run"}
            continue
        out[m] = {"by_k": {}, "complete": g["provenance"]["extra"].get("complete")}
        for k, v in g["k"].items():
            ind = v.get("indicators")
            for direction in ("break", "repair"):
                if ind:
                    x = [a - b for a, b in zip(ind[direction], ind[f"ctrl_{direction}"])]
                    e = {**mean_ci(x), "p": perm_p(x)}
                else:   # older artifact without per-pair indicators
                    e = {"est": float(v["margins"][direction]["flip_minus_control"]),
                         "lo": float("nan"), "hi": float("nan"), "n": int(v["n_pairs"]),
                         "p": float(v["p_values"][direction])}
                e["treat_rate_G"] = float(v[direction]["flip_rate"])
                e["layer_ctrl_rate"] = float(v["random_control"][direction]["flip_rate"])
                key = (m, k, direction)
                eff[key], pv[key] = e, e["p"]
    q = bh(pv)
    for (m, k, direction), e in eff.items():
        e["q"] = q[(m, k, direction)]
        base = _load(rd / f"e3_causal_{m}.json")
        bk = (base or {}).get("k", {}).get(k)
        if bk:
            e["random_ctrl_effect"] = float(bk["margins"][direction]["flip_minus_control"])
            e["treat_rate_original"] = float(bk[direction]["flip_rate"])
            decided = bool(bk["margins"][direction]["meets_margin"] and bk["bh_rejected"][direction])
        else:
            e["random_ctrl_effect"] = e["treat_rate_original"] = None
            decided = False
        e["random_ctrl_decision"] = decided
        if not decided:
            e["verdict"] = "no random-control effect at this budget"
        elif e["est"] >= MARGIN and e["q"] < 0.05:
            e["verdict"] = "listed heads carry the effect"
        elif e["est"] < EQUIV:
            e["verdict"] = "layer effect"
        else:
            e["verdict"] = "partly layer"
        out[m]["by_k"].setdefault(k, {})[direction] = e
    return out


# ---------------------------------------------------------------------------
# D - haystack
# ---------------------------------------------------------------------------

def analyze_D(summary_path: Path | None) -> dict:
    s = _load(summary_path) if summary_path else None
    if s is None:
        return {"status": "not run (pass --haystack-summary)"}
    models = s["models"]
    fb = {m: models.get(m, {}).get("fallback", {}).get("argmax") for m in D_FALLBACK_MODELS}
    if all(fb.values()):
        same = all(v["spearman_scores"] >= 0.99 and v["max_abs_score_diff"] <= 0.03 for v in fb.values())
        diff = all(v["spearman_scores"] <= 0.97 for v in fb.values())
        d1 = "fallback pool" if same else "not the fallback pool" if diff else "inconclusive"
    else:
        d1 = "not run"
    d2 = {}
    for m, row in models.items():
        w = row.get("wikitext")
        if not w:
            continue
        d2[m] = {}
        for det in ("argmax", "copy"):
            c = w.get(det)
            if not c:
                continue
            r = c["J_over_Rself"]
            d2[m][det] = {"J": c["jaccard"], "R_self": c["R_self"], "ratio": r,
                          "top10_overlap": c.get("top_k_overlap", {}).get("10"),
                          "verdict": ("haystack-robust" if r >= 0.8 else
                                      "haystack-dependent" if r < 0.5 else "partial")}
    return {"D1_pool": d1, "D1_detail": fb, "D2": d2,
            "D2_median_ratio": {det: (float(np.median([v[det]["ratio"] for v in d2.values() if det in v]))
                                      if any(det in v for v in d2.values()) else float("nan"))
                                for det in ("argmax", "copy")}}


# ---------------------------------------------------------------------------
# F - specificity of the signature, task transfer of the lists (+ F-3)
# ---------------------------------------------------------------------------

def analyze_F(rd: Path, models=None) -> dict:
    out, pvals = {}, {}
    for m in (models or F_MODELS):
        d = _load(rd / f"e9_answer_rows_{m}.json")
        if d is None:
            out[m] = {"status": "not run"}
            continue
        f1, f2 = d["F1_specificity"], d["F2_task_transfer"]
        pvals[m] = f1["mcnemar"]["p"]
        diff = f1["paired_difference"]
        top10 = f2.get("top_k_overlap", {}).get("10")
        rho = f2.get("score_spearman")
        inactive = sum(1 for v in f2.get("part1_top5_task_scores", {}).values() if v < 0.1)
        if top10 is None:
            v2 = "not computable"
        elif top10 >= 7 and rho is not None and rho >= 0.5:
            v2 = "list transfers"
        elif top10 <= 3:
            v2 = "does not transfer"
        else:
            v2 = "partial"
        res = {"F1": {"r_failure": f1["capture_rate_failure"], "r_success": f1["capture_rate_success"],
                      "diff": diff["est"], "lo": diff["lo"], "hi": diff["hi"],
                      "mcnemar": f1["mcnemar"], "n_pairs": f1["n_pairs"],
                      "distractor_mass_change": f1["distractor_mass_change"],
                      "needle_mass_change": f1["needle_mass_change"]},
               "F2": {"jaccard": f2["jaccard"], "top10_overlap": top10, "spearman": rho,
                      "n_part1": f2["n_part1"], "n_task": f2["n_task"],
                      "part1_top5_inactive": inactive, "verdict": v2,
                      "part1_top5_task_scores": f2.get("part1_top5_task_scores")}}
        # F-3 (if run): break effects of the task list vs the Part I list
        e3t = _load(rd / f"e3_causal_{m}_taskheads.json")
        e3p = _load(rd / f"e3_causal_{m}.json")
        if e3t:
            f3 = {}
            for k, v in e3t["k"].items():
                f3[k] = {"task_break": v["margins"]["break"]["flip_minus_control"],
                         "task_bh": bool(v["bh_rejected"]["break"]),
                         "part1_break": (e3p["k"][k]["margins"]["break"]["flip_minus_control"]
                                         if e3p and k in e3p["k"] else None)}
            reached = [k for k, v in f3.items()
                       if v["task_break"] >= MARGIN and v["task_bh"]
                       and (v["part1_break"] is None or v["part1_break"] < MARGIN)]
            stays = all(v["task_break"] < MARGIN for k, v in f3.items() if int(k) <= 20)
            res["F3"] = {"by_k": f3,
                         "verdict": ("null was a list-transfer effect" if reached else
                                     "dissociation holds with task-specific heads" if stays else
                                     "inconclusive")}
        out[m] = res
    q = bh(pvals)
    for m, qq in q.items():
        r = out[m]["F1"]
        r["q"] = qq
        r["verdict"] = ("failure-specific" if (r["diff"] >= MARGIN and r["lo"] > 0 and qq < 0.05) else
                        "not specific" if r["hi"] < 0.10 else "partially specific")
    return out


# ---------------------------------------------------------------------------
# E - Gemma-2 capture correction
# ---------------------------------------------------------------------------

def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and x != x)


def analyze_E(rd: Path) -> dict:
    old = _load(rd / "e2_signatures_gemma2_9b_it.json")
    new = _load(rd / "e2_signatures_gemma2_9b_it_capfix.json")
    chk = _load(rd / "capture_long_check_gemma2_9b_it.json")
    res = {"capture_long_check": None if chk is None else {"passed": chk["passed"],
                                                           "max_abs_diff": chk["max_abs_diff"]}}
    if old is None or new is None:
        res["status"] = "not run" if new is None else "old artifact missing"
        return res
    po, pn = old.get("pairs_by_cell", {}), new.get("pairs_by_cell", {})
    so = {(h, a, b) for h, v in po.items() for a, b in v}
    sn = {(h, a, b) for h, v in pn.items() for a, b in v}
    res["pairs"] = {"old": len(so), "new": len(sn), "identical": len(so & sn),
                    "pair_sets_equal": so == sn}
    res["rates_old"] = old["sample_level"]["bucket_rates"]
    res["rates_new"] = new["sample_level"]["bucket_rates"]
    res["nan_distractor_old"] = sum(_nan(f["distractor_mass"]) for f in old["failure_sample_masses"])
    res["nan_distractor_new"] = sum(_nan(f["distractor_mass"]) for f in new["failure_sample_masses"])
    res["per_head_new"] = {h: {k: v[k] for k in ("success_needle_mean", "failure_needle_mean",
                                                   "cliffs_delta_success_vs_failure", "bucket_counts")}
                           for h, v in new["per_head"].items()}
    wo, wn = _load(rd / "e2_signatures_gemma2_9b_it_wu.json"), _load(rd / "e2_signatures_gemma2_9b_it_wu_capfix.json")
    if wo and wn:
        res["wu_rates_old"] = wo["sample_level"]["bucket_rates"]
        res["wu_rates_new"] = wn["sample_level"]["bucket_rates"]
    res["E2_rule"] = ("E3 Gemma-2 results stand (identical pairs)" if res["pairs"]["pair_sets_equal"]
                      else "pair sets differ: label E3 Gemma-2 numbers as measured on the earlier pair set")
    return res


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def report(summary: dict) -> str:
    L = ["# Follow-up experiments A-G: results under the pre-specified rules", "",
         "Generated by `scripts/followup_analysis.py` from the run outputs; rules in "
         "`docs/FOLLOWUP_DECISION_RULES.md`.", ""]
    E = summary["E"]
    L += ["## E - Gemma-2 capture correction", ""]
    if "rates_new" in E:
        L.append(f"- capture-long check: {E['capture_long_check']}")
        L.append(f"- pairs identical: {E['pairs']['identical']}/{E['pairs']['old']} "
                 f"(equal sets: {E['pairs']['pair_sets_equal']}) -> {E['E2_rule']}")
        L.append("- NaN distractor masses: old {} -> new {}".format(E["nan_distractor_old"], E["nan_distractor_new"]))
        L.append("- sample-level rates old -> new: " + ", ".join(
            f"{b} {E['rates_old'].get(b, float('nan')):.3f} -> {E['rates_new'].get(b, float('nan')):.3f}"
            for b in BUCKETS))
        if "wu_rates_new" in E:
            L.append("- Wu heads old -> new: " + ", ".join(
                f"{b} {E['wu_rates_old'].get(b, float('nan')):.3f} -> {E['wu_rates_new'].get(b, float('nan')):.3f}"
                for b in BUCKETS))
    else:
        L.append(f"- {E.get('status')}")
    L += ["", "## F - signature specificity (F-1) and task transfer of the lists (F-2, F-3)", "",
          "| model | capture fail | capture succ | diff [CI] | F-1 | top10 overlap | Spearman | inactive top-5 | F-2 | F-3 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in summary["F"].items():
        if r.get("status") == "not run":
            L.append(f"| {m} | not run |||||||||")
            continue
        f1, f2 = r["F1"], r["F2"]
        L.append(f"| {m} | {f1['r_failure']:.2f} | {f1['r_success']:.2f} | {_f(f1['diff'])} "
                 f"[{_f(f1['lo'])}, {_f(f1['hi'])}] | {f1.get('verdict')} | {f2['top10_overlap']} | "
                 f"{_f(f2['spearman'])} | {f2['part1_top5_inactive']} | {f2['verdict']} | "
                 f"{r.get('F3', {}).get('verdict', '-')} |")
    L += ["", "## A - matched ablation (complete detected set, break direction)", "",
          "| model | pairs | E_all | E_context | E_answer | E_ansdec | D(ctx-ans) | A-1 | A-2 | E3 break (k) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in summary["A"].items():
        if r.get("status") == "not run":
            L.append(f"| {m} | not run |||||||||")
            continue
        e = r["effects_break_full"]

        def c(sc):
            return f"{_f(e[sc]['est'])} [{_f(e[sc]['lo'])}, {_f(e[sc]['hi'])}]" if sc in e else "n/a"
        dd = r.get("D_context_minus_answer", {})
        e3 = r.get("A4_e3_break_effect", {})
        L.append(f"| {m} | {r['n_pairs']} | {c('all')} | {c('context')} | {c('answer')} | {c('ansdec')} | "
                 f"{_f(dd.get('est'))} | {r['A1']} | {r['A2']} | {_f(e3.get('effect'))} ({e3.get('k')}) |")
    for m, r in summary["A"].items():
        if r.get("A5_flip_grades"):
            L.append(f"- {m} A-5 flip grades: " + "; ".join(
                f"{sc}: {dict(g)}" for sc, g in r["A5_flip_grades"].items()))
    for m, r in summary["A"].items():
        if "A3_scope_explanation" in r:
            L.append("")
            L.append(f"**A-3 (Section 7 scope explanation, {m}): {r['A3_scope_explanation']}**")
    L += ["", "## B - Part II under NF4 (parent's heads, identical probes)", "",
          "| NF4 model | B-1 dAcc | B-2 dCapture | B-3 k10 break | B-3 k10 repair | overall |",
          "|---|---|---|---|---|---|"]
    for q, r in summary["B"].items():
        if r.get("status") == "not run":
            L.append(f"| {q} | not run |||||")
            continue
        b1 = r.get("B1_behaviour", {}); b2 = r.get("B2_signature", {}); b3 = r.get("B3_causal", {})

        def cc(x):
            return (f"{_f(x.get('delta'))} [{_f(x.get('lo'))}, {_f(x.get('hi'))}] {x.get('verdict', '')}"
                    if x else "n/a")
        L.append(f"| {q} | {cc(b1)} | {cc(b2)} | {cc(b3.get('k10_break', {}))} | "
                 f"{cc(b3.get('k10_repair', {}))} | {r.get('B_overall')} |")
        if r.get("warning"):
            L.append(f"|  | warning: {r['warning']} |||||")
    L += ["", "## C - attention-pattern transplant (break, complete set)", "",
          "| model | pairs | activation | pattern | content | C-1 | C-2 | C-3 distractor share |",
          "|---|---|---|---|---|---|---|---|"]
    for m, r in summary["C"].items():
        if r.get("status") == "not run":
            L.append(f"| {m} | not run |||||||")
            continue
        k = r["k_full"]

        def ce(v):
            e = r["effects"].get(f"k{k}|{v}|break")
            return f"{_f(e['est'])} [{_f(e['lo'])}, {_f(e['hi'])}]" if e else "n/a"
        L.append(f"| {m} | {r['n_pairs']} | {ce('activation')} | {ce('pattern')} | {ce('content')} | "
                 f"{r.get('C1_full')} | {r.get('C2_full')} | {_f(r.get('C3_distractor_share'))} |")
    G = summary.get("G", {})
    if G:
        L += ["", "## G - layer-matched control (optional)", "",
              "| model | k | direction | effect vs random ctrl (orig.) | effect vs layer ctrl [CI] | q | verdict |",
              "|---|---|---|---|---|---|---|"]
        for m, r in G.items():
            if "by_k" not in r:
                L.append(f"| {m} | {r.get('status')} ||||||")
                continue
            for k in sorted(r["by_k"], key=int):
                for direction, e in r["by_k"][k].items():
                    L.append(f"| {m} | {k} | {direction} | {_f(e.get('random_ctrl_effect'))} | "
                             f"{_f(e['est'])} [{_f(e['lo'])}, {_f(e['hi'])}] | {e['q']:.3g} | {e['verdict']} |")
    D = summary["D"]
    L += ["", "## D - haystack source", ""]
    if "D1_pool" in D:
        L.append(f"- D-1 (pool of the stored sets): **{D['D1_pool']}**")
        L.append(f"- D-2 median J/R_self (WikiText): argmax {_f(D['D2_median_ratio']['argmax'])}, "
                 f"copy {_f(D['D2_median_ratio']['copy'])}")
        for m, v in D["D2"].items():
            L.append(f"  - {m}: " + "; ".join(f"{det} J={x['J']:.3f} ratio={x['ratio']:.2f} "
                                              f"top10={x['top10_overlap']} ({x['verdict']})"
                                              for det, x in v.items()))
    else:
        L.append(f"- {D.get('status')}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply the follow-up decision rules (A-G).")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--haystack-summary", default=None,
                    help="path to retrieval-head-profile .../robustness/D_haystack/summary.json")
    ap.add_argument("--b-tag", default="parentheads")
    ap.add_argument("--a-models", nargs="*", default=None, help="override (first = A-3 model)")
    ap.add_argument("--c-models", nargs="*", default=None)
    ap.add_argument("--f-models", nargs="*", default=None)
    ap.add_argument("--g-models", nargs="*", default=None)
    ap.add_argument("--b-pairs", nargs="*", default=None, help="parent:quant ...")
    args = ap.parse_args(argv)
    b_pairs = [tuple(x.split(":", 1)) for x in args.b_pairs] if args.b_pairs else None
    rd = Path(args.results_dir) if args.results_dir else Path(C.load_paths_cfg()["output"]["results_dir"])
    summary = {"rules": "docs/FOLLOWUP_DECISION_RULES.md (2026-10-15)",
               "E": analyze_E(rd), "F": analyze_F(rd, args.f_models),
               "A": analyze_A(rd, args.a_models),
               "B": analyze_B(rd, args.b_tag, b_pairs),
               "C": analyze_C(rd, args.c_models),
               "G": analyze_G(rd, args.g_models),
               "D": analyze_D(Path(args.haystack_summary) if args.haystack_summary else None)}
    C.write_json(rd / "followup_summary.json", summary)
    (rd / "followup_report.md").write_text(report(summary), encoding="utf-8")
    print(report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
