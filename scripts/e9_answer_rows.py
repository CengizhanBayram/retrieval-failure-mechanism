"""
E9 - signature specificity and task transfer of the head lists
(merged-paper follow-up F; EXPLORATORY).

Both questions are answered from the answer-step attention rows of ALL heads,
one prompt forward per probe (``followups.answer_step_all_heads``, checked
against the production per-head capture in the test suite).

F-1  Specificity of the failure signature. E2 classifies FAILING rows only. A
     capture-like row (strongest distractor >= 0.10 and >= needle + 0.05, the E2
     predicate) can also occur when the model answers correctly. Applying the
     same predicate to the paired SUCCESSES gives the base rate; the paired
     difference is the part of "capture at failure" that is failure-specific.
     Also reported: the paired change of the distractor and needle masses.

F-2  Task transfer of the lists. The Part I lists were detected on plain-text
     needle-in-a-haystack probes ("The secret passphrase is X", no chat
     template); Part II probes are chat-formatted with distractors. F-2 re-runs
     the argmax detector ON THE PART II TASK: per head, the fraction of probes
     whose answer-step argmax falls on the needle value. The probes come from
     held-out cells - cells with distractors that the model solves with accuracy
     >= 0.9, disjoint from the E2/E3 pairs - so the list is not selected on the
     samples it may later be patched on. The task list is written as a detection
     artifact ``<model>_taskII`` (results/detection_taskII/) for F-3, an E3 run
     with that list (set RFM_EXTRA_DETECTION_DIRS to the folder).

Outputs: e9_answer_rows_<model>.json (summary) and .npz (per-probe arrays).

Usage:
    python scripts/e9_answer_rows.py --model llama31_8b_instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _common as C  # noqa: E402

from failure_mech import panel as P  # noqa: E402
from failure_mech import prereg as PR  # noqa: E402
from failure_mech import detect, stats  # noqa: E402
from failure_mech import followups as FU  # noqa: E402
from failure_mech.probes import ProbeFactory, CellSpec  # noqa: E402
from failure_mech.provenance import make_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e9")
TOPK = (5, 10, 20)


def _cellspec(ax: dict, rec: dict) -> CellSpec:
    ctx = int(rec.get("effective_context_length", ax["context_length"]))
    return CellSpec(ax["model_key"], ctx, float(ax["needle_position"]),
                    int(ax["n_distractors"]), ax["similarity"])


def detection_cells(surface: dict, breaking: set, min_acc: float) -> list:
    """Held-out task cells: distractors present, accuracy >= min_acc, not breaking.
    Ordered by (context, position, count, similarity) for determinism."""
    out = []
    for h, rec in surface.items():
        ax = rec["axes"]
        acc = rec.get("accuracy")
        if h in breaking or acc is None or int(ax["n_distractors"]) < 2 or float(acc) < min_acc:
            continue
        out.append(h)
    return sorted(out, key=lambda h: (int(surface[h]["axes"]["context_length"]),
                                      float(surface[h]["axes"]["needle_position"]),
                                      int(surface[h]["axes"]["n_distractors"]),
                                      surface[h]["axes"]["similarity"], h))


def round_robin_samples(cells: list, surface: dict, n_total: int) -> list:
    """(cell, sample_idx) pairs, one sample per cell per round, sample indices
    within each cell's graded range, until ``n_total`` probes are chosen."""
    chosen, depth = [], 0
    while len(chosen) < n_total:
        progressed = False
        for h in cells:
            if depth < int(surface[h].get("n_total", 0)):
                chosen.append((h, depth))
                progressed = True
                if len(chosen) >= n_total:
                    break
        if not progressed:
            break
        depth += 1
    return chosen


def ranked(scores: np.ndarray, heads=None) -> list:
    L, H = scores.shape
    pool = heads if heads is not None else [(l, h) for l in range(L) for h in range(H)]
    return sorted(pool, key=lambda lh: (-float(scores[lh[0], lh[1]]), lh[0], lh[1]))


def sample_point(s: dict, heads: list) -> tuple[float, float, int]:
    """Sample-level (needle, distractor) means over the non-window-limited heads,
    exactly as E2 forms its sample-level point."""
    live = [(l, h) for (l, h) in heads if not s["window_limited"][l, h]]
    if not live:
        return float("nan"), float("nan"), 0
    nm = float(np.mean([s["needle"][l, h] for (l, h) in live]))
    dm = float(np.mean([s["dist_max"][l, h] for (l, h) in live]))
    return nm, dm, len(live)


def capture_pred(nm: float, dm: float, min_dist: float, margin: float) -> bool:
    return bool(dm >= min_dist and (dm - nm) >= margin)


def mcnemar_p(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def main(argv=None):
    ap = argparse.ArgumentParser(description="E9: signature specificity + task transfer (follow-up F).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--heads-from", default=None, help="Part I list to evaluate (default: --model)")
    ap.add_argument("--probe-key", default=None)
    ap.add_argument("--e2-tag", default=None, help="read pairs from a tagged E2 artifact")
    ap.add_argument("--max-pairs", type=int, default=300)
    ap.add_argument("--n-detect", type=int, default=200)
    ap.add_argument("--detect-min-acc", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    grid_cfg = C.load_yaml(C.config_path("grid.yaml"))
    decoding_cfg = C.load_yaml(C.config_path("decoding.yaml"))
    paths_cfg = C.load_paths_cfg()
    prereg = PR.load_prereg(args.prereg)
    seed = int(PR.get(prereg, "seeds.e1_surface"))
    level = float(PR.get(prereg, "statistics.ci"))
    k_heads = int(PR.get(prereg, "signature_rules.sample_level_k_heads"))
    min_dist = float(PR.get(prereg, "signature_rules.m2_capture.min_distractor_mass"))
    margin = float(PR.get(prereg, "signature_rules.m2_capture.distractor_over_needle_margin"))
    C.set_global_seed(seed)

    rd = Path(paths_cfg["output"]["results_dir"])
    e2_name = f"e2_signatures_{args.model}" + (f"_{args.e2_tag}" if args.e2_tag else "") + ".json"
    e2 = C.load_yaml(rd / e2_name)
    surface_doc = C.load_yaml(rd / f"e1_surface_{args.model}.json")
    surface = surface_doc["cells"]
    breaking = set(C.load_yaml(rd / f"e1_breaking_cells_{args.model}.json")["breaking_cells"])
    probe_key = C.resolve_probe_key(args.probe_key, surface_doc, args.model)
    heads_key = args.heads_from or args.model

    panel_reg = P.load_panel(paths_cfg)
    P.ensure_reuse_on_path(paths_cfg)
    det_seed = int(paths_cfg["detection_artifacts"]["seed"])
    det = detect.load_detection(P.detection_dir(paths_cfg), P.resolve_model_key(paths_cfg, heads_key),
                                det_seed)
    model, tokenizer, mcfg = P.load_model(
        paths_cfg, panel_reg, args.model,
        attn_implementation=decoding_cfg["attn_implementation"]["capture"],
        dtype=decoding_cfg["dtype"])
    factory = ProbeFactory(tokenizer, grid_cfg, probe_key)
    L = int(model.config.num_hidden_layers)
    H, n_kv = P.head_counts(model)

    # ---------------- F-2: detection on held-out Part II cells -----------------
    dcells = detection_cells(surface, breaking, args.detect_min_acc)
    dsamples = round_robin_samples(dcells, surface, args.n_detect)
    hits = np.zeros((L, H), dtype=np.float64)
    needle_sum = np.zeros((L, H), dtype=np.float64)
    for (h, i) in dsamples:
        p = factory.build(_cellspec(surface[h]["axes"], surface[h]), i, seed)
        s = FU.answer_step_all_heads(model, p.input_ids, p.needle_span.as_tuple(),
                                     [d.as_tuple() for d in p.distractor_spans])
        hits += (s["argmax_cat"] == FU.ARGMAX_NEEDLE)
        needle_sum += s["needle"]
    n_det = max(1, len(dsamples))
    task_scores = hits / n_det
    task_needle = needle_sum / n_det
    task_heads = [(l, h) for l in range(L) for h in range(H) if task_scores[l, h] >= args.tau]
    log.info("F-2: %d held-out probes from %d cells; %d task heads at tau=%.2f",
             len(dsamples), len(dcells), len(task_heads), args.tau)

    part1_ranked = det._ranked("argmax")
    part1_set = set(part1_ranked)
    task_ranked = ranked(task_scores, task_heads)
    from scipy.stats import spearmanr
    rho = float(spearmanr(det.argmax_scores.ravel(), task_scores.ravel())[0]) \
        if det.argmax_scores.shape == task_scores.shape else float("nan")
    union = part1_set | set(task_heads)
    f2 = {
        "n_probes": len(dsamples), "n_cells": len(dcells), "tau": args.tau,
        "n_part1": len(part1_ranked), "n_task": len(task_heads),
        "jaccard": len(part1_set & set(task_heads)) / len(union) if union else float("nan"),
        "score_spearman": rho,
        "top_k_overlap": {str(k): len(set(part1_ranked[:k]) & set(task_ranked[:k]))
                          for k in TOPK if k <= min(len(part1_ranked), len(task_ranked))},
        "part1_top5_task_scores": {f"L{l}_H{h}": float(task_scores[l, h]) for (l, h) in part1_ranked[:5]},
        "part1_top5_task_needle_mass": {f"L{l}_H{h}": float(task_needle[l, h]) for (l, h) in part1_ranked[:5]},
        "task_top10": [[int(l), int(h), float(task_scores[l, h])] for (l, h) in task_ranked[:10]],
    }

    # ---------------- F-1: success vs failure rows on the recorded pairs ---------
    cells_used = list(e2["cells_used"])
    chosen = C.select_pairs_round_robin(e2["pairs_by_cell"], cells_used, args.max_pairs)
    top = det.top_k_heads(k_heads, detector="argmax")
    arr = {k: [] for k in ("needle", "dist_max", "argmax_cat", "window_limited")}
    meta = []
    rows = []   # per pair: (nm_s, dm_s, nm_f, dm_f, cap_s, cap_f)
    per_head = {(l, h): {"s": 0, "f": 0, "n": 0} for (l, h) in top}
    for (hc, si, fi) in chosen:
        cell = _cellspec(surface[hc]["axes"], surface[hc])
        pts = []
        for role, idx in (("success", si), ("failure", fi)):
            p = factory.build(cell, idx, seed)
            s = FU.answer_step_all_heads(model, p.input_ids, p.needle_span.as_tuple(),
                                         [d.as_tuple() for d in p.distractor_spans])
            for k in arr:
                arr[k].append(s[k])
            meta.append({"cell": hc, "idx": int(idx), "role": role})
            pts.append(s)
        s_s, s_f = pts
        nm_s, dm_s, _ = sample_point(s_s, top)
        nm_f, dm_f, _ = sample_point(s_f, top)
        cap_s, cap_f = capture_pred(nm_s, dm_s, min_dist, margin), capture_pred(nm_f, dm_f, min_dist, margin)
        rows.append((nm_s, dm_s, nm_f, dm_f, cap_s, cap_f))
        for (l, h) in top:
            if s_s["window_limited"][l, h] or s_f["window_limited"][l, h]:
                continue
            per_head[(l, h)]["n"] += 1
            per_head[(l, h)]["s"] += capture_pred(float(s_s["needle"][l, h]), float(s_s["dist_max"][l, h]),
                                                  min_dist, margin)
            per_head[(l, h)]["f"] += capture_pred(float(s_f["needle"][l, h]), float(s_f["dist_max"][l, h]),
                                                  min_dist, margin)

    ok = [r for r in rows if not any(map(lambda v: v != v, r[:4]))]
    cs = np.array([int(r[4]) for r in ok]); cf = np.array([int(r[5]) for r in ok])
    b = int(((cs == 1) & (cf == 0)).sum()); c = int(((cs == 0) & (cf == 1)).sum())
    diff = (cf - cs).astype(float)
    dci = stats.bootstrap_ci(lambda x: float(x.mean()), diff, n=10000, level=level, seed=seed)
    d_dist = np.array([r[3] - r[1] for r in ok]); d_need = np.array([r[2] - r[0] for r in ok])

    def paired(x):
        ci = stats.bootstrap_ci(lambda a: float(a.mean()), x, n=10000, level=level, seed=seed)
        return {"mean": float(x.mean()) if x.size else float("nan"), "lo": ci["lo"], "hi": ci["hi"],
                "p_perm": float(stats.paired_permutation(list(x), n_perm=10000, two_sided=True,
                                                          seed=seed)["p"]) if x.size else float("nan")}

    f1 = {
        "n_pairs": len(ok), "k_heads": k_heads, "heads": [list(x) for x in top],
        "capture_rate_failure": float(cf.mean()) if cf.size else float("nan"),
        "capture_rate_success": float(cs.mean()) if cs.size else float("nan"),
        "paired_difference": {"est": float(diff.mean()) if diff.size else float("nan"),
                              "lo": dci["lo"], "hi": dci["hi"]},
        "mcnemar": {"success_only": b, "failure_only": c, "p": mcnemar_p(b, c)},
        "distractor_mass_change": paired(d_dist),
        "needle_mass_change": paired(d_need),
        "per_head": {f"L{l}_H{h}": {"n": v["n"],
                                    "capture_success": v["s"] / v["n"] if v["n"] else float("nan"),
                                    "capture_failure": v["f"] / v["n"] if v["n"] else float("nan")}
                     for (l, h), v in per_head.items()},
    }
    log.info("F-1: capture rate failures %.3f vs successes %.3f (paired diff %.3f [%.3f, %.3f]); "
             "McNemar b=%d c=%d", f1["capture_rate_failure"], f1["capture_rate_success"],
             f1["paired_difference"]["est"], dci["lo"], dci["hi"], b, c)

    # ---------------- outputs ----------------------------------------------------
    provenance = make_provenance(
        script="scripts/e9_answer_rows.py",
        config_paths=[C.config_path("grid.yaml"), C.config_path("decoding.yaml"),
                      C.config_path("paths.yaml"), PR.resolve_prereg_path(args.prereg)],
        model_key=args.model, model_sha=mcfg["revision"], seeds=seed,
        extra={"status": "exploratory follow-up F", "heads_from": heads_key, "probe_key": probe_key,
               "e2_artifact": e2_name, "max_pairs": args.max_pairs, "n_detect": args.n_detect,
               "detect_min_acc": args.detect_min_acc, "effective_attn": mcfg.get("effective_attn"),
               "quantization": mcfg.get("quantization"),
               "detection_samples": [[h, i] for (h, i) in dsamples],
               "pairs": [[h, si, fi] for (h, si, fi) in chosen]})
    suffix = f"_{args.tag}" if args.tag else ""
    C.write_json(rd / f"e9_answer_rows_{args.model}{suffix}.json",
                 {"provenance": provenance, "F1_specificity": f1, "F2_task_transfer": f2})
    np.savez_compressed(rd / f"e9_answer_rows_{args.model}{suffix}.npz",
                        needle=np.asarray(arr["needle"], dtype=np.float16),
                        dist_max=np.asarray(arr["dist_max"], dtype=np.float16),
                        argmax_cat=np.asarray(arr["argmax_cat"], dtype=np.int8),
                        window_limited=np.asarray(arr["window_limited"], dtype=bool),
                        task_scores=task_scores.astype(np.float32),
                        task_needle=task_needle.astype(np.float32),
                        meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8))
    art_dir = rd / "detection_taskII"
    art = {"model": f"{args.model}_taskII", "source": "scripts/e9_answer_rows.py (follow-up F-2)",
           "seed": det_seed, "n_layers": L, "n_heads": len(task_heads), "n_kv_heads": n_kv,
           "argmax_heads": [list(map(int, x)) for x in sorted(task_heads)],
           "copy_heads": [list(map(int, x)) for x in sorted(task_heads)],
           "argmax_scores": task_scores.tolist(), "copy_scores": task_scores.tolist(),
           "tau": args.tau, "n_probes": len(dsamples), "provenance": provenance}
    C.write_json(art_dir / f"{args.model}_taskII_seed{det_seed}.json", art)
    log.info("F-2: Part I vs task list Jaccard %.3f, top-10 overlap %s, score Spearman %.3f",
             f2["jaccard"], f2["top_k_overlap"].get("10"), rho)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
