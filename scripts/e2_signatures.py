"""
E2 — mechanistic signatures of failure (task §6).

Within BREAKING cells only, forms matched success/failure pairs (identical token
skeleton), captures each retrieval head's attention masses at the answer step,
builds the per-(head, cell) success reference distribution, classifies every
FAILURE sample into the four mechanistic buckets (per head and at sample level),
and cross-tabulates the behavioral error label against the mechanistic bucket.
Paired stats compare success-vs-failure needle mass. Measurement only (§12).

Usage:
    python scripts/e2_signatures.py --model llama31_8b_instruct
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _common as C  # noqa: E402

from failure_mech import panel as P  # noqa: E402
from failure_mech import prereg as PR  # noqa: E402
from failure_mech import grading, stats, classify  # noqa: E402
from failure_mech import capture as CAP  # noqa: E402
from failure_mech import detect  # noqa: E402
from failure_mech import patching  # noqa: E402
from failure_mech.probes import ProbeFactory, CellSpec  # noqa: E402
from failure_mech.provenance import make_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e2")
EXP = "e2"

WINDOW_LIMITED = "window_limited"  # excluded-from-silence label (§8)


def _cellspec_from_axes(ax: dict) -> CellSpec:
    return CellSpec(ax["model_key"], int(ax["context_length"]), float(ax["needle_position"]),
                    int(ax["n_distractors"]), ax["similarity"])


def _sample_masses(model, tokenizer, probe, heads, answer_steps):
    """Capture per-head masses (dict (l,h)->HeadMass) for one probe."""
    hm = CAP.capture_head_masses(
        model, tokenizer, probe.input_ids, heads,
        needle_span=probe.needle_span.as_tuple(),
        distractor_spans=[s.as_tuple() for s in probe.distractor_spans],
        answer_steps=answer_steps,
    )
    return {(m.layer, m.head): m for m in hm}


def _classify_head(mass: CAP.HeadMass, ref: classify.Reference, rules: dict) -> str:
    if mass.window_limited:
        return WINDOW_LIMITED
    pt = classify.MassPoint(mass.needle_mass, mass.distractor_mass)
    return classify.classify_point(pt, ref, rules)


def main(argv=None):
    ap = argparse.ArgumentParser(description="E2 signatures (§6).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--answer-steps", type=int, default=1)
    ap.add_argument("--detector", default=None, choices=["argmax", "copy"],
                    help="override detector (E5a: 'copy' = Wu head list)")
    ap.add_argument("--k-heads", type=int, default=None,
                    help="override sample_level_k_heads (E5 sensitivity)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override probe seed (E5 headline repeats)")
    ap.add_argument("--tag", default=None, help="output filename suffix for variants")
    args = ap.parse_args(argv)

    grid_cfg = C.load_yaml(C.config_path("grid.yaml"))
    decoding_cfg = C.load_yaml(C.config_path("decoding.yaml"))
    paths_cfg = C.load_paths_cfg()

    prereg = PR.load_prereg(args.prereg)
    k_heads = args.k_heads if args.k_heads is not None \
        else int(PR.get(prereg, "signature_rules.sample_level_k_heads"))
    pair_min = int(PR.get(prereg, "sampling.pair_min_per_cell"))
    rules = PR.get(prereg, "signature_rules")
    ref_dist = PR.get(prereg, "signature_rules.reference_distribution")
    seed = args.seed if args.seed is not None else int(PR.get(prereg, "seeds.e1_surface"))
    stat_ci = float(PR.get(prereg, "statistics.ci"))
    alpha = float(PR.get(prereg, "causal_criteria.alpha"))

    C.set_global_seed(seed)
    results_dir = Path(paths_cfg["output"]["results_dir"])
    breaking_path = results_dir / f"e1_breaking_cells_{args.model}.json"
    surface_path = results_dir / f"e1_surface_{args.model}.json"
    if not breaking_path.exists() or not surface_path.exists():
        raise SystemExit(f"E2 needs E1 outputs: {breaking_path} and {surface_path}. Run E1 first.")
    breaking = C.load_yaml(breaking_path)["breaking_cells"]
    surface = C.load_yaml(surface_path)["cells"]

    panel_reg = P.load_panel(paths_cfg)
    P.ensure_reuse_on_path(paths_cfg)
    det = detect.load_detection(
        P.detection_dir(paths_cfg), P.resolve_model_key(paths_cfg, args.model),
        int(paths_cfg["detection_artifacts"]["seed"]),
    )
    detector = args.detector or (ref_dist.get("detector", "argmax")
                                 if isinstance(ref_dist, dict) else "argmax")
    heads = det.top_k_heads(k_heads, detector=detector)

    model, tokenizer, mcfg = P.load_model(
        paths_cfg, panel_reg, args.model,
        attn_implementation=decoding_cfg["attn_implementation"]["capture"],
        dtype=decoding_cfg["dtype"],
    )
    factory = ProbeFactory(tokenizer, grid_cfg, args.model)

    per_head_acc: dict = {}         # (l,h) -> {"success_needle":[], "failure_needle":[], buckets Counter}
    sample_bucket_counts = {b: 0 for b in list(classify.BUCKETS) + [WINDOW_LIMITED]}
    crosstab: dict = {}             # behavior_label -> bucket -> count
    reference_stats: dict = {}
    cells_used, pairs_used = [], 0

    for h in breaking:
        rec = surface.get(h)
        if rec is None:
            continue
        cell = _cellspec_from_axes(rec["axes"])
        # 0-distractor breaking cells ARE included: M2 (capture) simply never
        # fires (distractor_mass = 0), but M1/correct_attend/residual still
        # classify from needle mass — the cleanest silence-vs-downstream signal.
        n = int(rec["n_total"])

        # Grading pass (batched, no hooks): identify success/failure per sample.
        probes = [factory.build(cell, i, seed) for i in range(n)]
        gens = patching.generate_plain_batch(
            model, tokenizer, [p.input_ids for p in probes], decoding_cfg)
        succ, fail = [], []
        masses_by_sample = {}
        grades_by_sample = {}
        for i, (probe, gen) in enumerate(zip(probes, gens)):
            g = grading.grade_generation(gen.text, probe.needle_value, probe.distractor_values)
            masses = _sample_masses(model, tokenizer, probe, heads, args.answer_steps)
            masses_by_sample[i] = masses
            grades_by_sample[i] = g
            (succ if g.correct else fail).append(i)

        n_pairs = min(len(succ), len(fail))
        if n_pairs < pair_min:
            log.info("cell %s excluded: %d pairs < pair_min %d", h, n_pairs, pair_min)
            continue
        cells_used.append(h)
        succ_sorted, fail_sorted = sorted(succ), sorted(fail)

        # per-(head,cell) reference from SUCCESS samples
        cell_ref: dict = {}
        for (l, hh) in heads:
            sm = [masses_by_sample[i][(l, hh)].needle_mass for i in succ_sorted]
            cell_ref[(l, hh)] = classify.build_reference(sm)
            reference_stats.setdefault(h, {})[f"L{l}_H{hh}"] = {
                "p5": cell_ref[(l, hh)].p5, "median": cell_ref[(l, hh)].median,
                "n_success": len(sm),
            }
        # sample-level reference from SUCCESS samples' top-k mean needle mass,
        # over the NON-window-limited heads only (§8: window truncation is not a
        # mechanism). Mirrors the failure-side sample-level logic below.
        succ_sl = []
        for i in succ_sorted:
            live = [masses_by_sample[i][hd] for hd in heads
                    if not masses_by_sample[i][hd].window_limited]
            if live:
                succ_sl.append(classify.sample_level_mass(live).needle_mass)
        sl_ref = classify.build_reference(succ_sl)

        # accumulate paired needle mass (per head) + classify FAILURE samples
        for si, fi in zip(succ_sorted[:n_pairs], fail_sorted[:n_pairs]):
            pairs_used += 1
            for (l, hh) in heads:
                acc = per_head_acc.setdefault((l, hh), {
                    "success_needle": [], "failure_needle": [],
                    "buckets": {b: 0 for b in list(classify.BUCKETS) + [WINDOW_LIMITED]}})
                acc["success_needle"].append(masses_by_sample[si][(l, hh)].needle_mass)
                acc["failure_needle"].append(masses_by_sample[fi][(l, hh)].needle_mass)
                bucket = _classify_head(masses_by_sample[fi][(l, hh)], cell_ref[(l, hh)], rules)
                acc["buckets"][bucket] += 1
            # sample-level bucket for the failure sample, over the NON-window-
            # limited top-k heads only (§8). A single local head beyond its
            # window no longer disqualifies the whole sample; the sample is
            # WINDOW_LIMITED only if EVERY top-k head is window-limited.
            live = [masses_by_sample[fi][hd] for hd in heads
                    if not masses_by_sample[fi][hd].window_limited]
            if not live:
                sl_bucket = WINDOW_LIMITED
            else:
                sl_bucket = classify.classify_point(
                    classify.sample_level_mass(live), sl_ref, rules)
            sample_bucket_counts[sl_bucket] += 1
            beh = grades_by_sample[fi].grade  # distractor_hit / other_wrong / empty
            crosstab.setdefault(beh, {b: 0 for b in list(classify.BUCKETS) + [WINDOW_LIMITED]})
            crosstab[beh][sl_bucket] += 1

    # ---- paired stats per head (BH within model) ----
    per_head_out, pvals, keys = {}, [], []
    for (l, hh), acc in per_head_acc.items():
        s = acc["success_needle"]; f = acc["failure_needle"]
        diffs = [a - b for a, b in zip(s, f)]
        perm = stats.paired_permutation(diffs, n_perm=10000, two_sided=True, seed=seed)
        delta = stats.cliffs_delta(s, f)
        ci = stats.bootstrap_ci(lambda x: float(x.mean()), diffs, n=10000, level=stat_ci, seed=seed)
        key = f"L{l}_H{hh}"
        keys.append((l, hh)); pvals.append(perm["p"])
        per_head_out[key] = {
            "n_pairs": len(diffs),
            "success_needle_mean": float(sum(s) / len(s)) if s else float("nan"),
            "failure_needle_mean": float(sum(f) / len(f)) if f else float("nan"),
            "paired_perm_p": perm["p"],
            "cliffs_delta_success_vs_failure": delta,
            "mean_diff_ci": {"lo": ci["lo"], "hi": ci["hi"], "level": ci["level"]},
            "bucket_counts": acc["buckets"],
        }
    bh = stats.bh_correct(pvals, alpha)
    for i, (l, hh) in enumerate(keys):
        per_head_out[f"L{l}_H{hh}"]["bh_rejected"] = bool(bh["rejected"][i])
        per_head_out[f"L{l}_H{hh}"]["bh_qvalue"] = float(bh["qvalues"][i])

    total_sl = sum(sample_bucket_counts.values()) or 1
    provenance = make_provenance(
        script="scripts/e2_signatures.py",
        config_paths=[C.config_path("grid.yaml"), C.config_path("decoding.yaml"),
                      C.config_path("paths.yaml"), PR.resolve_prereg_path(args.prereg)],
        model_key=args.model, model_sha=mcfg["revision"], seeds=seed,
        extra={"detection_seed": det.seed, "detection_artifact": det.path,
               "k_heads": k_heads, "answer_steps": args.answer_steps,
               "detector": detector, "seed_override": args.seed,
               "effective_attn": mcfg.get("effective_attn")},
    )
    out = {
        "provenance": provenance,
        "per_head": per_head_out,
        "sample_level": {
            "bucket_counts": sample_bucket_counts,
            "bucket_rates": {b: c / total_sl for b, c in sample_bucket_counts.items()},
        },
        "crosstab_behavior_x_mechanism": crosstab,
        "reference_stats": reference_stats,
        "pairs_used": pairs_used,
        "cells_used": cells_used,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    C.write_json(results_dir / f"e2_signatures_{args.model}{suffix}.json", out)
    log.info("E2 done: %d cells, %d pairs.", len(cells_used), pairs_used)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
