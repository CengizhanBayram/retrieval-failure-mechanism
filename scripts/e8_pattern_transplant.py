"""
E8 - attention-pattern transplant (merged-paper follow-up; EXPLORATORY).

Why: in E3 a success and a failure of the same cell share the token skeleton but
not their content (keys, needle value, distractor values). Transplanting a head's
output z_h therefore moves two things at once: WHERE the donor head attended
(its attention pattern, e.g. distractor capture) and WHAT it read there (the
donor's values). E8 separates them at the answer step, on the same pairs:

  activation : z' = z_donor                  (E3's activation patching)
  pattern    : z' = row_donor @ V_recipient   (donor's attention, recipient's content)
  content    : z' = row_recipient @ V_donor   (recipient's attention, donor's content)

``pattern`` is the clean causal test of the failure signature: if distractor
capture causes failures, forcing a success's heads to attend where the failure's
heads attended should make the success output ITS OWN distractor value (break),
and forcing a failure's heads to attend like the success should make it output
ITS OWN needle value (repair). ``content`` measures how much of the activation
effect is content transport. All three are patched exactly like E3 (z_h, last
prompt position, first step) and compared with an equal number of random
non-retrieval heads (one draw per pair and budget, seed recorded).

Gate: the self-transplant (recipient's own row with its own values) must
reproduce the head output (max relative error reported) and leave >= 90% of
generations token-identical, else the run aborts.

Supported: models whose caches are never window-truncated at the grid's
lengths (Qwen2.5, Llama-3.1, Mistral-7B-v0.3, OLMo-2, Phi-3.5). Gemma-2 (4,096-
token local window) is refused.

Usage:
    python scripts/e8_pattern_transplant.py --model llama31_8b_instruct
    python scripts/e8_pattern_transplant.py --model qwen25_3b_instruct --k-list 5 10 30
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _common as C  # noqa: E402

from failure_mech import panel as P  # noqa: E402
from failure_mech import prereg as PR  # noqa: E402
from failure_mech import grading, stats, detect, patching  # noqa: E402
from failure_mech import followups as FU  # noqa: E402
from failure_mech.probes import ProbeFactory  # noqa: E402
from failure_mech.provenance import make_provenance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e6_matched_ablation import _cellspec, select_pairs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e8")
EXP = "e8_transplant"
VARIANTS = ("activation", "pattern", "content")


def _grade(gen, probe):
    g = grading.grade_generation(gen.text, probe.needle_value, probe.distractor_values)
    return g.correct, g.grade


def _vectors(model, variant, heads, rv_recipient, rv_donor):
    """Patch vectors for ``heads`` given the recipient/donor captures."""
    out = {}
    for (l, h) in heads:
        if variant == "activation":
            out[(l, h)] = FU.actual_z(rv_donor, l, h)
        elif variant == "pattern":
            out[(l, h)] = FU.transplant_z(model, rv_donor.rows[(l, h)], rv_recipient, l, h)
        else:  # content
            out[(l, h)] = FU.transplant_z(model, rv_recipient.rows[(l, h)], rv_donor, l, h)
    return out


def summarize(records, k_list, variants, seed, level):
    out = {}
    for k in k_list:
        out[str(k)] = {}
        for v in variants:
            res = {}
            for direction, base_key in (("break", "base_s"), ("repair", "base_f")):
                tr, ct, lt, lc = [], [], Counter(), Counter()
                for r in records:
                    base = r[base_key]
                    t = r["cond"][f"{k}|{v}|treat|{direction}"]
                    c = r["cond"][f"{k}|{v}|ctrl|{direction}"]
                    if direction == "break":
                        ft, fc = int(base and not t["correct"]), int(base and not c["correct"])
                    else:
                        ft, fc = int((not base) and t["correct"]), int((not base) and c["correct"])
                    tr.append(ft); ct.append(fc)
                    if ft:
                        lt[t["grade"]] += 1
                    if fc:
                        lc[c["grade"]] += 1
                diffs = [a - b for a, b in zip(tr, ct)]
                perm = stats.paired_permutation(diffs, n_perm=10000, two_sided=True, seed=seed)
                ci_d = stats.bootstrap_ci(lambda x: float(x.mean()), diffs, n=10000, level=level, seed=seed)
                n = len(tr)
                res[direction] = {
                    "n_pairs": n,
                    "flip_rate": sum(tr) / n if n else float("nan"),
                    "ctrl_flip_rate": sum(ct) / n if n else float("nan"),
                    "effect": (sum(tr) - sum(ct)) / n if n else float("nan"),
                    "ci_effect": [ci_d["lo"], ci_d["hi"]], "p_perm": perm["p"],
                    "flip_grades_treat": dict(lt), "flip_grades_ctrl": dict(lc),
                }
            out[str(k)][v] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="E8 attention-pattern transplant (exploratory).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--max-pairs", type=int, default=120)
    ap.add_argument("--k-list", type=int, nargs="*", default=[10],
                    help="budgets; the full detected set is always added")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seed-base", type=int, default=80000)
    ap.add_argument("--gate-n", type=int, default=20,
                    help="probes used for the self-transplant gate")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    t_start = time.time()
    bad = [v for v in args.variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; expected {VARIANTS}")

    grid_cfg = C.load_yaml(C.config_path("grid.yaml"))
    decoding_cfg = C.load_yaml(C.config_path("decoding.yaml"))
    paths_cfg = C.load_paths_cfg()
    prereg = PR.load_prereg(args.prereg)
    seed = int(PR.get(prereg, "seeds.e1_surface"))
    level = float(PR.get(prereg, "statistics.ci"))
    C.set_global_seed(seed)

    results_dir = Path(paths_cfg["output"]["results_dir"])
    e2 = C.load_yaml(results_dir / f"e2_signatures_{args.model}.json")
    surface = C.load_yaml(results_dir / f"e1_surface_{args.model}.json")["cells"]
    cells_used = list(e2["cells_used"])
    chosen = select_pairs(e2, cells_used, args.max_pairs or None)

    panel_reg = P.load_panel(paths_cfg)
    P.ensure_reuse_on_path(paths_cfg)
    det = detect.load_detection(P.detection_dir(paths_cfg),
                                P.resolve_model_key(paths_cfg, args.model),
                                int(paths_cfg["detection_artifacts"]["seed"]))
    model, tokenizer, mcfg = P.load_model(
        paths_cfg, panel_reg, args.model,
        attn_implementation=decoding_cfg["attn_implementation"]["patching"],
        dtype=decoding_cfg["dtype"])
    FU.assert_transplant_supported(model)
    factory = ProbeFactory(tokenizer, grid_cfg, args.model)

    n_detected = len(det._ranked("argmax"))
    k_list = sorted({k for k in args.k_list if k < n_detected} | {n_detected})
    heads_max = det.top_k_heads(n_detected, detector="argmax")
    pool = det.non_retrieval_heads(detector="argmax")
    log.info("E8 %s: %d pairs, k=%s, variants=%s", args.model, len(chosen), k_list, args.variants)

    ckpt = C.CheckpointManager(
        results_dir, EXP + (f"_{args.tag}" if args.tag else ""), args.model, fresh=args.fresh,
        fingerprint=C.probe_fingerprint(
            [C.config_path("grid.yaml"), C.config_path("decoding.yaml"),
             PR.resolve_prereg_path(args.prereg)],
            # the plan is part of what a pair record contains: a resumed run with
            # other budgets/variants must not reuse records that lack them
            mcfg["revision"], f"{seed}|k={k_list}|v={sorted(args.variants)}|sb={args.seed_base}",
            extra_sources=[C.REPO_ROOT / "src/failure_mech/followups.py", Path(__file__)]))

    # ---- self-transplant gate --------------------------------------------------
    gate = {"n": 0, "identical": 0, "max_rel_err": 0.0}
    for (h, si, _fi) in chosen[:args.gate_n]:
        cell = _cellspec(surface[h]["axes"], surface[h])
        p = factory.build(cell, si, seed)
        rv = FU.capture_rows_values(model, p.input_ids, heads_max)
        err = FU.self_transplant_error(model, rv)["max_rel_err"]
        z_self = {lh: FU.transplant_z(model, rv.rows[lh], rv, *lh) for lh in heads_max}
        plain = patching.generate_plain(model, tokenizer, p.input_ids, decoding_cfg)
        same = patching.generate_with_patch(model, tokenizer, p.input_ids, heads_max, z_self,
                                            decoding_cfg, patch_mode="first_step", site="z_h")
        gate["n"] += 1
        gate["identical"] += int(plain.token_ids == same.token_ids)
        gate["max_rel_err"] = max(gate["max_rel_err"], err)
        del rv
    gate["identical_share"] = gate["identical"] / max(gate["n"], 1)
    log.info("self-transplant gate: %d/%d identical generations, max relative error %.3g",
             gate["identical"], gate["n"], gate["max_rel_err"])
    if gate["identical_share"] < 0.9:
        raise SystemExit(f"GATE FAILED: self-transplant left only {gate['identical_share']:.0%} "
                         "of generations identical. The manual row/value read does not reproduce "
                         "the head output for this model; do not interpret E8 here.")

    records = []
    for i, (h, si, fi) in enumerate(chosen, start=1):
        key = f"pair{i}"
        if ckpt.is_done(key):
            records.append(ckpt.load_cell(key)["record"])
            ckpt.note_skip(key)
            continue
        if args.max_hours and (time.time() - t_start) / 3600.0 >= args.max_hours:
            log.warning("time budget reached at pair %d; re-run to resume.", i)
            break
        cell = _cellspec(surface[h]["axes"], surface[h])
        sp, fp = factory.build(cell, si, seed), factory.build(cell, fi, seed)
        ctrl = {}
        for k in k_list:
            rng = random.Random(args.seed_base + 17 * i + k)
            ctrl[k] = rng.sample(pool, k)
        union = sorted(set(heads_max) | {lh for k in k_list for lh in ctrl[k]})
        rv_s = FU.capture_rows_values(model, sp.input_ids, union)
        rv_f = FU.capture_rows_values(model, fp.input_ids, union)
        base_s, _ = _grade(patching.generate_plain(model, tokenizer, sp.input_ids, decoding_cfg), sp)
        base_f, _ = _grade(patching.generate_plain(model, tokenizer, fp.input_ids, decoding_cfg), fp)
        rec = {"cell": h, "succ_idx": si, "fail_idx": fi, "base_s": base_s, "base_f": base_f,
               "ctrl_heads": {str(k): [list(x) for x in ctrl[k]] for k in k_list}, "cond": {}}
        for k in k_list:
            for v in args.variants:
                for arm, heads in (("treat", heads_max[:k]), ("ctrl", ctrl[k])):
                    # break: recipient = success, donor = failure; repair: the reverse
                    for direction, probe, rv_rec, rv_don in (("break", sp, rv_s, rv_f),
                                                             ("repair", fp, rv_f, rv_s)):
                        vecs = _vectors(model, v, heads, rv_rec, rv_don)
                        gen = patching.generate_with_patch(model, tokenizer, probe.input_ids, heads,
                                                           vecs, decoding_cfg,
                                                           patch_mode="first_step", site="z_h")
                        ok, grade = _grade(gen, probe)
                        rec["cond"][f"{k}|{v}|{arm}|{direction}"] = {"correct": bool(ok), "grade": grade}
        del rv_s, rv_f
        ckpt.save_cell(key, {"record": rec})
        records.append(rec)
        if i % 25 == 0:
            log.info("pair %d/%d done (%.2f h)", i, len(chosen), (time.time() - t_start) / 3600.0)

    summary = summarize(records, k_list, args.variants, seed, level) if records else {}
    provenance = make_provenance(
        script="scripts/e8_pattern_transplant.py",
        config_paths=[C.config_path("grid.yaml"), C.config_path("decoding.yaml"),
                      C.config_path("paths.yaml"), PR.resolve_prereg_path(args.prereg)],
        model_key=args.model, model_sha=mcfg["revision"], seeds=seed,
        extra={"status": "exploratory (post hoc follow-up for the merged paper)",
               "k_list": k_list, "variants": args.variants, "gate": gate,
               "n_pairs_chosen": len(chosen), "n_pairs_done": len(records),
               "seed_base": args.seed_base, "effective_attn": mcfg.get("effective_attn"),
               "complete": len(records) == len(chosen)})
    suffix = f"_{args.tag}" if args.tag else ""
    C.write_json(results_dir / f"e8_pattern_transplant_{args.model}{suffix}.json",
                 {"provenance": provenance, "summary": summary, "records": records})
    for k, by_v in summary.items():
        for v, r in by_v.items():
            log.info("  k=%-3s %-7s break eff %+.3f (p=%.4f) | repair eff %+.3f (p=%.4f)",
                     k, v, r["break"]["effect"], r["break"]["p_perm"],
                     r["repair"]["effect"], r["repair"]["p_perm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
