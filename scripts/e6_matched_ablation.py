"""
E6 - matched ablation on the E2/E3 pairs (merged-paper follow-up; EXPLORATORY).

Why: the merged paper compares two causal instruments on the same head lists -
the Part I knockout (zero-ablation of every detected head at every position and
decoding step, plain NIAH at 4k, 8-bit weights) and E3 patching (answer-step
z_h replacement, distractor probes, bf16). They differ in probes, precision and
scope at once. E6 removes the first two differences: it zero-ablates the same
heads on EXACTLY the success/failure probes E2 classified and E3 patched, in
E3's precision and attention backend, and it varies the scope:

  all     every prompt position and decoding step (= the knockout operation)
  context prompt positions 0..T-2 (what the answer step reads)
  answer  the last prompt position only (= E3's scope, zero donor)
  ansdec  the last prompt position plus every decoding step

For each pair the success probe gives a BREAK indicator (baseline correct ->
masked not correct) and the failure probe a REPAIR indicator (baseline not
correct -> masked correct); both are compared with an equal-size random set of
non-retrieval heads drawn once per pair (seed recorded) and reused across scopes.

Head sets: ``full`` = every detected argmax head (as in the knockout), ``top10``
= the pre-specified E3 budget. Pairs: E2's recorded list, subsampled
round-robin over cells to ``--max-pairs`` (deterministic; recorded).

Usage:
    python scripts/e6_matched_ablation.py --model llama31_8b_instruct
    python scripts/e6_matched_ablation.py --model qwen25_3b_instruct --max-pairs 150
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
from failure_mech.probes import ProbeFactory, CellSpec  # noqa: E402
from failure_mech.provenance import make_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e6")
EXP = "e6_ablation"

DEFAULT_PLAN = {"full": ["all", "context", "answer", "ansdec"], "top10": ["all", "answer"]}


def _cellspec(ax: dict, rec: dict | None = None) -> CellSpec:
    ctx = int(rec.get("effective_context_length", ax["context_length"])) if rec else int(ax["context_length"])
    return CellSpec(ax["model_key"], ctx, float(ax["needle_position"]),
                    int(ax["n_distractors"]), ax["similarity"])


def select_pairs(e2: dict, cells_used: list, max_pairs: int | None):
    """Round-robin over cells (in E2's order) through each cell's recorded pair
    list, until ``max_pairs`` pairs are chosen. Deterministic. (Shared with E3's
    ``--max-pairs`` and E8 via ``_common.select_pairs_round_robin``.)"""
    return C.select_pairs_round_robin(e2["pairs_by_cell"], cells_used, max_pairs)


def parse_plan(sets: list[str], scopes: list[str], top10_scopes: list[str]) -> dict:
    plan = {}
    for s in sets:
        if s == "full":
            plan["full"] = list(scopes)
        elif s == "top10":
            plan["top10"] = list(top10_scopes)
        else:
            raise SystemExit(f"unknown head set {s!r} (use full, top10)")
    for s, sc in plan.items():
        bad = [x for x in sc if x not in FU.SCOPES]
        if bad:
            raise SystemExit(f"unknown scope(s) {bad}; expected {FU.SCOPES}")
    return plan


def _grade(gen, probe):
    g = grading.grade_generation(gen.text, probe.needle_value, probe.distractor_values)
    return g.correct, g.grade


def summarize(records: list[dict], plan: dict, seed: int, level: float) -> dict:
    """Per (set, scope): break/repair rates for treatment and control, the
    control-subtracted effect, a paired sign-flip permutation p and bootstrap CIs,
    plus the outcome labels of the flipped generations."""
    out = {}
    for hs, scopes in plan.items():
        out[hs] = {}
        for sc in scopes:
            res = {}
            for direction, base_key in (("break", "base_s"), ("repair", "base_f")):
                tr, ct, labels_t, labels_c = [], [], Counter(), Counter()
                for r in records:
                    base = r[base_key]
                    t = r["cond"][f"{hs}|{sc}|treat|{direction}"]
                    c = r["cond"][f"{hs}|{sc}|ctrl|{direction}"]
                    if direction == "break":
                        ft, fc = int(base and not t["correct"]), int(base and not c["correct"])
                    else:
                        ft, fc = int((not base) and t["correct"]), int((not base) and c["correct"])
                    tr.append(ft); ct.append(fc)
                    if ft:
                        labels_t[t["grade"]] += 1
                    if fc:
                        labels_c[c["grade"]] += 1
                diffs = [a - b for a, b in zip(tr, ct)]
                perm = stats.paired_permutation(diffs, n_perm=10000, two_sided=True, seed=seed)
                ci_t = stats.bootstrap_ci(lambda x: float(x.mean()), tr, n=10000, level=level, seed=seed)
                ci_c = stats.bootstrap_ci(lambda x: float(x.mean()), ct, n=10000, level=level, seed=seed)
                ci_d = stats.bootstrap_ci(lambda x: float(x.mean()), diffs, n=10000, level=level, seed=seed)
                n = len(tr)
                res[direction] = {
                    "n_pairs": n,
                    "flip_rate": sum(tr) / n if n else float("nan"),
                    "ctrl_flip_rate": sum(ct) / n if n else float("nan"),
                    "effect": (sum(tr) - sum(ct)) / n if n else float("nan"),
                    "ci_flip": [ci_t["lo"], ci_t["hi"]], "ci_ctrl": [ci_c["lo"], ci_c["hi"]],
                    "ci_effect": [ci_d["lo"], ci_d["hi"]],
                    "p_perm": perm["p"],
                    "flip_grades_treat": dict(labels_t), "flip_grades_ctrl": dict(labels_c),
                }
            out[hs][sc] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="E6 matched ablation (exploratory).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--max-pairs", type=int, default=120,
                    help="cap on pairs (round-robin over cells); 0 = all recorded pairs")
    ap.add_argument("--sets", nargs="*", default=["full", "top10"])
    ap.add_argument("--scopes", nargs="*", default=DEFAULT_PLAN["full"],
                    help="scopes for the full detected set")
    ap.add_argument("--top10-scopes", nargs="*", default=DEFAULT_PLAN["top10"])
    ap.add_argument("--seed-base", type=int, default=60000,
                    help="random-control seed base (per pair: base + 17*i + |set|)")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--tag", default=None, help="output filename suffix")
    args = ap.parse_args(argv)
    t_start = time.time()

    grid_cfg = C.load_yaml(C.config_path("grid.yaml"))
    decoding_cfg = C.load_yaml(C.config_path("decoding.yaml"))
    paths_cfg = C.load_paths_cfg()
    prereg = PR.load_prereg(args.prereg)
    seed = int(PR.get(prereg, "seeds.e1_surface"))
    level = float(PR.get(prereg, "statistics.ci"))
    plan = parse_plan(args.sets, args.scopes, args.top10_scopes)
    C.set_global_seed(seed)

    results_dir = Path(paths_cfg["output"]["results_dir"])
    e2 = C.load_yaml(results_dir / f"e2_signatures_{args.model}.json")
    if "pairs_by_cell" not in e2:
        raise SystemExit("E2 output lacks pairs_by_cell; re-run E2 first.")
    surface = C.load_yaml(results_dir / f"e1_surface_{args.model}.json")["cells"]
    cells_used = list(e2["cells_used"])
    chosen = select_pairs(e2, cells_used, args.max_pairs or None)
    log.info("E6 %s: %d pairs chosen (of %d recorded) across %d cells; plan=%s",
             args.model, len(chosen), int(e2["pairs_used"]), len(cells_used), plan)

    panel_reg = P.load_panel(paths_cfg)
    P.ensure_reuse_on_path(paths_cfg)
    det = detect.load_detection(P.detection_dir(paths_cfg),
                                P.resolve_model_key(paths_cfg, args.model),
                                int(paths_cfg["detection_artifacts"]["seed"]))
    model, tokenizer, mcfg = P.load_model(
        paths_cfg, panel_reg, args.model,
        attn_implementation=decoding_cfg["attn_implementation"]["patching"],
        dtype=decoding_cfg["dtype"])
    factory = ProbeFactory(tokenizer, grid_cfg, args.model)

    n_detected = len(det._ranked("argmax"))
    head_sets = {"full": det.top_k_heads(n_detected, detector="argmax"),
                 "top10": det.top_k_heads(min(10, n_detected), detector="argmax")}
    pool = det.non_retrieval_heads(detector="argmax")

    ckpt = C.CheckpointManager(
        results_dir, EXP + (f"_{args.tag}" if args.tag else ""), args.model, fresh=args.fresh,
        fingerprint=C.probe_fingerprint(
            [C.config_path("grid.yaml"), C.config_path("decoding.yaml"),
             PR.resolve_prereg_path(args.prereg)],
            # the plan is part of a pair record: resuming with another plan must
            # not reuse records that lack its conditions
            mcfg["revision"],
            f"{seed}|plan={sorted((k, tuple(v)) for k, v in plan.items())}|sb={args.seed_base}",
            extra_sources=[C.REPO_ROOT / "src/failure_mech/followups.py",
                           Path(__file__)]))

    # ---- gate: the mask must write (never-op guard) and the empty mask must be a no-op
    h0, si0, fi0 = chosen[0]
    cell0 = _cellspec(surface[h0]["axes"], surface[h0])
    p0 = factory.build(cell0, si0, seed)
    plain0 = patching.generate_plain(model, tokenizer, p0.input_ids, decoding_cfg)
    empty_gen, _ = FU.generate_masked(model, tokenizer, p0.input_ids, [], "all", decoding_cfg)
    if empty_gen.token_ids != plain0.token_ids:
        raise SystemExit("GATE FAILED: an empty mask changed the generation.")
    _g, zn = FU.generate_masked(model, tokenizer, p0.input_ids, head_sets["full"], "all", decoding_cfg)
    if not zn > 0:
        raise SystemExit("GATE FAILED: the mask zeroed nothing (hook not on o_proj input).")
    log.info("gates passed: empty mask is a no-op; full mask zeroed norm %.3g.", zn)

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
        base_s, _ = _grade(patching.generate_plain(model, tokenizer, sp.input_ids, decoding_cfg), sp)
        base_f, _ = _grade(patching.generate_plain(model, tokenizer, fp.input_ids, decoding_cfg), fp)
        rec = {"cell": h, "succ_idx": si, "fail_idx": fi, "base_s": base_s, "base_f": base_f,
               "ctrl_heads": {}, "cond": {}}
        for hs, scopes in plan.items():
            treat = head_sets[hs]
            rng = random.Random(args.seed_base + 17 * i + len(treat))
            ctrl = rng.sample(pool, len(treat))
            rec["ctrl_heads"][hs] = [list(x) for x in ctrl]
            for sc in scopes:
                for arm, heads in (("treat", treat), ("ctrl", ctrl)):
                    for direction, probe in (("break", sp), ("repair", fp)):
                        gen, _ = FU.generate_masked(model, tokenizer, probe.input_ids, heads, sc,
                                                    decoding_cfg)
                        ok, grade = _grade(gen, probe)
                        rec["cond"][f"{hs}|{sc}|{arm}|{direction}"] = {"correct": bool(ok), "grade": grade}
        ckpt.save_cell(key, {"record": rec})
        records.append(rec)
        if i % 25 == 0:
            log.info("pair %d/%d done (%.2f h)", i, len(chosen), (time.time() - t_start) / 3600.0)

    complete = len(records) == len(chosen)
    summary = summarize(records, plan, seed, level) if records else {}
    provenance = make_provenance(
        script="scripts/e6_matched_ablation.py",
        config_paths=[C.config_path("grid.yaml"), C.config_path("decoding.yaml"),
                      C.config_path("paths.yaml"), PR.resolve_prereg_path(args.prereg)],
        model_key=args.model, model_sha=mcfg["revision"], seeds=seed,
        extra={"status": "exploratory (post hoc follow-up for the merged paper)",
               "plan": plan, "n_pairs_chosen": len(chosen), "n_pairs_done": len(records),
               "n_pairs_recorded_e2": int(e2["pairs_used"]), "max_pairs": args.max_pairs,
               "head_set_sizes": {k: len(v) for k, v in head_sets.items()},
               "seed_base": args.seed_base, "effective_attn": mcfg.get("effective_attn"),
               "complete": complete})
    suffix = f"_{args.tag}" if args.tag else ""
    C.write_json(results_dir / f"e6_matched_ablation_{args.model}{suffix}.json",
                 {"provenance": provenance, "summary": summary,
                  "pairs": [[h, si, fi] for (h, si, fi) in chosen[:len(records)]],
                  "records": records})
    log.info("E6 done for %s (%d/%d pairs).", args.model, len(records), len(chosen))
    for hs, by_scope in summary.items():
        for sc, r in by_scope.items():
            b, rp = r["break"], r["repair"]
            log.info("  %-5s %-7s break %.3f (ctrl %.3f, eff %+.3f, p=%.4f) | repair %.3f (ctrl %.3f, eff %+.3f)",
                     hs, sc, b["flip_rate"], b["ctrl_flip_rate"], b["effect"], b["p_perm"],
                     rp["flip_rate"], rp["ctrl_flip_rate"], rp["effect"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
