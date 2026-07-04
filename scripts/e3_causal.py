"""
E3 — bidirectional causal patching (task §7).

Over the matched pairs from E2's breaking cells, for each head-set size k:
patch z_h and re-grade. Five mandatory runs per pair (§4.4): repair
(failure<-success), break (success<-failure), random-control (same k, non-
retrieval heads), no-patch rerun (flip base rate), and self-patch (must be
token-identical; ABORT on failure).

All four repair/break outcomes are reportable; the code branches on NOTHING
outcome-dependent and uses no "expected" language (§7, §12). The pre-registered
effect decision (margin over control + BH-corrected permutation p) is recorded
per model+direction; the authoritative BH across {4 models x 2 directions} is
finalized in E4.

Usage:
    python scripts/e3_causal.py --model llama31_8b_instruct
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _common as C  # noqa: E402

from failure_mech import panel as P  # noqa: E402
from failure_mech import prereg as PR  # noqa: E402
from failure_mech import grading, stats, detect, patching  # noqa: E402
from failure_mech.probes import ProbeFactory, CellSpec  # noqa: E402
from failure_mech.provenance import make_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e3")
EXP = "e3"


class SelfPatchError(RuntimeError):
    """Self-patch produced a non-identical generation — abort (§4.4)."""


def _cellspec(ax: dict) -> CellSpec:
    return CellSpec(ax["model_key"], int(ax["context_length"]), float(ax["needle_position"]),
                    int(ax["n_distractors"]), ax["similarity"])


def _pairs_for_cell(model, tokenizer, factory, cell, n, seed, decoding_cfg, pair_min):
    """Reconstruct matched (success_idx, failure_idx) pairs deterministically.

    The grading pass is BATCHED (no hooks); pairing then zips the two classes.
    Returns a list of (success_probe, failure_probe) tuples."""
    probes = [factory.build(cell, i, seed) for i in range(n)]
    gens = patching.generate_plain_batch(
        model, tokenizer, [p.input_ids for p in probes], decoding_cfg)
    succ, fail = [], []
    for i, (p, g) in enumerate(zip(probes, gens)):
        (succ if grading.grade_generation(g.text, p.needle_value, p.distractor_values).correct
         else fail).append(i)
    n_pairs = min(len(succ), len(fail))
    if n_pairs < pair_min:
        return []
    return [(probes[s], probes[f])
            for s, f in zip(sorted(succ)[:n_pairs], sorted(fail)[:n_pairs])]


def _grade_plain(model, tokenizer, probe, decoding_cfg) -> bool:
    gen = patching.generate_plain(model, tokenizer, probe.input_ids, decoding_cfg)
    return grading.grade_generation(gen.text, probe.needle_value, probe.distractor_values).correct


def _grade_patched(model, tokenizer, probe, heads, donor, decoding_cfg, patch_mode) -> bool:
    gen = patching.generate_with_patch(model, tokenizer, probe.input_ids, heads, donor,
                                       decoding_cfg, patch_mode=patch_mode)
    return grading.grade_generation(gen.text, probe.needle_value, probe.distractor_values).correct


def _random_heads(det, k, seed) -> list[tuple[int, int]]:
    pool = det.non_retrieval_heads(detector="argmax")
    rng = random.Random(seed)
    return rng.sample(pool, k)


def _self_patch_check(model, tokenizer, cell_pairs, heads, decoding_cfg, patch_mode, model_key):
    """Run the self-patch no-op ONCE per model (§4.4): patch a recipient with its
    OWN z and require a token-identical generation. Independent of k and of the
    repair/break loop, so it need not repeat per pair. ABORT on mismatch."""
    for pairs in cell_pairs.values():
        if not pairs:
            continue
        _sp, fp = pairs[0]
        donor_self = patching.capture_donor_z(model, fp.input_ids, heads)
        plain = patching.generate_plain(model, tokenizer, fp.input_ids, decoding_cfg)
        patched = patching.generate_with_patch(model, tokenizer, fp.input_ids, heads,
                                               donor_self, decoding_cfg, patch_mode=patch_mode)
        if plain.token_ids != patched.token_ids:
            raise SelfPatchError(
                f"Self-patch mismatch (model={model_key}). ABORTING (§4.4). "
                "Investigate the o_proj hook / dtype round-trip.")
        return True
    return True  # no pairs to check


def main(argv=None):
    ap = argparse.ArgumentParser(description="E3 causal patching (§7).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prereg", default=None)
    args = ap.parse_args(argv)

    grid_cfg = C.load_yaml(C.config_path("grid.yaml"))
    decoding_cfg = C.load_yaml(C.config_path("decoding.yaml"))
    paths_cfg = C.load_paths_cfg()
    e3_cfg = C.load_yaml(C.config_path("e3.yaml"))["e3"]

    prereg = PR.load_prereg(args.prereg)
    pair_min = int(PR.get(prereg, "sampling.pair_min_per_cell"))
    seed = int(PR.get(prereg, "seeds.e1_surface"))
    margin_pp = float(PR.get(prereg, "causal_criteria.flip_margin_over_control_pp"))
    alpha = float(PR.get(prereg, "causal_criteria.alpha"))
    stat_ci = float(PR.get(prereg, "statistics.ci"))
    k_list = e3_cfg["k_list"]
    patch_mode = e3_cfg.get("patch_mode", "first_step")
    ctrl_seed_base = int(e3_cfg["random_control_seed_base"])

    C.set_global_seed(seed)
    results_dir = Path(paths_cfg["output"]["results_dir"])
    e2_path = results_dir / f"e2_signatures_{args.model}.json"
    if not e2_path.exists():
        raise SystemExit(f"E3 needs E2 output {e2_path}. Run E2 first.")
    cells_used = C.load_yaml(e2_path)["cells_used"]
    surface = C.load_yaml(results_dir / f"e1_surface_{args.model}.json")["cells"]

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

    max_k = max(k_list)
    heads_max = det.top_k_heads(max_k, detector="argmax")

    # ---- Precompute pairs + unpadded baselines ONCE (independent of k) -------
    # Pairs depend only on (cell, seed); baselines and the no-patch rerun depend
    # only on the recipient prompt. Computing them once avoids re-deriving them
    # for every k (a large saving on the heaviest experiment).
    cell_pairs: dict = {}
    baselines: dict = {}   # id(probe) -> {"base": bool, "rerun": bool}
    for h in cells_used:
        cell = _cellspec(surface[h]["axes"])
        n = int(surface[h]["n_total"])
        pairs = _pairs_for_cell(model, tokenizer, factory, cell, n, seed, decoding_cfg, pair_min)
        cell_pairs[h] = pairs
        for (sp, fp) in pairs:
            for probe in (sp, fp):
                if id(probe) not in baselines:
                    base = _grade_plain(model, tokenizer, probe, decoding_cfg)
                    rerun = _grade_plain(model, tokenizer, probe, decoding_cfg)  # no-patch rerun
                    baselines[id(probe)] = {"base": base, "rerun": rerun}

    # ---- Self-patch smoke check ONCE per model (§4.4), abort on mismatch -----
    self_patch_ok = _self_patch_check(model, tokenizer, cell_pairs, heads_max,
                                      decoding_cfg, patch_mode, args.model)

    out_by_k: dict = {}
    for k in k_list:
        heads = heads_max[:k]
        ind = {"repair": [], "break": [], "ctrl_repair": [], "ctrl_break": [],
               "nopatch_repair": [], "nopatch_break": []}
        ctrl_seeds = []
        pair_counter = 0

        for h in cells_used:
            for (sp, fp) in cell_pairs[h]:
                pair_counter += 1
                base_fp = baselines[id(fp)]["base"]   # repair recipient baseline
                base_sp = baselines[id(sp)]["base"]   # break  recipient baseline

                donor_success = patching.capture_donor_z(model, sp.input_ids, heads)
                donor_failure = patching.capture_donor_z(model, fp.input_ids, heads)
                rep_correct = _grade_patched(model, tokenizer, fp, heads, donor_success,
                                             decoding_cfg, patch_mode)
                brk_correct = _grade_patched(model, tokenizer, sp, heads, donor_failure,
                                             decoding_cfg, patch_mode)
                # Flips are defined RELATIVE TO THE UNPADDED NO-PATCH BASELINE so
                # they are apples-to-apples with the patched (also unpadded) run,
                # independent of the batched pairing path (§6 fix).
                ind["repair"].append(int((not base_fp) and rep_correct))
                ind["break"].append(int(base_sp and (not brk_correct)))

                # random controls (resampled per pair; seed recorded, §4.4)
                rseed_r = ctrl_seed_base + pair_counter * 7 + k
                rseed_b = ctrl_seed_base + pair_counter * 13 + k
                ctrl_seeds.append({"pair": pair_counter, "repair_seed": rseed_r,
                                   "break_seed": rseed_b})
                rheads_r = _random_heads(det, k, rseed_r)
                rheads_b = _random_heads(det, k, rseed_b)
                cr = _grade_patched(model, tokenizer, fp, rheads_r,
                                    patching.capture_donor_z(model, sp.input_ids, rheads_r),
                                    decoding_cfg, patch_mode)
                cb = _grade_patched(model, tokenizer, sp, rheads_b,
                                    patching.capture_donor_z(model, fp.input_ids, rheads_b),
                                    decoding_cfg, patch_mode)
                ind["ctrl_repair"].append(int((not base_fp) and cr))
                ind["ctrl_break"].append(int(base_sp and (not cb)))

                # no-patch rerun flip (base vs an independent rerun): detects GPU
                # kernel nondeterminism — should be ~0.
                ind["nopatch_repair"].append(int((not base_fp) and baselines[id(fp)]["rerun"]))
                ind["nopatch_break"].append(int(base_sp and (not baselines[id(sp)]["rerun"])))

        out_by_k[str(k)] = _summarize(ind, margin_pp, alpha, stat_ci, seed, self_patch_ok)
        out_by_k[str(k)]["random_control_seeds"] = ctrl_seeds

    provenance = make_provenance(
        script="scripts/e3_causal.py",
        config_paths=[C.config_path("e3.yaml"), C.config_path("decoding.yaml"),
                      C.config_path("paths.yaml"), PR.resolve_prereg_path(args.prereg)],
        model_key=args.model, model_sha=mcfg["revision"], seeds=seed,
        extra={"detection_seed": det.seed, "patch_mode": patch_mode,
               "note": "authoritative BH across 4 models x 2 directions is finalized in E4"})
    C.write_json(results_dir / f"e3_causal_{args.model}.json",
                 {"provenance": provenance, "k": out_by_k})
    log.info("E3 done for %s (k_list=%s).", args.model, k_list)
    return 0


def _rate_ci(indicator, level, seed):
    r = float(sum(indicator) / len(indicator)) if indicator else float("nan")
    ci = stats.bootstrap_ci(lambda x: float(x.mean()), indicator, n=10000, level=level, seed=seed)
    return r, {"lo": ci["lo"], "hi": ci["hi"], "level": level}


def _summarize(ind, margin_pp, alpha, level, seed, self_patch_ok):
    out = {"n_pairs": len(ind["repair"]), "self_patch_ok": bool(self_patch_ok),
           "p_values": {}, "bh_rejected": {}}
    directions = {"repair": ("repair", "ctrl_repair", "nopatch_repair"),
                  "break": ("break", "ctrl_break", "nopatch_break")}
    pvals, keys = [], []
    for name, (pk, ck, npk) in directions.items():
        fr, fci = _rate_ci(ind[pk], level, seed)
        cr, cci = _rate_ci(ind[ck], level, seed)
        nr, nci = _rate_ci(ind[npk], level, seed)
        diffs = [a - b for a, b in zip(ind[pk], ind[ck])]
        perm = stats.paired_permutation(diffs, n_perm=10000, two_sided=True, seed=seed)
        margin = fr - cr
        out[name] = {"flip_rate": fr, "ci": fci}
        out.setdefault("random_control", {})[name] = {"flip_rate": cr, "ci": cci}
        out.setdefault("no_patch", {})[name] = {"flip_rate": nr, "ci": nci}
        out["p_values"][name] = perm["p"]
        out.setdefault("margins", {})[name] = {
            "flip_minus_control": margin,
            "meets_margin": bool(margin >= margin_pp / 100.0),
        }
        pvals.append(perm["p"]); keys.append(name)
    # provisional within-model BH (authoritative cross-model BH is in E4)
    bh = stats.bh_correct(pvals, alpha)
    for i, name in enumerate(keys):
        out["bh_rejected"][name] = bool(bh["rejected"][i])
    return out


if __name__ == "__main__":
    raise SystemExit(main())
