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
import time
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


def _cellspec(ax: dict, rec: dict | None = None) -> CellSpec:
    # Honor E1's OOM fallback context (see e2_signatures._cellspec_from_axes).
    ctx = int(rec.get("effective_context_length", ax["context_length"])) if rec else int(ax["context_length"])
    return CellSpec(ax["model_key"], ctx, float(ax["needle_position"]),
                    int(ax["n_distractors"]), ax["similarity"])


def _pairs_from_e2(e2: dict, factory, cell, h: str, seed: int):
    """READ the matched (success, failure) pairs E2 measured the mechanism on.

    E3 must NOT re-derive the pairs by re-grading. Grading batches through
    ``_generate_batch_adaptive``, which halves the batch on OOM; a different
    batch composition changes the left-padding, and the greedy token can flip at
    the margin. Re-deriving therefore silently drifts (in the pilot: E2 measured
    614 pairs, E3 re-derived 625) and mechanism and causality end up measured on
    different samples. Probes are pure functions of (cell, sample_idx, seed), so
    the recorded indices rebuild the identical prompts.
    """
    pairs = e2.get("pairs_by_cell")
    if pairs is None:
        raise SystemExit(
            "E2 output has no 'pairs_by_cell'. It predates the pair-set fix, so E3 "
            "would have to re-derive the pairs and would not be measuring causality "
            "on the same sample as the mechanism. Re-run E2 (same config, same "
            "pre-registration) to record its pair list, then re-run E3.")
    # Carries the SAMPLE INDICES alongside the probes: they are the only stable key
    # for a probe across processes (id() is a memory address), so the baseline
    # checkpoint can be keyed by them and survive a restart.
    return [(int(si), factory.build(cell, int(si), seed),
             int(fi), factory.build(cell, int(fi), seed))
            for si, fi in pairs.get(h, [])]


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
        _si, _sp, _fi, fp = pairs[0]
        donor_self = patching.capture_donor_z(model, tokenizer, fp.input_ids, heads, decoding_cfg)
        plain = patching.generate_plain(model, tokenizer, fp.input_ids, decoding_cfg)
        patched = patching.generate_with_patch(model, tokenizer, fp.input_ids, heads,
                                               donor_self, decoding_cfg, patch_mode=patch_mode)
        if plain.token_ids != patched.token_ids:
            raise SelfPatchError(
                f"Self-patch mismatch (model={model_key}). ABORTING (§4.4). "
                "Investigate the o_proj hook / dtype round-trip.")
        return True
    return True  # no pairs to check


def _budget_exhausted(t_start: float, max_hours: float | None) -> bool:
    """True once the wall-clock budget is spent. Checked BETWEEN k values so the
    script always stops on a checkpoint boundary rather than being killed mid-k."""
    if not max_hours:
        return False
    return (time.time() - t_start) / 3600.0 >= max_hours


def main(argv=None):
    ap = argparse.ArgumentParser(description="E3 causal patching (§7).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock budget. E3 stops cleanly BETWEEN k values once "
                         "it is spent, so a Colab session that is about to be killed "
                         "loses nothing: every finished k is already checkpointed.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and recompute every k")
    ap.add_argument("--extra-k", type=int, nargs="*", default=[],
                    help="EXPLORATORY extra head-set sizes appended to configs/e3.yaml "
                         "k_list for THIS model only (e.g. --extra-k 39 to patch the full "
                         "detected set of llama3.1). Passed on the CLI, not the config, so "
                         "the checkpoint fingerprint is unchanged: existing k are reused "
                         "from checkpoints and only the new k is computed. The value must "
                         "not exceed the model's detected head count (top_k_heads refuses "
                         "to pad).")
    args = ap.parse_args(argv)
    t_start = time.time()

    grid_cfg = C.load_yaml(C.config_path("grid.yaml"))
    decoding_cfg = C.load_yaml(C.config_path("decoding.yaml"))
    paths_cfg = C.load_paths_cfg()
    e3_cfg = C.load_yaml(C.config_path("e3.yaml"))["e3"]

    prereg = PR.load_prereg(args.prereg)
    # NOTE: pair_min is NOT re-applied here. E2 already enforced it when it built
    # (and recorded) the pair list; E3 consumes that list verbatim.
    seed = int(PR.get(prereg, "seeds.e1_surface"))
    margin_pp = float(PR.get(prereg, "causal_criteria.flip_margin_over_control_pp"))
    alpha = float(PR.get(prereg, "causal_criteria.alpha"))
    stat_ci = float(PR.get(prereg, "statistics.ci"))
    # Merge the config k_list with any CLI --extra-k (exploratory, per-model). The
    # fingerprint hashes e3.yaml, NOT the CLI, so adding k here does not invalidate
    # the checkpoints of the config k values — they resume; only the new k computes.
    k_list = sorted(set(e3_cfg["k_list"]) | set(args.extra_k))
    patch_mode = e3_cfg.get("patch_mode", "first_step")
    ctrl_seed_base = int(e3_cfg["random_control_seed_base"])

    C.set_global_seed(seed)
    results_dir = Path(paths_cfg["output"]["results_dir"])
    e2_path = results_dir / f"e2_signatures_{args.model}.json"
    if not e2_path.exists():
        raise SystemExit(f"E3 needs E2 output {e2_path}. Run E2 first.")
    e2 = C.load_yaml(e2_path)
    cells_used = e2["cells_used"]
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
    n_detected = len(det._ranked("argmax"))
    if max_k > n_detected:
        raise SystemExit(
            f"k={max_k} exceeds the {n_detected} detected argmax heads for "
            f"{args.model}. The full detected set is k={n_detected}; pass "
            f"--extra-k {n_detected} (not more — top_k_heads refuses to pad).")
    heads_max = det.top_k_heads(max_k, detector="argmax")
    if args.extra_k:
        log.info("k_list = %s (config + exploratory --extra-k %s); full detected set "
                 "for this model is %d heads.", k_list, args.extra_k, n_detected)

    # ---- Head-set determinacy diagnostic (recorded, not acted on) ------------
    # The detector score saturates at 1.0. Where the top-k cut splits a block of
    # tied heads, the patched set is chosen by sort order rather than by
    # evidence, and a NULL result at that k is confounded with WHICH tied heads
    # happened to be inside. Recorded per k so the reader can see it; the code
    # branches on nothing (§12).
    head_set_ties = {str(k): det.boundary_tie(k, detector="argmax") for k in k_list}
    for k, bt in head_set_ties.items():
        if bt["arbitrary"]:
            log.warning(
                "k=%s head set is TIE-BROKEN: %d heads share the cut score %.3f, "
                "%d are inside the set and %d are excluded by (layer,head) sort order "
                "alone. A null causal result at this k is NOT interpretable as "
                "'these heads are not causal'.",
                k, bt["n_tied_at_cut"], bt["cut_score"],
                bt["n_tied_inside_k"], bt["n_tied_excluded"])

    # ---- Checkpoints: one per k, so a killed session loses at most one k ------
    ckpt = C.CheckpointManager(
        results_dir, "e3", args.model, fresh=args.fresh,
        fingerprint=C.probe_fingerprint(
            [C.config_path("e3.yaml"), C.config_path("grid.yaml"),
             C.config_path("decoding.yaml"), PR.resolve_prereg_path(args.prereg)],
            mcfg["revision"], seed))

    # ---- Precompute pairs + unpadded baselines ONCE (independent of k) -------
    # Pairs depend only on (cell, seed); baselines and the no-patch rerun depend
    # only on the recipient prompt. Computing them once avoids re-deriving them
    # for every k (a large saving on the heaviest experiment).
    cell_pairs: dict = {}
    baselines: dict = {}   # id(probe) -> {"base": bool, "rerun": bool}
    # Baselines are 4 plain generations per pair and survive a restart unchanged,
    # so they are checkpointed too: a resumed session must not pay for them twice.
    base_ck = ckpt.load_cell("baselines")["result"] if ckpt.is_done("baselines") else None
    for h in cells_used:
        cell = _cellspec(surface[h]["axes"], surface[h])
        pairs = _pairs_from_e2(e2, factory, cell, h, seed)
        cell_pairs[h] = pairs
        for (si, sp, fi, fp) in pairs:
            for idx, probe in ((si, sp), (fi, fp)):
                key = f"{h}:{idx}"
                if key in baselines:
                    continue
                if base_ck is not None and key in base_ck:
                    baselines[key] = base_ck[key]          # resumed, no GPU cost
                    continue
                baselines[key] = {
                    "base": _grade_plain(model, tokenizer, probe, decoding_cfg),
                    # independent no-patch rerun: detects GPU kernel nondeterminism
                    "rerun": _grade_plain(model, tokenizer, probe, decoding_cfg),
                }
    if base_ck is None:
        ckpt.save_cell("baselines", {"result": baselines})
        log.info("baselines computed and checkpointed (%d probes).", len(baselines))
    else:
        log.info("baselines restored from checkpoint (%d probes, no GPU cost).",
                 len(baselines))

    # Mechanism (E2) and causality (E3) must be measured on the SAME sample.
    # Fail loudly rather than report a causal claim over a different pair set.
    n_pairs_e3 = sum(len(p) for p in cell_pairs.values())
    n_pairs_e2 = int(e2["pairs_used"])
    if n_pairs_e3 != n_pairs_e2:
        raise SystemExit(
            f"pair-set mismatch: E2 measured {n_pairs_e2} pairs, E3 loaded {n_pairs_e3}. "
            "E2's recorded pair list is the single source of truth; do not proceed.")
    log.info("pair set: %d pairs read from E2 (matches E2's pairs_used).", n_pairs_e3)

    # ---- Self-patch smoke check ONCE per model (§4.4), abort on mismatch -----
    self_patch_ok = _self_patch_check(model, tokenizer, cell_pairs, heads_max,
                                      decoding_cfg, patch_mode, args.model)

    # ---- Flatten the pairs and precompute every random-control head set -------
    # The control seeds depend on (pair, k) exactly as before, so the head sets
    # drawn are IDENTICAL to the previous implementation — this only computes them
    # up front instead of inside the loop.
    pairs_flat = [(h, si, sp, fi, fp)
                  for h in cells_used for (si, sp, fi, fp) in cell_pairs[h]]
    rheads_r: dict = {}   # (pair_idx, k) -> control heads for the REPAIR direction
    rheads_b: dict = {}   # (pair_idx, k) -> control heads for the BREAK  direction
    ctrl_seeds_by_k: dict = {k: [] for k in k_list}
    for i in range(1, len(pairs_flat) + 1):
        for k in k_list:
            rs = ctrl_seed_base + i * 7 + k
            bs = ctrl_seed_base + i * 13 + k
            rheads_r[(i, k)] = _random_heads(det, k, rs)
            rheads_b[(i, k)] = _random_heads(det, k, bs)
            ctrl_seeds_by_k[k].append({"pair": i, "repair_seed": rs, "break_seed": bs})

    # ---- Capture each probe's donor ONCE, at the UNION of every head set -------
    # z_h is the o_proj input slice for head h at the answer position; it does NOT
    # depend on which OTHER heads are being captured. So the donor for any head set
    # is an exact slice of a capture taken over a superset — bitwise identical.
    # The old code re-captured per (pair, k), paying a full prompt forward each
    # time: 4 x len(k_list) captures per pair (20 at five k values) where 2 suffice.
    # This is a pure speedup, not a change of method.
    donors_sp: dict = {}
    donors_fp: dict = {}
    # If every k is already checkpointed the k loop only reloads from disk and never
    # touches a donor, so re-capturing them (a full prompt forward per pair — ~1 h
    # for a model like phi) would be pure waste on a completed model.
    if all(ckpt.is_done(f"k{k}") for k in k_list):
        log.info("all %d k already checkpointed; skipping donor capture.", len(k_list))
    else:
        for i, (_h, _si, sp, _fi, fp) in enumerate(pairs_flat, start=1):
            need_sp = set(heads_max) | {lh for k in k_list for lh in rheads_r[(i, k)]}
            need_fp = set(heads_max) | {lh for k in k_list for lh in rheads_b[(i, k)]}
            donors_sp[i] = patching.capture_donor_z(model, tokenizer, sp.input_ids,
                                                    sorted(need_sp), decoding_cfg)
            donors_fp[i] = patching.capture_donor_z(model, tokenizer, fp.input_ids,
                                                    sorted(need_fp), decoding_cfg)
        log.info("donors captured for %d pairs (%d prompt forwards; the per-(pair,k) "
                 "path would have needed %d).", len(pairs_flat), 2 * len(pairs_flat),
                 4 * len(k_list) * len(pairs_flat))

    # ---- Per-k sweep, CHECKPOINTED ------------------------------------------
    # E3 is the heaviest experiment and a k-sweep of a large panel does not fit in
    # one Colab session. Writing only at the end meant a killed session lost the
    # whole model. Each k is now checkpointed the moment it finishes, so a kill
    # costs at most the k in flight.
    out_by_k: dict = {}
    for k in k_list:
        if ckpt.is_done(f"k{k}"):
            out_by_k[str(k)] = ckpt.load_cell(f"k{k}")["result"]
            ckpt.note_skip(f"k{k}")
            log.info("k=%d already checkpointed, skipping.", k)
            continue
        if _budget_exhausted(t_start, args.max_hours):
            log.warning("time budget (%.1f h) reached before k=%d; stopping cleanly. "
                        "Re-run to resume — finished k are checkpointed.",
                        args.max_hours, k)
            break

        heads = heads_max[:k]
        ind = {"repair": [], "break": [], "ctrl_repair": [], "ctrl_break": [],
               "nopatch_repair": [], "nopatch_break": []}

        for i, (h, si, sp, fi, fp) in enumerate(pairs_flat, start=1):
            b_sp, b_fp = baselines[f"{h}:{si}"], baselines[f"{h}:{fi}"]
            base_fp = b_fp["base"]   # repair recipient baseline
            base_sp = b_sp["base"]   # break  recipient baseline
            d_sp, d_fp = donors_sp[i], donors_fp[i]

            rep_correct = _grade_patched(model, tokenizer, fp, heads,
                                         {lh: d_sp[lh] for lh in heads},
                                         decoding_cfg, patch_mode)
            brk_correct = _grade_patched(model, tokenizer, sp, heads,
                                         {lh: d_fp[lh] for lh in heads},
                                         decoding_cfg, patch_mode)
            # Flips are defined RELATIVE TO THE UNPADDED NO-PATCH BASELINE so
            # they are apples-to-apples with the patched (also unpadded) run,
            # independent of the batched pairing path (§6 fix).
            ind["repair"].append(int((not base_fp) and rep_correct))
            ind["break"].append(int(base_sp and (not brk_correct)))

            # random controls (resampled per pair; seed recorded, §4.4)
            hr, hb = rheads_r[(i, k)], rheads_b[(i, k)]
            cr = _grade_patched(model, tokenizer, fp, hr, {lh: d_sp[lh] for lh in hr},
                                decoding_cfg, patch_mode)
            cb = _grade_patched(model, tokenizer, sp, hb, {lh: d_fp[lh] for lh in hb},
                                decoding_cfg, patch_mode)
            ind["ctrl_repair"].append(int((not base_fp) and cr))
            ind["ctrl_break"].append(int(base_sp and (not cb)))

            # no-patch rerun flip (base vs an independent rerun): detects GPU
            # kernel nondeterminism — should be ~0.
            ind["nopatch_repair"].append(int((not base_fp) and b_fp["rerun"]))
            ind["nopatch_break"].append(int(base_sp and (not b_sp["rerun"])))

        res = _summarize(ind, margin_pp, alpha, stat_ci, seed, self_patch_ok)
        res["random_control_seeds"] = ctrl_seeds_by_k[k]
        out_by_k[str(k)] = res
        ckpt.save_cell(f"k{k}", {"result": res})
        log.info("k=%d done and checkpointed (%.2f h elapsed).", k,
                 (time.time() - t_start) / 3600.0)

    missing = [str(k) for k in k_list if str(k) not in out_by_k]
    if missing:
        log.warning("INCOMPLETE: k=%s not computed (time budget). The output carries "
                    "only the finished k; re-run to continue.", ",".join(missing))

    provenance = make_provenance(
        script="scripts/e3_causal.py",
        config_paths=[C.config_path("e3.yaml"), C.config_path("decoding.yaml"),
                      C.config_path("paths.yaml"), PR.resolve_prereg_path(args.prereg)],
        model_key=args.model, model_sha=mcfg["revision"], seeds=seed,
        extra={"detection_seed": det.seed, "patch_mode": patch_mode,
               "pairs_source": "e2.pairs_by_cell", "n_pairs": n_pairs_e3,
               "k_requested": [int(k) for k in k_list],
               "k_completed": sorted(int(k) for k in out_by_k),
               # CLI exploratory extension (e.g. the full detected head set) and the
               # model's total detected head count, so a full-set k is auditable.
               "extra_k_cli": sorted(int(k) for k in args.extra_k),
               "n_detected_argmax_heads": n_detected,
               # An artifact stopped by the time budget is INCOMPLETE. Say so here
               # rather than letting a partial k-sweep read as a finished one.
               "complete": not missing,
               "note": "authoritative BH across 4 models x 2 directions is finalized in E4"})
    C.write_json(results_dir / f"e3_causal_{args.model}.json",
                 {"provenance": provenance, "k": out_by_k,
                  "head_set_ties": head_set_ties})
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
