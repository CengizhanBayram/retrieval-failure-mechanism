"""
E1 - breaking-surface mapping (task §5).

Sweeps the (context, needle-position, distractor count x similarity) grid, grades
greedy generations, and identifies BREAKING cells (accuracy inside the pre-
registered band) that E2/E3 then dissect. Two-stage: a cheap stage-1 pass over
every cell, then a stage-2 top-up of only the breaking cells.

Nothing here is interpreted (§12) - it measures accuracy and behavioral-grade
counts per cell, with a Wilson CI, and records which cells break.

Usage:
    python scripts/e1_breaking_surface.py --model llama31_8b_instruct \
        --config configs/grid.yaml [--dry-run] [--stage 1|2|auto]
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
from failure_mech import grading  # noqa: E402
from failure_mech import stats  # noqa: E402
from failure_mech import patching  # noqa: E402
from failure_mech.probes import ProbeFactory  # noqa: E402
from failure_mech.provenance import make_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e1")

EXP = "e1"


# ---------------------------------------------------------------------------
# Breaking-band logic (all thresholds from prereg; §1.2, §5)
# ---------------------------------------------------------------------------

def _band_bounds(band) -> tuple[float, float]:
    if isinstance(band, dict):
        return float(band["lo"]), float(band["hi"])
    if isinstance(band, (list, tuple)) and len(band) == 2:
        return float(band[0]), float(band[1])
    raise PR.PreregError(f"breaking_band_accuracy must be [lo, hi] or {{lo, hi}}; got {band!r}")


def _is_breaking(acc: float, band) -> bool:
    lo, hi = _band_bounds(band)
    return lo <= acc <= hi


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------

def _grade_counts(grades) -> dict:
    counts = {g: 0 for g in grading.GRADES}
    for gr in grades:
        counts[gr.grade] += 1
    return counts


def _cell_record(cell, grades) -> dict:
    n = len(grades)
    n_correct = sum(1 for g in grades if g.correct)
    acc = n_correct / n if n else float("nan")
    wilson = stats.wilson_ci(n_correct, n, level=0.95)
    return {
        "axes": cell.axes(),
        "cell_hash": cell.cell_hash(),
        "n_stage1": None,   # filled by caller
        "n_total": n,
        "accuracy": acc,
        "wilson_ci95": {"lo": wilson["lo"], "hi": wilson["hi"]},
        "behavioral_grade_counts": _grade_counts(grades),
    }


def _generate_and_grade(model, tokenizer, factory, cell, n_samples, seed, decoding_cfg,
                        start_idx=0):
    """Greedy generation (token-budgeted batches, no hooks) + grading.

    Uses each probe's CANONICAL ``input_ids`` (built once by the factory, chat
    template + special tokens already applied) rather than re-tokenising text -
    so E1 and the E2/E3 single-sequence paths tokenise identically (§4.1, §1.4).
    """
    probes = [factory.build(cell, i, seed) for i in range(start_idx, start_idx + n_samples)]
    tokenizer.padding_side = decoding_cfg.get("padding_side", "left")
    grades = []
    for batch in C.token_budget_batches(
            probes, length_of=lambda p: p.answer_prompt_len,
            max_tokens=decoding_cfg["batch"]["max_tokens_per_batch"],
            max_samples=decoding_cfg["batch"]["max_samples_per_batch"],
            min_samples=decoding_cfg["batch"].get("min_samples_per_batch", 1)):
        gens = patching.generate_plain_batch(
            model, tokenizer, [p.input_ids for p in batch], decoding_cfg)
        for p, gen in zip(batch, gens):
            grades.append(grading.grade_generation(gen.text, p.needle_value, p.distractor_values))
    return grades


def _gen_grade_oom_safe(model, tokenizer, factory, cell, n, seed, decoding_cfg, grid_cfg,
                        start_idx=0):
    """Generate+grade with the ONE documented OOM fallback (§5, §1.9): on a CUDA
    OOM at the fallback context (16384) retry at 12288, log loudly, and flag it.
    Returns (grades, fallback_applied, effective_ctx, eval_cell)."""
    import torch
    from failure_mech.probes import CellSpec
    _OOM = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                             getattr(torch.cuda, "OutOfMemoryError", None)) if t) or (RuntimeError,)
    try:
        grades = _generate_and_grade(model, tokenizer, factory, cell, n, seed,
                                     decoding_cfg, start_idx=start_idx)
        return grades, False, cell.context_length, cell
    except _OOM:
        fb_ctx = C.apply_oom_fallback_ctx(grid_cfg, cell.context_length)
        if fb_ctx == cell.context_length:
            raise  # no fallback defined for this context -> fail loudly (§1.9)
        torch.cuda.empty_cache()
        log.warning("OOM at ctx=%d (cell %s). DOCUMENTED FALLBACK -> ctx=%d (§5).",
                    cell.context_length, cell.cell_hash(), fb_ctx)
        fb_cell = CellSpec(cell.model_key, fb_ctx, cell.needle_position,
                           cell.n_distractors, cell.similarity)
        grades = _generate_and_grade(model, tokenizer, factory, fb_cell, n, seed,
                                     decoding_cfg, start_idx=start_idx)
        return grades, True, fb_ctx, fb_cell


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="E1 breaking-surface mapping (§5).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(C.config_path("grid.yaml")))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage", choices=["1", "2", "auto"], default="auto")
    ap.add_argument("--prereg", default=None, help="override prereg path (tests only)")
    ap.add_argument("--fresh", action="store_true",
                    help="recompute + OVERWRITE every cell, ignoring existing checkpoints")
    args = ap.parse_args(argv)

    grid_cfg = C.load_yaml(args.config)
    decoding_cfg = C.load_yaml(C.config_path("decoding.yaml"))
    paths_cfg = C.load_paths_cfg()

    # HARD GATE: no analysis without a complete pre-registration (§1.2).
    prereg = PR.load_prereg(args.prereg)
    stage1_n = int(PR.get(prereg, "sampling.stage1_n_per_cell"))
    stage2_n = int(PR.get(prereg, "sampling.stage2_topup_n_breaking_cells"))
    band = PR.get(prereg, "breaking_band_accuracy")
    fb = PR.get(prereg, "breaking_band_fallback")
    seed = int(PR.get(prereg, "seeds.e1_surface"))

    cells = C.enumerate_cells(grid_cfg, args.model)
    log.info("E1 grid for %s: %d cells; stage1_n=%d stage2_n=%d seed=%d",
             args.model, len(cells), stage1_n, stage2_n, seed)

    if args.dry_run:
        _dry_run(args.model, cells, grid_cfg, prereg, stage1_n, stage2_n, band)
        return 0

    C.set_global_seed(seed)
    panel_reg = P.load_panel(paths_cfg)
    P.ensure_reuse_on_path(paths_cfg)
    model, tokenizer, mcfg = P.load_model(
        paths_cfg, panel_reg, args.model,
        attn_implementation=decoding_cfg["attn_implementation"]["e1"],
        dtype=decoding_cfg["dtype"],
    )
    factory = ProbeFactory(tokenizer, grid_cfg, args.model)
    fingerprint = C.probe_fingerprint(
        [args.config, C.config_path("decoding.yaml")], mcfg["revision"], seed)
    ckpt = C.CheckpointManager(paths_cfg["output"]["results_dir"], EXP, args.model,
                               fingerprint=fingerprint, fresh=args.fresh)
    if args.fresh:
        log.warning("--fresh: recomputing and OVERWRITING all cells for %s.", args.model)

    surface: dict[str, dict] = {}
    # ---- Stage 1 ----
    for cell in cells:
        h = cell.cell_hash()
        if ckpt.is_done(h):
            rec = ckpt.load_cell(h)
            surface[h] = rec
            ckpt.note_skip(h)
            continue
        grades, fb_applied, eff_ctx, _ = _gen_grade_oom_safe(
            model, tokenizer, factory, cell, stage1_n, seed, decoding_cfg, grid_cfg)
        rec = _cell_record(cell, grades)   # axes keep the grid context; fallback flagged below
        rec["n_stage1"] = stage1_n
        rec["oom_fallback_applied"] = fb_applied
        rec["effective_context_length"] = eff_ctx
        surface[h] = rec
        ckpt.save_cell(h, rec)
        log.info("stage1 %s acc=%.3f (n=%d)%s", h, rec["accuracy"], rec["n_total"],
                 f" [OOM fallback -> {eff_ctx}]" if fb_applied else "")

    breaking, fallback_applied = _select_breaking(surface, band, fb)

    # ---- Stage 2 (top up breaking cells only) ----
    if args.stage in ("2", "auto"):
        for cell in cells:
            h = cell.cell_hash()
            if h not in breaking:
                continue
            rec = surface[h]
            have = rec["n_total"]
            if have >= stage2_n:
                continue
            extra = stage2_n - have
            grades, fb_applied, eff_ctx, _ = _gen_grade_oom_safe(
                model, tokenizer, factory, cell, extra, seed, decoding_cfg, grid_cfg,
                start_idx=have)
            # merge counts
            merged = _merge_records(rec, cell, grades)
            if fb_applied:
                merged["oom_fallback_applied"] = True
                merged["effective_context_length"] = eff_ctx
            surface[h] = merged
            ckpt.save_cell(h, merged)
            log.info("stage2 %s topped to n=%d acc=%.3f", h, merged["n_total"], merged["accuracy"])
        breaking, fallback_applied = _select_breaking(surface, band, fb)

    out_dir = Path(paths_cfg["output"]["results_dir"])
    provenance = make_provenance(
        script="scripts/e1_breaking_surface.py",
        config_paths=[args.config, C.config_path("decoding.yaml"),
                      C.config_path("paths.yaml"), PR.resolve_prereg_path(args.prereg)],
        model_key=args.model, model_sha=mcfg["revision"], seeds=seed,
        extra={"effective_attn": mcfg.get("effective_attn")},
    )
    C.write_json(out_dir / f"e1_surface_{args.model}.json",
                 {"provenance": provenance, "cells": surface})
    C.write_json(out_dir / f"e1_breaking_cells_{args.model}.json",
                 {"provenance": provenance, "breaking_cells": sorted(breaking),
                  "fallback_applied": fallback_applied})
    log.info("E1 done: %d breaking cells (fallback_applied=%s)", len(breaking), fallback_applied)
    return 0


def _merge_records(rec, cell, new_grades):
    counts = dict(rec["behavioral_grade_counts"])
    for g in new_grades:
        counts[g.grade] += 1
    n_total = sum(counts.values())
    n_correct = counts[grading.CORRECT]
    wilson = stats.wilson_ci(n_correct, n_total, level=0.95)
    merged = dict(rec)
    merged["behavioral_grade_counts"] = counts
    merged["n_total"] = n_total
    merged["accuracy"] = n_correct / n_total if n_total else float("nan")
    merged["wilson_ci95"] = {"lo": wilson["lo"], "hi": wilson["hi"]}
    return merged


def _select_breaking(surface, band, fb) -> tuple[set, bool]:
    """Primary band selects breaking cells. The widened fallback band is applied
    ONLY when the researcher pre-registered a trigger AND it fires (§5). Never a
    silent default (§1.9)."""
    breaking = {h for h, r in surface.items() if _is_breaking(r["accuracy"], band)}
    widened = fb.get("widened_band")
    trigger = fb.get("min_breaking_cells")  # optional researcher-authored trigger
    if trigger is None:
        log.info("Fallback band disabled (no breaking_band_fallback.min_breaking_cells "
                 "pre-registered).")
        return breaking, False
    if len(breaking) >= int(trigger):
        return breaking, False
    log.warning("FALLBACK BAND APPLIED: %d breaking cells < trigger %s; widening band to %s.",
                len(breaking), trigger, widened)
    breaking = {h for h, r in surface.items() if _is_breaking(r["accuracy"], widened)}
    for h in breaking:
        surface[h]["fallback_applied"] = True
    return breaking, True


def _dry_run(model_key, cells, grid_cfg, prereg, stage1_n, stage2_n, band):
    lo, hi = _band_bounds(band)
    print(f"\n=== E1 DRY RUN - model={model_key} ===")
    print(f"cells: {len(cells)}  (context x position x [distractor count x similarity])")
    print(f"stage-1 samples/cell: {stage1_n}   stage-2 top-up (breaking): {stage2_n}")
    print(f"breaking band (accuracy): [{lo}, {hi}]  (from preregistration.yaml)")
    print(f"stage-1 total generations: {len(cells) * stage1_n}")
    from collections import Counter
    by_ctx = Counter(c.context_length for c in cells)
    print(f"cells by context length: {dict(sorted(by_ctx.items()))}")
    print("\noutput schema  e1_surface_{model}.json.cells[<cell_hash>]:")
    print("""  {
    "axes": {"model_key","context_length","needle_position","n_distractors","similarity"},
    "cell_hash": "<16-hex>",
    "n_stage1": <int>,
    "n_total": <int>,
    "accuracy": <float in [0,1]>,
    "wilson_ci95": {"lo": <float>, "hi": <float>},
    "behavioral_grade_counts": {"correct","distractor_hit","other_wrong","empty"},
    "oom_fallback_applied": <bool>,            # §5 context OOM fallback 16384->12288
    "effective_context_length": <int>          # actual context used (== axes unless OOM fallback)
  }""")
    print("output  e1_breaking_cells_{model}.json: "
          '{"breaking_cells": [cell_hash,...], "fallback_applied": <bool>}   # band-widen fallback')
    ex = cells[0]
    print(f"\nexample cell: {ex.axes()}  hash={ex.cell_hash()}")
    print("(no model loaded in --dry-run; accuracy/Wilson fields populate on a real run)\n")


if __name__ == "__main__":
    raise SystemExit(main())
