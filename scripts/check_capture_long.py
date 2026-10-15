"""
Long-context capture check (merged-paper follow-up; run before E2 re-runs).

The test suite verifies the manual attention row against the model's own eager
row on SHORT prompts. That check could not see the Gemma-2 indexing error fixed
on 2026-10-15 (capture._row_for_head): once a prompt is longer than Gemma-2's
4,096-token local window, HF's HybridCache keeps only the last 4,096 keys of a
local layer and the row was read as if it started at position 0.

This script repeats the check on REAL probes from the model's breaking cells, at
their real length, on the real (bf16 or NF4) weights: for each probe it runs the
production ``capture.capture_head_masses`` for the top-k heads and compares the
needle and per-distractor masses with masses computed from the model's own eager
rows (``followups.reference_rows_last_position``; one layer's attention matrix
alive at a time, so 8k-token prompts fit on a 40 GB A100).

Writes ``capture_long_check_<model>.json`` and exits non-zero if any mass differs
by more than ``--tol + --rel-tol x mass`` (window-limited heads are compared too:
their capture mass must be 0 when the needle lies outside the window, as in the
model).

Usage:
    python scripts/check_capture_long.py --model gemma2_9b_it
    python scripts/check_capture_long.py --model gemma2_9b_it_bnb4 --heads-from gemma2_9b_it
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
from failure_mech import capture as CAP  # noqa: E402
from failure_mech import detect  # noqa: E402
from failure_mech import followups as FU  # noqa: E402
from failure_mech.probes import ProbeFactory, CellSpec  # noqa: E402
from failure_mech.provenance import make_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("capture_long")


def _cellspec(ax: dict, rec: dict) -> CellSpec:
    ctx = int(rec.get("effective_context_length", ax["context_length"]))
    return CellSpec(ax["model_key"], ctx, float(ax["needle_position"]),
                    int(ax["n_distractors"]), ax["similarity"])


def pick_cells(surface: dict, breaking: list, max_ctx: int, n_cells: int) -> list:
    """Longest breaking cells not above ``max_ctx`` (the window problem appears
    only above 4,096 tokens), one per (context, needle position), longest first."""
    seen, out = set(), []
    order = sorted(breaking, key=lambda h: (-int(surface[h].get("effective_context_length",
                                                               surface[h]["axes"]["context_length"])),
                                            surface[h]["axes"]["needle_position"], h))
    for h in order:
        rec = surface[h]
        ctx = int(rec.get("effective_context_length", rec["axes"]["context_length"]))
        key = (ctx, rec["axes"]["needle_position"])
        if ctx > max_ctx or key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= n_cells:
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Long-context capture check (follow-up).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--heads-from", default=None)
    ap.add_argument("--probe-key", default=None)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--k-heads", type=int, default=5)
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--samples-per-cell", type=int, default=1)
    ap.add_argument("--max-ctx", type=int, default=8192,
                    help="skip longer cells (the eager reference materialises one "
                         "layer's [heads, T, T] matrix at a time)")
    ap.add_argument("--tol", type=float, default=FU.ABS_TOL,
                    help="absolute part of the allowed |mass difference|")
    ap.add_argument("--rel-tol", type=float, default=FU.REL_TOL,
                    help="relative part (x max mass): eager rows round logits to bf16; "
                         "see followups.tolerance")
    args = ap.parse_args(argv)

    grid_cfg = C.load_yaml(C.config_path("grid.yaml"))
    decoding_cfg = C.load_yaml(C.config_path("decoding.yaml"))
    paths_cfg = C.load_paths_cfg()
    prereg = PR.load_prereg(args.prereg)
    seed = int(PR.get(prereg, "seeds.e1_surface"))
    results_dir = Path(paths_cfg["output"]["results_dir"])
    surface_doc = C.load_yaml(results_dir / f"e1_surface_{args.model}.json")
    surface = surface_doc["cells"]
    breaking = C.load_yaml(results_dir / f"e1_breaking_cells_{args.model}.json")["breaking_cells"]
    probe_key = C.resolve_probe_key(args.probe_key, surface_doc, args.model)

    panel_reg = P.load_panel(paths_cfg)
    P.ensure_reuse_on_path(paths_cfg)
    det = detect.load_detection(P.detection_dir(paths_cfg),
                                P.resolve_model_key(paths_cfg, args.heads_from or args.model),
                                int(paths_cfg["detection_artifacts"]["seed"]))
    heads = det.top_k_heads(args.k_heads, detector="argmax")
    model, tokenizer, mcfg = P.load_model(
        paths_cfg, panel_reg, args.model,
        attn_implementation=decoding_cfg["attn_implementation"]["capture"],
        dtype=decoding_cfg["dtype"])
    factory = ProbeFactory(tokenizer, grid_cfg, probe_key)

    cells = pick_cells(surface, breaking, args.max_ctx, args.n_cells)
    if not cells:
        raise SystemExit(f"no breaking cell with context <= {args.max_ctx}")
    sw = CAP.sliding_window_size(model)
    records, worst = [], 0.0
    for h in cells:
        cell = _cellspec(surface[h]["axes"], surface[h])
        for i in range(args.samples_per_cell):
            p = factory.build(cell, i, seed)
            needle = p.needle_span.as_tuple()
            dspans = [s.as_tuple() for s in p.distractor_spans]
            got = {(m.layer, m.head): m for m in CAP.capture_head_masses(
                model, tokenizer, p.input_ids, heads, needle_span=needle,
                distractor_spans=dspans, answer_steps=1)}
            ref_rows = FU.reference_rows_last_position(model, p.input_ids, heads)
            for (l, hh) in heads:
                row = ref_rows[(l, hh)]
                ref_needle = float(row[needle[0]:needle[1]].sum())
                ref_d = [float(row[a:b].sum()) for (a, b) in dspans]
                m = got[(l, hh)]
                d_needle = abs(m.needle_mass - ref_needle)
                d_dist = max([abs(x - y) for x, y in zip(m.distractor_mass_per, ref_d)] or [0.0])
                # NaN never passes (comparisons with NaN are False)
                ok_needle = bool(d_needle <= float(FU.tolerance(m.needle_mass, ref_needle,
                                                                 args.tol, args.rel_tol)))
                ok_dist = all(bool(abs(x - y) <= float(FU.tolerance(x, y, args.tol, args.rel_tol)))
                              for x, y in zip(m.distractor_mass_per, ref_d))
                bad = not (ok_needle and ok_dist)
                worst = max(worst, d_needle if d_needle == d_needle else 1.0,
                            d_dist if d_dist == d_dist else 1.0)
                records.append({
                    "cell": h, "sample": i, "T": len(p.input_ids), "layer": l, "head": hh,
                    "local_layer": bool(CAP.layer_uses_sliding(model, l)),
                    "window_limited": bool(m.window_limited),
                    "needle_capture": m.needle_mass, "needle_reference": ref_needle,
                    "max_distractor_abs_diff": d_dist, "ok": not bad})
            log.info("cell %s (T=%d): max diff so far %.3g", h, len(p.input_ids), worst)

    passed = all(r["ok"] for r in records)
    out = {
        "provenance": make_provenance(
            script="scripts/check_capture_long.py",
            config_paths=[C.config_path("grid.yaml"), C.config_path("decoding.yaml"),
                          PR.resolve_prereg_path(args.prereg)],
            model_key=args.model, model_sha=mcfg["revision"], seeds=seed,
            extra={"heads": [list(x) for x in heads], "sliding_window": sw,
                   "effective_attn": mcfg.get("effective_attn"),
                   "quantization": mcfg.get("quantization"),
                   "tol": {"abs": args.tol, "rel": args.rel_tol}}),
        "passed": passed, "max_abs_diff": worst, "records": records,
    }
    C.write_json(results_dir / f"capture_long_check_{args.model}.json", out)
    log.info("capture check %s: %d head-probe comparisons, max |diff| %.3g (tol %.2g + %.2g x mass)",
             "PASSED" if passed else "FAILED", len(records), worst, args.tol, args.rel_tol)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
