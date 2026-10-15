"""
E7 - prepare the NF4 (bitsandbytes 4-bit) re-run of Part II on the bf16
parent's breaking cells (merged-paper follow-up B; EXPLORATORY).

Why: the merged paper's two halves never meet on the same weights. Part I shows
that retrieval-head IDENTITY survives NF4 quantization; Part II dissects the
FAILURE mechanism (E2 signatures, E3 causal patching) in bf16 only. Experiment B
re-runs E2/E3 on the NF4 model, on the bf16 parent's breaking cells and on
byte-identical probes, so that "does the failure mechanism survive deployment
quantization?" is answered with the same instrument.

What this script does (no generation; the behavioural outcomes come from the
E2 grading pass, run with ``--save-grades``):

  1. checks that the quantized entry and the parent share hf_id + revision
     (same tokenizer -> identical token ids);
  2. writes ``e1_surface_<quant>.json`` / ``e1_breaking_cells_<quant>.json``
     holding the parent's breaking cells (same cell hashes, same contexts,
     same n), with ``probe_key`` = the parent's key so E2/E3 build the parent's
     prompts, and ``accuracy`` = null (not measured here; the parent's numbers
     are kept as ``parent_*``);
  3. compares the parent's and the NF4 model's own detected head lists
     (Jaccard, top-k overlap, tie diagnostics);
  4. GATE (needs the GPU, skipped with --skip-gate): loads the NF4 model,
     asserts real 4-bit modules, and checks on a short prefix of a real probe
     that the manual attention row used by E2 matches the model's own eager row
     for the parent's top heads (|diff| <= 0.02 + 0.08 x p per position; see
     followups.tolerance for why bf16 eager rows need a relative band). On failure nothing is written and the script
     exits non-zero, so E2 cannot run on a broken capture.

Run order for one model (see notebooks/B_quant_part2.ipynb):

    python scripts/e7_quant_cells.py --model llama31_8b_instruct_bnb4 --parent llama31_8b_instruct
    python scripts/e2_signatures.py  --model llama31_8b_instruct_bnb4 \
        --heads-from llama31_8b_instruct --tag parentheads --save-grades
    python scripts/e3_causal.py      --model llama31_8b_instruct_bnb4 \
        --heads-from llama31_8b_instruct --e2-tag parentheads --tag parentheads \
        --k-list 10 20 --max-pairs 250
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _common as C  # noqa: E402

from failure_mech import panel as P  # noqa: E402
from failure_mech import prereg as PR  # noqa: E402
from failure_mech import detect  # noqa: E402
from failure_mech import followups as FU  # noqa: E402
from failure_mech.probes import ProbeFactory, CellSpec  # noqa: E402
from failure_mech.provenance import make_provenance, hash_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e7")


def _cellspec(ax: dict, rec: dict) -> CellSpec:
    ctx = int(rec.get("effective_context_length", ax["context_length"]))
    return CellSpec(ax["model_key"], ctx, float(ax["needle_position"]),
                    int(ax["n_distractors"]), ax["similarity"])


def transfer_cells(parent_surface: dict, parent_breaking: dict, parent_key: str) -> tuple[dict, str]:
    """Build the quantized model's E1-format cell records from the parent's.

    Returns (cells, probe_key). Every record keeps the axes (so the cell hash and
    the prompts are the parent's), the evaluated context and n; accuracy fields
    are moved to ``parent_*`` and the live ones set to None.
    """
    cells, keys = {}, set()
    for h in parent_breaking["breaking_cells"]:
        rec = parent_surface["cells"].get(h)
        if rec is None:
            raise SystemExit(f"parent surface has no record for breaking cell {h}")
        ax = rec["axes"]
        keys.add(ax["model_key"])
        # The hash is a pure function of the GRID axes (not the OOM-fallback
        # context): recompute it to make sure the record is what it claims.
        grid_cell = CellSpec(ax["model_key"], int(ax["context_length"]),
                             float(ax["needle_position"]), int(ax["n_distractors"]),
                             ax["similarity"])
        if grid_cell.cell_hash() != h:
            raise SystemExit(f"cell {h}: recomputed hash {grid_cell.cell_hash()} differs - "
                             "the parent surface was not written by this code version.")
        cells[h] = {
            "axes": dict(ax),
            "cell_hash": h,
            "n_stage1": rec.get("n_stage1"),
            "n_total": int(rec["n_total"]),
            "effective_context_length": int(rec.get("effective_context_length",
                                                    ax["context_length"])),
            "oom_fallback_applied": bool(rec.get("oom_fallback_applied", False)),
            "accuracy": None,
            "behavioral_grade_counts": None,
            "wilson_ci95": None,
            "parent_accuracy": rec.get("accuracy"),
            "parent_behavioral_grade_counts": rec.get("behavioral_grade_counts"),
            "parent_wilson_ci95": rec.get("wilson_ci95"),
            "accuracy_source": ("not measured by e7; the quantized model's per-sample "
                                "outcomes are E2's grades_by_cell (--save-grades)"),
        }
    if len(keys) != 1:
        raise SystemExit(f"parent breaking cells carry several model keys {sorted(keys)}")
    probe_key = keys.pop()
    if probe_key != parent_key:
        log.warning("parent cells were built under key '%s' (task key '%s'); using '%s' "
                    "as probe_key.", probe_key, parent_key, probe_key)
    return cells, probe_key


def head_list_comparison(det_parent, det_quant, ks=(5, 10, 20)) -> dict:
    ra, rb = det_parent._ranked("argmax"), det_quant._ranked("argmax")
    sa, sb = set(ra), set(rb)
    out = {"n_parent": len(ra), "n_quant": len(rb),
           "jaccard_full": len(sa & sb) / len(sa | sb) if (sa | sb) else float("nan"),
           "top_k": {}}
    for k in ks:
        if k > min(len(ra), len(rb)):
            continue
        ta, tb = set(ra[:k]), set(rb[:k])
        out["top_k"][str(k)] = {
            "overlap": len(ta & tb),
            "parent_only": sorted([list(x) for x in ta - tb]),
            "quant_only": sorted([list(x) for x in tb - ta]),
            "parent_tie_at_cut": bool(det_parent.boundary_tie(k)["arbitrary"]),
            "quant_tie_at_cut": bool(det_quant.boundary_tie(k)["arbitrary"]),
        }
    return out


def run_gate(paths_cfg, panel_reg, decoding_cfg, grid_cfg, args, cells, probe_key,
             parent_heads, seed) -> dict:
    """Load the quantized model and check capture validity on a real prompt."""
    model, tokenizer, mcfg = P.load_model(
        paths_cfg, panel_reg, args.model,
        attn_implementation=decoding_cfg["attn_implementation"]["capture"],
        dtype=decoding_cfg["dtype"])
    factory = ProbeFactory(tokenizer, grid_cfg, probe_key)
    # smallest-context breaking cell -> cheapest real prompt
    h0 = min(cells, key=lambda h: (cells[h]["effective_context_length"], h))
    cell0 = _cellspec(cells[h0]["axes"], cells[h0])
    probe = factory.build(cell0, 0, seed)
    prefix = list(probe.input_ids[: args.gate_tokens])
    rows = FU.manual_vs_eager_rows(model, prefix, parent_heads[: args.gate_heads])
    probe_hashes = {}
    for h in sorted(cells):
        p = factory.build(_cellspec(cells[h]["axes"], cells[h]), 0, seed)
        probe_hashes[h] = {
            "sha1_input_ids_sample0": hashlib.sha1(
                ",".join(map(str, p.input_ids)).encode()).hexdigest(),
            "len_sample0": len(p.input_ids)}
    footprint = None
    if hasattr(model, "get_memory_footprint"):
        footprint = round(model.get_memory_footprint() / 1e9, 3)
    gate = {
        "quantization": mcfg.get("quantization"),
        "memory_footprint_gb": footprint,
        "effective_attn": mcfg.get("effective_attn"),
        "row_check": rows,
        "row_tolerance": {"abs": FU.ABS_TOL, "rel": FU.REL_TOL},
        "passed": bool(rows["within_tol"]),
        "probe_hashes": probe_hashes,
        "gate_cell": h0,
    }
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return gate


def main(argv=None):
    ap = argparse.ArgumentParser(description="E7: NF4 cell transfer + capture gate (follow-up B).")
    ap.add_argument("--model", required=True, help="quantized panel key, e.g. llama31_8b_instruct_bnb4")
    ap.add_argument("--parent", required=True, help="bf16 parent key whose E1 cells are used")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--skip-gate", action="store_true",
                    help="do not load the model (CPU dry preparation). The E2/E3 runs still "
                         "re-check self-patching; the row gate is then NOT recorded.")
    ap.add_argument("--gate-tokens", type=int, default=512)
    ap.add_argument("--gate-heads", type=int, default=5)
    args = ap.parse_args(argv)

    grid_cfg = C.load_yaml(C.config_path("grid.yaml"))
    decoding_cfg = C.load_yaml(C.config_path("decoding.yaml"))
    paths_cfg = C.load_paths_cfg()
    prereg = PR.load_prereg(args.prereg)
    seed = int(PR.get(prereg, "seeds.e1_surface"))
    k_sample = int(PR.get(prereg, "signature_rules.sample_level_k_heads"))

    panel_reg = P.load_panel(paths_cfg)
    P.ensure_reuse_on_path(paths_cfg)
    cfg_q = P.model_cfg(paths_cfg, panel_reg, args.model)
    cfg_p = P.model_cfg(paths_cfg, panel_reg, args.parent)
    if cfg_q.get("quant") != "bnb4":
        raise SystemExit(f"{args.model} is not a bnb4 panel entry (quant={cfg_q.get('quant')!r}).")
    if cfg_p.get("quant"):
        raise SystemExit(f"parent {args.parent} is itself quantized ({cfg_p.get('quant')}).")
    if (cfg_q["hf_id"], cfg_q["revision"]) != (cfg_p["hf_id"], cfg_p["revision"]):
        raise SystemExit(
            f"{args.model} ({cfg_q['hf_id']}@{cfg_q['revision'][:10]}) and {args.parent} "
            f"({cfg_p['hf_id']}@{cfg_p['revision'][:10]}) are not the same checkpoint; the "
            "tokenizer - and therefore the probes - could differ.")

    results_dir = Path(paths_cfg["output"]["results_dir"])
    ps_path = results_dir / f"e1_surface_{args.parent}.json"
    pb_path = results_dir / f"e1_breaking_cells_{args.parent}.json"
    if not ps_path.exists() or not pb_path.exists():
        raise SystemExit(f"need the parent's E1 outputs {ps_path} and {pb_path}")
    parent_surface = C.load_yaml(ps_path)
    parent_breaking = C.load_yaml(pb_path)
    cells, probe_key = transfer_cells(parent_surface, parent_breaking, args.parent)

    det_dir = P.detection_dir(paths_cfg)
    det_seed = int(paths_cfg["detection_artifacts"]["seed"])
    det_p = detect.load_detection(det_dir, P.resolve_model_key(paths_cfg, args.parent), det_seed)
    det_q = detect.load_detection(det_dir, P.resolve_model_key(paths_cfg, args.model), det_seed)
    heads_cmp = head_list_comparison(det_p, det_q)
    parent_heads = det_p.top_k_heads(max(k_sample, args.gate_heads), detector="argmax")
    log.info("head lists: Jaccard %.3f; top-5 overlap %s; top-10 overlap %s",
             heads_cmp["jaccard_full"], heads_cmp["top_k"].get("5", {}).get("overlap"),
             heads_cmp["top_k"].get("10", {}).get("overlap"))

    gate = None
    if not args.skip_gate:
        gate = run_gate(paths_cfg, panel_reg, decoding_cfg, grid_cfg, args, cells,
                        probe_key, parent_heads, seed)
        log.info("gate: max |manual-eager| = %.4g over %d heads (tol %.2g + %.2g x p) -> %s",
                 gate["row_check"]["max_abs_diff"], len(gate["row_check"]["per_head"]),
                 FU.ABS_TOL, FU.REL_TOL, "PASS" if gate["passed"] else "FAIL")

    provenance = make_provenance(
        script="scripts/e7_quant_cells.py",
        config_paths=[C.config_path("grid.yaml"), C.config_path("decoding.yaml"),
                      C.config_path("paths.yaml"), PR.resolve_prereg_path(args.prereg)],
        model_key=args.model, model_sha=cfg_q["revision"], seeds=seed,
        extra={"status": "exploratory follow-up B", "parent": args.parent,
               "probe_key": probe_key, "quant": cfg_q.get("quant"),
               "parent_surface_sha256": hash_file(ps_path),
               "gate_skipped": gate is None})
    report = {"provenance": provenance, "parent": args.parent, "probe_key": probe_key,
              "n_cells": len(cells), "head_lists": heads_cmp, "gate": gate}
    C.write_json(results_dir / f"e7_quant_cells_{args.model}.json", report)

    if gate is not None and not gate["passed"]:
        log.error("capture gate FAILED - E1-format files NOT written; E2/E3 must not run.")
        return 2

    surface_doc = {"provenance": provenance, "cells": cells, "probe_key": probe_key,
                   "transferred_from": args.parent}
    breaking_doc = {"provenance": provenance, "breaking_cells": sorted(cells),
                    "fallback_applied": parent_breaking.get("fallback_applied", False),
                    "transferred_from": args.parent}
    C.write_json(results_dir / f"e1_surface_{args.model}.json", surface_doc)
    C.write_json(results_dir / f"e1_breaking_cells_{args.model}.json", breaking_doc)
    log.info("E7 done: %d parent breaking cells transferred to %s (probe_key=%s).",
             len(cells), args.model, probe_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
