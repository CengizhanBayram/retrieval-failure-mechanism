"""
E4 — cross-family comparison table (task §8).

Assembles ONE table across the four models of the E2/E3 headline metrics and
finalizes the authoritative BH correction across {4 models x 2 directions} for
the E3 causal decision. Measurement only (§12).

GEMMA-2 handling (§8): each retrieval head is annotated local|global from the
model config's sliding-window pattern. A LOCAL head whose needle lies beyond its
window for a given context is marked ``window_limited`` upstream (E2/capture) and
EXCLUDED from silence statistics — window truncation is never counted as M1. The
per-head local/global annotation is added here from AutoConfig (no weights, no
GPU).

Usage:
    python scripts/e4_families.py [--models a b c d]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _common as C  # noqa: E402

from failure_mech import panel as P  # noqa: E402
from failure_mech import prereg as PR, stats, detect  # noqa: E402
from failure_mech.provenance import make_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e4")

DEFAULT_MODELS = ["llama31_8b_instruct", "gemma2_9b_it", "mistral_7b_instruct",
                  "qwen25_7b_instruct", "olmo2_7b_instruct", "phi35_mini",
                  "qwen25_3b_instruct"]


def _gemma_sliding_annotation(paths_cfg, panel_reg, model_key, det) -> dict | None:
    """For Gemma-2: annotate each detected retrieval head local|global (no GPU)."""
    cfg = P.model_cfg(paths_cfg, panel_reg, model_key)
    if cfg.get("family") != "gemma":
        return None
    from transformers import AutoConfig
    hf = AutoConfig.from_pretrained(cfg["hf_id"], revision=cfg["revision"], trust_remote_code=True)
    layer_types = getattr(hf, "layer_types", None)

    def is_sliding(li):
        if layer_types is not None:
            return str(layer_types[li]).lower().startswith("sliding")
        return (li % 2) == 0

    ann = {}
    for (l, h) in det.top_k_heads(min(30, len(det.argmax_heads)), detector="argmax"):
        ann[f"L{l}_H{h}"] = "local" if is_sliding(l) else "global"
    return {"sliding_window": getattr(hf, "sliding_window", None), "heads": ann}


def main(argv=None):
    ap = argparse.ArgumentParser(description="E4 family table (§8).")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--prereg", default=None)
    args = ap.parse_args(argv)

    paths_cfg = C.load_paths_cfg()
    prereg = PR.load_prereg(args.prereg)
    alpha = float(PR.get(prereg, "causal_criteria.alpha"))
    margin_pp = float(PR.get(prereg, "causal_criteria.flip_margin_over_control_pp"))
    results_dir = Path(paths_cfg["output"]["results_dir"])
    panel_reg = P.load_panel(paths_cfg)
    P.ensure_reuse_on_path(paths_cfg)

    table = {}
    e3_pool = {}  # (model, direction, k) -> {p, margin_met}
    for m in args.models:
        e2p = results_dir / f"e2_signatures_{m}.json"
        e3p = results_dir / f"e3_causal_{m}.json"
        row = {"model": m}
        if e2p.exists():
            e2 = C.load_yaml(e2p)
            row["e2_sample_bucket_rates"] = e2["sample_level"]["bucket_rates"]
            row["e2_pairs_used"] = e2["pairs_used"]
            row["e2_crosstab"] = e2["crosstab_behavior_x_mechanism"]
        if e3p.exists():
            e3 = C.load_yaml(e3p)
            row["e3"] = {}
            for k, kd in e3["k"].items():
                row["e3"][k] = {
                    "repair_flip_rate": kd.get("repair", {}).get("flip_rate"),
                    "break_flip_rate": kd.get("break", {}).get("flip_rate"),
                    "self_patch_ok": kd.get("self_patch_ok"),
                }
                for direction in ("repair", "break"):
                    e3_pool[(m, direction, k)] = {
                        "p": kd["p_values"][direction],
                        "margin_met": kd["margins"][direction]["meets_margin"],
                    }
        # Gemma window annotation
        det_ = None
        try:
            det_ = detect.load_detection(P.detection_dir(paths_cfg),
                                         P.resolve_model_key(paths_cfg, m),
                                         int(paths_cfg["detection_artifacts"]["seed"]))
            ann = _gemma_sliding_annotation(paths_cfg, panel_reg, m, det_)
            if ann:
                row["gemma_local_global"] = ann
        except detect.DetectionError as exc:
            row["detection_note"] = str(exc)
        table[m] = row

    # ---- authoritative BH across {models x directions}, per k (§7) ----
    # Iterate every k that ACTUALLY appears in the E3 artifacts (the union across
    # models), not just configs/e3.yaml k_list — otherwise a per-model exploratory
    # extension (e.g. llama3.1 --extra-k 39, gemma2 --extra-k 77) would be silently
    # dropped from the causal decision. BH's family at a k is the tests present at
    # that k, so a full-set k with a single model is a 2-test (2-direction) family.
    all_ks = sorted({key[2] for key in e3_pool}, key=lambda s: int(s))
    causal_decision = {}
    for k in all_ks:
        keys = [key for key in e3_pool if key[2] == k]
        pvals = [e3_pool[key]["p"] for key in keys]
        if not pvals:
            continue
        bh = stats.bh_correct(pvals, alpha)
        for i, key in enumerate(keys):
            m, direction, kk = key
            causal_decision.setdefault(k, {}).setdefault(m, {})[direction] = {
                "p": e3_pool[key]["p"],
                "bh_rejected": bool(bh["rejected"][i]),
                "margin_met": e3_pool[key]["margin_met"],
                "effect": bool(bh["rejected"][i] and e3_pool[key]["margin_met"]),
            }

    provenance = make_provenance(
        script="scripts/e4_families.py",
        config_paths=[C.config_path("paths.yaml"), C.config_path("e3.yaml"),
                      PR.resolve_prereg_path(args.prereg)],
        seeds=None,
        extra={"models": args.models, "alpha": alpha, "flip_margin_over_control_pp": margin_pp,
               "bh_scope": "models x directions per k"})
    C.write_json(results_dir / "e4_families.json",
                 {"provenance": provenance, "table": table, "causal_decision": causal_decision})
    log.info("E4 done for models %s.", args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
