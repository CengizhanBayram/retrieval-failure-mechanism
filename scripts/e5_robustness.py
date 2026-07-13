"""
E5 — robustness of the E2/E3 headlines (task §9).

(a) Recompute E2 headline metrics on the Wu-detector head list (``--detector
    copy``) and at ``sensitivity_k_heads``.
(b) ``seeds.e2_e3_headline_repeats`` full repeats of the E2/E3 headlines (seed
    varies probe instantiation); report mean +/- range.
(c) R_self ceiling on breaking cells: test-retest agreement of the sample-level
    bucket rates across the seed repeats (the reliability ceiling the E2 signal
    is measured against).
(d) ``capture.answer_steps=3`` sensitivity.

E5 orchestrates E2/E3 variant runs (each a clean subprocess with the pinned
model) and aggregates their JSON. Measurement only (§12).

Usage:
    python scripts/e5_robustness.py --model llama31_8b_instruct
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _common as C  # noqa: E402

from failure_mech import prereg as PR  # noqa: E402
from failure_mech.provenance import make_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e5")

E2 = str(C.REPO_ROOT / "scripts" / "e2_signatures.py")


def _run(cmd: list[str]) -> bool:
    """Run one E2 variant. Returns True on success; a failing variant (e.g. a
    model with fewer copy-heads than k) is logged and skipped rather than killing
    the whole robustness run (it just won't contribute to the aggregate)."""
    log.info("RUN %s", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        log.warning("variant FAILED (%d), skipping: %s", r.returncode, " ".join(cmd))
        return False
    return True


def _seeds(spec) -> list[int]:
    if isinstance(spec, int):
        return list(range(spec))
    return list(spec)


def _bucket_rates(results_dir: Path, model: str, tag: str | None) -> dict:
    suffix = f"_{tag}" if tag else ""
    p = results_dir / f"e2_signatures_{model}{suffix}.json"
    if not p.exists():
        return {}
    return C.load_yaml(p)["sample_level"]["bucket_rates"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="E5 robustness (§9).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--skip-runs", action="store_true",
                    help="aggregate existing variant outputs without re-running")
    ap.add_argument("--variants", default="all",
                    help="comma-separated subset of robustness checks to run: "
                         "wu (Wu/copy detector head list), ksens (sensitivity k), "
                         "steps3 (answer_steps=3), seeds (headline seed repeats), "
                         "m2sens (alternative M2 distractor-mass floor). "
                         "'all' runs every check. Each variant is a full E2 re-run, "
                         "so a subset keeps the panel tractable.")
    ap.add_argument("--m2-alt-floor", type=float, default=None,
                    help="alternative M2 min_distractor_mass for the 'm2sens' variant. "
                         "The PRE-REGISTERED floor stays the primary analysis; this "
                         "only produces a side-by-side sensitivity artifact.")
    args = ap.parse_args(argv)

    paths_cfg = C.load_paths_cfg()
    prereg = PR.load_prereg(args.prereg)
    sens_k = int(PR.get(prereg, "signature_rules.sensitivity_k_heads"))
    repeats = _seeds(PR.get(prereg, "seeds.e2_e3_headline_repeats"))
    results_dir = Path(paths_cfg["output"]["results_dir"])
    base = [sys.executable, E2, "--model", args.model]
    if args.prereg:
        base += ["--prereg", args.prereg]

    want = {v.strip() for v in args.variants.split(",")} if args.variants != "all" \
        else {"wu", "ksens", "steps3", "seeds", "m2sens"}
    known = {"wu", "ksens", "steps3", "seeds", "m2sens"}
    unknown = want - known
    if unknown:
        raise SystemExit(f"--variants: unknown {sorted(unknown)}; known: {sorted(known)}")
    if "m2sens" in want and args.m2_alt_floor is None:
        raise SystemExit("--variants m2sens requires --m2-alt-floor (the alternative "
                         "M2 floor to report ALONGSIDE the pre-registered one).")

    variants = {}
    if "wu" in want:
        variants["wu"] = base + ["--detector", "copy", "--tag", "wu"]          # (a)
    if "ksens" in want:
        variants["ksens"] = base + ["--k-heads", str(sens_k), "--tag", "ksens"]  # (a)
    if "steps3" in want:
        variants["steps3"] = base + ["--answer-steps", "3", "--tag", "steps3"]   # (d)
    if "seeds" in want:
        for i, s in enumerate(repeats):
            variants[f"seed{i}"] = base + ["--seed", str(s), "--tag", f"seed{i}"]  # (b)
    if "m2sens" in want:
        # (e) M2 THRESHOLD SENSITIVITY. The pre-registered floor stays the primary
        # analysis; this is a tagged side-by-side artifact, never an overwrite.
        variants["m2sens"] = base + ["--m2-min-distractor-mass", str(args.m2_alt_floor),
                                     "--tag", "m2sens"]

    log.info("E5 variants to run for %s: %s", args.model, list(variants) or "(none)")
    if not args.skip_runs:
        for name, cmd in variants.items():
            _run(cmd)

    # ---- aggregate ----
    agg = {"headline_variants": {}, "seed_repeats": {}, "R_self": {}}
    # "primary" = the pre-registered analysis, carried alongside every variant so
    # the sensitivity comparison is always read side-by-side, never as a
    # replacement (§ amendment: the M2 floor is NOT re-tuned post hoc).
    agg["primary"] = _bucket_rates(results_dir, args.model, None)
    for name in ("wu", "ksens", "steps3", "m2sens"):
        agg["headline_variants"][name] = _bucket_rates(results_dir, args.model, name)
    agg["m2_floor"] = {
        "preregistered": float(PR.get(prereg, "signature_rules.m2_capture.min_distractor_mass")),
        "alternative": args.m2_alt_floor,
    }

    repeat_rates = [_bucket_rates(results_dir, args.model, f"seed{i}")
                    for i in range(len(repeats))]
    repeat_rates = [r for r in repeat_rates if r]
    if repeat_rates:
        buckets = sorted({b for r in repeat_rates for b in r})
        summary = {}
        for b in buckets:
            vals = [r.get(b, 0.0) for r in repeat_rates]
            summary[b] = {"mean": sum(vals) / len(vals),
                          "min": min(vals), "max": max(vals),
                          "range": max(vals) - min(vals)}
        agg["seed_repeats"] = {"per_bucket": summary, "n_repeats": len(repeat_rates)}
        # (c) R_self: test-retest agreement = 1 - mean pairwise range across seeds,
        # averaged over buckets (a reliability ceiling; 1.0 = perfectly stable).
        mean_range = sum(summary[b]["range"] for b in buckets) / len(buckets) if buckets else float("nan")
        agg["R_self"] = {"bucket_rate_stability": 1.0 - mean_range,
                         "mean_bucket_range": mean_range,
                         "definition": "1 - mean over buckets of (max-min sample-level rate) across seed repeats"}

    provenance = make_provenance(
        script="scripts/e5_robustness.py",
        config_paths=[C.config_path("paths.yaml"), PR.resolve_prereg_path(args.prereg)],
        model_key=args.model, seeds=repeats,
        extra={"sensitivity_k_heads": sens_k, "variants": list(variants)})
    C.write_json(results_dir / f"e5_robustness_{args.model}.json",
                 {"provenance": provenance, **agg})
    log.info("E5 done for %s.", args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
