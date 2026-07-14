"""
Reproduction check for the E2 pair-recording re-run (amendment §I).

The re-run is configuration-identical to the run that produced the pre-fix
artifacts — same pre-registration, same M2 floor, same seed. The ONLY intended
difference is that it now records ``pairs_by_cell``. It must therefore reproduce
the pre-fix numbers exactly.

If it does not, E2 is not deterministic run-to-run (the adaptive OOM batch-halving
in the grading pass is the suspect: a different batch composition changes the
left-padding and a greedy token can flip at the margin). That would outrank every
other open question, and nothing may be called confirmatory until it is resolved.

Compares each ``e2_signatures_{model}.json`` against its
``.pre_pairfix`` backup. Pure JSON — no GPU, no model load. Measurement only (§12).

Usage:
    python scripts/check_e2_repro.py
    RFM_RESULTS_DIR=/path/to/rfm_results python scripts/check_e2_repro.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _common as C  # noqa: E402

VARIANT_TAGS = ("_wu", "_ksens", "_seed", "_steps3", "_m2sens")


def _is_primary(p: Path) -> bool:
    return not any(t in p.name for t in VARIANT_TAGS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E2 re-run reproduction check (§I).")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir) if args.results_dir \
        else Path(C.load_paths_cfg()["output"]["results_dir"])

    backups = sorted(p for p in results_dir.glob("e2_signatures_*.json.pre_pairfix")
                     if _is_primary(p))
    if not backups:
        print(f"No .pre_pairfix backups in {results_dir}. Run notebook 09 section 1 "
              "first (it snapshots the pre-fix artifacts).")
        return 1

    print(f"{'model':22s} {'pairs before -> after':24s} {'bucket rates':11s} verdict")
    drift, pending, ok_models = [], [], []
    for bak in backups:
        cur = bak.with_suffix("")            # strip .pre_pairfix
        model = cur.name[len("e2_signatures_"):-len(".json")]
        old = json.loads(bak.read_text(encoding="utf-8"))
        if not cur.exists():
            pending.append(model)
            print(f"{model:22s} {'(current artifact missing)':24s}")
            continue
        new = json.loads(cur.read_text(encoding="utf-8"))
        if "pairs_by_cell" not in new:
            pending.append(model)
            print(f"{model:22s} {'(not re-run yet)':24s}")
            continue

        same_pairs = new["pairs_used"] == old["pairs_used"]
        nb = new["sample_level"]["bucket_rates"]
        ob = old["sample_level"]["bucket_rates"]
        same_rates = all(abs(nb.get(k, 0.0) - ob.get(k, 0.0)) < 1e-9
                         for k in set(nb) | set(ob))
        # A recorded pair list that disagrees with its own pairs_used would mean the
        # recording itself is wrong -- check it too.
        n_recorded = sum(len(v) for v in new["pairs_by_cell"].values())
        consistent = n_recorded == new["pairs_used"]

        good = same_pairs and same_rates and consistent
        (ok_models if good else drift).append(model)
        note = "reproduced" if good else "*** DRIFT ***"
        if not consistent:
            note += f" (pairs_by_cell has {n_recorded}, pairs_used says {new['pairs_used']})"
        print(f"{model:22s} {old['pairs_used']:>8d} -> {new['pairs_used']:<12d} "
              f"{'identical' if same_rates else 'CHANGED':11s} {note}")

    print()
    if pending:
        print("still to re-run:", pending)
    if drift:
        print("*** E2 did NOT reproduce for:", drift)
        print("*** E2 is not deterministic run-to-run. Do NOT treat any result as")
        print("*** confirmatory until this is explained. The prime suspect is the")
        print("*** adaptive OOM batch-halving in the grading pass (padding changes ->")
        print("*** a greedy token flips at the margin).")
        return 2
    if not pending:
        print(f"E2 reproduced exactly for all {len(ok_models)} models. The recorded pair")
        print("list is the single source of truth; E3 will measure causality on exactly")
        print("the pairs E2 measured the mechanism on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
