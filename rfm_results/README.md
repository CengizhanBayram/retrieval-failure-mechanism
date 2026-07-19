# Published results

The result JSONs for the paper. Committed so the numbers can be read without a GPU;
heavy/internal artifacts (`checkpoints/`, `*.npz` donor stores, `*.pre_pairfix` E2
backups) are regenerated, not published.

Every file carries a top-level `provenance` block (producing script, git commit,
config hashes, model SHA, seeds, timestamp). The per-file output schema is documented
in [`../docs/FINDINGS.md`](../docs/FINDINGS.md); what the numbers mean is in
[`../docs/FINDINGS.md`](../docs/FINDINGS.md) and the paper.

| pattern | experiment |
|---|---|
| `e1_surface_{model}.json`, `e1_breaking_cells_{model}.json` | E1 — breaking surface |
| `e2_signatures_{model}.json` | E2 — mechanism buckets + `pairs_by_cell` |
| `e2_signatures_{model}_{wu,ksens,steps3,seed0..2}.json` | E5 — tagged robustness variants |
| `e3_causal_{model}.json` | E3 — causal k-sweep (incl. full-set `--extra-k`) |
| `e3_causal_{model}_site-{v,mlp}.json` | E3 — value/MLP patch-site follow-up (exploratory) |
| `e4_families.json` | E4 — family table + authoritative `causal_decision` |
| `e5_robustness_{model}.json` | E5 — robustness aggregate |
| `gate_eager_reference.json` | capture-correctness gate (manual row vs eager, fp32) |

All k > 10 results, the full-set patch, and the value/MLP sites are **exploratory**;
the pre-registered result is k ≤ 10. See the amendment for the status of each run.
