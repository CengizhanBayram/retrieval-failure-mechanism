# Pre-registration schema (`configs/preregistration.yaml`)

> **This file is authored and committed BY THE RESEARCHER, before any analysis
> runs.** The codebase never creates, modifies, or defaults it (task §1.2). On a
> missing file or any missing/`null` key, `failure_mech.prereg.load_prereg`
> raises `PreregError` naming the exact key and every script exits nonzero - no
> results are produced. This document describes the required keys and their
> semantics so you can author the file; it deliberately supplies **no numeric
> values** (every value below is `<FILL>` - a placeholder that fails the gate
> until you replace it with your pre-registered decision).

The gate checks presence of all keys in `failure_mech.prereg.REQUIRED_KEYS`.
The classifier (`classify.py`) additionally reads the sub-fields documented under
`signature_rules.*` - if you name a bucket in `mode_precedence`, provide its rule
block.

```yaml
# configs/preregistration.yaml  -  RESEARCHER-AUTHORED. Replace every <FILL>.

sampling:
  stage1_n_per_cell: <FILL>              # E1 stage-1 samples per cell (int)
  stage2_topup_n_breaking_cells: <FILL>  # E1 top-up target for breaking cells (int)
  pair_min_per_cell: <FILL>              # min matched pairs to keep a cell in E2/E3 (int)

breaking_band_accuracy:                  # a cell "breaks" if its accuracy is in this band
  lo: <FILL>                             # float in [0,1]
  hi: <FILL>                             # float in [0,1]

breaking_band_fallback:
  widened_band: {lo: <FILL>, hi: <FILL>} # used ONLY if the trigger below fires
  min_breaking_cells: <FILL>             # OPTIONAL trigger: if fewer breaking cells than
                                         # this, widen the band (logged + flagged). OMIT
                                         # this key to disable the fallback entirely.

signature_rules:
  reference_distribution:                # per-(head,cell) success needle-mass reference
    source: success_samples
    stats: [p5, median]
    detector: argmax                     # which detector's heads to profile (argmax|copy)
  m1_silence:                            # needle_mass <= max(ref[reference]*factor, absolute_floor)
    reference: p5                        # p5 | median
    factor: <FILL>                       # float
    absolute_floor: <FILL>               # float or null
  m2_capture:                            # distractor_mass >= min AND (distractor-needle) >= margin
    min_distractor_mass: <FILL>          # float
    distractor_over_needle_margin: <FILL># float
  correct_attend:                        # needle_mass >= ref[reference]*factor
    reference: median                    # p5 | median
    factor: <FILL>                       # float
  residual: {}                           # terminal catch-all (no threshold)
  mode_precedence: [m2_capture, m1_silence, correct_attend, residual]
  sample_level_k_heads: <FILL>           # top-k heads for sample-level classification (int)
  sensitivity_k_heads: <FILL>            # alternate k for the E5 sensitivity check (int)

causal_criteria:
  flip_margin_over_control_pp: <FILL>    # required (flip_rate - control) margin, PERCENTAGE POINTS
  test: paired_permutation               # the pre-registered test name
  alpha: <FILL>                          # float (e.g. 0.05)
  correction: benjamini_hochberg         # multiple-comparison correction name

seeds:
  e1_surface: <FILL>                     # master seed for probe instantiation / E1/E2/E3 (int)
  e2_e3_headline_repeats: <FILL>         # int (number of repeats) OR list of seeds for E5(b)

statistics:
  effect_size: cliffs_delta              # the pre-registered effect-size measure
  ci: <FILL>                             # CI level, e.g. 0.95
```

## Semantics recap (where each key is consumed)

| Key | Used in | Meaning |
|---|---|---|
| `sampling.stage1_n_per_cell` | E1 | samples graded per cell in stage 1 |
| `sampling.stage2_topup_n_breaking_cells` | E1 | breaking cells topped up to this many samples |
| `sampling.pair_min_per_cell` | E2, E3 | cells with fewer matched pairs are excluded (logged) |
| `breaking_band_accuracy` | E1 | accuracy band that defines a breaking cell |
| `breaking_band_fallback.widened_band` | E1 | widened band applied only under the trigger |
| `signature_rules.*` | E2 | four-bucket predicates + reference + k (see `classify.py`) |
| `causal_criteria.flip_margin_over_control_pp` | E3, E4 | effect requires this margin over control |
| `causal_criteria.alpha` / `correction` | E3, E4 | BH across {4 models × 2 directions} |
| `seeds.e1_surface` | all | deterministic probe/run seed |
| `seeds.e2_e3_headline_repeats` | E5 | headline repeats (mean ± range) |
| `statistics.effect_size` / `ci` | E2, E3 | Cliff's delta + bootstrap CI level |
