"""
Pre-registration gate (task §1.2). THE hard gate for the whole suite.

All decision thresholds come from ``configs/preregistration.yaml``, which is
authored and committed BY THE RESEARCHER. This module never creates, modifies,
or defaults that file. On a missing file or ANY missing required key it raises
``PreregError`` naming the exact missing key and the caller exits nonzero. No
analysis output may be produced without a complete, valid pre-registration.

The set of required keys (dotted paths) below is fixed by the task spec §1.2;
their SEMANTICS are documented in README and in §5-§7 of the spec. This module
validates presence and basic shape only - it does not invent values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# The canonical pre-registration path, relative to the repository root. The gate
# reads it; it is NEVER written by this codebase.
DEFAULT_PREREG_PATH = "configs/preregistration.yaml"

# Every required key, as a dotted path into the YAML. Order is the §1.2 order.
REQUIRED_KEYS: tuple[str, ...] = (
    "sampling.stage1_n_per_cell",
    "sampling.stage2_topup_n_breaking_cells",
    "sampling.pair_min_per_cell",
    "breaking_band_accuracy",
    "breaking_band_fallback.widened_band",
    "signature_rules.reference_distribution",
    "signature_rules.m1_silence",
    "signature_rules.m2_capture",
    "signature_rules.correct_attend",
    "signature_rules.residual",
    "signature_rules.mode_precedence",
    "signature_rules.sample_level_k_heads",
    "signature_rules.sensitivity_k_heads",
    "causal_criteria.flip_margin_over_control_pp",
    "causal_criteria.test",
    "causal_criteria.alpha",
    "causal_criteria.correction",
    "seeds.e1_surface",
    "seeds.e2_e3_headline_repeats",
    "statistics.effect_size",
    "statistics.ci",
)


class PreregError(RuntimeError):
    """Raised when the pre-registration file is missing or incomplete (§1.2)."""


def _get_dotted(d: Any, dotted: str) -> tuple[bool, Any]:
    """Return (present, value) for a dotted path in nested mappings.

    A key that exists but maps to ``None`` counts as MISSING - a pre-registered
    threshold must have an actual value, not a null placeholder.
    """
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    if cur is None:
        return False, None
    return True, cur


def resolve_prereg_path(path: str | Path | None = None) -> Path:
    """Resolve the pre-registration file path (default: configs/preregistration.yaml
    relative to the repo root)."""
    if path is not None:
        return Path(path)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / DEFAULT_PREREG_PATH


def load_prereg(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the pre-registration file.

    Returns the parsed dict on success. Raises ``PreregError`` naming:
      * the missing FILE, if it does not exist; or
      * the first missing/null required KEY (its exact dotted path).

    This function NEVER writes or defaults the file.
    """
    p = resolve_prereg_path(path)
    if not p.exists():
        raise PreregError(
            f"Pre-registration file not found: {p}\n"
            "It must be authored and committed by the researcher before any "
            "analysis runs (task §1.2). This codebase will not create or "
            "default it. Required keys:\n  " + "\n  ".join(REQUIRED_KEYS)
        )
    with open(p, encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:  # explicit; not a silent fallback (§1.9)
            raise PreregError(f"Pre-registration YAML failed to parse ({p}): {exc}") from exc
    if not isinstance(cfg, dict):
        raise PreregError(f"Pre-registration file {p} did not parse to a mapping.")

    missing = [k for k in REQUIRED_KEYS if not _get_dotted(cfg, k)[0]]
    if missing:
        raise PreregError(
            f"Pre-registration file {p} is missing required key(s): "
            + ", ".join(missing)
            + ".\nAuthor these in the file (task §1.2); this codebase will not "
            "default them."
        )
    return cfg


def get(cfg: dict[str, Any], dotted: str) -> Any:
    """Fetch a required, validated key. Raises ``PreregError`` if absent - use
    this so a typo'd key surfaces loudly at the call site rather than as a
    silent ``None`` (§1.9)."""
    present, value = _get_dotted(cfg, dotted)
    if not present:
        raise PreregError(f"Pre-registration key not present: {dotted}")
    return value
