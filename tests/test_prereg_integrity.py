"""Guards on the pre-registration boundary and on head-set determinacy.

These lock in two amendment decisions (2026-07-13):

1. The M2 distractor-mass floor may NOT be changed for the primary analysis after
   the results are known (HARKing). An alternative floor is reportable only as a
   TAGGED sensitivity variant, never as an overwrite of the primary artifact.
2. Where the top-k cut splits a block of tied detector scores, the patched head
   set is chosen by sort order rather than by evidence, and a null causal result
   at that k is not interpretable. That condition must be detected, not silent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from failure_mech.detect import Detection, DetectionError

REPO = Path(__file__).resolve().parents[1]


def _detection(scores: np.ndarray) -> Detection:
    n_layers, n_heads = scores.shape
    heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    return Detection(
        model_key="fake", seed=42, path="(test)",
        n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_heads,
        argmax_heads=heads, copy_heads=heads,
        argmax_scores=scores, copy_scores=scores,
    )


def test_boundary_tie_flags_an_arbitrary_head_set():
    """gemma2's real shape: many heads saturated at exactly 1.000, so a top-k cut
    inside that block admits some and excludes others on sort order alone."""
    scores = np.zeros((2, 4))
    scores[0, :] = 1.0          # 4 heads tied at the ceiling
    scores[1, :] = [0.5, 0.4, 0.3, 0.2]
    det = _detection(scores)

    bt = det.boundary_tie(2, detector="argmax")   # cut lands INSIDE the 4-way tie
    assert bt["arbitrary"] is True
    assert bt["n_tied_at_cut"] == 4
    assert bt["n_tied_inside_k"] == 2
    assert bt["n_tied_excluded"] == 2

    bt = det.boundary_tie(4, detector="argmax")   # cut lands exactly AT the tie edge
    assert bt["arbitrary"] is False
    assert bt["n_tied_excluded"] == 0

    bt = det.boundary_tie(5, detector="argmax")   # cut below the tie: determined
    assert bt["arbitrary"] is False


def test_boundary_tie_rejects_out_of_range_k():
    det = _detection(np.linspace(1.0, 0.1, 8).reshape(2, 4))
    with pytest.raises(DetectionError):
        det.boundary_tie(99)


def test_top_k_tie_break_is_deterministic():
    """Arbitrary is not the same as unstable: the same artifact must always yield
    the same set, so a run is reproducible even where the cut splits a tie."""
    scores = np.ones((3, 4))                       # everything tied
    det = _detection(scores)
    assert det.top_k_heads(5) == det.top_k_heads(5)
    assert det.top_k_heads(5) == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]


def test_m2_floor_override_refuses_to_overwrite_the_primary():
    """Changing the M2 floor without a --tag would silently replace the
    pre-registered primary artifact with a post-hoc re-tuned one. Refuse."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "e2_signatures.py"),
         "--model", "llama31_8b_instruct", "--m2-min-distractor-mass", "0.15"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0, "an untagged M2-floor override must not be allowed"
    assert "requires --tag" in (r.stdout + r.stderr)
