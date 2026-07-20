"""The E3 speedup must be a speedup and nothing else.

E3 used to call ``capture_donor_z`` once per (pair, k) - a full prompt forward
each time. It now captures ONCE per probe over the union of every head set that
probe will ever donate through, and slices per k.

That is only legitimate if z_h is independent of which OTHER heads are captured.
It is: z_h is the o_proj input slice for head h at the answer position, and the
head list only selects which slices are recorded. These tests pin that invariant,
because if it ever stopped holding, every E3 number would change silently and
nothing else in the suite would notice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from failure_mech import patching

REPO = Path(__file__).resolve().parents[1]
DECODING = yaml.safe_load((REPO / "configs" / "decoding.yaml").read_text(encoding="utf-8"))

TEXT = "The access code for the golden lantern is K7QW2Z. What is the access code?"


def test_donor_is_bitwise_identical_when_captured_with_a_superset(tiny_llama):
    """The donor for a subset of heads must be EXACTLY the same array whether it
    was captured alone or as part of a larger capture - bitwise, not 'close'."""
    model, tok = tiny_llama
    ids = tok(TEXT, add_special_tokens=True)["input_ids"]

    subset = [(0, 0), (1, 2)]
    superset = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]

    alone = patching.capture_donor_z(model, tok, ids, subset, DECODING)
    together = patching.capture_donor_z(model, tok, ids, superset, DECODING)

    assert set(subset) <= set(together)
    for lh in subset:
        assert np.array_equal(alone[lh], together[lh]), (
            f"donor for head {lh} changed depending on which other heads were "
            "captured - the E3 union-capture optimisation is NOT sound")


def test_patching_with_a_sliced_donor_matches_patching_with_a_fresh_one(tiny_llama):
    """End-to-end: the generation produced by patching with a sliced donor must be
    token-identical to patching with a donor captured for exactly those heads."""
    model, tok = tiny_llama
    donor_ids = tok(TEXT, add_special_tokens=True)["input_ids"]
    recipient_ids = tok("Numbers drift across a quiet field near the mill. What is the access code?",
                        add_special_tokens=True)["input_ids"]

    heads = [(0, 0), (1, 2)]
    superset = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]

    fresh = patching.capture_donor_z(model, tok, donor_ids, heads, DECODING)
    union = patching.capture_donor_z(model, tok, donor_ids, superset, DECODING)
    sliced = {lh: union[lh] for lh in heads}

    g_fresh = patching.generate_with_patch(model, tok, recipient_ids, heads, fresh, DECODING)
    g_sliced = patching.generate_with_patch(model, tok, recipient_ids, heads, sliced, DECODING)

    assert g_fresh.token_ids == g_sliced.token_ids, (
        "patching with a sliced donor diverged from patching with a freshly captured "
        "one - E3's per-k results would silently change")
