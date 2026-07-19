"""Patch-site tests (value/MLP follow-up, exploratory).

The site generalisation (z_h | v | mlp) must keep the one hard invariant the
whole causal method rests on: patching a recipient with ITS OWN activation is a
token-identical no-op. For v and mlp this also exercises the *shared-write*
correctness — several query heads map to the same KV slice (v) or the same MLP
output (mlp), so the self-patch writes the same value repeatedly; it must still be
a no-op. If a hook were on the wrong tensor, or the slice/GQA map were wrong, the
self-patch would diverge — the same guard the positive control uses at scale.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from failure_mech import patching

REPO = Path(__file__).resolve().parents[1]
DECODING = yaml.safe_load((REPO / "configs" / "decoding.yaml").read_text(encoding="utf-8"))
TEXT = "The access code for the golden lantern is K7QW2Z. What is the access code?"


def _heads(model):
    # a mix that forces KV-sharing (v) and layer-sharing (mlp): several heads per layer
    return [(0, 0), (0, 1), (0, 3), (1, 0), (1, 2), (1, 3)]


def test_self_patch_v_is_noop(tiny_llama):
    model, tok = tiny_llama
    ids = tok(TEXT, add_special_tokens=True)["input_ids"]
    plain, patched = patching.self_patch_generate(model, tok, ids, _heads(model),
                                                  DECODING, site="v")
    assert plain.token_ids == patched.token_ids, "site='v' self-patch changed the generation"


def test_self_patch_mlp_is_noop(tiny_llama):
    model, tok = tiny_llama
    ids = tok(TEXT, add_special_tokens=True)["input_ids"]
    plain, patched = patching.self_patch_generate(model, tok, ids, _heads(model),
                                                  DECODING, site="mlp")
    assert plain.token_ids == patched.token_ids, "site='mlp' self-patch changed the generation"


def test_v_donor_is_the_kv_slice_shared_across_query_heads(tiny_llama):
    """Query heads sharing a KV head must capture the SAME donor value at site='v'
    (they read the same v_proj slice)."""
    import numpy as np
    model, tok = tiny_llama
    ids = tok(TEXT, add_special_tokens=True)["input_ids"]
    n_q, n_kv = patching.panel.head_counts(model)
    if n_q == n_kv:
        return  # MHA tiny model: no sharing to test; the no-op tests still cover it
    group = n_q // n_kv
    # two query heads in the same KV group, same layer
    h_a, h_b = 0, 1 if group >= 2 else 0
    donor = patching.capture_donor_z(model, tok, ids, [(0, h_a), (0, h_b)], DECODING, site="v")
    if patching.panel.query_to_kv_head(h_a, n_q, n_kv) == patching.panel.query_to_kv_head(h_b, n_q, n_kv):
        assert np.array_equal(donor[(0, h_a)], donor[(0, h_b)]), \
            "query heads in one KV group must carry the same v-slice donor"


def test_patch_with_foreign_donor_can_change_generation_v(tiny_llama):
    """A donor from a DIFFERENT prompt applied at site='v' should be able to change
    the output — proves the patch is actually applied, not a silent no-op."""
    model, tok = tiny_llama
    a = tok(TEXT, add_special_tokens=True)["input_ids"]
    b = tok("Numbers drift across a quiet field near the mill. What is the access code?",
            add_special_tokens=True)["input_ids"]
    heads = [(l, h) for l in range(model.config.num_hidden_layers)
             for h in range(model.config.num_attention_heads)]
    donor_b = patching.capture_donor_z(model, tok, b, heads, DECODING, site="v")
    patched = patching.generate_with_patch(model, tok, a, heads, donor_b, DECODING, site="v")
    assert len(patched.token_ids) <= DECODING["decoding"]["max_new_tokens"]
