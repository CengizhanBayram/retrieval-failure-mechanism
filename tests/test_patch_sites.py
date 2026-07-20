"""Patch-site tests (value/MLP follow-up, exploratory).

Two invariants the whole causal method rests on, per site:
  * self-patch (recipient <- its OWN activation) is a token-identical no-op;
  * a FOREIGN donor actually changes the target tensor (the never-op detector -
    what self-patch cannot catch, because for self-patch 'no change' is correct).

site='v' is a KV-CACHE swap at the CONTEXT span positions (retrieval reads the
value vectors there, not at the answer step), so its tests pass explicit
positions. If a hook were on the wrong tensor, the wrong positions, or the GQA map
were wrong, one of these two invariants would break - the same guards the E3
positive control uses at scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from failure_mech import patching

REPO = Path(__file__).resolve().parents[1]
DECODING = yaml.safe_load((REPO / "configs" / "decoding.yaml").read_text(encoding="utf-8"))
TEXT = "The access code for the golden lantern is K7QW2Z. What is the access code?"


def _heads(model):
    # several heads per layer -> forces KV-sharing (v) and layer-sharing (mlp)
    return [(0, 0), (0, 1), (0, 3), (1, 0), (1, 2), (1, 3)]


def _ctx_positions(tok):
    n = len(tok(TEXT, add_special_tokens=True)["input_ids"])
    return [p for p in (2, 3, 4, 5) if p < n - 1]   # a few context (non-answer) tokens


def test_self_patch_v_is_noop(tiny_llama):
    model, tok = tiny_llama
    ids = tok(TEXT, add_special_tokens=True)["input_ids"]
    plain, patched = patching.self_patch_generate(
        model, tok, ids, _heads(model), DECODING, site="v", positions=_ctx_positions(tok))
    assert plain.token_ids == patched.token_ids, "site='v' self-patch changed the generation"


def test_self_patch_mlp_is_noop(tiny_llama):
    model, tok = tiny_llama
    ids = tok(TEXT, add_special_tokens=True)["input_ids"]
    plain, patched = patching.self_patch_generate(model, tok, ids, _heads(model),
                                                  DECODING, site="mlp")
    assert plain.token_ids == patched.token_ids, "site='mlp' self-patch changed the generation"


def test_foreign_donor_changes_tensor_every_site(tiny_llama):
    """The never-op detector: a donor from a DIFFERENT prompt must change the target
    tensor at every site. Guards against a hook that fires but never writes what the
    model reads (the exact bug the first v implementation had)."""
    model, tok = tiny_llama
    a = tok(TEXT, add_special_tokens=True)["input_ids"]
    b = tok("Numbers drift across a quiet field near the mill. What is the access code?",
            add_special_tokens=True)["input_ids"]
    heads = [(l, h) for l in range(model.config.num_hidden_layers)
             for h in range(model.config.num_attention_heads)]
    for site, pos in (("z_h", None), ("mlp", None), ("v", _ctx_positions(tok))):
        donor_b = patching.capture_donor_z(model, tok, b, heads, DECODING, site=site,
                                           positions=pos)
        changed = patching.patch_changes_tensor(model, tok, a, heads, donor_b, DECODING,
                                                site=site, positions=pos)
        assert changed, f"site={site}: foreign donor did NOT change the tensor (never-op)"


def test_v_donor_is_the_kv_slice_over_positions(tiny_llama):
    """At site='v' the donor is the value slice over the given positions, and query
    heads sharing a KV head capture the same slice."""
    model, tok = tiny_llama
    ids = tok(TEXT, add_special_tokens=True)["input_ids"]
    pos = _ctx_positions(tok)
    n_q, n_kv = patching.panel.head_counts(model)
    donor = patching.capture_donor_z(model, tok, ids, [(0, 0), (0, 1)], DECODING,
                                     site="v", positions=pos)
    assert donor[(0, 0)].shape[0] == len(pos), "v donor must span the patched positions"
    if n_q != n_kv and patching.panel.query_to_kv_head(0, n_q, n_kv) == \
            patching.panel.query_to_kv_head(1, n_q, n_kv):
        assert np.array_equal(donor[(0, 0)], donor[(0, 1)]), \
            "query heads in one KV group must carry the same v-slice donor"
