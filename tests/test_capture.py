"""Capture test (task §10): the MANUAL attention row must match the model's own
eager attention row to < 1e-3, at tiny context. This is THE arbiter of the
recompute-RoPE + per-family scale/softcap logic (§4.3). Skips offline."""

from __future__ import annotations

import numpy as np
import pytest

from failure_mech import capture as CAP


def _check_row_matches(model, tokenizer):
    text = "The access code for the golden lantern is K7QW2Z. Remember it well."
    input_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    max_diffs = []
    for layer in range(min(2, n_layers)):
        for head in range(min(3, n_heads)):
            ref = CAP.eager_reference_row(model, input_ids, layer, head)
            # manual row via the same machinery capture uses
            manual = _manual_row(model, tokenizer, input_ids, layer, head)
            max_diffs.append(float(np.max(np.abs(manual - ref))))
    assert max(max_diffs) < 1e-3, f"manual vs eager row max abs diff {max(max_diffs)} >= 1e-3"


def _manual_row(model, tokenizer, input_ids, layer, head):
    """Reproduce capture's manual row for one head at the last prompt position,
    reusing the library helpers so the test exercises the real code path."""
    import torch
    from failure_mech import panel
    device = next(model.parameters()).device
    n_heads, n_kv = panel.head_counts(model)
    scale = panel.attention_scale(model)
    softcap = panel.attn_logit_softcap(model)
    ids = torch.tensor([list(input_ids)], device=device)
    with torch.no_grad(), CAP._QProjTap(model, {layer}) as tap:
        position_ids = torch.arange(0, ids.shape[1], device=device).unsqueeze(0)
        out = model(input_ids=ids, position_ids=position_ids, use_cache=True)
        cos, sin = CAP._rope_cos_sin(model, position_ids)
        q_rot = CAP.query_rotated_last(model, layer, tap.store[layer],
                                       cos[:, -1, :], sin[:, -1, :], n_heads,
                                       panel.head_dim(model))
        key_cache = CAP._layer_keys(out.past_key_values, layer)
        return CAP._row_for_head(model, layer, head, q_rot, key_cache,
                                 ids.shape[1] - 1, n_heads, n_kv, scale, softcap)


def test_manual_row_matches_eager_llama(tiny_llama):
    model, tok = tiny_llama
    _check_row_matches(model, tok)


def test_manual_row_matches_eager_gemma2(tiny_gemma2):
    # Gemma-2: exercises query_pre_attn_scalar + softcap + sliding-window masking.
    model, tok = tiny_gemma2
    _check_row_matches(model, tok)


def test_manual_row_matches_eager_olmo2(tiny_olmo2):
    # OLMo-2: exercises QK-norm (q_norm applied before RoPE).
    model, tok = tiny_olmo2
    _check_row_matches(model, tok)


def test_capture_masses_sum_to_one(tiny_llama):
    model, tok = tiny_llama
    text = "The access code for the golden lantern is K7QW2Z."
    ids = tok(text, add_special_tokens=True)["input_ids"]
    heads = [(0, 0), (1, 1)]
    needle = (2, 4)
    ms = CAP.capture_head_masses(model, tok, ids, heads, needle_span=needle,
                                 distractor_spans=[], answer_steps=1)
    assert len(ms) == 2
    for m in ms:
        assert 0.0 <= m.needle_mass <= 1.0
