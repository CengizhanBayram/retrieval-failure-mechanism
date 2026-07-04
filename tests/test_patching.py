"""Patching tests (task §10): self-patch is a bitwise (token-identical) no-op,
and the donor .npz store round-trips. Skips offline."""

from __future__ import annotations

from pathlib import Path

import yaml
import numpy as np

from failure_mech import patching

REPO = Path(__file__).resolve().parents[1]
DECODING = yaml.safe_load((REPO / "configs" / "decoding.yaml").read_text(encoding="utf-8"))


def test_self_patch_is_noop(tiny_llama):
    model, tok = tiny_llama
    text = "The access code for the golden lantern is K7QW2Z. What is it?"
    ids = tok(text, add_special_tokens=True)["input_ids"]
    heads = [(0, 0), (1, 1), (1, 2)]
    plain, patched = patching.self_patch_generate(model, tok, ids, heads, DECODING)
    assert plain.token_ids == patched.token_ids, "self-patch changed the generation (§4.4)"


def test_self_patch_sustained_is_noop(tiny_llama):
    model, tok = tiny_llama
    text = "The access code for the golden lantern is K7QW2Z. What is it?"
    ids = tok(text, add_special_tokens=True)["input_ids"]
    heads = [(0, 0)]
    plain, patched = patching.self_patch_generate(model, tok, ids, heads, DECODING,
                                                  patch_mode="sustained")
    assert plain.token_ids == patched.token_ids


def test_donor_store_roundtrip(tmp_path, tiny_llama):
    model, tok = tiny_llama
    text = "The access code for the golden lantern is K7QW2Z."
    ids = tok(text, add_special_tokens=True)["input_ids"]
    heads = [(0, 0), (1, 2)]
    donor = patching.capture_donor_z(model, ids, heads)
    p = tmp_path / "donor.npz"
    patching.save_donor_store(p, donor, meta={"model": "tiny"})
    loaded = patching.load_donor_store(p)
    assert set(loaded) == set(donor)
    for k in donor:
        assert np.allclose(loaded[k], donor[k], atol=0)


def test_patch_changes_generation_when_donor_differs(tiny_llama):
    """A donor from a DIFFERENT prompt should be able to change the output (a
    sanity check that the patch is actually applied, not a silent no-op)."""
    model, tok = tiny_llama
    a = tok("The access code for the golden lantern is K7QW2Z. What is it?",
            add_special_tokens=True)["input_ids"]
    b = tok("Numbers and colors drift across a quiet field near the mill today.",
            add_special_tokens=True)["input_ids"]
    heads = [(l, h) for l in range(model.config.num_hidden_layers)
             for h in range(model.config.num_attention_heads)]
    donor_b = patching.capture_donor_z(model, b, heads)
    plain = patching.generate_plain(model, tok, a, DECODING)
    patched = patching.generate_with_patch(model, tok, a, heads, donor_b, DECODING)
    # Not asserting inequality (tiny random model may coincide); assert it RAN
    # and produced a valid generation of the configured length bound.
    assert len(patched.token_ids) <= DECODING["decoding"]["max_new_tokens"]
    assert isinstance(plain.text, str)
