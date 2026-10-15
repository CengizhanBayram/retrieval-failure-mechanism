"""Mechanics of the follow-up interventions on tiny offline models (no hub).

* ScopedHeadMask writes zeros exactly where its scope says, nowhere else, and
  the 'all' scope equals weight surgery on o_proj (the Part I knockout).
* The manual answer-step row equals the model's own eager row for Llama/Qwen2/
  OLMo-2/Gemma-2 (the NF4 gate helper), and capture_rows_values + transplant_z
  reproduce the head output (self-transplant).
* A pattern transplant writes row_donor @ V_recipient into z_h at the last
  prompt position (checked with an independent tap).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import harness  # noqa: E402
from failure_mech import followups as FU  # noqa: E402
from failure_mech import capture as CAP  # noqa: E402
from failure_mech import panel, patching  # noqa: E402

DEC = {"decoding": {"max_new_tokens": 4}, "batch": {"max_tokens_per_batch": 100000,
                                                     "max_samples_per_batch": 8}}


def _ids(tok, n=48, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(10, len(tok) - 1, size=n).tolist()
    return [tok.bos_token_id] + ids


class _Tap:
    """Records every o_proj input AS CONSUMED by o_proj, i.e. after every
    forward-pre-hook (mask, patcher) has run, whatever their registration order:
    a forward (post) hook receives the possibly-replaced input args."""

    def __init__(self, model, layer):
        self.mod = panel.attn_module(model, layer).o_proj
        self.seen = []

    def __enter__(self):
        self.h = self.mod.register_forward_hook(
            lambda _m, args, _out: self.seen.append(args[0].detach().clone()))
        return self

    def __exit__(self, *e):
        self.h.remove()


@pytest.mark.parametrize("scope", FU.SCOPES)
def test_scoped_mask_zero_pattern(scope):
    model, tok = harness.tiny_model("llama")
    hd = panel.head_dim(model)
    heads = [(1, 1), (1, 3)]
    ids = _ids(tok)
    T = len(ids)
    with FU.ScopedHeadMask(model, heads, scope) as m, _Tap(model, 1) as tap:
        patching.generate_plain(model, tok, ids, DEC)
    assert m.n_prompt_calls == 1 and m.n_decode_calls >= 1
    prompt, decs = tap.seen[0], tap.seen[1:]
    assert prompt.shape[1] == T
    masked_cols = [slice(h * hd, (h + 1) * hd) for (_l, h) in heads]
    other_cols = slice(0, hd)                       # head 0 is never masked

    def zero(x):
        return bool(torch.all(x == 0))

    exp_prompt = {"all": range(T), "context": range(T - 1),
                  "answer": [T - 1], "ansdec": [T - 1]}[scope]
    for t in range(T):
        for cs in masked_cols:
            assert zero(prompt[0, t, cs]) == (t in exp_prompt), (scope, t)
        assert not zero(prompt[0, t, other_cols])
    for x in decs:
        for cs in masked_cols:
            assert zero(x[0, :, cs]) == (scope in ("all", "ansdec")), scope
    assert m.zeroed_norm > 0


def test_scope_all_equals_weight_surgery():
    model, tok = harness.tiny_model("llama")
    hd = panel.head_dim(model)
    heads = [(0, 2), (2, 1)]
    ids = torch.tensor([_ids(tok, 40, seed=3)])
    with torch.no_grad(), FU.ScopedHeadMask(model, heads, "all"):
        masked = model(input_ids=ids).logits
    with torch.no_grad():
        saved = {}
        for (l, h) in heads:
            w = panel.attn_module(model, l).o_proj.weight
            saved[(l, h)] = w[:, h * hd:(h + 1) * hd].clone()
            w[:, h * hd:(h + 1) * hd] = 0
        surgery = model(input_ids=ids).logits
        for (l, h), v in saved.items():
            panel.attn_module(model, l).o_proj.weight[:, h * hd:(h + 1) * hd] = v
    assert torch.allclose(masked, surgery, atol=1e-6)


def test_answer_scope_leaves_earlier_positions_untouched():
    model, tok = harness.tiny_model("qwen2")
    ids = torch.tensor([_ids(tok, 30, seed=5)])
    with torch.no_grad():
        base = model(input_ids=ids).logits
        with FU.ScopedHeadMask(model, [(0, 0), (1, 2)], "answer"):
            ans = model(input_ids=ids).logits
    assert torch.allclose(base[0, :-1], ans[0, :-1], atol=1e-6)
    assert not torch.allclose(base[0, -1], ans[0, -1], atol=1e-6)


def test_empty_mask_is_noop():
    model, tok = harness.tiny_model("llama")
    ids = _ids(tok)
    plain = patching.generate_plain(model, tok, ids, DEC)
    gen, zn = FU.generate_masked(model, tok, ids, [], "all", DEC)
    assert gen.token_ids == plain.token_ids and zn == 0.0


@pytest.mark.parametrize("family", ["llama", "qwen2", "olmo2", "gemma2"])
def test_manual_rows_match_eager(family):
    model, tok = harness.tiny_model(family)
    n_heads, _ = panel.head_counts(model)
    heads = [(l, h) for l in range(model.config.num_hidden_layers) for h in range(n_heads)]
    # gemma2's window is 64: use a longer prompt so the sliding mask is exercised
    n = 100 if family == "gemma2" else 40
    res = FU.manual_vs_eager_rows(model, _ids(tok, n, seed=7), heads)
    assert res["max_abs_diff"] < 1e-5, res["max_abs_diff"]
    for v in res["per_head"].values():
        assert abs(v["row_sum"] - 1.0) < 1e-4


@pytest.mark.parametrize("family", ["llama", "qwen2", "olmo2"])
def test_rows_values_self_transplant(family):
    model, tok = harness.tiny_model(family, attn="sdpa")
    n_heads, _ = panel.head_counts(model)
    heads = [(l, h) for l in range(model.config.num_hidden_layers) for h in range(n_heads)]
    ids = _ids(tok, 60, seed=11)
    rv = FU.capture_rows_values(model, ids, heads)
    assert rv.seq_len == len(ids)
    err = FU.self_transplant_error(model, rv)
    assert err["max_rel_err"] < 1e-4, err["max_rel_err"]
    # rows equal the eager reference rows
    for (l, h) in [(0, 0), (1, 3), (2, 1)]:
        ref = CAP.eager_reference_row(model, ids, l, h)
        assert np.max(np.abs(rv.rows[(l, h)].cpu().numpy() - ref)) < 1e-5


def test_transplant_refuses_sliding_window():
    model, tok = harness.tiny_model("gemma2")
    with pytest.raises(NotImplementedError):
        FU.capture_rows_values(model, _ids(tok, 20), [(0, 0)])


def test_pattern_transplant_writes_expected_vector():
    model, tok = harness.tiny_model("llama", attn="sdpa")
    heads = [(1, 0), (1, 1), (2, 3)]
    rec, don = _ids(tok, 50, seed=1), _ids(tok, 50, seed=2)
    rv_r = FU.capture_rows_values(model, rec, heads)
    rv_d = FU.capture_rows_values(model, don, heads)
    vec = {lh: FU.transplant_z(model, rv_d.rows[lh], rv_r, *lh) for lh in heads}
    # by construction: row_donor @ V_recipient (GQA: shared kv head)
    n_q, n_kv = panel.head_counts(model)
    for (l, h) in heads:
        kv = panel.query_to_kv_head(h, n_q, n_kv)
        expect = (rv_d.rows[(l, h)].double() @ rv_r.values[(l, kv)].double()).numpy()
        assert np.allclose(vec[(l, h)], expect, atol=1e-5)
    hd = panel.head_dim(model)
    with _Tap(model, 1) as tap:
        patching.generate_with_patch(model, tok, rec, heads, vec, DEC,
                                     patch_mode="first_step", site="z_h")
    prompt = tap.seen[0]
    for (l, h) in heads:
        if l == 1:
            got = prompt[0, -1, h * hd:(h + 1) * hd].numpy()
            assert np.allclose(got, vec[(l, h)], atol=1e-6)
    # the content variant differs from the pattern variant
    vec_c = {lh: FU.transplant_z(model, rv_r.rows[lh], rv_d, *lh) for lh in heads}
    assert any(not np.allclose(vec[lh], vec_c[lh]) for lh in heads)


def test_transplant_rejects_length_mismatch():
    model, tok = harness.tiny_model("llama", attn="sdpa")
    rv_a = FU.capture_rows_values(model, _ids(tok, 30), [(0, 0)])
    rv_b = FU.capture_rows_values(model, _ids(tok, 31), [(0, 0)])
    with pytest.raises(ValueError):
        FU.transplant_z(model, rv_a.rows[(0, 0)], rv_b, 0, 0)


# ---------------------------------------------------------------------------
# Regression: Gemma-2 local layers with a window-truncated cache (fix 2026-10-15)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tokens", [50, 100, 150])   # < window, 1-2x window, > 2x window
def test_capture_masses_match_eager_beyond_window(n_tokens):
    """capture_head_masses on Gemma-2 must read the needle mass at the right
    ABSOLUTE positions when the prompt is longer than the sliding window (the
    HybridCache keeps only the last `window` keys of a local layer)."""
    model, tok = harness.tiny_model("gemma2")       # sliding_window = 64
    ids = _ids(tok, n_tokens, seed=13)
    T = len(ids)
    needle = (T - 20, T - 12)                        # inside the window of the answer step
    distractors = [(T - 40, T - 32)]
    heads = [(l, h) for l in range(model.config.num_hidden_layers) for h in range(2)]
    ms = {(m.layer, m.head): m for m in CAP.capture_head_masses(
        model, tok, ids, heads, needle_span=needle, distractor_spans=distractors,
        answer_steps=1)}
    for (l, h) in heads:
        ref = CAP.eager_reference_row(model, ids, l, h)
        m = ms[(l, h)]
        assert not m.window_limited
        assert abs(m.needle_mass - float(ref[needle[0]:needle[1]].sum())) < 1e-5, (l, h, T)
        assert abs(m.distractor_mass - float(ref[distractors[0][0]:distractors[0][1]].sum())) < 1e-5


def test_capture_flags_needle_outside_window():
    model, tok = harness.tiny_model("gemma2")
    ids = _ids(tok, 150, seed=17)
    needle = (5, 13)                                 # far outside the 64-token window
    ms = CAP.capture_head_masses(model, tok, ids, [(0, 0), (1, 0)], needle_span=needle,
                                 distractor_spans=[], answer_steps=1)
    by = {m.layer: m for m in ms}
    assert by[0].window_limited                      # layer 0 is local (sliding)
    assert not by[1].window_limited                  # layer 1 is global
    assert by[0].needle_mass == 0.0 or abs(by[0].needle_mass) < 1e-12
    ref = CAP.eager_reference_row(model, ids, 1, 0)
    assert abs(by[1].needle_mass - float(ref[5:13].sum())) < 1e-5


@pytest.mark.parametrize("family", ["llama", "gemma2", "olmo2"])
def test_reference_rows_low_memory_equal_eager(family):
    model, tok = harness.tiny_model(family, attn="eager")
    ids = _ids(tok, 150, seed=19)
    heads = [(0, 0), (1, 2), (2, 3)]
    rows = FU.reference_rows_last_position(model, ids, heads)
    for (l, h) in heads:
        ref = CAP.eager_reference_row(model, ids, l, h)
        assert rows[(l, h)].shape == ref.shape
        assert np.max(np.abs(rows[(l, h)] - ref)) < 1e-6


def test_old_cache_indexing_is_detected(monkeypatch):
    """The pre-fix row (cache index read as absolute position) must FAIL the
    eager comparison beyond the window - i.e. the regression tests above, and
    scripts/check_capture_long.py, would have caught the original error."""
    import torch

    def old_row(model, layer, head, q_rot_last, key_cache, position_id, n_heads, n_kv, scale,
                softcap):
        kv_head = panel.query_to_kv_head(head, n_heads, n_kv)
        logits = scale * torch.einsum("bd,bsd->bs", q_rot_last[:, head, :].float(),
                                      key_cache[:, kv_head, :, :].float())
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        if CAP.layer_uses_sliding(model, layer):
            first_visible = max(0, position_id - int(CAP.sliding_window_size(model)) + 1)
            if first_visible > 0:
                logits[:, :first_visible] = float("-inf")
        return torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    monkeypatch.setattr(CAP, "_row_for_head", old_row)
    model, tok = harness.tiny_model("gemma2")          # window 64, layer 0 local
    ids = _ids(tok, 100, seed=13)                       # between one and two windows
    T = len(ids)
    needle = (T - 20, T - 12)
    m = CAP.capture_head_masses(model, tok, ids, [(0, 0)], needle_span=needle,
                                distractor_spans=[], answer_steps=1)[0]
    ref = CAP.eager_reference_row(model, ids, 0, 0)
    ref_mass = float(ref[needle[0]:needle[1]].sum())
    assert not (abs(m.needle_mass - ref_mass) <= float(FU.tolerance(m.needle_mass, ref_mass)))


@pytest.mark.parametrize("family,n", [("llama", 60), ("qwen2", 60), ("olmo2", 60),
                                      ("gemma2", 50), ("gemma2", 150)])
def test_all_heads_summary_matches_per_head_capture(family, n):
    """The vectorised all-head summary (experiment F) equals the production
    per-head capture for every head, including Gemma-2 beyond its window."""
    model, tok = harness.tiny_model(family)
    ids = _ids(tok, n, seed=23)
    T = len(ids)
    needle = (T - 20, T - 12)
    dspans = [(T - 40, T - 32), (T - 30, T - 24)]
    s = FU.answer_step_all_heads(model, ids, needle, dspans)
    n_heads, _ = panel.head_counts(model)
    heads = [(l, h) for l in range(model.config.num_hidden_layers) for h in range(n_heads)]
    ms = {(m.layer, m.head): m for m in CAP.capture_head_masses(
        model, tok, ids, heads, needle_span=needle, distractor_spans=dspans, answer_steps=1)}
    for (l, h) in heads:
        assert abs(s["needle"][l, h] - ms[(l, h)].needle_mass) < 1e-5
        assert abs(s["dist_max"][l, h] - ms[(l, h)].distractor_mass) < 1e-5
        assert bool(s["window_limited"][l, h]) == ms[(l, h)].window_limited
    ref = CAP.eager_reference_row(model, ids, 1, 0)
    cat = 1 if needle[0] <= int(np.argmax(ref)) < needle[1] else (
        2 if any(a <= int(np.argmax(ref)) < b for a, b in dspans) else 0)
    assert int(s["argmax_cat"][1, 0]) == cat
