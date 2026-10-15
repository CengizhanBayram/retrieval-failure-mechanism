"""
Follow-up interventions for the merged paper (exploratory, added after the
pre-specified E1-E5 analysis was complete).

Two mechanisms, both keyed by query head ``(layer, head)`` like the rest of the
package, both implemented as forward-PRE-hooks on ``o_proj`` (the same tensor
the E3 patcher writes, so the head slice convention is identical):

* :class:`ScopedHeadMask` - zero-ablation of selected heads' outputs (``z_h``)
  restricted to a SCOPE of positions:

      all     every prompt position and every decoding step
              (the Part I knockout operation)
      context prompt positions 0..T-2 (the positions the answer step reads)
      answer  the last prompt position only (E3's intervention scope)
      ansdec  the last prompt position and every decoding step

  The prompt forward is recognised as the call whose sequence length is > 1;
  decoding steps (with a KV cache) have length 1. Generation is single-sequence
  (no padding), exactly as in E3.

* :func:`capture_rows_values` + :func:`transplant_z` - the ingredients of an
  ATTENTION-PATTERN TRANSPLANT. For a head h at the answer step the output is
  ``z_h = softmax_row_h @ V_kv(h)``. Combining the DONOR's attention row with the
  RECIPIENT's value vectors gives the output the recipient would produce if it
  attended where the donor attended, while reading its own content. Patching
  that vector into ``z_h`` (first step, as in E3) removes the content-transport
  confound of ordinary activation patching, where the donor's output also
  carries the donor's needle/distractor values.

The attention row is recomputed manually with the validated machinery of
:mod:`failure_mech.capture` (post-RoPE query from the q_proj tap, cached keys,
per-family scale/softcap). Values are read from the KV cache (values are not
rotated). Models with sliding-window layers are refused by the transplant
helpers because their caches are window-truncated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from failure_mech import capture as CAP
from failure_mech import panel

SCOPES = ("all", "context", "answer", "ansdec")


# ---------------------------------------------------------------------------
# Scoped zero-ablation
# ---------------------------------------------------------------------------

class ScopedHeadMask:
    """Zero the o_proj input slice of ``heads`` at the positions selected by
    ``scope`` (see module docstring). Use as a context manager around a
    generation call. ``zeroed_norm`` accumulates the L2 norm of everything that
    was zeroed, so callers can assert the mask actually wrote (never-op guard)."""

    def __init__(self, model, heads, scope: str):
        if scope not in SCOPES:
            raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")
        self.model = model
        self.scope = scope
        self.by_layer: dict[int, list[int]] = {}
        for (l, h) in heads:
            self.by_layer.setdefault(int(l), []).append(int(h))
        self.head_dim = panel.head_dim(model)
        self.zeroed_norm = 0.0
        self.n_prompt_calls = 0
        self.n_decode_calls = 0
        self._handles = []

    def _positions(self, seq_len: int, is_prompt: bool):
        """Return a slice over the sequence axis to zero, or None."""
        if is_prompt:
            if self.scope == "all":
                return slice(0, seq_len)
            if self.scope == "context":
                return slice(0, seq_len - 1)
            return slice(seq_len - 1, seq_len)          # answer / ansdec
        # decoding step (length 1)
        if self.scope in ("all", "ansdec"):
            return slice(0, seq_len)
        return None

    def _hook(self, layer):
        def pre_hook(_m, args):
            if not args:
                return None
            x = args[0]
            is_prompt = x.shape[1] > 1
            if is_prompt:
                self.n_prompt_calls += 1
            else:
                self.n_decode_calls += 1
            sl = self._positions(x.shape[1], is_prompt)
            if sl is None or sl.stop <= sl.start:
                return None
            x = x.clone()
            for h in self.by_layer[layer]:
                a, b = h * self.head_dim, (h + 1) * self.head_dim
                seg = x[:, sl, a:b]
                self.zeroed_norm += float(seg.float().norm().item())
                x[:, sl, a:b] = 0
            return (x,) + tuple(args[1:])
        return pre_hook

    def __enter__(self):
        for layer in self.by_layer:
            o_proj = panel.attn_module(self.model, layer).o_proj
            self._handles.append(o_proj.register_forward_pre_hook(self._hook(layer)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def generate_masked(model, tokenizer, input_ids, heads, scope, decoding_cfg):
    """Greedy single-sequence generation with ``heads`` zeroed in ``scope``.
    Returns (Generation, zeroed_norm)."""
    from failure_mech import patching
    with ScopedHeadMask(model, heads, scope) as m:
        gen = patching.generate_plain(model, tokenizer, input_ids, decoding_cfg)
    return gen, m.zeroed_norm


# ---------------------------------------------------------------------------
# Attention-pattern transplant
# ---------------------------------------------------------------------------

@dataclass
class RowsValues:
    """Answer-step attention rows and KV-group values of one prompt, kept on the
    model's device (a 16k-token value slice is a few MB per KV group)."""
    rows: dict            # (layer, head) -> torch.Tensor [T] float32 (softmax row)
    values: dict          # (layer, kv)   -> torch.Tensor [T, head_dim] (model dtype)
    z_actual: dict        # (layer, head) -> torch.Tensor [head_dim] float32 (o_proj input)
    seq_len: int


def _layer_values(past, layer_idx):
    value_cache = getattr(past, "value_cache", None)
    if value_cache is not None:
        return value_cache[layer_idx]
    return past[layer_idx][1]


MIN_TRANSPLANT_WINDOW = 32768   # > the longest probe of the grid (16,384 tokens)


def assert_transplant_supported(model):
    """The transplant reads full-length caches; refuse models whose sliding
    window is shorter than the probes (Gemma-2's 4,096-token local layers keep
    only the last window of keys). Phi-3.5's 262k window never truncates and is
    accepted. ``capture_rows_values`` additionally checks every cache length."""
    sw = CAP.sliding_window_size(model)
    if sw is not None and int(sw) < MIN_TRANSPLANT_WINDOW:
        raise NotImplementedError(
            "attention-pattern transplant needs full-length caches; "
            f"{type(model).__name__} uses a {sw}-token sliding window.")


class _ZTap:
    """Pre-hook on o_proj recording the last-position input slice of heads
    during a full-prompt forward."""

    def __init__(self, model, heads):
        self.model = model
        self.by_layer: dict[int, list[int]] = {}
        for (l, h) in heads:
            self.by_layer.setdefault(int(l), []).append(int(h))
        self.head_dim = panel.head_dim(model)
        self.store: dict = {}
        self._handles = []

    def __enter__(self):
        for layer in self.by_layer:
            o_proj = panel.attn_module(self.model, layer).o_proj

            def make(idx):
                def pre_hook(_m, args):
                    x = args[0]
                    if x.shape[1] > 1:
                        for h in self.by_layer[idx]:
                            a, b = h * self.head_dim, (h + 1) * self.head_dim
                            self.store[(idx, h)] = x[0, -1, a:b].detach().float()
                    return None
                return pre_hook

            self._handles.append(o_proj.register_forward_pre_hook(make(layer)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def capture_rows_values(model, input_ids, heads) -> RowsValues:
    """One prompt forward: answer-step attention rows for ``heads`` (manual,
    validated recomputation), the value cache of their KV groups, and the actual
    o_proj input slices (for the self-transplant gate). Everything stays on the
    model's device; the full cache is released when this returns."""
    import torch

    assert_transplant_supported(model)
    device = next(model.parameters()).device
    n_heads, n_kv = panel.head_counts(model)
    scale = panel.attention_scale(model)
    softcap = panel.attn_logit_softcap(model)
    hd = panel.head_dim(model)
    layers = {int(l) for (l, _h) in heads}
    ids = torch.tensor([list(input_ids)], device=device, dtype=torch.long)
    T = ids.shape[1]
    rows, values = {}, {}
    with torch.no_grad(), CAP._QProjTap(model, layers) as tap, _ZTap(model, heads) as ztap:
        position_ids = torch.arange(0, T, device=device).unsqueeze(0)
        # The base model only: the rows need the q_proj tap and the cache, not
        # the LM head (whose [T, vocab] logits are several GB at 16k tokens).
        out = model.model(input_ids=ids, position_ids=position_ids, use_cache=True)
        past = out.past_key_values
        if tap.pos_emb is not None:
            cos, sin = tap.pos_emb
        else:
            cos, sin = CAP._rope_cos_sin(model, position_ids)
        cos_p, sin_p = cos[:, -1, :], sin[:, -1, :]
        for li in sorted(layers):
            q_rot = CAP.query_rotated_last(model, li, tap.store[li], cos_p, sin_p, n_heads, hd)
            key_cache = CAP._layer_keys(past, li)
            val_cache = _layer_values(past, li)
            if key_cache.shape[2] != T or val_cache.shape[2] != T:
                raise RuntimeError(
                    f"cache length {key_cache.shape[2]} != prompt length {T} at layer {li}; "
                    "the transplant needs a full-length cache.")
            for (l2, h2) in heads:
                if int(l2) != li:
                    continue
                kv = panel.query_to_kv_head(int(h2), n_heads, n_kv)
                q_h = q_rot[:, int(h2), :]
                K = key_cache[:, kv, :, :]
                logits = scale * torch.einsum("bd,bsd->bs", q_h.float(), K.float())
                if softcap is not None:
                    logits = softcap * torch.tanh(logits / softcap)
                rows[(li, int(h2))] = torch.softmax(logits, dim=-1)[0]          # [T] fp32
                if (li, kv) not in values:
                    values[(li, kv)] = val_cache[0, kv, :, :].detach().clone()  # [T, hd]
        del past, out
    return RowsValues(rows=rows, values=values, z_actual=dict(ztap.store), seq_len=T)


def actual_z(rv: RowsValues, layer: int, head: int):
    """The head's real answer-step output (o_proj input slice) captured with the
    rows - the donor vector of ordinary activation patching (E3)."""
    return rv.z_actual[(int(layer), int(head))].cpu().numpy().astype(np.float32)


def transplant_z(model, row, rv_recipient: RowsValues, layer: int, head: int):
    """z' = row @ V_recipient[kv(head)] as a float32 numpy vector [head_dim]."""
    n_heads, n_kv = panel.head_counts(model)
    kv = panel.query_to_kv_head(int(head), n_heads, n_kv)
    V = rv_recipient.values[(int(layer), kv)]
    if row.shape[0] != V.shape[0]:
        raise ValueError(f"row length {row.shape[0]} != value length {V.shape[0]}: "
                         "donor and recipient must share the cell's token skeleton.")
    return (row.float() @ V.float()).cpu().numpy().astype(np.float32)


def manual_vs_eager_rows(model, input_ids, heads) -> dict:
    """Validity gate for a new weight format (experiment B, NF4).

    Recomputes the answer-step attention row of each head with the production
    capture machinery (q_proj tap, the model's own RoPE, cached keys, per-family
    scale/softcap/sliding mask) and compares it with the model's OWN eager row
    (``output_attentions=True``). The test suite does this on tiny fp32 models;
    this runs it on the real quantized model, so a hook that stopped seeing the
    query projection (e.g. a fused 4-bit module) fails loudly. Keep the prompt
    short: the eager pass materialises every layer's full attention matrix.

    Returns {"max_abs_diff", "argmax_agree", "per_head": {...}}.
    """
    import torch

    device = next(model.parameters()).device
    n_heads, n_kv = panel.head_counts(model)
    scale = panel.attention_scale(model)
    softcap = panel.attn_logit_softcap(model)
    hd = panel.head_dim(model)
    layers = {int(l) for (l, _h) in heads}
    ids = torch.tensor([list(input_ids)], device=device, dtype=torch.long)
    T = ids.shape[1]
    manual = {}
    with torch.no_grad(), CAP._QProjTap(model, layers) as tap:
        position_ids = torch.arange(0, T, device=device).unsqueeze(0)
        out = model(input_ids=ids, position_ids=position_ids, use_cache=True)
        cos, sin = tap.pos_emb if tap.pos_emb is not None else CAP._rope_cos_sin(model, position_ids)
        for li in sorted(layers):
            q_rot = CAP.query_rotated_last(model, li, tap.store[li], cos[:, -1, :],
                                           sin[:, -1, :], n_heads, hd)
            key_cache = CAP._layer_keys(out.past_key_values, li)
            for (l2, h2) in heads:
                if int(l2) == li:
                    manual[(li, int(h2))] = CAP._row_for_head(
                        model, li, int(h2), q_rot, key_cache, T - 1, n_heads, n_kv,
                        scale, softcap)
        del out
    with torch.no_grad():
        ref_out = model(input_ids=ids, output_attentions=True, use_cache=False)
    per_head, diffs, agree, excess = {}, [], [], []
    for (l, h), row in manual.items():
        ref = ref_out.attentions[l][0, h, -1, :].float().cpu().numpy()
        d = float(np.max(np.abs(row - ref)))
        x = float(np.max(np.abs(row - ref) - tolerance(row, ref)))
        a = bool(int(np.argmax(row)) == int(np.argmax(ref)))
        per_head[f"L{l}_H{h}"] = {"max_abs_diff": d, "max_excess_over_tol": x,
                                  "argmax_agree": a, "row_sum": float(row.sum())}
        diffs.append(d)
        excess.append(x)
        agree.append(a)
    del ref_out
    return {"max_abs_diff": max(diffs) if diffs else float("nan"),
            "max_excess_over_tol": max(excess) if excess else float("nan"),
            "within_tol": bool(excess) and max(excess) <= 0.0,
            "argmax_agree": all(agree), "n_tokens": int(T), "per_head": per_head}


ABS_TOL, REL_TOL = 0.02, 0.08


def tolerance(a, b, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL):
    """Allowed |a - b| between a manual (fp32-softmax) and an eager (bf16)
    attention probability or mass: ``abs_tol + rel_tol * max(|a|, |b|)``.

    Eager attention rounds the logits to bf16 before its softmax (spacing
    0.06-0.125 for logits of 10-20), which moves individual probabilities by up
    to ~0.02 and span masses by up to ~6% relative in simulation; the sdpa kernels
    the model actually runs keep fp32 logits, like the manual row. The gates look
    for indexing errors (NaN rows, shifted windows, zeros where the model attends),
    which exceed this band by an order of magnitude."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return abs_tol + rel_tol * np.maximum(np.abs(a), np.abs(b))


def reference_rows_last_position(model, input_ids, heads) -> dict:
    """The model's OWN attention rows at the last prompt position for ``heads``,
    at full prompt length, with bounded memory.

    ``output_attentions=True`` makes every decoder layer return its full
    [batch, heads, T, T] weights and the model keeps all of them, which does not
    fit at 8k tokens. Here a forward hook on each decoder layer copies out only
    the last-position rows of the requested heads and replaces the layer's
    returned weights with None, so at most one layer's matrix is alive at a time.
    (sdpa-loaded models fall back to their eager path when attentions are
    requested, so the rows are the eager reference in every case.)

    Returns {(layer, head): np.ndarray [T] float32}.
    """
    import torch

    device = next(model.parameters()).device
    by_layer: dict[int, list[int]] = {}
    for (l, h) in heads:
        by_layer.setdefault(int(l), []).append(int(h))
    rows: dict = {}
    handles = []
    layers = model.model.layers
    for li, layer in enumerate(layers):
        def make(idx):
            def hook(_m, _args, output):
                if not isinstance(output, tuple) or len(output) < 2:
                    return None
                w = output[1]
                if w is not None and idx in by_layer:
                    for h in by_layer[idx]:
                        rows[(idx, h)] = w[0, h, -1, :].detach().float().cpu().numpy()
                return (output[0], None) + tuple(output[2:])
            return hook
        handles.append(layer.register_forward_hook(make(li)))
    ids = torch.tensor([list(input_ids)], device=device, dtype=torch.long)
    try:
        with torch.no_grad():
            model.model(input_ids=ids, output_attentions=True, use_cache=False)
    finally:
        for hd in handles:
            hd.remove()
    missing = [lh for lh in heads if (int(lh[0]), int(lh[1])) not in rows]
    if missing:
        raise RuntimeError(f"no attention weights returned for {missing[:3]}... - the "
                           "decoder layers did not output attentions.")
    return rows


def self_transplant_error(model, rv: RowsValues) -> dict:
    """Max |row@V - z_actual| / max|z_actual| over heads: checks that the manual
    row and the value read reproduce the head's real output."""
    errs = {}
    for (l, h), row in rv.rows.items():
        z_hat = transplant_z(model, row, rv, l, h)
        z = rv.z_actual[(l, h)].cpu().numpy()
        denom = float(np.max(np.abs(z))) or 1.0
        errs[(l, h)] = float(np.max(np.abs(z_hat - z)) / denom)
    return {"max_rel_err": max(errs.values()) if errs else float("nan"),
            "per_head": {f"L{l}_H{h}": e for (l, h), e in errs.items()}}


# ---------------------------------------------------------------------------
# All-head answer-step summaries (follow-up F)
# ---------------------------------------------------------------------------

ARGMAX_OTHER, ARGMAX_NEEDLE, ARGMAX_DISTRACTOR = 0, 1, 2


def answer_step_all_heads(model, input_ids, needle_span, distractor_spans) -> dict:
    """Answer-step attention summaries for EVERY head of the model, from one
    prompt forward.

    The row of each head is the same quantity ``capture._row_for_head`` computes
    (post-RoPE query from the q_proj tap, cached keys, per-family scale and
    softcap, absolute-position alignment of window-truncated caches, sliding
    mask), vectorised over the heads of a layer; the test suite checks the two
    against each other. Returns numpy arrays of shape [n_layers, n_heads]:

      needle      attention mass on the needle value span
      dist_max    largest mass on a single distractor value span (0 if none)
      argmax_cat  0 other / 1 needle / 2 distractor (where the row's argmax falls)
      window_limited  True if a local layer cannot see the needle at this step
    """
    import torch

    device = next(model.parameters()).device
    n_heads, n_kv = panel.head_counts(model)
    group = n_heads // n_kv
    scale = panel.attention_scale(model)
    softcap = panel.attn_logit_softcap(model)
    hd = panel.head_dim(model)
    n_layers = int(model.config.num_hidden_layers)
    ids = torch.tensor([list(input_ids)], device=device, dtype=torch.long)
    T = ids.shape[1]
    ns, ne = int(needle_span[0]), int(needle_span[1])
    dsp = [(int(a), int(b)) for (a, b) in distractor_spans]
    out = {k: np.zeros((n_layers, n_heads), dtype=np.float32) for k in ("needle", "dist_max")}
    out["argmax_cat"] = np.zeros((n_layers, n_heads), dtype=np.int8)
    out["window_limited"] = np.zeros((n_layers, n_heads), dtype=bool)
    sw = CAP.sliding_window_size(model)
    with torch.no_grad(), CAP._QProjTap(model, set(range(n_layers))) as tap:
        position_ids = torch.arange(0, T, device=device).unsqueeze(0)
        res = model.model(input_ids=ids, position_ids=position_ids, use_cache=True)
        past = res.past_key_values
        cos, sin = tap.pos_emb if tap.pos_emb is not None else CAP._rope_cos_sin(model, position_ids)
        cos_p, sin_p = cos[:, -1, :], sin[:, -1, :]
        for li in range(n_layers):
            q = CAP.query_rotated_last(model, li, tap.store[li], cos_p, sin_p, n_heads, hd)[0]
            K = CAP._layer_keys(past, li)[0]                        # [n_kv, Lc, hd]
            Kq = K.repeat_interleave(group, dim=0)                  # kv(h) = h // group
            logits = scale * torch.einsum("hd,hsd->hs", q.float(), Kq.float())
            if softcap is not None:
                logits = softcap * torch.tanh(logits / softcap)
            Lc = logits.shape[-1]
            if Lc < T:                                              # window-truncated cache
                pad = torch.full((n_heads, T - Lc), float("-inf"), device=logits.device,
                                 dtype=logits.dtype)
                logits = torch.cat([pad, logits], dim=-1)
            elif Lc > T:
                logits = logits[:, :T]
            limited = False
            if CAP.layer_uses_sliding(model, li):
                first_visible = max(0, T - 1 - int(sw) + 1)
                if first_visible > 0:
                    logits[:, :first_visible] = float("-inf")
                limited = ns < first_visible
            rows = torch.softmax(logits, dim=-1)                    # [n_heads, T]
            out["needle"][li] = rows[:, ns:ne].sum(-1).cpu().numpy()
            if dsp:
                dm = torch.stack([rows[:, a:b].sum(-1) for (a, b) in dsp], dim=-1)
                out["dist_max"][li] = dm.max(-1).values.cpu().numpy()
            am = rows.argmax(-1).cpu().numpy()
            cat = np.zeros(n_heads, dtype=np.int8)
            cat[(am >= ns) & (am < ne)] = ARGMAX_NEEDLE
            for (a, b) in dsp:
                cat[(am >= a) & (am < b)] = ARGMAX_DISTRACTOR
            out["argmax_cat"][li] = cat
            out["window_limited"][li] = limited
        del past, res
    return out
