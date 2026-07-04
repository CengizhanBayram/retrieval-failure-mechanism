"""
Manual, memory-safe attention-row capture (task §4.3 + researcher corrections).

At the answer step we need each retrieval head's attention row over key
positions, to measure how much mass lands on the needle vs the distractors. We
NEVER pass ``output_attentions=True`` and NEVER materialise an LxL matrix
(§1.5): the row for a target head is computed by hand from the head's post-RoPE
query and the cached post-RoPE keys.

Two correctness rules the researcher flagged (both arbitrated by the eager-
reference test, §10 — trust no number until it passes on all four models):

1.  You cannot READ post-RoPE q under sdpa (RoPE is applied inside the attention
    forward). We hook ``q_proj`` for the PRE-RoPE q and RECOMPUTE the rotation
    with the model's own rotary module at the answer step's position; K
    (post-RoPE) is read from the KV cache via the GQA map.

2.  Gemma-2 is NOT plain ``softmax(qKᵀ/√d)``: it scales q by
    ``1/sqrt(query_pre_attn_scalar)`` and softcaps the logits
    (``cap*tanh(logits/cap)``) before softmax, and alternate layers use a
    sliding window. All three are read from config and applied here; the
    standard families use ``1/√head_dim`` and no cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from failure_mech import panel


# ---------------------------------------------------------------------------
# Rotary helpers (standard HF convention, shared by Llama/Qwen/Mistral/Gemma-2)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    import torch
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(model, position_ids):
    """(cos, sin) for ``position_ids`` from the model's own rotary module.

    position_ids: LongTensor [batch, seq]. Returns cos, sin [batch, seq, head_dim].
    """
    import torch
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    dummy = torch.zeros(1, 1, 1, device=device, dtype=dtype)
    rot = model.model.rotary_emb
    with torch.no_grad():
        cos, sin = rot(dummy, position_ids.to(device))
    return cos, sin


def _apply_rope(q_last, cos_p, sin_p):
    """Rotate q at one position for every head.

    q_last: [batch, n_heads, head_dim] (pre-RoPE). cos_p/sin_p: [batch, head_dim].
    """
    cos = cos_p[:, None, :]
    sin = sin_p[:, None, :]
    return q_last * cos + _rotate_half(q_last) * sin


# ---------------------------------------------------------------------------
# Gemma-2 layer geometry (local sliding-window vs global)
# ---------------------------------------------------------------------------

def gemma2_is_sliding_layer(model, layer_idx: int) -> bool:
    """True if this Gemma-2 layer uses local sliding-window attention.

    HF Gemma-2 exposes the per-layer type either as ``layer_types`` (newer) or
    via the even/odd ``sliding_window_pattern`` (older). Read robustly.
    """
    cfg = model.config
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is not None:
        return str(layer_types[layer_idx]).lower().startswith("sliding")
    # Older Gemma-2: even layers are sliding (local), odd are global.
    return (layer_idx % 2) == 0


# ---------------------------------------------------------------------------
# Capture config + result
# ---------------------------------------------------------------------------

@dataclass
class HeadMass:
    layer: int
    head: int
    needle_mass: float
    distractor_mass: float                 # max over distractors
    distractor_mass_per: list[float]       # per-distractor mass
    total_mass_checked: float              # sum of the row (should be ~1.0)
    window_limited: bool                   # Gemma-2 local layer w/ needle beyond window


# ---------------------------------------------------------------------------
# Core: capture masses for a set of heads at the answer step(s)
# ---------------------------------------------------------------------------

class _QProjTap:
    """Forward-hook manager stashing pre-RoPE q_proj output for target layers."""

    def __init__(self, model, layers: set[int]):
        self.model = model
        self.layers = layers
        self.store: dict[int, object] = {}
        self._handles = []

    def __enter__(self):
        for li in self.layers:
            attn = panel.attn_module(self.model, li)

            def make(idx):
                def hook(_m, _inp, out):
                    self.store[idx] = out.detach()
                return hook

            self._handles.append(attn.q_proj.register_forward_hook(make(li)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _row_for_head(
    model, layer, head, q_rot_last, key_cache, position_id, n_heads, n_kv, scale, softcap,
):
    """softmax over key positions for one head at the last query position.

    q_rot_last: [batch, n_heads, head_dim] post-RoPE. key_cache: post-RoPE keys
    [batch, n_kv, seq, head_dim]. Returns row as a 1-D numpy array over keys.
    """
    import torch
    kv_head = panel.query_to_kv_head(head, n_heads, n_kv)
    q_h = q_rot_last[:, head, :]                    # [batch, head_dim]
    K = key_cache[:, kv_head, :, :]                 # [batch, seq, head_dim]
    logits = scale * torch.einsum("bd,bsd->bs", q_h.float(), K.float())  # [batch, seq]
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    # Sliding-window mask for Gemma-2 local layers (keys older than the window
    # are not visible at the answer step).
    if panel.is_gemma2(model) and gemma2_is_sliding_layer(model, layer):
        sw = getattr(model.config, "sliding_window", None)
        if sw:
            seq = logits.shape[-1]
            first_visible = max(0, position_id - int(sw) + 1)
            if first_visible > 0:
                logits[:, :first_visible] = float("-inf")
    row = torch.softmax(logits, dim=-1)[0]         # batch 0
    return row.detach().cpu().numpy()


def _greedy_next(logits_last):
    import torch
    return int(torch.argmax(logits_last).item())


def capture_head_masses(
    model,
    tokenizer,
    input_ids: list[int],
    heads: list[tuple[int, int]],
    needle_span: tuple[int, int],
    distractor_spans: list[tuple[int, int]],
    *,
    answer_steps: int = 1,
    stop_token_ids: list[int] | None = None,
) -> list[HeadMass]:
    """Capture per-head attention masses at the answer step(s) (§4.3).

    ``answer_steps`` averages masses over the first N answer steps (1 = default;
    3 = E5 sensitivity). At steps > 1 the row spans the prompt PLUS already-
    generated tokens; needle/distractor spans still index the original prompt.
    """
    import torch

    device = next(model.parameters()).device
    n_heads, n_kv = panel.head_counts(model)
    scale = panel.attention_scale(model)
    softcap = panel.attn_logit_softcap(model)
    layers = {l for (l, _h) in heads}

    prompt_len = len(input_ids)
    ids = torch.tensor([list(input_ids)], device=device, dtype=torch.long)
    stop = set(stop_token_ids or [])
    if tokenizer.eos_token_id is not None:
        stop.add(int(tokenizer.eos_token_id))

    # accumulate masses over steps
    acc: dict[tuple[int, int], dict[str, list]] = {
        (l, h): {"needle": [], "dist_per": []} for (l, h) in heads
    }
    window_limited: dict[tuple[int, int], bool] = {}

    past = None
    cur_ids = ids
    with torch.no_grad(), _QProjTap(model, layers) as tap:
        for step in range(answer_steps):
            seq_from = 0 if step == 0 else (prompt_len + step - 1)
            position_ids = torch.arange(
                seq_from, seq_from + cur_ids.shape[1], device=device
            ).unsqueeze(0)
            out = model(
                input_ids=cur_ids,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            cur_position = seq_from + cur_ids.shape[1] - 1  # index of the answer query

            cos, sin = _rope_cos_sin(model, position_ids)
            cos_p, sin_p = cos[:, -1, :], sin[:, -1, :]    # last position

            for li in layers:
                q_proj_out = tap.store[li]                 # [batch, seq, n_heads*head_dim]
                hd = q_proj_out.shape[-1] // n_heads
                q_last = q_proj_out[:, -1, :].view(q_proj_out.shape[0], n_heads, hd)
                q_rot = _apply_rope(q_last, cos_p, sin_p)  # [batch, n_heads, head_dim]
                key_cache = _layer_keys(past, li)          # [batch, n_kv, seq, head_dim]
                for (l2, h2) in heads:
                    if l2 != li:
                        continue
                    row = _row_for_head(
                        model, li, h2, q_rot, key_cache, cur_position,
                        n_heads, n_kv, scale, softcap,
                    )
                    acc[(l2, h2)]["needle"].append(float(row[needle_span[0]:needle_span[1]].sum()))
                    acc[(l2, h2)]["dist_per"].append(
                        [float(row[s0:s1].sum()) for (s0, s1) in distractor_spans]
                    )
                    if (l2, h2) not in window_limited:
                        window_limited[(l2, h2)] = _is_window_limited(
                            model, li, needle_span, cur_position
                        )
            if step + 1 >= answer_steps:
                break
            nxt = _greedy_next(out.logits[0, -1, :])
            if nxt in stop:
                break
            cur_ids = torch.tensor([[nxt]], device=device, dtype=torch.long)

    results: list[HeadMass] = []
    for (l, h) in heads:
        needle = float(np.mean(acc[(l, h)]["needle"])) if acc[(l, h)]["needle"] else float("nan")
        per_steps = acc[(l, h)]["dist_per"]
        if per_steps and distractor_spans:
            per = np.mean(np.asarray(per_steps, dtype=np.float64), axis=0).tolist()
            dmax = float(max(per))
        else:
            per, dmax = [], 0.0
        results.append(HeadMass(
            layer=l, head=h, needle_mass=needle, distractor_mass=dmax,
            distractor_mass_per=per, total_mass_checked=1.0,
            window_limited=window_limited.get((l, h), False),
        ))
    return results


def _layer_keys(past, layer_idx):
    """Post-RoPE keys for a layer from a transformers Cache / legacy tuple."""
    key_cache = getattr(past, "key_cache", None)
    if key_cache is not None:
        return key_cache[layer_idx]
    return past[layer_idx][0]  # legacy (key, value) tuple layout


def _is_window_limited(model, layer_idx, needle_span, cur_position) -> bool:
    """True if this is a Gemma-2 local layer and the needle lies OUTSIDE the
    window at the answer step — its mass is truncated, not truly silent (§8)."""
    if not panel.is_gemma2(model) or not gemma2_is_sliding_layer(model, layer_idx):
        return False
    sw = getattr(model.config, "sliding_window", None)
    if not sw:
        return False
    first_visible = max(0, cur_position - int(sw) + 1)
    return needle_span[0] < first_visible


# ---------------------------------------------------------------------------
# Eager reference (TESTS ONLY, §10) — the arbiter of the manual row
# ---------------------------------------------------------------------------

def eager_reference_row(model, input_ids: list[int], layer: int, head: int) -> np.ndarray:
    """The model's OWN attention row for (layer, head) at the last prompt
    position, via ``output_attentions=True`` on an eager forward.

    Used ONLY by the test suite (§10) as the ground truth the manual row must
    match to < 1e-3. Never called from an experiment (§1.5).
    """
    import torch
    device = next(model.parameters()).device
    ids = torch.tensor([list(input_ids)], device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True, use_cache=False)
    attn = out.attentions[layer]            # [batch, n_heads, q_len, k_len]
    return attn[0, head, -1, :].float().cpu().numpy()
