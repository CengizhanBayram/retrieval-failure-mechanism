"""
Head-output (z_h) activation patching + donor store (task §4.4).

``z_h`` is the per-head attention output BEFORE o_proj — i.e. the input to
``o_proj``, laid out as ``[..., n_heads * head_dim]`` with head ``h`` occupying
columns ``[h*head_dim, (h+1)*head_dim)``. We capture it with a forward-PRE-hook
on ``o_proj`` (which sees exactly that tensor) and patch it by returning a
modified input — never touching model weights.

Runs (§4.4): repair (failure<-success), break (success<-failure), random-control
(same k, non-retrieval heads), no-patch rerun (flip base rate) and SELF-PATCH
(recipient<-its own z: MUST be bitwise-identical generation; unit test + a hard
runtime assert). ``patch_mode``: "first_step" (default, headline) patches only
generation step 1; "sustained" (E5) patches every generated step.

Decoding is the one greedy spec (§1.4); the manual loop here mirrors it so the
patch can be scoped to specific steps.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from failure_mech import panel


@dataclass
class Generation:
    token_ids: list[int]
    text: str


class _OProjPatcher:
    """Forward-pre-hook controller on o_proj: captures and/or overwrites the
    per-head z at the LAST position of the current forward."""

    def __init__(self, model, heads: list[tuple[int, int]]):
        self.model = model
        self.heads = heads
        self.by_layer: dict[int, list[int]] = {}
        for (l, h) in heads:
            self.by_layer.setdefault(l, []).append(h)
        self.head_dim = panel.head_dim(model)
        # runtime state
        self.capture = False           # capture donor z at the answer step?
        self.patch_enabled = False     # overwrite z with the donor?
        self.patch_mode = "first_step" # "first_step" | "sustained"
        self.donor: dict[tuple[int, int], object] = {}   # (l,h) -> tensor [head_dim]
        self.captured: dict[tuple[int, int], object] = {}
        self._handles = []

    def __enter__(self):
        for li in self.by_layer:
            o_proj = panel.attn_module(self.model, li).o_proj

            def make(idx):
                def pre_hook(_m, args):
                    x = args[0]
                    # The ANSWER step is the last position of the PROMPT forward
                    # (seq_len > 1). Incremental generation steps have seq_len == 1.
                    is_prompt_forward = x.shape[1] > 1
                    if self.capture and is_prompt_forward:
                        for h in self.by_layer[idx]:
                            sl = x[:, -1, h * self.head_dim:(h + 1) * self.head_dim]
                            self.captured[(idx, h)] = sl[0].detach().clone()
                    if self.patch_enabled and (self.patch_mode == "sustained" or is_prompt_forward):
                        x = x.clone()
                        for h in self.by_layer[idx]:
                            dv = self.donor.get((idx, h))
                            if dv is not None:
                                vec = dv.to(device=x.device, dtype=x.dtype)
                                x[:, -1, h * self.head_dim:(h + 1) * self.head_dim] = vec
                        return (x,) + tuple(args[1:])
                    return None
                return pre_hook

            self._handles.append(o_proj.register_forward_pre_hook(make(li)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _generate_hf(model, tokenizer, input_ids, decoding_cfg) -> "Generation":
    """One-prompt greedy generation via ``model.generate`` (robust cache handling
    for every architecture, incl. Gemma-2's fixed-size HybridCache near its max
    position — which the old manual token loop mishandled). Any active o_proj
    hook fires during generate, so it works for both plain and patched runs."""
    return generate_plain_batch(model, tokenizer, [list(input_ids)], decoding_cfg)[0]


def generate_plain(model, tokenizer, input_ids, decoding_cfg) -> Generation:
    """No-patch greedy generation (flip base rate / self-patch reference)."""
    return _generate_hf(model, tokenizer, input_ids, decoding_cfg)


def _oom_error_types():
    import torch
    oom = getattr(torch, "OutOfMemoryError", None)
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    types = tuple(t for t in (oom, cuda_oom) if t is not None)
    return types or (RuntimeError,)


def _free_cuda(model=None):
    """Release CUDA memory after an OOM. Critically, transformers PERSISTS a
    reusable generation cache on ``model._cache``; if we don't drop it, each OOM
    retry starts with less free memory and the split-and-retry spirals to a hard
    OOM (observed on A100). Drop it, then empty the allocator cache."""
    import torch
    if model is not None and getattr(model, "_cache", None) is not None:
        try:
            model._cache = None
        except Exception:  # noqa: BLE001
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_batch_adaptive(model, tokenizer, input_ids_list, decoding_cfg):
    """One left-padded greedy batch; on OOM halve and retry (order-preserving),
    freeing the model cache between tries. A single sample that still OOMs
    propagates so the caller's context-level fallback (§5) can act."""
    import torch
    device = next(model.parameters()).device
    dec = decoding_cfg["decoding"]
    pad_id = tokenizer.pad_token_id
    maxlen = max(len(x) for x in input_ids_list)
    input_ids, attn = [], []
    for ids in input_ids_list:                    # LEFT pad (decoder-only)
        pad = maxlen - len(ids)
        input_ids.append([pad_id] * pad + list(ids))
        attn.append([0] * pad + [1] * len(ids))
    input_ids_t = torch.tensor(input_ids, device=device, dtype=torch.long)
    attn_t = torch.tensor(attn, device=device, dtype=torch.long)
    try:
        with torch.no_grad():
            out = model.generate(input_ids=input_ids_t, attention_mask=attn_t,
                                 do_sample=False, num_beams=1,
                                 max_new_tokens=int(dec["max_new_tokens"]),
                                 pad_token_id=pad_id)
    except _oom_error_types():
        del input_ids_t, attn_t
        _free_cuda(model)
        if len(input_ids_list) == 1:
            raise
        mid = len(input_ids_list) // 2
        return (_generate_batch_adaptive(model, tokenizer, input_ids_list[:mid], decoding_cfg)
                + _generate_batch_adaptive(model, tokenizer, input_ids_list[mid:], decoding_cfg))
    gen = out[:, input_ids_t.shape[1]:]
    results = []
    for row in gen:
        toks = [int(t) for t in row.tolist()]
        results.append(Generation(token_ids=toks,
                                   text=tokenizer.decode(toks, skip_special_tokens=True)))
    return results


def generate_plain_batch(model, tokenizer, input_ids_list, decoding_cfg) -> list["Generation"]:
    """Greedy generation for many prompts, TOKEN-BUDGETED and order-preserving.

    Used for the no-hook GRADING passes (E1, and the E2/E3 pairing step). The
    caller may pass ALL of a cell's probes at once — this chunks them by
    ``batch.max_tokens_per_batch`` (length-sorted for packing, then mapped back
    to input order) so a 100-probe x 16k cell never becomes one 1.6M-token batch
    (that OOMs even an A100). Each chunk is GPU-adaptive (§ _generate_batch_
    adaptive). Results are returned in the SAME order as ``input_ids_list`` so
    callers can zip them with their probes.
    """
    if not input_ids_list:
        return []
    bcfg = decoding_cfg.get("batch", {})
    max_tokens = int(bcfg.get("max_tokens_per_batch", 12288))
    max_samples = int(bcfg.get("max_samples_per_batch", 8))
    results: list = [None] * len(input_ids_list)
    order = sorted(range(len(input_ids_list)), key=lambda i: len(input_ids_list[i]))

    def flush(chunk_idx):
        gens = _generate_batch_adaptive(
            model, tokenizer, [input_ids_list[i] for i in chunk_idx], decoding_cfg)
        for i, g in zip(chunk_idx, gens):
            results[i] = g

    chunk_idx: list = []
    tok_sum = 0
    for i in order:
        L = len(input_ids_list[i])
        if chunk_idx and (tok_sum + L > max_tokens or len(chunk_idx) >= max_samples):
            flush(chunk_idx)
            chunk_idx, tok_sum = [], 0
        chunk_idx.append(i)
        tok_sum += L
    if chunk_idx:
        flush(chunk_idx)
    return results


def capture_donor_z(model, tokenizer, input_ids, heads, decoding_cfg,
                    *, answer_steps: int = 1) -> dict:
    """Capture donor z_h at the answer step (per head) for ``heads``.

    Captured during the SAME ``model.generate`` prompt forward that
    ``generate_with_patch`` patches, so the donor value matches exactly and
    self-patch stays bitwise-identical. Returns ``{(layer, head):
    np.ndarray[head_dim] float32}``.
    """
    ctrl = _OProjPatcher(model, heads)
    with ctrl:
        ctrl.capture = True
        ctrl.patch_enabled = False
        _generate_hf(model, tokenizer, input_ids, decoding_cfg)  # prompt forward captures step-0 z
    return {lh: v.float().cpu().numpy() for lh, v in ctrl.captured.items()}


def generate_with_patch(
    model, tokenizer, input_ids, heads, donor_z: dict, decoding_cfg,
    *, patch_mode: str = "first_step",
) -> Generation:
    """Greedy generation with the target heads' z_h replaced by ``donor_z``."""
    import torch
    ctrl = _OProjPatcher(model, heads)
    for (l, h) in heads:
        v = donor_z.get((l, h))
        if v is not None:
            ctrl.donor[(l, h)] = torch.as_tensor(np.asarray(v))
    with ctrl:
        ctrl.capture = False
        ctrl.patch_enabled = True
        ctrl.patch_mode = patch_mode
        return _generate_hf(model, tokenizer, input_ids, decoding_cfg)


def self_patch_generate(model, tokenizer, input_ids, heads, decoding_cfg,
                        *, patch_mode: str = "first_step") -> tuple[Generation, Generation]:
    """Self-patch: patch the recipient with ITS OWN z. Returns (plain, patched).
    The two MUST be token-identical (§4.4); callers assert it and abort on
    failure."""
    donor = capture_donor_z(model, tokenizer, input_ids, heads, decoding_cfg)
    plain = generate_plain(model, tokenizer, input_ids, decoding_cfg)
    patched = generate_with_patch(model, tokenizer, input_ids, heads, donor,
                                  decoding_cfg, patch_mode=patch_mode)
    return plain, patched


# ---------------------------------------------------------------------------
# Donor store (.npz)
# ---------------------------------------------------------------------------

def _key(layer: int, head: int) -> str:
    return f"L{layer}_H{head}"


def save_donor_store(path: str | Path, donor_z: dict, meta: dict | None = None) -> None:
    """Persist a donor z map to a compact .npz (fp32; §4.4)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {_key(l, h): np.asarray(v, dtype=np.float32) for (l, h), v in donor_z.items()}
    if meta:
        import json
        arrays["__meta__"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
    np.savez(p, **arrays)


def load_donor_store(path: str | Path) -> dict:
    """Load a donor z map from .npz -> ``{(layer, head): np.ndarray float32}``."""
    p = Path(path)
    with np.load(p) as z:
        out: dict[tuple[int, int], np.ndarray] = {}
        for k in z.files:
            if k == "__meta__":
                continue
            l, h = k[1:].split("_H")
            out[(int(l), int(h))] = z[k].astype(np.float32)
    return out
