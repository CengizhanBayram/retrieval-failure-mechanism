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


class PatchError(RuntimeError):
    """Raised when a patch site is unsupported for a model (e.g. site='v' on a
    fused-qkv architecture with no separate v_proj)."""


SITES = ("z_h", "v", "mlp")


def _mlp_module(model, layer_idx: int):
    return model.model.layers[layer_idx].mlp


class _SitePatcher:
    """Capture and/or overwrite a donor activation at the LAST position of the
    prompt forward (the answer step), at one of three sites. Keyed throughout by
    the QUERY head ``(layer, head)`` so the caller (E3) is unchanged across sites;
    the site only changes WHICH tensor slice that key maps to:

      site="z_h" : forward-PRE-hook on ``o_proj``; slice ``[h*hd:(h+1)*hd]`` of its
                   input (the concatenated per-head outputs). The original path —
                   bitwise-identical to before.
      site="v"   : forward-hook on ``v_proj`` OUTPUT; slice ``[kv*hd:(kv+1)*hd]``
                   where ``kv = query_to_kv_head(h)``. GQA: query heads sharing a
                   KV head map to the SAME slice, so capturing/patching them writes
                   the same value idempotently (correct, and it keeps the caller's
                   head-keyed donor dict intact).
      site="mlp" : forward-hook on the layer MLP OUTPUT; the WHOLE vector (the MLP
                   is not head-indexed), so every head in a layer maps to that
                   layer's single MLP output — again an idempotent shared write.
    """

    def __init__(self, model, heads: list[tuple[int, int]], site: str = "z_h",
                 positions=None):
        if site not in SITES:
            raise PatchError(f"unknown site {site!r}; expected one of {SITES}")
        self.model = model
        self.site = site
        self.heads = heads
        # v is a KV-CACHE swap: retrieval reads the value vectors at the CONTEXT
        # positions (needle + distractor tokens), NOT at the answer step, so v
        # captures/patches those positions of the PROMPT forward. Skeletons are
        # token-aligned within a cell, so donor and recipient share these indices.
        self.positions = list(positions) if positions is not None else None
        self.by_layer: dict[int, list[int]] = {}
        for (l, h) in heads:
            self.by_layer.setdefault(l, []).append(h)
        self.head_dim = panel.head_dim(model)
        self.n_q, self.n_kv = panel.head_counts(model)
        self.capture = False
        self.patch_enabled = False
        self.patch_mode = "first_step"
        self.verify = False            # never-op detector: assert the tensor changed
        self.tensor_changed = False    # set True once a differing donor actually altered x
        self.donor: dict[tuple[int, int], object] = {}
        self.captured: dict[tuple[int, int], object] = {}
        self._handles = []
        if site == "v" and self.positions is None:
            raise PatchError("site='v' needs the context positions to swap (needle + "
                             "distractor spans); pass positions=...")

    def _slice(self, h: int):
        """(start, end) columns for query head ``h`` at this site, or None for the
        whole vector (mlp)."""
        if self.site == "z_h":
            return (h * self.head_dim, (h + 1) * self.head_dim)
        if self.site == "v":
            kv = panel.query_to_kv_head(h, self.n_q, self.n_kv)
            return (kv * self.head_dim, (kv + 1) * self.head_dim)
        return None  # mlp: whole vector

    def _target(self, layer: int):
        """(module, hook_kind) for this site. z_h hooks the o_proj INPUT (pre);
        v/mlp hook the module OUTPUT (post)."""
        if self.site == "z_h":
            return panel.attn_module(self.model, layer).o_proj, "pre"
        if self.site == "v":
            attn = panel.attn_module(self.model, layer)
            if not hasattr(attn, "v_proj"):
                raise PatchError(
                    f"site='v' needs a separate v_proj, but {type(attn).__name__} "
                    "has none (fused qkv, e.g. Phi-3). Not supported for this model.")
            return attn.v_proj, "post"
        return _mlp_module(self.model, layer), "post"

    def _apply(self, x, layer: int):
        """Capture (read-only) and/or patch (return modified). Returns the modified
        tensor when patching, else None. The site sets WHICH positions:
          v          -> the context span positions of the PROMPT forward (cache swap);
          z_h / mlp  -> the answer step (last position of the prompt forward)."""
        import torch
        is_prompt = x.shape[1] > 1
        if self.site == "v":
            if not is_prompt:
                return None            # V is written to the cache on the prompt forward only
            pos = [p for p in self.positions if p < x.shape[1]]
            if self.capture:
                for h in self.by_layer[layer]:
                    a, b = self._slice(h)
                    self.captured[(layer, h)] = x[0, pos, a:b].detach().clone()  # [n_pos, hd]
            if self.patch_enabled:
                x = x.clone()
                for h in self.by_layer[layer]:
                    dv = self.donor.get((layer, h))
                    if dv is None:
                        continue
                    a, b = self._slice(h)
                    vec = dv.to(device=x.device, dtype=x.dtype)      # [n_pos, hd]
                    n = min(vec.shape[0], len(pos))
                    if self.verify and not self.tensor_changed and n:
                        if not torch.equal(x[0, pos[:n], a:b], vec[:n]):
                            self.tensor_changed = True
                    for j in range(n):
                        x[:, pos[j], a:b] = vec[j]
                return x
            return None

        # z_h / mlp: answer step (last position)
        if self.capture and is_prompt:
            for h in self.by_layer[layer]:
                sl = self._slice(h)
                v = x[:, -1, sl[0]:sl[1]] if sl else x[:, -1, :]
                self.captured[(layer, h)] = v[0].detach().clone()
        if self.patch_enabled and (self.patch_mode == "sustained" or is_prompt):
            x = x.clone()
            for h in self.by_layer[layer]:
                dv = self.donor.get((layer, h))
                if dv is None:
                    continue
                vec = dv.to(device=x.device, dtype=x.dtype)
                sl = self._slice(h)
                target = x[:, -1, sl[0]:sl[1]] if sl else x[:, -1, :]
                if self.verify and not self.tensor_changed and not torch.equal(target, vec):
                    self.tensor_changed = True
                if sl:
                    x[:, -1, sl[0]:sl[1]] = vec
                else:
                    x[:, -1, :] = vec
            return x
        return None

    def __enter__(self):
        for layer in self.by_layer:
            mod, kind = self._target(layer)

            def make(idx):
                if kind == "pre":
                    def pre_hook(_m, args):
                        out = self._apply(args[0], idx)
                        return None if out is None else (out,) + tuple(args[1:])
                    return pre_hook

                def post_hook(_m, _inp, output):
                    t = output[0] if isinstance(output, tuple) else output
                    out = self._apply(t, idx)
                    if out is None:
                        return None
                    return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
                return post_hook

            handle = (mod.register_forward_pre_hook(make(layer)) if kind == "pre"
                      else mod.register_forward_hook(make(layer)))
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# Backward-compatible alias: the z_h-only controller callers may still reference.
def _OProjPatcher(model, heads):
    return _SitePatcher(model, heads, site="z_h")


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
    gen = out[:, maxlen:]          # maxlen == input_ids_t.shape[1]; input_ids_t is
                                   # deleted on the OOM path, which always raises or
                                   # returns, but referencing it here reads as unsafe.
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
                    *, answer_steps: int = 1, site: str = "z_h", positions=None) -> dict:
    """Capture the donor activation for ``heads`` at ``site`` (z_h | v | mlp).

    z_h / mlp: the answer-step vector. v: the value slice at the CONTEXT
    ``positions`` (needle + distractor spans) — the cache the answer step reads.
    Captured during the SAME ``model.generate`` prompt forward that
    ``generate_with_patch`` patches, so self-patch stays bitwise-identical. Keyed
    by query head; for v/mlp heads sharing a KV head / layer carry the same value.
    """
    ctrl = _SitePatcher(model, heads, site=site, positions=positions)
    with ctrl:
        ctrl.capture = True
        ctrl.patch_enabled = False
        _generate_hf(model, tokenizer, input_ids, decoding_cfg)  # prompt forward captures
    return {lh: v.float().cpu().numpy() for lh, v in ctrl.captured.items()}


def generate_with_patch(
    model, tokenizer, input_ids, heads, donor_z: dict, decoding_cfg,
    *, patch_mode: str = "first_step", site: str = "z_h", positions=None,
) -> Generation:
    """Greedy generation with the target heads' activation at ``site`` replaced by
    ``donor_z`` (z_h | v | mlp). For v, ``positions`` are the context tokens whose
    value vectors are swapped in the KV cache."""
    import torch
    ctrl = _SitePatcher(model, heads, site=site, positions=positions)
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
                        *, patch_mode: str = "first_step",
                        site: str = "z_h", positions=None) -> tuple[Generation, Generation]:
    """Self-patch: patch the recipient with ITS OWN activation at ``site``. Returns
    (plain, patched); the two MUST be token-identical (§4.4) — callers assert it
    and abort on failure. Works for every site (the write replaces a value with
    itself, idempotent even where heads share a KV slice / MLP output / positions)."""
    donor = capture_donor_z(model, tokenizer, input_ids, heads, decoding_cfg,
                            site=site, positions=positions)
    plain = generate_plain(model, tokenizer, input_ids, decoding_cfg)
    patched = generate_with_patch(model, tokenizer, input_ids, heads, donor,
                                  decoding_cfg, patch_mode=patch_mode, site=site,
                                  positions=positions)
    return plain, patched


def patch_changes_tensor(model, tokenizer, recipient_ids, heads, foreign_donor,
                         decoding_cfg, *, site: str = "z_h", positions=None) -> bool:
    """Never-op detector (§ item 2): patch the recipient with a FOREIGN donor
    (donor != recipient) and report whether the target tensor actually changed.
    Self-patch cannot catch a hook that never writes (its donor equals the
    recipient, so 'no change' is correct); this can. Returns True iff the patched
    slice differed from the original somewhere."""
    import torch
    ctrl = _SitePatcher(model, heads, site=site, positions=positions)
    for (l, h) in heads:
        v = foreign_donor.get((l, h))
        if v is not None:
            ctrl.donor[(l, h)] = torch.as_tensor(np.asarray(v))
    with ctrl:
        ctrl.capture = False
        ctrl.patch_enabled = True
        ctrl.patch_mode = "first_step"
        ctrl.verify = True
        _generate_hf(model, tokenizer, recipient_ids, decoding_cfg)
    return bool(ctrl.tensor_changed)


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
