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
        self.active = False            # patch on this forward?
        self.capture = False           # capture donor on this forward?
        self.donor: dict[tuple[int, int], object] = {}   # (l,h) -> tensor [head_dim]
        self.captured: dict[tuple[int, int], object] = {}
        self._handles = []

    def __enter__(self):
        for li in self.by_layer:
            o_proj = panel.attn_module(self.model, li).o_proj

            def make(idx):
                def pre_hook(_m, args):
                    x = args[0]
                    if self.capture:
                        for h in self.by_layer[idx]:
                            sl = x[:, -1, h * self.head_dim:(h + 1) * self.head_dim]
                            self.captured[(idx, h)] = sl[0].detach().clone()
                    if self.active:
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


def _newline_token_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for probe in ("\n", " \n", "\n\n"):
        for t in tokenizer(probe, add_special_tokens=False)["input_ids"]:
            ids.add(int(t))
    return ids


def _greedy_loop(model, tokenizer, input_ids, decoding_cfg, controller, patch_mode):
    """Deterministic greedy generation (§1.4) with a step-scoped patch."""
    import torch
    device = next(model.parameters()).device
    max_new = int(decoding_cfg["decoding"]["max_new_tokens"])
    stop_nl = bool(decoding_cfg["decoding"].get("stop_on_newline", True))
    newline_ids = _newline_token_ids(tokenizer) if stop_nl else set()
    eos = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None

    prompt_len = len(input_ids)
    cur = torch.tensor([list(input_ids)], device=device, dtype=torch.long)
    past = None
    generated: list[int] = []
    with torch.no_grad():
        for step in range(max_new):
            if controller is not None:
                controller.active = (patch_mode == "sustained") or (step == 0)
            seq_from = 0 if step == 0 else prompt_len + step - 1
            position_ids = torch.arange(
                seq_from, seq_from + cur.shape[1], device=device
            ).unsqueeze(0)
            out = model(input_ids=cur, position_ids=position_ids,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(torch.argmax(out.logits[0, -1, :]).item())
            generated.append(nxt)
            if eos is not None and nxt == eos:
                break
            if nxt in newline_ids:
                break
            cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return Generation(token_ids=generated, text=text)


def generate_plain(model, tokenizer, input_ids, decoding_cfg) -> Generation:
    """No-patch greedy generation (flip base rate / self-patch reference)."""
    return _greedy_loop(model, tokenizer, input_ids, decoding_cfg, None, "first_step")


def generate_plain_batch(model, tokenizer, input_ids_list, decoding_cfg) -> list["Generation"]:
    """Batched, left-padded greedy generation for many prompts at once.

    Used for the no-hook GRADING passes (E1, and the E2/E3 pairing step) where
    throughput matters. Hooked capture/patching stay single-sequence. Greedy +
    max_new_tokens from the one decoding spec (§1.4); the grader reads the first
    line, so per-sequence newline stopping is unnecessary here.
    """
    import torch
    if not input_ids_list:
        return []
    device = next(model.parameters()).device
    dec = decoding_cfg["decoding"]
    pad_id = tokenizer.pad_token_id
    maxlen = max(len(x) for x in input_ids_list)
    input_ids, attn = [], []
    for ids in input_ids_list:                    # LEFT pad (decoder-only)
        pad = maxlen - len(ids)
        input_ids.append([pad_id] * pad + list(ids))
        attn.append([0] * pad + [1] * len(ids))
    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
    attn = torch.tensor(attn, device=device, dtype=torch.long)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attn,
                             do_sample=False, num_beams=1,
                             max_new_tokens=int(dec["max_new_tokens"]),
                             pad_token_id=pad_id)
    gen = out[:, input_ids.shape[1]:]
    results = []
    for row in gen:
        toks = [int(t) for t in row.tolist()]
        results.append(Generation(token_ids=toks,
                                   text=tokenizer.decode(toks, skip_special_tokens=True)))
    return results


def capture_donor_z(model, input_ids, heads, *, answer_steps: int = 1) -> dict:
    """Capture donor z_h at generation step 1 (per head) for ``heads``.

    Returns ``{(layer, head): np.ndarray[head_dim] float32}``. (Only step-1
    vectors for the headline; sustained donors reuse the step-1 vector where a
    later step is unavailable, per §4.4.)
    """
    import torch
    device = next(model.parameters()).device
    ids = torch.tensor([list(input_ids)], device=device, dtype=torch.long)
    ctrl = _OProjPatcher(model, heads)
    with ctrl:
        ctrl.capture = True
        ctrl.active = False
        position_ids = torch.arange(0, ids.shape[1], device=device).unsqueeze(0)
        with torch.no_grad():
            model(input_ids=ids, position_ids=position_ids, use_cache=True)
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
        return _greedy_loop(model, tokenizer, input_ids, decoding_cfg, ctrl, patch_mode)


def self_patch_generate(model, tokenizer, input_ids, heads, decoding_cfg,
                        *, patch_mode: str = "first_step") -> tuple[Generation, Generation]:
    """Self-patch: patch the recipient with ITS OWN z. Returns (plain, patched).
    The two MUST be token-identical (§4.4); callers assert it and abort on
    failure."""
    donor = capture_donor_z(model, input_ids, heads)
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
