"""
Pinned model loading + panel access (task §1.3, §2, §3).

Reuses the Part-2 pinned registry (``configs/panel.yaml`` in retrieval-head-
profile) and puts the Part-1 ``src`` and Part-2 ``rhp`` packages on ``sys.path``
so their proven code is importable (never edited — §0, §12). Paths come from
``configs/paths.yaml`` with env-var overrides so the same configs run on Colab
and locally.

Loading is deterministic and PINNED: the weight commit SHA is read from
panel.yaml and asserted to be a real 40-hex commit (``revision="main"`` is a bug,
§1.3). Weights load in bf16 (§1.10) with an explicit attention backend chosen
per experiment (§1.5). The inherited Part-1 loader only does fp16/8-bit, so this
module loads directly against the same pinned cfg rather than through it, and
documents that choice.

GQA bookkeeping (§2): head identity is always ``(layer, query_head)``; the K for
a query head lives in ``kv_head = query_head // (n_query_heads // n_kv_heads)``.
The division is asserted exact at load.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class PanelError(RuntimeError):
    """Raised on unpinned revisions, unknown keys, or bad GQA divisions (§1.9)."""


# ---------------------------------------------------------------------------
# Path resolution (Part-1 / Part-2 repos, panel.yaml, detection artifacts)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_paths_cfg(paths_yaml: str | Path | None = None) -> dict:
    p = Path(paths_yaml) if paths_yaml else _repo_root() / "configs" / "paths.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_repo(spec: dict, looks_like) -> Path:
    """Resolve a repo directory from a paths.yaml sub-spec (env -> path ->
    sibling autodiscovery by dirname). ``looks_like(path)->bool`` validates a
    candidate."""
    candidates: list[Path] = []
    env = spec.get("env")
    if env and os.environ.get(env):
        candidates.append(Path(os.environ[env]))
    if spec.get("path"):
        candidates.append((_repo_root() / spec["path"]).resolve())
        candidates.append(Path(spec["path"]))
    dirname = spec.get("dirname")
    if dirname:
        # search siblings of the repo root and of Desktop-style parents
        for base in (_repo_root().parent, _repo_root().parent.parent):
            candidates.append(base / dirname)
            # also one nested level (e.g. rope-dimensions+/retrieval-head-profile)
            if base.exists():
                for child in base.iterdir():
                    if child.is_dir():
                        candidates.append(child / dirname)
    for c in candidates:
        try:
            c = c.resolve()
        except OSError:
            continue
        if c.exists() and looks_like(c):
            return c
    raise PanelError(
        f"Could not resolve repo (env={spec.get('env')}, dirname={dirname}). "
        f"Set the {env} environment variable. Tried: {[str(c) for c in candidates]}"
    )


def resolve_part1_repo(paths_cfg: dict) -> Path:
    return _resolve_repo(
        paths_cfg["part1_repo"],
        lambda p: (p / "src" / "retrieval_head_detector.py").exists(),
    )


def resolve_part2_repo(paths_cfg: dict) -> Path:
    return _resolve_repo(
        paths_cfg["part2_repo"],
        lambda p: (p / "rhp" / "panel.py").exists(),
    )


def ensure_reuse_on_path(paths_cfg: dict) -> tuple[Path, Path]:
    """Put Part-1 ``src`` and Part-2 ``rhp`` on ``sys.path``; return both roots.
    Also sets ``RHP_PART1_REPO`` so ``rhp._paths`` resolves consistently."""
    part2 = resolve_part2_repo(paths_cfg)
    part1 = resolve_part1_repo(paths_cfg)
    os.environ.setdefault("RHP_PART1_REPO", str(part1))
    for root in (str(part1), str(part2)):
        if root not in sys.path:
            sys.path.insert(0, root)
    return part1, part2


def panel_yaml_path(paths_cfg: dict) -> Path:
    spec = paths_cfg["panel_yaml"]
    if spec.get("env") and os.environ.get(spec["env"]):
        return Path(os.environ[spec["env"]])
    part2 = resolve_part2_repo(paths_cfg)
    return part2 / spec["relpath"]


def detection_dir(paths_cfg: dict) -> Path:
    spec = paths_cfg["detection_artifacts"]
    if spec.get("env") and os.environ.get(spec["env"]):
        return Path(os.environ[spec["env"]])
    part2 = resolve_part2_repo(paths_cfg)
    return part2 / spec["relpath"]


# ---------------------------------------------------------------------------
# Panel / model config
# ---------------------------------------------------------------------------

def load_panel(paths_cfg: dict) -> dict:
    with open(panel_yaml_path(paths_cfg), encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_key(paths_cfg: dict, task_key: str) -> str:
    """Map a task-spec model key (§2) to the panel.yaml key."""
    mapping = paths_cfg.get("models", {})
    if task_key in mapping:
        return mapping[task_key]
    return task_key  # allow passing a panel key directly


def model_cfg(paths_cfg: dict, panel: dict, task_key: str) -> dict:
    """Fully-resolved, PINNED per-model config. Asserts a real commit SHA (§1.3)."""
    key = resolve_model_key(paths_cfg, task_key)
    defaults = dict(panel.get("defaults", {}))
    pool = {**panel.get("models", {}), **panel.get("quant_models", {})}
    if key not in pool:
        raise PanelError(f"Unknown model key '{key}'. Known: {sorted(pool)}")
    cfg = {**defaults, **pool[key]}
    rev = cfg.get("revision")
    if not rev or rev == "main" or not _SHA_RE.match(str(rev)):
        raise PanelError(
            f"Model '{key}' revision is not a pinned 40-hex commit SHA "
            f"(got {rev!r}). revision='main' is a bug (§1.3)."
        )
    cfg["panel_key"] = key
    return cfg


# ---------------------------------------------------------------------------
# GQA head mapping (§2)
# ---------------------------------------------------------------------------

def gqa_group_size(n_query_heads: int, n_kv_heads: int) -> int:
    """Query heads per KV group; asserts the division is exact (§2)."""
    if n_kv_heads <= 0 or n_query_heads % n_kv_heads != 0:
        raise PanelError(
            f"n_query_heads ({n_query_heads}) not divisible by n_kv_heads "
            f"({n_kv_heads}); GQA mapping undefined."
        )
    return n_query_heads // n_kv_heads


def query_to_kv_head(query_head: int, n_query_heads: int, n_kv_heads: int) -> int:
    """Map a query head to its shared KV head (§2)."""
    return query_head // gqa_group_size(n_query_heads, n_kv_heads)


def head_counts(model) -> tuple[int, int]:
    """(num_attention_heads, num_key_value_heads) from the model config; MHA ->
    kv == heads."""
    cfg = model.config
    n_heads = int(getattr(cfg, "num_attention_heads"))
    n_kv = getattr(cfg, "num_key_value_heads", None)
    n_kv = int(n_kv) if n_kv is not None else n_heads
    return n_heads, n_kv


def head_dim(model) -> int:
    cfg = model.config
    if getattr(cfg, "head_dim", None):
        return int(cfg.head_dim)
    return int(cfg.hidden_size // cfg.num_attention_heads)


# ---------------------------------------------------------------------------
# Per-family attention metadata (for the manual capture row, §4.3)
# ---------------------------------------------------------------------------

def is_gemma2(model) -> bool:
    return type(model).__name__.lower().startswith("gemma2") or \
        getattr(model.config, "model_type", "").lower() == "gemma2"


def attention_scale(model) -> float:
    """The multiplicative scale applied to q.Kᵀ BEFORE softmax.

    Standard (Llama/Mistral/Qwen): 1/sqrt(head_dim). Gemma-2 uses
    1/sqrt(query_pre_attn_scalar) when that config field is present (it may
    differ from head_dim). Read from config; never assumed.
    """
    import math
    cfg = model.config
    qpre = getattr(cfg, "query_pre_attn_scalar", None)
    if is_gemma2(model) and qpre:
        return 1.0 / math.sqrt(float(qpre))
    return 1.0 / math.sqrt(head_dim(model))


def attn_logit_softcap(model) -> float | None:
    """Gemma-2 attention-logit softcapping value (tanh squash before softmax),
    or None for models that do not softcap (§4.3, researcher note)."""
    cap = getattr(model.config, "attn_logit_softcapping", None)
    return float(cap) if cap else None


def attn_module(model, layer_idx: int):
    """The self-attention module for a decoder layer (Llama/Qwen/Mistral/Gemma-2
    all expose ``model.model.layers[i].self_attn``)."""
    return model.model.layers[layer_idx].self_attn


# ---------------------------------------------------------------------------
# Loading (pinned, bf16, explicit attention backend)
# ---------------------------------------------------------------------------

def load_model(
    paths_cfg: dict,
    panel: dict,
    task_key: str,
    *,
    attn_implementation: str = "sdpa",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple[Any, Any, dict]:
    """Load (model, tokenizer, resolved_cfg) at the PINNED revision in bf16.

    Gemma-2 with softcapping is auto-promoted to eager by HF regardless of the
    requested backend; that is expected (README) and we surface the effective
    backend in ``resolved_cfg['effective_attn']``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = model_cfg(paths_cfg, panel, task_key)
    hf_id, rev = cfg["hf_id"], cfg["revision"]
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=rev, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, revision=rev, torch_dtype=torch_dtype, device_map=device_map,
        attn_implementation=attn_implementation, trust_remote_code=True,
    )
    model.eval()

    n_heads, n_kv = head_counts(model)
    gqa_group_size(n_heads, n_kv)  # assert exact (§2)
    cfg["n_query_heads"] = n_heads
    cfg["n_kv_heads"] = n_kv
    cfg["head_dim"] = head_dim(model)
    cfg["effective_attn"] = getattr(model.config, "_attn_implementation", attn_implementation)
    return model, tokenizer, cfg
