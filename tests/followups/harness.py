"""
Offline harness for the follow-up experiments (A-D): a locally TRAINED fast
tokenizer (byte-level BPE over the probe vocabulary, with a chat template) and
tiny RANDOM models built from config (Llama/Qwen2 GQA, Gemma-2 softcap + sliding
window, OLMo-2 QK-norm). Nothing is downloaded, so these tests run in a sandbox
without Hugging Face access - unlike tests/conftest.py's hub fixtures, which skip
offline.

The models are untrained: they exercise the mechanics (hooks, scopes, rows,
transplants, script plumbing), never behaviour.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def _grid():
    with open(REPO / "configs" / "grid.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _corpus(grid) -> list[str]:
    p = grid["probe"]
    words = p["adjectives"] + p["nouns"] + p["topic_words"]
    lines = [p["preamble"], p["question_template"], p["needle_template"]]
    lines += list(p["distractor_templates"].values()) + list(p["filler_templates"])
    rng = random.Random(0)
    for _ in range(400):
        lines.append(" ".join(rng.choice(words) for _ in range(12)))
        lines.append(rng.choice(p["filler_templates"]).format(TOPIC=rng.choice(p["topic_words"])))
        lines.append(f"the {rng.choice(words)} apple. the {rng.choice(p['adjectives'])} "
                     f"{rng.choice(p['nouns'])} is here.")
    for w in words:                       # every slot word becomes one ' word' token
        lines += [f"the {w} apple."] * 20
    # Values: learn the merged-leading-space token ' X' for every value glyph (as
    # Llama-3/Qwen byte-level BPE do) but no glyph-glyph merges, so every value is
    # exactly char_len tokens and never merges with the following period.
    for c in p["value"]["alphabet"]:
        lines += [f"is {c}."] * 30
    return lines


@lru_cache(maxsize=1)
def tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    grid = _grid()
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    special = ["<|bos|>", "<|eos|>", "<|pad|>", "<|user|>", "<|assistant|>", "<|end|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=1200, min_frequency=2, special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(_corpus(grid), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>")
    fast.chat_template = CHAT_TEMPLATE
    return fast


def tiny_model(family: str = "llama", *, n_layers: int = 3, seed: int = 0, attn: str = "eager"):
    """Untrained tiny model of ``family`` in fp32 with the tokenizer's vocab."""
    import torch
    import transformers as T

    tok = tiny_tokenizer()
    common = dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                  num_hidden_layers=n_layers, num_attention_heads=4,
                  max_position_embeddings=8192, bos_token_id=tok.bos_token_id,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    if family == "llama":
        cfg = T.LlamaConfig(num_key_value_heads=2, rope_theta=500000.0, **common)
        cls = T.LlamaForCausalLM
    elif family == "qwen2":
        cfg = T.Qwen2Config(num_key_value_heads=2, rope_theta=1e6, use_sliding_window=False,
                            **common)
        cls = T.Qwen2ForCausalLM
    elif family == "gemma2":
        cfg = T.Gemma2Config(num_key_value_heads=2, head_dim=16, sliding_window=64,
                             attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
                             query_pre_attn_scalar=16, **common)
        cls = T.Gemma2ForCausalLM
    elif family == "olmo2":
        cfg = T.Olmo2Config(num_key_value_heads=4, **common)
        cls = T.Olmo2ForCausalLM
    else:
        raise ValueError(family)
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    model = cls(cfg)
    # sharpen the random init a little so attention rows are not uniform
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("q_proj.weight", "k_proj.weight")):
                p.mul_(4.0)
    model.eval()
    return model, tok


def detection_artifact(model, path: Path, panel_key: str, seed: int = 42,
                       n_retrieval: int | None = None) -> Path:
    """Synthetic detection artifact: deterministic pseudo-scores over
    (layer, head); the top ``n_retrieval`` are the 'detected' heads."""
    import numpy as np
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    kv = getattr(model.config, "num_key_value_heads", H)
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, size=(L, H))
    n_retrieval = n_retrieval or max(2, (L * H) // 2)
    flat = sorted(((float(scores[l, h]), l, h) for l in range(L) for h in range(H)),
                  reverse=True)
    heads = sorted([[l, h] for (_s, l, h) in flat[:n_retrieval]])
    art = {"model": panel_key, "seed": seed, "n_layers": L, "n_heads": len(heads),
           "n_kv_heads": kv, "argmax_heads": heads, "copy_heads": heads,
           "argmax_scores": scores.tolist(), "copy_scores": scores.tolist()}
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"{panel_key}_seed{seed}.json"
    import json
    out.write_text(json.dumps(art), encoding="utf-8")
    return out
