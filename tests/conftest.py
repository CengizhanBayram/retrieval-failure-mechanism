"""
Shared test fixtures.

Puts ``src/`` on the path, provides a VALID pre-registration dict (a test
fixture only — NEVER the real configs/preregistration.yaml, §1.2), and cached
loaders for tiny public models used by the mechanism tests (skipped when a
download is unavailable so the CPU-only suite still runs offline).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


# A complete, valid pre-registration (fixture only). Mirrors the §1.2 schema and
# the classify.py sub-schema. This is written to a tmp file by `prereg_file`.
VALID_PREREG = {
    "sampling": {
        "stage1_n_per_cell": 30,
        "stage2_topup_n_breaking_cells": 120,
        "pair_min_per_cell": 5,
    },
    "breaking_band_accuracy": {"lo": 0.2, "hi": 0.8},
    "breaking_band_fallback": {"widened_band": {"lo": 0.1, "hi": 0.9}, "min_breaking_cells": 10},
    "signature_rules": {
        "reference_distribution": {"source": "success_samples", "stats": ["p5", "median"],
                                   "detector": "argmax"},
        "m1_silence": {"reference": "p5", "factor": 1.0, "absolute_floor": 0.02},
        "m2_capture": {"min_distractor_mass": 0.10, "distractor_over_needle_margin": 0.0},
        "correct_attend": {"reference": "median", "factor": 1.0},
        "residual": {},
        "mode_precedence": ["m2_capture", "m1_silence", "correct_attend", "residual"],
        "sample_level_k_heads": 5,
        "sensitivity_k_heads": 10,
    },
    "causal_criteria": {
        "flip_margin_over_control_pp": 20,
        "test": "paired_permutation",
        "alpha": 0.05,
        "correction": "benjamini_hochberg",
    },
    "seeds": {"e1_surface": 42, "e2_e3_headline_repeats": [42, 43, 44]},
    "statistics": {"effect_size": "cliffs_delta", "ci": 0.95},
}


@pytest.fixture
def valid_prereg_dict():
    import copy
    return copy.deepcopy(VALID_PREREG)


@pytest.fixture
def prereg_file(tmp_path, valid_prereg_dict):
    p = tmp_path / "preregistration.yaml"
    p.write_text(yaml.safe_dump(valid_prereg_dict), encoding="utf-8")
    return p


def _try_load_tokenizer(repo_id):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id)
        if not getattr(tok, "is_fast", False):
            return None
        return tok
    except Exception:  # noqa: BLE001 - offline / gated -> skip
        return None


def _try_load_model(repo_id, **kw):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float32, **kw)
        model.eval()
        return model, tok
    except Exception:  # noqa: BLE001
        return None


# Small open fast tokenizer for probe/span skeleton tests.
_FAST_TOK_CANDIDATES = ["Qwen/Qwen2.5-0.5B-Instruct", "hf-internal-testing/llama-tokenizer"]
# Tiny random models for capture/patching/gqa mechanism tests.
_TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_TINY_GEMMA2 = "hf-internal-testing/tiny-random-Gemma2ForCausalLM"
_TINY_OLMO2 = "hf-internal-testing/tiny-random-Olmo2ForCausalLM"


@pytest.fixture(scope="session")
def fast_tokenizer():
    for cand in _FAST_TOK_CANDIDATES:
        tok = _try_load_tokenizer(cand)
        if tok is not None:
            return tok
    pytest.skip("no fast tokenizer downloadable (offline/gated)")


@pytest.fixture(scope="session")
def tiny_llama():
    got = _try_load_model(_TINY_LLAMA, attn_implementation="eager")
    if got is None:
        pytest.skip(f"{_TINY_LLAMA} not downloadable")
    return got


@pytest.fixture(scope="session")
def tiny_gemma2():
    got = _try_load_model(_TINY_GEMMA2, attn_implementation="eager")
    if got is None:
        pytest.skip(f"{_TINY_GEMMA2} not downloadable")
    return got


@pytest.fixture(scope="session")
def tiny_olmo2():
    got = _try_load_model(_TINY_OLMO2, attn_implementation="eager")
    if got is None:
        pytest.skip(f"{_TINY_OLMO2} not downloadable")
    return got
