"""GQA mapping tests (task §10): query_head -> kv_head, pure math always, plus a
tiny-model forward when a model is downloadable."""

from __future__ import annotations

import pytest

from failure_mech import panel as P
from failure_mech.panel import PanelError


def test_group_size_exact():
    assert P.gqa_group_size(32, 8) == 4
    assert P.gqa_group_size(28, 4) == 7
    assert P.gqa_group_size(16, 16) == 1  # MHA


def test_group_size_inexact_raises():
    with pytest.raises(PanelError):
        P.gqa_group_size(30, 8)


def test_query_to_kv_head_llama31_layout():
    # Llama-3.1-8B: 32 query heads, 8 kv heads, group=4.
    for qh in range(32):
        assert P.query_to_kv_head(qh, 32, 8) == qh // 4
    assert P.query_to_kv_head(0, 32, 8) == 0
    assert P.query_to_kv_head(31, 32, 8) == 7


def test_tiny_model_mapping_consistent(tiny_llama):
    model, _tok = tiny_llama
    n_heads, n_kv = P.head_counts(model)
    P.gqa_group_size(n_heads, n_kv)  # asserts exact
    for qh in range(n_heads):
        kv = P.query_to_kv_head(qh, n_heads, n_kv)
        assert 0 <= kv < n_kv
