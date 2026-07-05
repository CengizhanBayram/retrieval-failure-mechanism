"""Probe tests (task §10): determinism + token-skeleton alignment + span
decode-verify. Uses a small downloadable fast tokenizer; skips offline."""

from __future__ import annotations

from pathlib import Path

import yaml

from failure_mech.probes import ProbeFactory, CellSpec

REPO = Path(__file__).resolve().parents[1]
GRID = yaml.safe_load((REPO / "configs" / "grid.yaml").read_text(encoding="utf-8"))


def _factory(tok):
    return ProbeFactory(tok, GRID, "test_model")


def test_determinism_same_bytes(fast_tokenizer):
    f = _factory(fast_tokenizer)
    cell = CellSpec("test_model", 512, 0.5, 2, "shell_same")
    a = f.build(cell, 3, seed=42)
    b = f.build(cell, 3, seed=42)
    assert a.text == b.text
    assert a.input_ids == b.input_ids
    assert a.needle_span.as_tuple() == b.needle_span.as_tuple()
    assert [s.as_tuple() for s in a.distractor_spans] == [s.as_tuple() for s in b.distractor_spans]


def test_different_sample_idx_differs(fast_tokenizer):
    f = _factory(fast_tokenizer)
    cell = CellSpec("test_model", 512, 0.5, 2, "shell_same")
    a = f.build(cell, 0, seed=42)
    b = f.build(cell, 1, seed=42)
    assert a.needle_value != b.needle_value  # contents differ


def test_skeleton_alignment_across_samples(fast_tokenizer):
    """20 samples of one cell -> identical token length + span indices (§4.1)."""
    f = _factory(fast_tokenizer)
    cell = CellSpec("test_model", 512, 0.5, 5, "shell_diff")
    skels = set()
    for i in range(20):
        p = f.build(cell, i, seed=7)
        skels.add(p.skeleton())
    assert len(skels) == 1, f"skeleton drifted across samples: {skels}"


def test_span_decode_verify(fast_tokenizer):
    """Decoded spans equal the intended strings (§4.2). locate_value_span already
    calls verify_span, so a successful build IS the guarantee; assert explicitly."""
    f = _factory(fast_tokenizer)
    cell = CellSpec("test_model", 512, 0.3, 2, "shell_same")
    p = f.build(cell, 0, seed=1)
    dec = fast_tokenizer.decode(
        p.input_ids[p.needle_span.start:p.needle_span.end], skip_special_tokens=True
    ).strip()
    assert dec == p.needle_value
    for dv, sp in zip(p.distractor_values, p.distractor_spans):
        d = fast_tokenizer.decode(p.input_ids[sp.start:sp.end], skip_special_tokens=True).strip()
        assert d == dv


def test_degenerate_placement_no_hang(fast_tokenizer):
    """Tiny context + many distractors (fewer gaps than items) must NOT hang in
    gap placement and must still align (regression for the infinite-loop bug)."""
    f = _factory(fast_tokenizer)
    cell = CellSpec("test_model", 96, 0.5, 10, "shell_share")
    skels = {f.build(cell, i, seed=3).skeleton() for i in range(6)}
    assert len(skels) == 1
    p = f.build(cell, 0, seed=3)
    assert len(p.distractor_spans) == 10


def test_zero_distractor_baseline_has_no_distractor_spans(fast_tokenizer):
    f = _factory(fast_tokenizer)
    cell = CellSpec("test_model", 512, 0.5, 0, "none")
    p = f.build(cell, 0, seed=1)
    assert p.distractor_spans == [] and p.distractor_values == []


def test_chat_template_applied_and_deterministic(fast_tokenizer):
    """With use_chat_template on (default), the wrapper is present, skeleton is
    still stable across samples, and a fixed date keeps it byte-identical."""
    import copy
    if getattr(fast_tokenizer, "chat_template", None) is None:
        import pytest
        pytest.skip("tokenizer has no chat template")
    grid_ct = copy.deepcopy(GRID)
    grid_ct["probe"]["use_chat_template"] = True
    f = ProbeFactory(fast_tokenizer, grid_ct, "test_model")
    cell = CellSpec("test_model", 512, 0.5, 2, "shell_same")
    a = f.build(cell, 0, seed=5)
    b = f.build(cell, 0, seed=5)
    assert a.text == b.text and a.input_ids == b.input_ids
    # skeleton stable across different samples of the cell
    skels = {f.build(cell, i, seed=5).skeleton() for i in range(8)}
    assert len(skels) == 1
    # value span still decodes correctly through the wrapper
    dec = fast_tokenizer.decode(
        a.input_ids[a.needle_span.start:a.needle_span.end], skip_special_tokens=True).strip()
    assert dec == a.needle_value
