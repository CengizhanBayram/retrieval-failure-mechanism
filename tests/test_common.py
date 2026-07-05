"""Scaffolding tests: checkpoint fingerprint invalidation + token-budget batching."""

from __future__ import annotations

import _common as C


def test_probe_fingerprint_sensitive_to_inputs(tmp_path):
    a = tmp_path / "a.yaml"; a.write_text("x: 1", encoding="utf-8")
    fp1 = C.probe_fingerprint([a], "sha1", 42)
    fp2 = C.probe_fingerprint([a], "sha1", 42)
    assert fp1 == fp2                       # deterministic
    assert C.probe_fingerprint([a], "sha1", 43) != fp1   # seed matters
    assert C.probe_fingerprint([a], "sha2", 42) != fp1   # model SHA matters
    a.write_text("x: 2", encoding="utf-8")
    assert C.probe_fingerprint([a], "sha1", 42) != fp1   # config bytes matter


def test_checkpoint_fingerprint_invalidates_stale(tmp_path):
    ck1 = C.CheckpointManager(tmp_path, "e1", "m", fingerprint="fpA")
    ck1.save_cell("cellX", {"accuracy": 0.5})
    assert ck1.is_done("cellX")
    # same fingerprint -> reused
    ck_same = C.CheckpointManager(tmp_path, "e1", "m", fingerprint="fpA")
    assert ck_same.is_done("cellX")
    # different fingerprint (config/model/seed changed) -> NOT reused
    ck_diff = C.CheckpointManager(tmp_path, "e1", "m", fingerprint="fpB")
    assert not ck_diff.is_done("cellX")
    # no fingerprint -> existence-only (back-compat)
    ck_none = C.CheckpointManager(tmp_path, "e1", "m", fingerprint=None)
    assert ck_none.is_done("cellX")


def test_checkpoint_roundtrip_preserves_data(tmp_path):
    ck = C.CheckpointManager(tmp_path, "e1", "m", fingerprint="fp")
    ck.save_cell("c", {"accuracy": 0.42, "n_total": 24})
    loaded = ck.load_cell("c")
    assert loaded["accuracy"] == 0.42 and loaded["_fingerprint"] == "fp"


def test_token_budget_batches_respects_limits():
    items = [("a", 100), ("b", 100), ("c", 100), ("d", 5000)]
    batches = list(C.token_budget_batches(
        items, length_of=lambda x: x[1], max_tokens=250, max_samples=8))
    for b in batches:
        assert sum(x[1] for x in b) <= 250 or len(b) == 1  # single oversized item allowed alone
    # every item appears exactly once
    flat = [x for b in batches for x in b]
    assert sorted(flat) == sorted(items)
