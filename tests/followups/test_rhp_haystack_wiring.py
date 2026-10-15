"""
Wiring of follow-up D (scripts/rhp_haystack_check.py), which lives in this repository
but runs the Part I detector of a retrieval-head-profile clone. No model is loaded:
the test checks that the clone is found, that its own scripts/_common.py is used (not
this repository's), that the panel config and the stored profiles are read from the
clone, and that the fallback pool is installed and verified. Skipped when the sibling
clones are not available (RHP_PART2_REPO / RHP_PART1_REPO, as set by the Colab setup).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rhp_haystack_check.py"
RHP = os.environ.get("RHP_PART2_REPO")
PART1 = os.environ.get("RHP_PART1_REPO")

pytestmark = pytest.mark.skipif(
    not (RHP and PART1 and Path(RHP, "rhp").is_dir() and Path(PART1, "src").is_dir()),
    reason="retrieval-head-profile / Part-1 clones not available (set RHP_PART2_REPO, RHP_PART1_REPO)")


def _load():
    spec = importlib.util.spec_from_file_location("rhp_haystack_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolves_clone_and_its_helpers(tmp_path):
    D = _load()
    root = D.rhp_root(RHP)
    assert root == Path(RHP).resolve() and str(root) in sys.path
    common = D.rhp_common(root)
    assert Path(common.__file__).resolve() == (root / "scripts" / "_common.py").resolve()
    assert hasattr(common, "bootstrap") and hasattr(common, "save_json")
    with pytest.raises(SystemExit):
        D.rhp_root(str(tmp_path))                       # not a clone -> refused


def test_fallback_pool_installs_and_is_verified():
    D = _load()
    root = D.rhp_root(RHP)
    D.rhp_common(root).bootstrap(PART1)
    pool, meta = D.build_pool("fallback")
    assert meta["n_sentences"] == len(pool) and meta["n_unique"] <= 50
    sha = D.install_pool(pool)
    D.assert_pool(sha)                                  # unchanged cache passes
    import src.corpus as corpus
    corpus._CORPUS_CACHE = list(pool[:-1])
    with pytest.raises(RuntimeError):
        D.assert_pool(sha)                              # a changed cache is detected


def test_main_reads_config_from_clone_and_summarizes(tmp_path):
    D = _load()
    with pytest.raises(SystemExit, match="no models"):  # gets past bootstrap + load_panel
        D.main(["--rhp-repo", RHP, "--part1-repo", PART1, "--haystack", "fallback",
                "--out-dir", str(tmp_path)])
    assert D.main(["--rhp-repo", RHP, "--summarize", "--out-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "D_haystack" / "summary.json").read_text())
    assert "models" in summary
