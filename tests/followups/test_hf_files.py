"""
failure_mech.hf_files: a Hub file whose download is refused is seeded from a
byte-identical sibling, verified against the file's LFS sha256 (the Mistral-7B-v0.3
tokenizer.model case). Offline: the Hub calls are replaced by fakes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import huggingface_hub as hh  # noqa: E402
from huggingface_hub.utils import EntryNotFoundError  # noqa: E402

from failure_mech import hf_files  # noqa: E402

REPO, REV = "org/model", "c" * 40
BLOB = b"sentencepiece model bytes"


class Refused(Exception):
    pass


def _fake_hub(monkeypatch, tmp_path, *, sibling=BLOB, target_status="refused"):
    files = {"tokenizer.model.v3": sibling, "tokenizer.json": b"{}"}
    sha = hashlib.sha256(BLOB).hexdigest()

    def download(repo_id, filename, revision=None, cache_dir=None):
        if filename == "tokenizer.model":
            if target_status == "missing":
                raise EntryNotFoundError("404")
            if target_status == "ok":
                p = tmp_path / "ok_tokenizer.model"
                p.write_bytes(BLOB)
                return str(p)
            raise Refused("403 Forbidden")
        p = tmp_path / "dl" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(files[filename])
        return str(p)

    tree = [SimpleNamespace(path="tokenizer.model", size=len(BLOB), lfs=SimpleNamespace(sha256=sha)),
            SimpleNamespace(path="tokenizer.model.v3", size=len(sibling), lfs=None),
            SimpleNamespace(path="tokenizer.json", size=2, lfs=None)]

    class Api:
        def list_repo_tree(self, repo_id, revision=None):
            return iter(tree)

    def from_cache(repo_id, filename, cache_dir=None, revision=None):
        p = Path(cache_dir, "models--" + repo_id.replace("/", "--"), "snapshots", revision, filename)
        return str(p) if p.exists() else None

    monkeypatch.setattr(hh, "hf_hub_download", download)
    monkeypatch.setattr(hh, "HfApi", Api)
    monkeypatch.setattr(hh, "try_to_load_from_cache", from_cache)
    return tmp_path / "cache"


def test_download_that_works_is_returned_unchanged(monkeypatch, tmp_path):
    cache = _fake_hub(monkeypatch, tmp_path, target_status="ok")
    assert hf_files.ensure_hub_file(REPO, "tokenizer.model", REV, cache_dir=str(cache)).endswith("ok_tokenizer.model")


def test_missing_file_is_none(monkeypatch, tmp_path):
    cache = _fake_hub(monkeypatch, tmp_path, target_status="missing")
    assert hf_files.ensure_hub_file(REPO, "tokenizer.model", REV, cache_dir=str(cache)) is None


def test_refused_file_is_seeded_from_verified_sibling(monkeypatch, tmp_path):
    cache = _fake_hub(monkeypatch, tmp_path)
    got = hf_files.ensure_hub_file(REPO, "tokenizer.model", REV, cache_dir=str(cache))
    assert Path(got) == cache / "models--org--model" / "snapshots" / REV / "tokenizer.model"
    assert Path(got).read_bytes() == BLOB


def test_sibling_with_other_bytes_is_rejected(monkeypatch, tmp_path):
    cache = _fake_hub(monkeypatch, tmp_path, sibling=b"x" * len(BLOB))   # same size, other hash
    with pytest.raises(Refused):
        hf_files.ensure_hub_file(REPO, "tokenizer.model", REV, cache_dir=str(cache))
    assert not (cache / "models--org--model" / "snapshots" / REV / "tokenizer.model").exists()


def test_unpinned_revision_is_not_seeded(monkeypatch, tmp_path):
    cache = _fake_hub(monkeypatch, tmp_path)
    with pytest.raises(Refused):
        hf_files.ensure_hub_file(REPO, "tokenizer.model", "main", cache_dir=str(cache))


def test_tokenizer_wrapper_never_raises(monkeypatch, tmp_path):
    cache = _fake_hub(monkeypatch, tmp_path, sibling=b"x" * len(BLOB))
    hf_files.ensure_tokenizer_files(REPO, REV, cache_dir=str(cache))      # logs, does not raise
