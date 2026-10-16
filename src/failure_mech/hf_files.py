"""
Hub files the CDN refuses: seed them from a byte-identical sibling.

Why: at the pinned revision of ``mistralai/Mistral-7B-Instruct-v0.3``
(c170c708...) the Xet bridge now answers 403 for ``tokenizer.model``. The same
bytes are in the repository as ``tokenizer.model.v3`` (a plain git file, same
size, sha256 equal to tokenizer.model's LFS oid), which downloads normally.
Because the repository's tokenizer_config sets ``add_prefix_space``, the fast
tokenizer is rebuilt from the slow one and needs ``tokenizer.model``;
transformers swallows the 403, passes ``vocab_file=None`` on and fails with
``TypeError: expected str, bytes or os.PathLike object, not NoneType``. The
same pinned revision loaded without problems in the July 2026 E1-E5 runs.

``ensure_hub_file`` downloads the file normally when it can. When the download
is refused, it looks for a sibling of the same size whose sha256 equals the
file's LFS sha256 and copies it into the Hub cache as that file, at the pinned
commit. The content is verified against the Hub's own hash, never assumed, so
whatever loads afterwards reads exactly the pinned bytes. Nothing changes for
repositories where the download works or where the file does not exist.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil

log = logging.getLogger(__name__)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_hub_file(repo_id: str, filename: str, revision: str,
                    cache_dir: str | None = None) -> str | None:
    """Local path of ``filename`` at ``revision`` in the Hub cache, or None when
    the repository has no such file. A refused download is replaced by a
    verified byte-identical sibling; if none exists the original error is raised."""
    import huggingface_hub as hh
    from huggingface_hub.utils import EntryNotFoundError

    try:
        return hh.hf_hub_download(repo_id, filename, revision=revision, cache_dir=cache_dir)
    except EntryNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001  - any refusal: try the verified sibling
        refused = exc

    if not _SHA_RE.match(str(revision)):
        raise refused                       # the cache slot needs the pinned commit
    tree = {f.path: f for f in hh.HfApi().list_repo_tree(repo_id, revision=revision)
            if getattr(f, "size", None) is not None}
    target = tree.get(filename)
    lfs = getattr(target, "lfs", None)
    want = getattr(lfs, "sha256", None)
    if target is None or not want:
        raise refused                       # nothing to verify a sibling against
    for path, f in sorted(tree.items()):
        if path == filename or f.size != target.size:
            continue
        src = hh.hf_hub_download(repo_id, path, revision=revision, cache_dir=cache_dir)
        if _sha256(src) != want:
            continue
        root = os.path.join(cache_dir or hh.constants.HF_HUB_CACHE,
                            "models--" + repo_id.replace("/", "--"))
        dst = os.path.join(root, "snapshots", revision, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".seeding"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        marker = os.path.join(root, ".no_exist", revision, filename)
        if os.path.exists(marker):
            os.remove(marker)
        got = hh.try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir, revision=revision)
        if not isinstance(got, str) or _sha256(got) != want:
            raise RuntimeError(f"{repo_id}/{filename}: seeded copy not visible in the Hub cache")
        log.warning("%s/%s @ %s: download refused (%s); seeded from %s "
                    "(sha256 %s… matches the file's LFS hash).",
                    repo_id, filename, revision[:12], type(refused).__name__, path, want[:12])
        return got
    raise refused


def ensure_tokenizer_files(repo_id: str, revision: str, cache_dir: str | None = None) -> None:
    """Make the sentencepiece model loadable before ``AutoTokenizer`` runs.

    Never raises: when the file cannot be provided, ``AutoTokenizer`` behaves
    exactly as before (it only needs the file when it rebuilds from the slow
    tokenizer). No-op for repositories without one (Llama-3, Qwen2)."""
    try:
        ensure_hub_file(repo_id, "tokenizer.model", revision, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001
        log.info("%s: tokenizer.model not provided (%s: %s); leaving it to AutoTokenizer.",
                 repo_id, type(exc).__name__, str(exc)[:200])
