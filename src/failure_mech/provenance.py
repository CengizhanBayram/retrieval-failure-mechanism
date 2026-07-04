"""
Provenance stamping (task §1.6). ONE implementation, reused by every script.

Every JSON record written to ``results/`` embeds a provenance block so any
number can be traced back to: the producing script, the exact code commit, the
byte-hashes of every config that fed it, the model key + pinned weight SHA, the
seed(s), and an ISO-8601 UTC timestamp. See README "PROVENANCE" for the audit
chain.

No interpretation, no analysis — just faithful recording of what produced a
number.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run_git(args: list[str], repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def code_commit(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return the current code commit + dirty flag for ``repo_root``.

    ``repo_root`` defaults to the repository containing this file. A missing git
    or a non-repo yields ``{"commit": None, ...}`` rather than raising — the
    provenance block still records that the commit was unavailable (which is
    itself audit information), and callers that require a clean commit assert on
    it explicitly.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    commit = _run_git(["rev-parse", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "repo_root": str(root),
    }


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes (used for config hashes)."""
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(paths: list[str | Path]) -> dict[str, str]:
    """Map each config path -> its SHA-256 (missing files raise; §1.9)."""
    out: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Config to hash does not exist: {p}")
        out[str(p)] = hash_file(p)
    return out


def package_versions() -> dict[str, str]:
    """Versions of the numerics/model stack that shaped a number."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "torch", "transformers"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - version probe must never break a run
            versions[mod] = "not-installed"
    return versions


def make_provenance(
    *,
    script: str,
    config_paths: list[str | Path],
    model_key: str | None = None,
    model_sha: str | None = None,
    seeds: list[int] | int | None = None,
    repo_root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance block embedded in every results JSON (§1.6).

    Args:
        script: producing script name, e.g. "scripts/e2_signatures.py".
        config_paths: every config file that fed this run; each is byte-hashed.
        model_key: task/panel model key (§2), when a model was involved.
        model_sha: the PINNED weight commit SHA the model was loaded at (§1.3).
        seeds: seed or seeds used.
        repo_root: code repo root (defaults to this repo).
        extra: any additional audit fields to record verbatim.
    """
    if isinstance(seeds, int):
        seeds = [seeds]
    block: dict[str, Any] = {
        "script": script,
        "code": code_commit(repo_root),
        "config_hashes": hash_files(config_paths),
        "model_key": model_key,
        "model_sha": model_sha,
        "seeds": seeds,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    if extra:
        block["extra"] = extra
    return block
