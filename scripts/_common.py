"""
Shared script scaffolding: path bootstrap, determinism, config loading, grid
enumeration, and per-cell checkpointing/resumability (task §1.7, §1.8).

Importing this module puts ``src/`` on ``sys.path`` so ``failure_mech`` is
available to every script without an install step.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from failure_mech.probes import CellSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Artifact freshness - the "resume skipped a stale run" trap
# ---------------------------------------------------------------------------

def artifact_is_current(path: str | Path) -> bool:
    """True iff ``path`` exists AND the config hashes recorded in its provenance
    still match the config files on disk.

    Resume logic must never ask "does an output file exist?" - after any earlier
    campaign, an output file exists for EVERY model, so that question answers
    "yes" for stale results and the resume silently skips work it never did. This
    bit twice: notebook 09 skipped the whole panel (every model already had an
    ``e2_signatures_*.json`` from the pre-fix runs), and notebook 10 would have
    skipped E3 models still carrying the old ``k_list = [1, 5, 10]``.

    Every artifact records ``provenance.config_hashes`` (config path -> SHA-256),
    so the honest question is "was this produced under the configs I am running
    now?". Changing grid.yaml (shell_share) or e3.yaml (k_list) therefore
    invalidates exactly the artifacts it should, automatically.

    Paths are compared by BASENAME: the recorded key is absolute and differs
    between a Colab clone and a local checkout. Any recorded config that no longer
    exists, or whose bytes changed, makes the artifact stale.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as f:
            recorded = json.load(f)["provenance"]["config_hashes"]
    except (json.JSONDecodeError, KeyError, OSError):
        return False        # unreadable or pre-provenance artifact -> not current
    if not recorded:
        return False

    from failure_mech.provenance import hash_file
    for cfg_path, old_hash in recorded.items():
        current = REPO_ROOT / "configs" / Path(cfg_path.replace("\\", "/")).name
        if not current.exists() or hash_file(current) != old_hash:
            return False
    return True


# ---------------------------------------------------------------------------
# Determinism (§1.7)
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Seed python/numpy/torch globally. GPU kernel-level nondeterminism is
    documented in README (unavoidable, §1.7)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_path(name: str) -> Path:
    return REPO_ROOT / "configs" / name


def load_paths_cfg() -> dict:
    """paths.yaml with an optional ``RFM_RESULTS_DIR`` env override for the output
    directory (Colab redirects results to Google Drive without editing configs)."""
    cfg = load_yaml(config_path("paths.yaml"))
    override = os.environ.get("RFM_RESULTS_DIR")
    if override:
        cfg.setdefault("output", {})
        cfg["output"]["results_dir"] = override
        cfg["output"]["checkpoints_dir"] = str(Path(override) / "checkpoints")
        cfg["output"]["donor_store_dir"] = str(Path(override) / "donors")
    return cfg


def time_guard(start: float, model_times: list[float], first_est_h: float,
               hard_cap_h: float = 23.0) -> tuple[bool, float, float]:
    """Adaptive 24h-safe guard (mirrors the prior detector). Returns (ok_to_start_next,
    elapsed_h, est_next_h): won't green-light a model that can't finish under the
    hard cap. ``model_times`` are per-model wall-clock hours already recorded."""
    import time
    elapsed_h = (time.time() - start) / 3600.0
    est_h = (sum(model_times) / len(model_times)) if model_times else first_est_h
    return (elapsed_h + est_h <= hard_cap_h), elapsed_h, est_h


# ---------------------------------------------------------------------------
# Grid enumeration (§5)
# ---------------------------------------------------------------------------

def enumerate_cells(grid_cfg: dict, model_key: str) -> list[CellSpec]:
    """All E1 cells for a model. The 0-distractor row is a single cell tagged
    similarity='none' (capture-free baseline); count>0 rows are per-similarity."""
    g = grid_cfg["grid"]
    cells: list[CellSpec] = []
    for ctx in g["context_length"]:
        for pos in g["needle_position"]:
            for nd in g["distractor_count"]:
                if nd == 0:
                    cells.append(CellSpec(model_key, ctx, float(pos), 0, "none"))
                else:
                    for sim in g["similarity"]:
                        cells.append(CellSpec(model_key, ctx, float(pos), int(nd), sim))
    return cells


def apply_oom_fallback_ctx(grid_cfg: dict, ctx: int) -> int:
    fb = grid_cfg.get("oom_fallback", {})
    if ctx == fb.get("from_context_length"):
        return int(fb["to_context_length"])
    return ctx


# ---------------------------------------------------------------------------
# Checkpointing / resumability (§1.8)
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Per-cell checkpoints + a manifest under
    ``results/checkpoints/{exp}_{model}/``. A killed job resumes without
    recomputing finished cells and logs what it skipped."""

    def __init__(self, results_dir: str | Path, exp: str, model_key: str,
                 fingerprint: str | None = None, fresh: bool = False):
        self.root = Path(results_dir) / "checkpoints" / f"{exp}_{model_key}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self._load_manifest()
        # A checkpoint is only reused if its fingerprint matches - so changing the
        # probe config (e.g. turning on the chat template), the pinned model SHA
        # or the seed INVALIDATES stale cells instead of silently mixing them
        # with new ones (§1.7, §1.8).
        self.fingerprint = fingerprint
        # fresh=True recomputes and OVERWRITES every cell, ignoring existing
        # checkpoints (a clean run, no resume).
        self.fresh = fresh

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as f:
                return json.load(f)
        return {"done": [], "skipped_on_resume": [], "fingerprint": None}

    def cell_path(self, cell_hash: str) -> Path:
        return self.root / f"cell_{cell_hash}.json"

    def is_done(self, cell_hash: str) -> bool:
        if self.fresh:                 # overwrite mode: nothing counts as done
            return False
        p = self.cell_path(cell_hash)
        if not p.exists():
            return False
        if self.fingerprint is None:
            return True
        try:
            with open(p, encoding="utf-8") as f:
                stored = json.load(f).get("_fingerprint")
        except (json.JSONDecodeError, OSError):
            return False
        return stored == self.fingerprint

    def load_cell(self, cell_hash: str) -> dict:
        with open(self.cell_path(cell_hash), encoding="utf-8") as f:
            return json.load(f)

    def save_cell(self, cell_hash: str, data: dict) -> None:
        if self.fingerprint is not None:
            data = {**data, "_fingerprint": self.fingerprint}
        with open(self.cell_path(cell_hash), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)
        if cell_hash not in self.manifest["done"]:
            self.manifest["done"].append(cell_hash)
        self.manifest["fingerprint"] = self.fingerprint
        self._flush()

    def note_skip(self, cell_hash: str) -> None:
        self.manifest["skipped_on_resume"].append(cell_hash)

    def _flush(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if hasattr(o, "as_tuple"):
        return o.as_tuple()
    raise TypeError(f"Not JSON-serialisable: {type(o)}")


def probe_fingerprint(config_paths: list, model_sha, seed, extra_sources=()) -> str:
    """Fingerprint of everything that determines a cell's per-sample results: the
    config bytes (grid/decoding - templates, vocab, chat-template flag, decoding
    spec), the pinned model SHA, the seed, AND the probe-generation source
    (probes.py, spans.py). Including the code means a change to e.g. the
    single-token filter invalidates stale checkpoints instead of silently mixing
    old- and new-filter probes.

    ``extra_sources`` adds more source files to the hash - E3 passes patching.py for
    a non-z_h --site run, so changing the patcher (e.g. the v cache-swap fix)
    invalidates the OLD site checkpoints, while the z_h fingerprint (which does not
    include patching.py) is unchanged and its completed sweep is not recomputed."""
    import hashlib
    h = hashlib.sha256()
    for p in config_paths:
        h.update(Path(p).read_bytes())
    for src in ("src/failure_mech/probes.py", "src/failure_mech/spans.py"):
        sp = REPO_ROOT / src
        if sp.exists():
            h.update(sp.read_bytes())
    for src in extra_sources:
        sp = Path(src)
        if sp.exists():
            h.update(sp.read_bytes())
    h.update(str(model_sha).encode("utf-8"))
    h.update(str(seed).encode("utf-8"))
    return h.hexdigest()[:16]


def write_json(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Length-sorted, token-budgeted batching (§1.10)
# ---------------------------------------------------------------------------

def token_budget_batches(
    items: list, length_of, max_tokens: int, max_samples: int, min_samples: int = 1,
) -> Iterator[list]:
    """Yield length-sorted batches whose summed prompt length stays under
    ``max_tokens`` (auto-shrinking as context grows). ``length_of(item)->int``."""
    ordered = sorted(items, key=length_of)
    batch: list = []
    tok = 0
    for it in ordered:
        L = length_of(it)
        if batch and (tok + L > max_tokens or len(batch) >= max_samples) and len(batch) >= min_samples:
            yield batch
            batch, tok = [], 0
        batch.append(it)
        tok += L
    if batch:
        yield batch
