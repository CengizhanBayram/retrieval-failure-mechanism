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
    """Adaptive 24h-safe guard (mirrors Part-2). Returns (ok_to_start_next,
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

    def __init__(self, results_dir: str | Path, exp: str, model_key: str):
        self.root = Path(results_dir) / "checkpoints" / f"{exp}_{model_key}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as f:
                return json.load(f)
        return {"done": [], "skipped_on_resume": []}

    def cell_path(self, cell_hash: str) -> Path:
        return self.root / f"cell_{cell_hash}.json"

    def is_done(self, cell_hash: str) -> bool:
        return self.cell_path(cell_hash).exists()

    def load_cell(self, cell_hash: str) -> dict:
        with open(self.cell_path(cell_hash), encoding="utf-8") as f:
            return json.load(f)

    def save_cell(self, cell_hash: str, data: dict) -> None:
        with open(self.cell_path(cell_hash), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)
        if cell_hash not in self.manifest["done"]:
            self.manifest["done"].append(cell_hash)
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
