"""
Detection-artifact reader (task §0, §6, §9).

Reads the Part-2 detection artifact for a model — one JSON per (model, seed) at
``datas/results/profile/{panel_key}_seed{seed}.json`` — and exposes a canonical
top-k ranked retrieval-head list. Part 3 NEVER re-detects (§6); it consumes this.

The artifact stores ``argmax_heads``/``copy_heads`` SORTED BY (layer, head), not
by score, so a canonical top-k requires ranking the detected heads by their
score matrix here (descending, tie-broken by (layer, head) for determinism).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DetectionError(RuntimeError):
    """Raised when an artifact is missing or a requested k exceeds the list (§1.9)."""


@dataclass
class Detection:
    model_key: str
    seed: int
    path: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    argmax_heads: list[tuple[int, int]]
    copy_heads: list[tuple[int, int]]
    argmax_scores: np.ndarray  # (n_layers, n_heads)
    copy_scores: np.ndarray

    # -- ranking ------------------------------------------------------------

    def _ranked(self, detector: str) -> list[tuple[int, int]]:
        if detector == "argmax":
            heads, scores = self.argmax_heads, self.argmax_scores
        elif detector == "copy":
            heads, scores = self.copy_heads, self.copy_scores
        else:
            raise DetectionError(f"Unknown detector '{detector}' (use 'argmax'|'copy').")
        # Sort detected heads by score DESC, tie-break (layer, head) ASC.
        return sorted(heads, key=lambda lh: (-float(scores[lh[0], lh[1]]), lh[0], lh[1]))

    def top_k_heads(self, k: int, detector: str = "argmax") -> list[tuple[int, int]]:
        """Top-k retrieval heads by detector score. Fails loudly if the detected
        list is shorter than k (§6)."""
        ranked = self._ranked(detector)
        if k > len(ranked):
            raise DetectionError(
                f"Requested top-{k} {detector} heads but only {len(ranked)} were "
                f"detected for '{self.model_key}' (artifact {self.path}). "
                "Refusing to pad (§6)."
            )
        return ranked[:k]

    def retrieval_head_set(self, detector: str = "argmax") -> set[tuple[int, int]]:
        heads = self.argmax_heads if detector == "argmax" else self.copy_heads
        return {(int(l), int(h)) for (l, h) in heads}

    def non_retrieval_heads(self, detector: str = "argmax") -> list[tuple[int, int]]:
        """All (layer, head) NOT in the retrieval set — the random-control pool
        (§4.4)."""
        retr = self.retrieval_head_set(detector)
        return [
            (l, h)
            for l in range(self.n_layers)
            for h in range(self.n_heads)
            if (l, h) not in retr
        ]


def artifact_path(detection_dir: str | Path, panel_key: str, seed: int) -> Path:
    return Path(detection_dir) / f"{panel_key}_seed{seed}.json"


def load_detection(detection_dir: str | Path, panel_key: str, seed: int) -> Detection:
    p = artifact_path(detection_dir, panel_key, seed)
    if not p.exists():
        raise DetectionError(
            f"Detection artifact not found: {p}. Part 3 consumes the Part-2 "
            "profile artifact and never re-detects (§6)."
        )
    with open(p, encoding="utf-8") as f:
        d = json.load(f)

    def _heads(key: str) -> list[tuple[int, int]]:
        return [(int(l), int(h)) for (l, h) in d[key]]

    argmax_scores = np.asarray(d["argmax_scores"], dtype=np.float64)
    copy_scores = np.asarray(d["copy_scores"], dtype=np.float64)
    n_layers = int(d.get("n_layers", argmax_scores.shape[0]))
    n_heads = int(d.get("n_heads_per_layer", argmax_scores.shape[1]))
    # NOTE: the artifact's top-level "n_heads" is the COUNT of detected heads,
    # not heads-per-layer; use the score-matrix width for the layout.
    n_heads = int(argmax_scores.shape[1])
    return Detection(
        model_key=panel_key,
        seed=int(d.get("seed", seed)),
        path=str(p),
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=int(d["n_kv_heads"]),
        argmax_heads=_heads("argmax_heads"),
        copy_heads=_heads("copy_heads"),
        argmax_scores=argmax_scores,
        copy_scores=copy_scores,
    )
