"""
Detection-artifact reader (task §0, §6, §9).

Reads the prior detector's detection artifact for a model - one JSON per (model, seed) at
``datas/results/profile/{panel_key}_seed{seed}.json`` - and exposes a canonical
top-k ranked retrieval-head list. this repo NEVER re-detects (§6); it consumes this.

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

    def boundary_tie(self, k: int, detector: str = "argmax") -> dict:
        """Diagnose whether the top-k CUT lands inside a block of tied scores.

        The detector score saturates at 1.0, so several heads can share the top
        score exactly. When that happens the top-k set is not selected by
        evidence - it is selected by the (layer, head) tie-break, i.e. by sort
        order. Any causal claim over such a set (in particular a NULL: "patching
        the top-k heads did nothing") is confounded with the arbitrariness of
        WHICH tied heads got patched.

        Observed in the pilot: gemma2_9b has 13 heads at exactly 1.000, so its
        top-10 set is 10 arbitrary members of a 13-way tie.

        Returns the score at the cut, how many heads tie it, and how many of
        those made it in - ``arbitrary`` is True when the cut splits a tie.
        """
        scores = self.argmax_scores if detector == "argmax" else self.copy_scores
        ranked = self._ranked(detector)
        if k > len(ranked) or k < 1:
            raise DetectionError(f"boundary_tie: k={k} outside 1..{len(ranked)}.")
        val = [float(scores[l, h]) for (l, h) in ranked]
        cut = val[k - 1]
        tied = [i for i, v in enumerate(val) if abs(v - cut) < 1e-9]
        n_in = sum(1 for i in tied if i < k)
        return {
            "k": k,
            "detector": detector,
            "cut_score": cut,
            "n_tied_at_cut": len(tied),
            "n_tied_inside_k": n_in,
            "n_tied_excluded": len(tied) - n_in,
            # True <=> the cut splits a tie: some heads with the SAME score are in
            # and some are out, decided only by sort order.
            "arbitrary": len(tied) > n_in,
        }

    def retrieval_head_set(self, detector: str = "argmax") -> set[tuple[int, int]]:
        heads = self.argmax_heads if detector == "argmax" else self.copy_heads
        return {(int(l), int(h)) for (l, h) in heads}

    def non_retrieval_heads(self, detector: str = "argmax") -> list[tuple[int, int]]:
        """All (layer, head) NOT in the retrieval set - the random-control pool
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
            f"Detection artifact not found: {p}. this repo consumes the prior detector "
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
