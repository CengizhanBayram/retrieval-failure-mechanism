"""
Probe construction (task §4.1). All text is procedurally generated from
``configs/grid.yaml`` — no external corpora (§12).

Anatomy: a constant preamble, filler padding, one NEEDLE sentence carrying the
key->value fact, ``n_distractors`` DISTRACTOR sentences (shell_same or
shell_diff), and a QUESTION suffix asking for the needle's value.

TOKEN-SKELETON ALIGNMENT (critical for E3). Within a cell every probe shares an
IDENTICAL token skeleton; only the slot CONTENTS differ:
  * Filler is a deterministic function of the CELL (identical across the cell's
    samples), so it never shifts positions.
  * ADJ/NOUN come from a per-tokenizer vocab pre-filtered to single-token words
    (with a leading space), so each key slot is a fixed token count.
  * VALUE is rejection-sampled so its in-context token span equals the cell's
    canonical length and does not merge with the following delimiter.
Every ``build`` re-locates the spans and HARD-FAILS if the token length or any
span index differs from the cell's canonical skeleton (§4.1).

A ``Probe`` is a pure function of ``(cell, sample_idx, seed)`` and the
tokenizer: same inputs -> same bytes (§1.7).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from failure_mech.spans import Span, SpanError, locate_value_span


class ProbeError(RuntimeError):
    """Raised on any unrecoverable probe-construction problem (§1.9)."""


# ---------------------------------------------------------------------------
# Cell + Probe data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellSpec:
    """One E1 grid cell (§5). ``similarity`` is 'none' for the 0-distractor
    baseline rows."""
    model_key: str
    context_length: int
    needle_position: float
    n_distractors: int
    similarity: str  # "shell_same" | "shell_diff" | "none"

    def cell_hash(self) -> str:
        payload = (
            f"{self.model_key}|ctx={self.context_length}|pos={self.needle_position}"
            f"|nd={self.n_distractors}|sim={self.similarity}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def axes(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "context_length": self.context_length,
            "needle_position": self.needle_position,
            "n_distractors": self.n_distractors,
            "similarity": self.similarity,
        }


@dataclass
class Probe:
    text: str
    input_ids: list[int]
    needle_value: str
    needle_span: Span
    distractor_spans: list[Span]
    key_strings: dict[str, str]           # {"adj": ..., "noun": ...} of the needle
    distractor_values: list[str]
    answer_prompt_len: int                # number of prompt tokens (answer step index)
    cell_hash: str
    sample_idx: int
    seed: int

    def skeleton(self) -> tuple:
        """Positional fingerprint used for the cross-sample alignment assertion."""
        return (
            self.answer_prompt_len,
            self.needle_span.as_tuple(),
            tuple(s.as_tuple() for s in self.distractor_spans),
        )


# ---------------------------------------------------------------------------
# Probe factory
# ---------------------------------------------------------------------------

class ProbeFactory:
    """Builds probes for ONE tokenizer, caching per-tokenizer vocab, the value
    calibration, and per-cell layouts + canonical skeletons."""

    def __init__(self, tokenizer, grid_cfg: dict, model_key: str) -> None:
        self.tok = tokenizer
        self.model_key = model_key
        pcfg = grid_cfg["probe"]
        self.preamble: str = pcfg["preamble"]
        self.needle_template: str = pcfg["needle_template"]
        self.distractor_templates: dict = pcfg["distractor_templates"]
        self.question_template: str = pcfg["question_template"]
        self.value_alphabet: str = pcfg["value"]["alphabet"]
        self.value_char_len: int = int(pcfg["value"]["char_len"])
        self.max_value_tries: int = int(pcfg["value"]["max_value_tries"])
        self.min_vocab: int = int(pcfg["min_vocab_after_filter"])
        self.filler_templates: list[str] = list(pcfg["filler_templates"])
        self.topic_words: list[str] = list(pcfg["topic_words"])
        self.use_chat_template: bool = bool(pcfg.get("use_chat_template", False))
        self.chat_template_date: str = pcfg.get("chat_template_date", "01 Jan 2025")
        if self.use_chat_template and getattr(self.tok, "chat_template", None) is None:
            raise ProbeError(
                f"use_chat_template is true but tokenizer for '{model_key}' has no "
                "chat_template. Set use_chat_template: false for base models.")

        if not getattr(self.tok, "is_fast", False):
            raise ProbeError(
                f"Probe construction needs a fast tokenizer (offset_mapping) for "
                f"model '{model_key}'; got a slow tokenizer."
            )

        self.adjectives = self._filter_single_token(pcfg["adjectives"])
        self.nouns = self._filter_single_token(pcfg["nouns"])
        for name, pool in (("adjectives", self.adjectives), ("nouns", self.nouns)):
            if len(pool) < self.min_vocab:
                raise ProbeError(
                    f"After single-token filtering, only {len(pool)} {name} survive "
                    f"for '{model_key}' (< min_vocab_after_filter={self.min_vocab}). "
                    "Add more vocab to grid.yaml."
                )
        self._canonical_value_len = self._calibrate_value_len()
        self._layouts: dict[str, _Layout] = {}
        self._skeletons: dict[str, tuple] = {}

    # -- vocab / value calibration -----------------------------------------

    def _n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _filter_single_token(self, words: list[str]) -> list[str]:
        """Keep words that tokenise to exactly one token WITH a leading space,
        preserving config order (determinism)."""
        keep = []
        for w in words:
            if self._n_tokens(" " + w) == 1:
                keep.append(w)
        return keep

    def _value_span_len(self, value: str) -> int | None:
        """In-context token length of ``value`` under the needle template's local
        neighbourhood, or ``None`` if it merges with the trailing delimiter."""
        local = f"is {value}."
        enc = self.tok(local, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        char_start = local.index(value)
        try:
            span = locate_value_span(
                self.tok, local, enc["input_ids"], offsets, value, char_start,
                require_clean_boundary=True,
            )
        except SpanError:
            return None
        return len(span)

    def _calibrate_value_len(self) -> int:
        """Modal in-context value token length over deterministic candidates."""
        rng = random.Random(f"calib|{self.model_key}")
        counts: dict[int, int] = {}
        for _ in range(256):
            v = "".join(rng.choice(self.value_alphabet) for _ in range(self.value_char_len))
            n = self._value_span_len(v)
            if n is not None:
                counts[n] = counts.get(n, 0) + 1
        if not counts:
            raise ProbeError(
                f"Could not calibrate a canonical VALUE token length for "
                f"'{self.model_key}' (every candidate merged with the delimiter)."
            )
        # Most frequent length; tie-break to the smaller length (determinism).
        return min(counts, key=lambda k: (-counts[k], k))

    def _sample_value(self, rng: random.Random) -> str:
        for _ in range(self.max_value_tries):
            v = "".join(rng.choice(self.value_alphabet) for _ in range(self.value_char_len))
            if self._value_span_len(v) == self._canonical_value_len:
                return v
        raise ProbeError(
            f"Exhausted {self.max_value_tries} tries sampling a VALUE of canonical "
            f"token length {self._canonical_value_len} for '{self.model_key}'."
        )

    # -- cell layout (filler, fixed per cell) ------------------------------

    def _make_filler(self, rng: random.Random) -> str:
        tmpl = rng.choice(self.filler_templates)
        return tmpl.format(TOPIC=rng.choice(self.topic_words))

    def _reference_slot_text(self, cell: CellSpec, rng: random.Random) -> dict:
        """Reference slot strings (canonical lengths) used only to SIZE the cell
        layout; contents are irrelevant to positions because every slot is
        length-matched."""
        adj = self.adjectives[0]
        noun = self.nouns[0]
        val = self._sample_value(rng)
        distractor = (self._render_distractor(cell.similarity, adj, noun, val)
                      if cell.n_distractors > 0 else "")
        return {"needle": self._render_needle(adj, noun, val), "distractor": distractor}

    def _cell_layout(self, cell: CellSpec) -> "_Layout":
        h = cell.cell_hash()
        if h in self._layouts:
            return self._layouts[h]
        rng = random.Random(f"layout|{h}")
        ref = self._reference_slot_text(cell, random.Random(f"refval|{h}"))
        needle_ref = ref["needle"]
        distractor_ref = ref["distractor"]

        # Grow filler sentence count until the assembled prompt reaches the
        # target context. Estimate first, then adjust (few re-tokenisations).
        target = cell.context_length
        fillers: list[str] = []

        def assemble(n_filler: int) -> tuple[str, list[str]]:
            local_rng = random.Random(f"filler|{h}")
            fl = [self._make_filler(local_rng) for _ in range(n_filler)]
            body = self._assemble_body(cell, fl, needle_ref, [distractor_ref] * cell.n_distractors)
            return body, fl

        def prompt_len(body: str) -> int:
            # measure the ACTUAL prompt the model sees (chat wrapper included)
            text, add_special = self._wrap(self.preamble + body + self._render_question("x", "y"))
            return len(self.tok(text, add_special_tokens=add_special)["input_ids"])

        # crude estimate
        sample_len = max(1, self._n_tokens(self._make_filler(random.Random(f"est|{h}"))) + 1)
        n = max(1, target // sample_len)
        body, fillers = assemble(n)
        total = prompt_len(body)
        # Grow / shrink to reach target (prompt should be >= target, minimal over).
        guard = 0
        while total < target and guard < 100000:
            n += max(1, (target - total) // sample_len)
            body, fillers = assemble(n)
            total = prompt_len(body)
            guard += 1
        while n > 0:
            body_try, fillers_try = assemble(n - 1)
            t = prompt_len(body_try)
            if t < target:
                break
            n, body, fillers, total = n - 1, body_try, fillers_try, t

        layout = _Layout(cell=cell, n_filler=n, filler=fillers)
        self._layouts[h] = layout
        return layout

    # -- text assembly ------------------------------------------------------

    def _render_needle(self, adj: str, noun: str, value: str) -> str:
        return self.needle_template.format(ADJ=adj, NOUN=noun, VALUE=value)

    def _render_distractor(self, similarity: str, adj: str, noun: str, value: str) -> str:
        tmpl = self.distractor_templates[similarity]
        return tmpl.format(ADJ=adj, NOUN=noun, VALUE=value)

    def _render_question(self, adj: str, noun: str) -> str:
        return self.question_template.format(ADJ=adj, NOUN=noun)

    def _wrap(self, user_msg: str) -> tuple[str, bool]:
        """Return (prompt_text, add_special_tokens).

        With the chat template on, the template already carries the model's
        special tokens (so ``add_special_tokens=False`` when tokenising) and a
        generation prompt is appended so the model answers. A FIXED date is
        injected for templates that stamp the current date (Llama-3), keeping the
        skeleton deterministic (§1.7); templates that don't accept ``date_string``
        fall through unchanged.
        """
        if not self.use_chat_template:
            return user_msg, True
        messages = [{"role": "user", "content": user_msg}]
        try:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                date_string=self.chat_template_date)
        except TypeError:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return text, False

    def _placements(self, cell: CellSpec, n_filler: int) -> tuple[int, list[int]]:
        """Gap indices (0..n_filler) at which the needle and each distractor are
        inserted between filler sentences. Deterministic; spread out; distractors
        never collide with the needle gap or each other."""
        needle_gap = int(round(cell.needle_position * n_filler))
        needle_gap = min(max(needle_gap, 0), n_filler)
        gaps: list[int] = []
        if cell.n_distractors > 0:
            for j in range(cell.n_distractors):
                # even spread across the full span
                g = int(round((j + 1) * n_filler / (cell.n_distractors + 1)))
                g = min(max(g, 0), n_filler)
                while g == needle_gap or g in gaps:
                    g = (g + 1) % (n_filler + 1)
                gaps.append(g)
        return needle_gap, gaps

    def _assemble_body(
        self, cell: CellSpec, fillers: list[str],
        needle_text: str, distractor_texts: list[str],
    ) -> str:
        """Interleave filler sentences with the needle + distractors at their
        gaps. Sentences are joined by single spaces."""
        n_filler = len(fillers)
        needle_gap, dist_gaps = self._placements(cell, n_filler)
        # gap -> list of inserted sentences (needle first if colliding, but gaps
        # are unique by construction).
        inserts: dict[int, list[str]] = {}
        inserts.setdefault(needle_gap, []).append(needle_text)
        for g, dt in zip(dist_gaps, distractor_texts):
            inserts.setdefault(g, []).append(dt)
        pieces: list[str] = []
        for gap in range(n_filler + 1):
            for ins in inserts.get(gap, []):
                pieces.append(ins)
            if gap < n_filler:
                pieces.append(fillers[gap])
        return " ".join(pieces)

    # -- public API ---------------------------------------------------------

    def build(self, cell: CellSpec, sample_idx: int, seed: int) -> Probe:
        layout = self._cell_layout(cell)
        rng = random.Random(f"slots|{cell.cell_hash()}|{sample_idx}|{seed}")

        # Draw distinct keys for needle + distractors (without replacement).
        n_keys = 1 + cell.n_distractors
        adjs = rng.sample(self.adjectives, n_keys)
        nouns = rng.sample(self.nouns, n_keys)
        needle_adj, needle_noun = adjs[0], nouns[0]
        needle_value = self._sample_value(rng)

        distractor_values: list[str] = []
        distractor_texts: list[str] = []
        for j in range(cell.n_distractors):
            dv = self._sample_value(rng)
            distractor_values.append(dv)
            distractor_texts.append(
                self._render_distractor(cell.similarity, adjs[j + 1], nouns[j + 1], dv)
            )

        needle_text = self._render_needle(needle_adj, needle_noun, needle_value)
        question = self._render_question(needle_adj, needle_noun)
        body = self._assemble_body(cell, layout.filler, needle_text, distractor_texts)
        user_msg = self.preamble + body + question

        text, add_special = self._wrap(user_msg)
        enc = self.tok(text, add_special_tokens=add_special, return_offsets_mapping=True)
        input_ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]

        needle_span = self._locate(text, input_ids, offsets, needle_value)
        distractor_spans = [
            self._locate(text, input_ids, offsets, dv) for dv in distractor_values
        ]

        probe = Probe(
            text=text,
            input_ids=input_ids,
            needle_value=needle_value,
            needle_span=needle_span,
            distractor_spans=distractor_spans,
            key_strings={"adj": needle_adj, "noun": needle_noun},
            distractor_values=distractor_values,
            answer_prompt_len=len(input_ids),
            cell_hash=cell.cell_hash(),
            sample_idx=sample_idx,
            seed=seed,
        )
        self._assert_skeleton(cell, probe)
        return probe

    def _locate(self, text: str, input_ids, offsets, value: str) -> Span:
        char_start = text.find(value)
        if char_start < 0:
            raise ProbeError(f"VALUE {value!r} not present in assembled probe text.")
        return locate_value_span(self.tok, text, input_ids, offsets, value, char_start)

    def _assert_skeleton(self, cell: CellSpec, probe: Probe) -> None:
        """Every sample of a cell must share the token length + span indices
        (skeleton alignment, §4.1). First build records the skeleton; later ones
        must match exactly."""
        h = cell.cell_hash()
        sk = probe.skeleton()
        if h not in self._skeletons:
            self._skeletons[h] = sk
            return
        if self._skeletons[h] != sk:
            raise ProbeError(
                f"Skeleton mismatch in cell {h}: sample {probe.sample_idx} has "
                f"{sk} but the cell's canonical skeleton is {self._skeletons[h]}. "
                "Token-skeleton alignment (§4.1) violated."
            )


@dataclass
class _Layout:
    cell: CellSpec
    n_filler: int
    filler: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Convenience wrapper matching the §4.1 signature
# ---------------------------------------------------------------------------

def build_probe(tokenizer, grid_cfg: dict, model_key: str, cell: CellSpec,
                sample_idx: int, seed: int) -> Probe:
    """One-shot probe build (constructs a fresh factory — for interactive use;
    scripts reuse a single :class:`ProbeFactory` per model for the cache)."""
    return ProbeFactory(tokenizer, grid_cfg, model_key).build(cell, sample_idx, seed)
