"""
Answer grading + behavioral error taxonomy (task §4.5).

Measurement only. Given a generation and the probe's needle/distractor values,
assign one of four mutually-exclusive labels. The accuracy "correct flag" is
``grade == "correct"``. The 4-way label is stored per sample and cross-tabulated
against mechanistic buckets in E2 (§6).

Grade definitions (verbatim §4.5), applied in this precedence so the labels
partition every generation exactly once:

    empty          -> no alphanumeric content
    distractor_hit -> any distractor_value present (record which)
    correct        -> needle_value present AND no distractor_value present
    other_wrong    -> neither value present, non-empty

Note the precedence resolves the overlap: "correct" requires NO distractor
present, so a generation containing both the needle and a distractor value is a
distractor_hit, not correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Everything that is not a Unicode letter or digit is stripped before matching.
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

CORRECT = "correct"
DISTRACTOR_HIT = "distractor_hit"
OTHER_WRONG = "other_wrong"
EMPTY = "empty"

GRADES = (CORRECT, DISTRACTOR_HIT, OTHER_WRONG, EMPTY)


def normalize(ans: str) -> str:
    """Strip whitespace/punctuation and casefold, leaving only [0-9a-z].

    Only the text before the first newline is considered - greedy decoding stops
    on newline, and any trailing tokens after a newline are not part of the
    answer (§1.4).
    """
    if ans is None:
        return ""
    head = ans.split("\n", 1)[0]
    return _NON_ALNUM.sub("", head.casefold())


@dataclass
class Grade:
    grade: str
    correct: bool
    distractors_hit: list[int] = field(default_factory=list)  # distractor indices present
    normalized_answer: str = ""

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "correct": self.correct,
            "distractors_hit": list(self.distractors_hit),
            "normalized_answer": self.normalized_answer,
        }


def grade_generation(
    generation: str,
    needle_value: str,
    distractor_values: list[str],
) -> Grade:
    """Grade one generation against a probe's needle + distractor values (§4.5).

    ``distractors_hit`` records the INDICES (into ``distractor_values``) whose
    value appears in the normalized generation.
    """
    norm = normalize(generation)
    needle_norm = normalize(needle_value)

    if norm == "":
        return Grade(EMPTY, False, [], norm)

    hits = [
        i for i, dv in enumerate(distractor_values)
        if (dnorm := normalize(dv)) != "" and dnorm in norm
    ]
    if hits:
        return Grade(DISTRACTOR_HIT, False, hits, norm)

    if needle_norm != "" and needle_norm in norm:
        return Grade(CORRECT, True, [], norm)

    return Grade(OTHER_WRONG, False, [], norm)
