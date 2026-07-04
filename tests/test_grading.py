"""Grading tests (task §10): value normalization + distractor-hit precedence."""

from __future__ import annotations

from failure_mech import grading as G


def test_normalize_strips_and_casefolds():
    # normalize keeps [0-9a-z]; letters in words are kept (value still substring-
    # matches downstream), punctuation/space removed, first line only.
    assert G.normalize("  Code: AB2-9XQ. ") == "codeab29xq"
    assert G.normalize("AB2\nsecond line ignored") == "ab2"
    assert G.normalize("") == ""
    assert G.normalize("!!! ???") == ""


def test_correct_when_needle_only():
    g = G.grade_generation("The code is K7QW2Z", "K7QW2Z", ["ZZ9AAB"])
    assert g.grade == G.CORRECT and g.correct is True


def test_distractor_hit_records_index():
    g = G.grade_generation("X9ABCD", "K7QW2Z", ["QQ0", "X9ABCD"])
    assert g.grade == G.DISTRACTOR_HIT and g.correct is False
    assert g.distractors_hit == [1]


def test_distractor_precedence_over_needle():
    # both needle and a distractor present -> distractor_hit (not correct)
    g = G.grade_generation("K7QW2Z and also X9ABCD", "K7QW2Z", ["X9ABCD"])
    assert g.grade == G.DISTRACTOR_HIT and g.correct is False


def test_other_wrong_when_neither():
    g = G.grade_generation("banana", "K7QW2Z", ["X9ABCD"])
    assert g.grade == G.OTHER_WRONG and g.correct is False


def test_empty_when_no_alnum():
    g = G.grade_generation("   \n  ", "K7QW2Z", ["X9ABCD"])
    assert g.grade == G.EMPTY and g.correct is False


def test_partition_is_total_and_exclusive():
    cases = ["K7QW2Z", "X9ABCD", "banana", "", "K7QW2Z X9ABCD"]
    for c in cases:
        g = G.grade_generation(c, "K7QW2Z", ["X9ABCD"])
        assert g.grade in G.GRADES
