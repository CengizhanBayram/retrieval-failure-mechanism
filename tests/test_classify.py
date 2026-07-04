"""Four-bucket classifier tests (task §10): every synthetic failure case lands
in exactly one bucket under precedence; totality via the terminal catch-all."""

from __future__ import annotations

import pytest

from failure_mech import classify as CL
from failure_mech.prereg import PreregError


@pytest.fixture
def rules(valid_prereg_dict):
    return valid_prereg_dict["signature_rules"]


@pytest.fixture
def ref():
    # success needle-mass distribution: p5 ~ 0.10, median ~ 0.50
    return CL.build_reference([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9])


def test_m2_capture_wins(rules, ref):
    # big distractor mass beats needle by the margin -> M2 (first in precedence)
    pt = CL.MassPoint(needle_mass=0.05, distractor_mass=0.6)
    assert CL.classify_point(pt, ref, rules) == CL.M2


def test_m1_silence(rules, ref):
    # needle below floor (p5*factor and absolute_floor), no big distractor -> M1
    pt = CL.MassPoint(needle_mass=0.01, distractor_mass=0.0)
    assert CL.classify_point(pt, ref, rules) == CL.M1


def test_correct_attend(rules, ref):
    # needle at/above median, no capture -> correct_attend
    pt = CL.MassPoint(needle_mass=0.7, distractor_mass=0.0)
    assert CL.classify_point(pt, ref, rules) == CL.CORRECT_ATTEND


def test_residual_catch_all(rules, ref):
    # needle between floor and median, no capture -> residual (terminal)
    pt = CL.MassPoint(needle_mass=0.2, distractor_mass=0.05)
    assert CL.classify_point(pt, ref, rules) == CL.RESIDUAL


def test_partition_total_and_unique(rules, ref):
    grid = [CL.MassPoint(n, d) for n in (0.0, 0.01, 0.2, 0.5, 0.9)
            for d in (0.0, 0.05, 0.3, 0.6, 0.95)]
    for pt in grid:
        bucket = CL.classify_point(pt, ref, rules)
        assert bucket in CL.BUCKETS  # exactly one label, always defined


def test_missing_threshold_raises(ref, valid_prereg_dict):
    rules = valid_prereg_dict["signature_rules"]
    del rules["m2_capture"]["min_distractor_mass"]
    pt = CL.MassPoint(0.05, 0.6)
    with pytest.raises(PreregError) as exc:
        CL.classify_point(pt, ref, rules)
    assert "m2_capture.min_distractor_mass" in str(exc.value)


def test_precedence_order_respected(ref, valid_prereg_dict):
    rules = valid_prereg_dict["signature_rules"]
    # a point that satisfies BOTH m2 and (would-be) silence: m2 precedes
    pt = CL.MassPoint(needle_mass=0.0, distractor_mass=0.6)
    assert CL.classify_point(pt, ref, rules) == CL.M2
    # flip precedence so silence is first
    rules["mode_precedence"] = ["m1_silence", "m2_capture", "correct_attend", "residual"]
    assert CL.classify_point(pt, ref, rules) == CL.M1
