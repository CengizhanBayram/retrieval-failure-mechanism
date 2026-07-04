"""
Four-bucket mechanistic classifier (task §7 rules).

Buckets: ``m2_capture`` (distractor capture), ``m1_silence`` (head goes quiet),
``correct_attend`` (attends the needle correctly -> failure is downstream), and
``residual`` (residual attenuation; the terminal catch-all). Assignment follows
the researcher's ``mode_precedence`` (default M2 -> M1 -> correct_attend ->
residual) and is a TOTAL partition: every point lands in exactly one bucket
(§10).

The predicates' SEMANTICS are fixed here and documented in README; every NUMBER
comes from ``preregistration.yaml`` ``signature_rules`` (§1.2) — this module
defaults nothing. A missing threshold raises ``PreregError`` naming it. No
interpretation of the resulting labels happens anywhere (§12).

Expected ``signature_rules`` sub-schema (researcher-authored):

    reference_distribution: {source: success_samples, stats: [p5, median]}
    m1_silence:      {reference: p5|median, factor: <float>, absolute_floor: <float|null>}
    m2_capture:      {min_distractor_mass: <float>, distractor_over_needle_margin: <float>}
    correct_attend:  {reference: p5|median, factor: <float>}
    residual:        {}                      # terminal catch-all
    mode_precedence: [m2_capture, m1_silence, correct_attend, residual]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from failure_mech.prereg import PreregError

M1 = "m1_silence"
M2 = "m2_capture"
CORRECT_ATTEND = "correct_attend"
RESIDUAL = "residual"
BUCKETS = (M2, M1, CORRECT_ATTEND, RESIDUAL)


@dataclass(frozen=True)
class MassPoint:
    needle_mass: float
    distractor_mass: float


@dataclass(frozen=True)
class Reference:
    p5: float
    median: float

    def stat(self, name: str) -> float:
        if name == "p5":
            return self.p5
        if name == "median":
            return self.median
        raise PreregError(f"reference stat '{name}' not available (use p5|median).")


def build_reference(success_needle_masses: list[float]) -> Reference:
    """Reference distribution per (head, cell): needle_mass over the cell's
    SUCCESS samples; store p5 and median (§6)."""
    arr = np.asarray(success_needle_masses, dtype=np.float64)
    if arr.size == 0:
        return Reference(p5=float("nan"), median=float("nan"))
    return Reference(p5=float(np.percentile(arr, 5)), median=float(np.percentile(arr, 50)))


def _req(rule: dict, key: str, ctx: str):
    if key not in rule or rule[key] is None:
        raise PreregError(f"signature_rules.{ctx}.{key} missing (§1.2).")
    return rule[key]


def _pred_m1(mass: MassPoint, ref: Reference, rule: dict) -> bool:
    """Silence: needle_mass at/below a floor. Floor = ref[reference]*factor, and
    additionally the absolute_floor if provided (silence if below EITHER is
    configured — the researcher sets whichever they pre-registered)."""
    reference = _req(rule, "reference", "m1_silence")
    factor = float(_req(rule, "factor", "m1_silence"))
    floor = ref.stat(reference) * factor
    thresholds = [floor]
    if rule.get("absolute_floor") is not None:
        thresholds.append(float(rule["absolute_floor"]))
    return mass.needle_mass <= max(t for t in thresholds if not np.isnan(t)) \
        if any(not np.isnan(t) for t in thresholds) else False


def _pred_m2(mass: MassPoint, ref: Reference, rule: dict) -> bool:
    """Capture: a distractor holds enough mass AND out-masses the needle by the
    pre-registered margin."""
    min_dist = float(_req(rule, "min_distractor_mass", "m2_capture"))
    margin = float(_req(rule, "distractor_over_needle_margin", "m2_capture"))
    return (mass.distractor_mass >= min_dist) and \
           (mass.distractor_mass - mass.needle_mass >= margin)


def _pred_correct_attend(mass: MassPoint, ref: Reference, rule: dict) -> bool:
    """Correct-attend: needle_mass is at/above the reference level (the head
    still points at the needle; the failure is downstream)."""
    reference = _req(rule, "reference", "correct_attend")
    factor = float(_req(rule, "factor", "correct_attend"))
    return mass.needle_mass >= ref.stat(reference) * factor


_PREDICATES = {M1: _pred_m1, M2: _pred_m2, CORRECT_ATTEND: _pred_correct_attend}


def classify_point(mass: MassPoint, ref: Reference, signature_rules: dict) -> str:
    """Assign one bucket under ``mode_precedence``. Total partition: the LAST
    bucket in precedence is the terminal catch-all (residual), so exactly one
    bucket always fires (§10)."""
    precedence = _req(signature_rules, "mode_precedence", "")
    if not isinstance(precedence, list) or not precedence:
        raise PreregError("signature_rules.mode_precedence must be a non-empty list.")
    for i, bucket in enumerate(precedence):
        is_last = (i == len(precedence) - 1)
        if is_last:
            return bucket  # terminal catch-all guarantees totality
        pred = _PREDICATES.get(bucket)
        if pred is None:
            raise PreregError(
                f"mode_precedence names '{bucket}', which has no predicate "
                f"(known: {sorted(_PREDICATES)} + terminal catch-all)."
            )
        rule = signature_rules.get(bucket)
        if rule is None:
            raise PreregError(f"signature_rules.{bucket} missing (§1.2).")
        if pred(mass, ref, rule):
            return bucket
    return precedence[-1]  # unreachable; precedence non-empty


def sample_level_mass(head_masses: list[MassPoint]) -> MassPoint:
    """Sample-level point: mean needle/distractor mass over the top-k heads (§6)."""
    if not head_masses:
        return MassPoint(float("nan"), float("nan"))
    nm = float(np.mean([m.needle_mass for m in head_masses]))
    dm = float(np.mean([m.distractor_mass for m in head_masses]))
    return MassPoint(nm, dm)
