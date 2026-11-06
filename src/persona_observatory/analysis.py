"""Frozen experiment-10 analysis: paired contrasts over locked scores.

Operates on the per_row records produced by evaluation.score_rows. All weighting
is question-blocked: every held-out question contributes one mean D-Brier per
observer branch, branches are then averaged equally. No significance claim;
the primary decision is a sign rule on a finite authored question set, and the
progression rule is the documented practical threshold.
"""
from __future__ import annotations

import math
from typing import Any

from .storage import require

TIE_TOLERANCE = 1e-12
PROGRESSION_DELTA = -0.005
CONTRASTS = (
    ("joint", "evidence"),   # primary
    ("setting", "global"),   # prespecified auxiliary
    ("joint", "setting"),    # prespecified auxiliary
    ("joint", "global"),     # calibration to pooled baseline
    ("evidence", "global"),  # calibration to pooled baseline
)


def _question_means(per_row: list[dict[str, Any]], questions: dict[str, dict[str, str]],
                    metric: str) -> dict[tuple, float]:
    """Mean metric per (observer, question, predictor); equal weight per generation row."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in per_row:
        sid = row["sample_id"]
        require(sid in questions, "Score row without question metadata: " + sid)
        groups.setdefault((row["observer"], questions[sid]["question_id"], row["predictor"]), []).append(row)
    return {key: math.fsum(r[metric] for r in rows) / len(rows)
            for key, rows in groups.items()}


def paired_delta(per_row: list[dict[str, Any]], questions: dict[str, dict[str, str]],
                 left: str, right: str, metric: str = "gate_brier") -> dict[str, Any]:
    """Question-blocked paired gate-Brier difference left - right, negative is better.

    ``questions`` maps sample_id -> {question_id, ...}; both sides share the
    same generations, so the pairing is exact. Missing (observer, question)
    cells are reported, never silently dropped: the primary judgment is
    INCOMPLETE unless coverage is full.
    """
    means = _question_means(per_row, questions, metric)
    by_observer: dict[str, dict[str, tuple[float, float]]] = {}
    for (observer, qid, predictor), value in means.items():
        if predictor in (left, right):
            entry = by_observer.setdefault(observer, {}).setdefault(qid, {})
            entry[predictor] = value
    per_observer = {}
    incomplete = []
    for observer, qs in by_observer.items():
        deltas = []
        for qid, pair in qs.items():
            if left not in pair or right not in pair:
                incomplete.append((observer, qid))
                continue
            deltas.append((qid, pair[left] - pair[right]))
        per_observer[observer] = {
            "questions": len(deltas),
            "delta": math.fsum(d for _, d in deltas) / len(deltas) if deltas else None,
            "per_question": {qid: d for qid, d in deltas},
        }
    both = [o["delta"] for o in per_observer.values() if o["delta"] is not None]
    return {
        "contrast": f"{left}-{right}",
        "metric": metric,
        "delta_ab_mean": math.fsum(both) / len(both) if both else None,
        "observers": per_observer,
        "incomplete_cells": len(incomplete),
    }


def primary_report(per_row: list[dict[str, Any]], questions: dict[str, dict[str, str]]) -> dict[str, Any]:
    contrasts = {f"{l}-{r}": paired_delta(per_row, questions, l, r) for l, r in CONTRASTS}
    primary = contrasts["joint-evidence"]
    if primary["incomplete_cells"] or primary["delta_ab_mean"] is None:
        decision = "INCOMPLETE"
    else:
        decision = "SUPPORT" if primary["delta_ab_mean"] < -TIE_TOLERANCE else "NOT_SUPPORTED"
    progression = {"met": False, "checks": {}}
    if decision == "SUPPORT":
        nll = paired_delta(per_row, questions, "joint", "evidence", metric="gate_nll")
        contrasts["joint-evidence-nll"] = nll
        branches_ok = all(o["delta"] is not None and o["delta"] < -TIE_TOLERANCE
                          for o in primary["observers"].values())
        progression["checks"] = {
            "delta_leq_-0.005": primary["delta_ab_mean"] <= PROGRESSION_DELTA,
            "both_branches_negative": branches_ok,
            "joint_minus_global_not_worse": contrasts["joint-global"]["delta_ab_mean"] <= TIE_TOLERANCE,
            "joint_minus_evidence_nll_not_worse": nll["delta_ab_mean"] <= TIE_TOLERANCE,
        }
        progression["met"] = all(progression["checks"].values())
    return {
        "schema_version": "pobs-exp10-analysis-v1",
        "primary": primary,
        "contrasts": contrasts,
        "primary_decision": decision,
        "progression": progression,
        "note": "Descriptive sign rule on a finite authored question set; not a significance test. Observer branches measure the same generations.",
    }
