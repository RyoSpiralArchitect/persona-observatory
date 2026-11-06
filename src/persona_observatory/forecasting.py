"""External fitting and blocked selection of gate-conditioning predictors.

This module implements the proposed experiment-10 predictor family: direct
judgment probability D is shrunk toward a pooled rate within a grouping
(global, evidence stratum, instruction setting, or their joint cells), while
the four-category judgment composition stays pooled for every predictor.
It does not establish question novelty, development provenance, or chronology;
those are recorded by the surrounding protocol and locks.

Fitting counts are computed per observer branch. The shrinkage strength rho is
selected once per predictor, shared across branches and all cells, by
question-blocked development cross-validation.
"""
from __future__ import annotations

import math
from typing import Any

from .evaluation import CATEGORY_ORDER
from .storage import require

RHO_CANDIDATES = (0.25, 1.0, 4.0, math.inf)
RHO_TIE_TOLERANCE = 1e-12
JUDGMENT_ORDER = CATEGORY_ORDER[:4]
PREDICTORS = ("global", "evidence", "setting", "joint")

_DEV_FIELDS = {"question_id", "evidence_stratum", "setting_id", "direct_event", "category"}


def _group_key(predictor: str, row: dict[str, Any]) -> Any:
    if predictor == "global":
        return "all"
    if predictor == "evidence":
        return row["evidence_stratum"]
    if predictor == "setting":
        return row["setting_id"]
    return (row["setting_id"], row["evidence_stratum"])


def validate_dev_rows(rows: list[dict[str, Any]]) -> None:
    require(isinstance(rows, list) and rows, "Development rows must be a nonempty list")
    for row in rows:
        require(isinstance(row, dict) and _DEV_FIELDS <= set(row), "Missing development fields")
        require(type(row["direct_event"]) is bool, "direct_event must be Boolean")
        require(row["category"] in CATEGORY_ORDER, "Unknown five-category code")
        require(row["direct_event"] == (row["category"] != "no_direct"),
                "direct_event and category disagree")


def base_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled gate rate and judgment composition with the fixed 0.5 smoothing."""
    validate_dev_rows(rows)
    n = len(rows)
    n_d = sum(1 for row in rows if row["direct_event"])
    g_d = (n_d + 0.5) / (n + 1)
    h = {k: (sum(1 for row in rows if row["category"] == k) + 0.5) / (n_d + 2)
         for k in JUDGMENT_ORDER}
    return {"n": n, "n_d": n_d, "g_d": g_d, "judgment_composition": h}


def cell_rate(direct_count: int, cell_count: int, g_d: float, rho: float) -> float:
    """Shrunk direct-judgment rate for one cell; empty cells and rho=inf pool fully."""
    if cell_count == 0 or math.isinf(rho):
        return g_d
    return (direct_count + rho * cell_count * g_d) / (cell_count * (1 + rho))


def fit_predictor(rows: list[dict[str, Any]], predictor: str, rho: float) -> dict[str, Any]:
    require(predictor in PREDICTORS, "Unknown predictor: " + str(predictor))
    require(rho > 0 or math.isinf(rho), "rho must be positive or inf")
    base = base_rates(rows)
    counts: dict[Any, list[int]] = {}
    for row in rows:
        cell = counts.setdefault(_group_key(predictor, row), [0, 0])
        cell[0] += int(row["direct_event"])
        cell[1] += 1
    if predictor == "global":
        # The declared baseline is exactly the smoothed pooled rate; rho is inert.
        rates = {"all": base["g_d"]}
    else:
        rates = {key: cell_rate(d, n, base["g_d"], rho) for key, (d, n) in counts.items()}
    return {"predictor": predictor, "rho": rho, "base": base, "cell_rates": rates}


def gate_probability(fitted: dict[str, Any], row: dict[str, Any]) -> float:
    key = _group_key(fitted["predictor"], row)
    return fitted["cell_rates"].get(key, fitted["base"]["g_d"])


def category_probabilities(fitted: dict[str, Any], row: dict[str, Any]) -> list[float]:
    """Five-category forecast in CATEGORY_ORDER; composition is pooled for every predictor."""
    r = gate_probability(fitted, row)
    h = fitted["base"]["judgment_composition"]
    return [r * h[k] for k in JUDGMENT_ORDER] + [1.0 - r]


def _question_brier(rows: list[dict[str, Any]], fitted_by_observer: dict[str, dict[str, Any]]) -> float:
    """Mean D Brier over a question's rows; each branch is scored by its own fit."""
    total = 0.0
    for row in rows:
        fitted = fitted_by_observer[row["observer"]]
        total += (gate_probability(fitted, row) - int(row["direct_event"])) ** 2
    return total / len(rows)


def cv_score(rows: list[dict[str, Any]], fold_of: dict[str, Any], predictor: str, rho: float) -> float:
    """Question-blocked CV with branch-specific fits and equal question weights.

    ``fold_of`` maps every question_id to its fold. Each observer branch is
    fitted on its own training rows — counts are never pooled across branches —
    and scores only its own held-out labels. Both branches stay in the same
    fold and enter the question mean with equal weight, so no branch is
    upweighted by generation count.
    """
    folds: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        require(row["question_id"] in fold_of, "No fold assigned for " + str(row["question_id"]))
        folds.setdefault(fold_of[row["question_id"]], []).append(row)
    require(len(folds) >= 2, "Cross-validation needs at least two folds")
    question_means = []
    for held_out in folds.values():
        held_ids = {row["question_id"] for row in held_out}
        train = [row for rows_in_fold in folds.values() for row in rows_in_fold
                 if row["question_id"] not in held_ids]
        branches = {row["observer"] for row in held_out}
        fitted = {obs: fit_predictor([r for r in train if r["observer"] == obs], predictor, rho)
                  for obs in branches}
        by_question: dict[str, list[dict[str, Any]]] = {}
        for row in held_out:
            by_question.setdefault(row["question_id"], []).append(row)
        question_means.extend(_question_brier(q_rows, fitted) for q_rows in by_question.values())
    return math.fsum(question_means) / len(question_means)


def select_rho(rows: list[dict[str, Any]], fold_of: dict[str, Any], predictor: str) -> dict[str, Any]:
    """One rho per predictor, shared across cells and observer branches.

    The caller passes development rows of all branches; each branch's labels
    contribute separate rows but identical question folds. Ties within 1e-12
    choose the stronger shrinkage in the fixed order inf, 4, 1, 0.25.
    """
    scores = {rho: cv_score(rows, fold_of, predictor, rho) for rho in RHO_CANDIDATES}
    best = min(scores.values())
    chosen = next(rho for rho in (math.inf, 4.0, 1.0, 0.25)
                  if scores[rho] <= best + RHO_TIE_TOLERANCE)
    return {"predictor": predictor, "rho": chosen, "cv_gate_brier": scores}


def build_forecasts(fitted_by_branch: dict[str, dict[str, dict[str, Any]]],
                    samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Emit the complete sample x observer x predictor forecast grid.

    ``fitted_by_branch`` maps observer -> predictor -> fitted model. ``samples``
    rows need sample_id, evidence_stratum and setting_id.
    """
    require(isinstance(fitted_by_branch, dict) and fitted_by_branch, "No fitted branches")
    for observer, fitted in fitted_by_branch.items():
        require(set(fitted) == set(PREDICTORS), "Every branch needs all four predictors: " + str(observer))
    rows = []
    for sample in samples:
        sid = sample["sample_id"]
        for observer, fitted in fitted_by_branch.items():
            for predictor in PREDICTORS:
                rows.append({
                    "sample_id": sid,
                    "observer": observer,
                    "predictor": predictor,
                    "category_probabilities": category_probabilities(fitted[predictor], sample),
                })
    return rows
