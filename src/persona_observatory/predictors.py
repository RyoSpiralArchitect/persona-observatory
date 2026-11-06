"""Development-only gate forecasts; no inference or automatic semantic labels."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

from .evaluation import CATEGORY_ORDER, validate_forecasts
from .observations import derived_outcome
from .storage import require

ALGORITHMS = ("Global", "Evidence", "Setting", "Joint")
RHO_GRID = (0.25, 1.0, 4.0, "inf")
TOLERANCE = 1e-12


def _key(row, algorithm):
    if algorithm == "Global": return ()
    if algorithm == "Evidence": return (row["evidence_stratum"],)
    if algorithm == "Setting": return (row["setting_id"],)
    if algorithm == "Joint": return (row["setting_id"], row["evidence_stratum"])
    raise ValueError("Unknown predictor")


def validate_development(rows, labels_by_observer):
    require(isinstance(rows, list) and bool(rows), "Development records required")
    ids = [r["sample_id"] for r in rows]
    require(len(ids) == len(set(ids)), "Duplicate development sample")
    questions = {}
    for r in rows:
        require(type(r["fold"]) is int and r["fold"] >= 0, "Invalid development fold")
        require(all(isinstance(r[k], str) and r[k] for k in ("question_id", "setting_id", "evidence_stratum")),
                "Invalid development design field")
        value = r["fold"], r["evidence_stratum"]
        require(r["question_id"] not in questions or questions[r["question_id"]] == value,
                "Question crosses folds or evidence strata")
        questions[r["question_id"]] = value
    require(len({r["fold"] for r in rows}) >= 2, "At least two development folds required")
    require(set(labels_by_observer) == {"observer_a", "observer_b"}, "Separate A/B labels required")
    outcomes = {}
    for observer, labels in labels_by_observer.items():
        require(len(labels) == len(ids) and {x["sample_id"] for x in labels} == set(ids), "Label coverage differs")
        outcomes[observer] = {x["sample_id"]: derived_outcome(x)["category"] for x in labels}
    return outcomes


def fit_profile(rows, outcomes, algorithm, rho):
    require(bool(rows) and algorithm in ALGORITHMS and type(rho) in (str, int, float) and rho in RHO_GRID,
            "Invalid fit specification")
    counts = Counter(outcomes[r["sample_id"]] for r in rows)
    require(set(counts) <= set(CATEGORY_ORDER), "Invalid semantic category")
    direct = len(rows) - counts["no_direct"]
    global_d = (direct + 0.5) / (len(rows) + 1)
    composition = [(counts[k] + 0.5) / (direct + 2) for k in CATEGORY_ORDER[:-1]]
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        c = cells[_key(r, algorithm)]
        c[0] += 1
        c[1] += outcomes[r["sample_id"]] != "no_direct"
    return {"algorithm": algorithm, "rho": rho, "n": len(rows), "n_direct": direct,
            "global_d": global_d, "composition": composition,
            "cells": [{"key": list(k), "n": n, "d": d} for k, (n, d) in sorted(cells.items())]}


def predict(profile, row):
    key = _key(row, profile["algorithm"])
    cell = next((c for c in profile["cells"] if tuple(c["key"]) == key), None)
    r = profile["global_d"]
    if profile["algorithm"] != "Global" and profile["rho"] != "inf" and cell is not None:
        rho = profile["rho"]
        r = (cell["d"] + rho * cell["n"] * r) / (cell["n"] * (1 + rho))
    return [r * h for h in profile["composition"]] + [1 - r]


def select_and_fit(rows, labels_by_observer):
    outcomes = validate_development(rows, labels_by_observer)
    folds = sorted({r["fold"] for r in rows})
    cv = []
    selection = {}
    profiles = {}
    for algorithm in ALGORITHMS:
        candidates = ("inf",) if algorithm == "Global" else RHO_GRID
        scores = []
        for rho in candidates:
            block_losses = defaultdict(list)
            for fold in folds:
                train = [r for r in rows if r["fold"] != fold]
                heldout = [r for r in rows if r["fold"] == fold]
                for observer in sorted(outcomes):
                    profile = fit_profile(train, outcomes[observer], algorithm, rho)
                    for row in heldout:
                        p = predict(profile, row)
                        d = outcomes[observer][row["sample_id"]] != "no_direct"
                        block_losses[(observer, row["question_id"])].append((1 - p[-1] - d)**2)
            means = {o: math.fsum(math.fsum(v)/len(v) for (ob, q), v in block_losses.items() if ob == o)
                     / len({q for ob, q in block_losses if ob == o}) for o in outcomes}
            mean = math.fsum(means.values()) / len(means)
            row = {"algorithm": algorithm, "rho": rho, "gate_brier": mean, "observers": means,
                   "folds": len(folds), "question_blocks": len({r['question_id'] for r in rows})}
            cv.append(row); scores.append(row)
        minimum = min(s["gate_brier"] for s in scores)
        tied = [s for s in scores if s["gate_brier"] <= minimum + TOLERANCE]
        selected = max(tied, key=lambda s: math.inf if s["rho"] == "inf" else s["rho"])
        selection[algorithm] = selected["rho"]
        profiles[algorithm] = {o: fit_profile(rows, outcomes[o], algorithm, selected["rho"]) for o in outcomes}
    return {"schema_version": "persona-observatory-gate-fit-v1", "development_samples": len(rows),
            "selection": selection, "cv": cv, "profiles": profiles,
            "note": "Selected on development data only. CV scores are selection quantities, not fresh evaluation."}


def forecast_targets(fitted, targets, development_rows):
    require(not ({r["sample_id"] for r in targets} & {r["sample_id"] for r in development_rows}),
            "Evaluation sample overlaps development")
    require(not ({r["question_id"] for r in targets} & {r["question_id"] for r in development_rows}),
            "Evaluation question ID overlaps development")
    forecasts = [{"sample_id": row["sample_id"], "observer": observer, "predictor": algorithm,
                  "category_probabilities": predict(fitted["profiles"][algorithm][observer], row)}
                 for row in targets for observer in ("observer_a", "observer_b") for algorithm in ALGORITHMS]
    validate_forecasts(forecasts, [r["sample_id"] for r in targets])
    return forecasts
