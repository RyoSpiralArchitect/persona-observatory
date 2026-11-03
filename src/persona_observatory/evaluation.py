"""Pure scoring of externally fixed probabilities and semantic labels.

This module does not fit or choose a predictor. It cannot establish chronology:
the CLI must verify the run, forecast and observation locks before calling it.
All observer branches measure the same subject generations; their rows are not
additional independent trials.
"""
from __future__ import annotations

import math
import re
from typing import Any

from .observations import derived_outcome


CATEGORY_ORDER = ("support", "reject", "withhold", "mixed", "no_direct")
FORECAST_FIELDS = frozenset({"sample_id", "observer", "predictor", "category_probabilities"})
PROBABILITY_SUM_TOLERANCE = 1e-9
METRICS = ("category_brier", "category_nll", "gate_brier", "gate_nll")
_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_BRANCH_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def _safe_identifier(value: Any, field: str) -> str:
    pattern = _SAMPLE_ID if field == "sample_id" else _BRANCH_ID
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"Invalid safe {field}: use a nonempty alphanumeric identifier with '_' or '-'")
    return value


def validate_forecasts(rows: list[dict], sample_ids: list[str]) -> dict[str, Any]:
    """Validate a complete sample × observer × predictor forecast grid.

The branch declarations are the observer and predictor identifiers present in
the forecast file. Every combination must occur exactly once for every supplied
sample. No probability normalization, clipping or row filling is performed.
"""
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError("Forecast validation requires a nonempty list of sample_ids")
    expected_samples = [_safe_identifier(sid, "sample_id") for sid in sample_ids]
    if len(set(expected_samples)) != len(expected_samples):
        raise ValueError("Expected sample_ids must be unique")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Forecasts must be a nonempty list of records")
    sample_set = set(expected_samples)
    observers: set[str] = set()
    predictors: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != FORECAST_FIELDS:
            raise ValueError("Every forecast must have exactly sample_id, observer, predictor, category_probabilities")
        sid = _safe_identifier(row["sample_id"], "sample_id")
        observer = _safe_identifier(row["observer"], "observer")
        predictor = _safe_identifier(row["predictor"], "predictor")
        if sid not in sample_set:
            raise ValueError(f"Unknown forecast sample_id: {sid}")
        key = sid, observer, predictor
        if key in seen:
            raise ValueError(f"Duplicate forecast row: {key}")
        seen.add(key)
        observers.add(observer)
        predictors.add(predictor)
        probabilities = row["category_probabilities"]
        if not isinstance(probabilities, list) or len(probabilities) != len(CATEGORY_ORDER):
            raise ValueError(f"Exactly five category probabilities are required: {key}")
        for probability in probabilities:
            if type(probability) not in (int, float):
                raise ValueError(f"Category probabilities must be numbers, not Booleans or strings: {key}")
            try:
                valid = math.isfinite(probability) and 0 < probability < 1
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"Category probabilities must be finite and strictly between zero and one: {key}")
        if abs(math.fsum(probabilities) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"Category probabilities must sum to one within 1e-9: {key}")
    expected_count = len(expected_samples) * len(observers) * len(predictors)
    if len(seen) != expected_count:
        raise ValueError("Forecasts do not cover the complete sample × observer × predictor grid")
    # All keys use expected samples and declared branches; unique key count at
    # the Cartesian-product size therefore proves exact coverage.
    return {
        "sample_ids": list(expected_samples),
        "samples": len(expected_samples),
        "observers": sorted(observers),
        "predictors": sorted(predictors),
        "forecast_rows": len(rows),
        "category_order": list(CATEGORY_ORDER),
    }


def score_rows(forecasts: list[dict], labels_by_observer: dict[str, list[dict]]) -> dict[str, Any]:
    """Score complete, schema-valid labels against fixed positive forecasts.

``gate_probability`` is defined as ``1 - p_no_direct``. The gate NLL uses
``log(p_no_direct)`` / ``log1p(-p_no_direct)`` directly, preserving finite NLL
even when the binary complement rounds to one. Observer summaries remain
separate; no confidence interval, pooled independent N or significance test is
calculated.
    """
    if not isinstance(labels_by_observer, dict) or not labels_by_observer:
        raise ValueError("Provide a nonempty mapping of observers to complete labels")
    outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    sample_ids: list[str] | None = None
    for observer, labels in labels_by_observer.items():
        _safe_identifier(observer, "observer")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Observer labels must be a nonempty list: {observer}")
        branch: dict[str, dict[str, Any]] = {}
        for label in labels:
            if not isinstance(label, dict) or "sample_id" not in label:
                raise ValueError(f"Every label must be an object with sample_id: {observer}")
            sid = _safe_identifier(label["sample_id"], "sample_id")
            if sid in branch:
                raise ValueError(f"Duplicate label sample_id for {observer}: {sid}")
            # This call enforces the full eight-field rubric and enum relations.
            branch[sid] = derived_outcome(label)
        if sample_ids is None:
            sample_ids = sorted(branch)
        elif set(branch) != set(sample_ids):
            raise ValueError("All observers must label the same exact sample identifiers")
        outcomes[observer] = branch
    assert sample_ids is not None
    dimensions = validate_forecasts(forecasts, sample_ids)
    if set(dimensions["observers"]) != set(outcomes):
        raise ValueError("Forecast observers and label observers must match exactly")
    per_row = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    category_indices = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    for forecast in forecasts:
        sid, observer, predictor = (forecast[key] for key in ("sample_id", "observer", "predictor"))
        outcome = outcomes[observer][sid]
        category_index = category_indices[outcome["category"]]
        p = forecast["category_probabilities"]
        no_direct_probability = p[-1]
        is_direct = outcome["direct_event"]
        gate_probability = 1.0 - no_direct_probability
        row = {
            "sample_id": sid,
            "observer": observer,
            "predictor": predictor,
            "category": outcome["category"],
            "direct_event": is_direct,
            "category_probabilities": list(p),
            "gate_probability": gate_probability,
            "category_brier": math.fsum((probability - int(index == category_index)) ** 2
                                        for index, probability in enumerate(p)),
            "category_nll": -math.log(p[category_index]),
            "gate_brier": (no_direct_probability if is_direct else gate_probability) ** 2,
            "gate_nll": -math.log1p(-no_direct_probability) if is_direct else -math.log(no_direct_probability),
        }
        if any(not math.isfinite(row[metric]) for metric in METRICS):
            raise ValueError(f"Nonfinite score for {(sid, observer, predictor)}")
        per_row.append(row)
        groups.setdefault((observer, predictor), []).append(row)
    summary = []
    for (observer, predictor), rows in sorted(groups.items()):
        summary.append({
            "observer": observer,
            "predictor": predictor,
            "n": len(rows),
            **{metric: math.fsum(row[metric] for row in rows) / len(rows) for metric in METRICS},
        })
    return {
        "schema_version": "persona-observatory-scores-v1",
        "dimensions": dimensions,
        "per_row": per_row,
        "summary": summary,
        "measurement_note": (
            "Observer branches code the same subject generations. The number of unique subject samples "
            "is dimensions.samples; multiplying by observers or predictors does not add independent trials. "
            "Scores are descriptive means with natural-log NLL; no p-values or population intervals are computed."
        ),
    }
