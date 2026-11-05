"""Fit the experiment-10 predictor family on a development pool and emit forecasts.

Inputs:
  --dev-pool     JSONL rows from extract_dev_pool.py (observer, question_id,
                 evidence_stratum, setting_id, direct_event, category, fold)
  --samples      JSONL/JSON rows with sample_id, evidence_stratum, setting_id
                 (e.g. `pobs design` output or a run plan)
  --out          predictions.jsonl in the pobs forecast schema
  --report       selection/fit report JSON

One rho per predictor is selected by question-blocked 13-fold CV, shared across
cells and observer branches. Per-branch counts produce separate forecasts;
the forecast grid covers sample x observer x predictor exactly once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from persona_observatory.forecasting import (  # noqa: E402
    PREDICTORS, build_forecasts, fit_predictor, select_rho)


def read_rows(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return data
    for key in ("plan", "samples", "rows"):
        if isinstance(data.get(key), list):
            return data[key]
    raise SystemExit("Cannot find a row list in " + str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-pool", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    pool = read_rows(args.dev_pool)
    samples = read_rows(args.samples)
    fold_of = {}
    for row in pool:
        q, f = row["question_id"], row["fold"]
        if q in fold_of and fold_of[q] != f:
            raise SystemExit("Inconsistent fold for " + q)
        fold_of[q] = f

    selection = {p: select_rho(pool, fold_of, p) for p in PREDICTORS}
    branches = sorted({r["observer"] for r in pool})
    fitted = {obs: {p: fit_predictor([r for r in pool if r["observer"] == obs], p,
                                    selection[p]["rho"]) for p in PREDICTORS}
              for obs in branches}
    forecasts = build_forecasts(fitted, samples)

    with open(args.out, "w") as f:
        for row in forecasts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "pobs-exp10-fit-v1",
        "dev_rows": len(pool), "observers": branches,
        "selection": {p: {"rho": ("inf" if selection[p]["rho"] == float("inf") else selection[p]["rho"]),
                          "cv_gate_brier": {("inf" if k == float("inf") else k): v
                                            for k, v in selection[p]["cv_gate_brier"].items()}}
                      for p in PREDICTORS},
        "forecasts": len(forecasts),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({p: report["selection"][p]["rho"] for p in PREDICTORS}))


if __name__ == "__main__":
    main()
