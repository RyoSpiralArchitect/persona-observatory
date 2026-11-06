"""Fit the experiment-10 predictor family on a development pool and emit forecasts.

Usage:
  python3 scripts/fit_forecasts.py --dev-pool POOL.jsonl --samples SAMPLES.jsonl \
      --attempt-dir research/experiment_10/fits/<name>

The attempt directory is created exclusively (it must not already exist) and
receives: dev_pool.sha256, predictions.jsonl, report.json, manifest.json.
The manifest binds SHA-256 of the dev pool, the sample design, this script, the
installed package source, and both outputs, so the frozen forecasts trace to
exact inputs. One rho per predictor is selected by question-blocked 13-fold CV,
shared across cells and observer branches; per-branch counts produce separate
forecasts covering sample x observer x predictor exactly once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from persona_observatory import __version__  # noqa: E402
from persona_observatory.forecasting import (  # noqa: E402
    PREDICTORS, build_forecasts, fit_predictor, select_rho)
from persona_observatory.storage import (  # noqa: E402
    require, sha_file, write_json_exclusive, write_jsonl_exclusive)


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
    ap.add_argument("--dev-pool", required=True, type=Path)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--attempt-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=("gate", "category"), default="gate",
                    help="gate: E10 direct-rate x pooled-composition family; category: E12 per-cell five-category distribution")
    args = ap.parse_args()

    pool = read_rows(args.dev_pool)
    samples = read_rows(args.samples)
    fold_of, sample_branches = {}, {}
    for row in pool:
        q, f = row["question_id"], row["fold"]
        require(q not in fold_of or fold_of[q] == f, "Inconsistent fold for " + q)
        fold_of[q] = f
        branches = sample_branches.setdefault(row["sample_id"], set())
        require(row["observer"] not in branches,
                "Duplicate dev row: " + row["sample_id"] + "/" + row["observer"])
        branches.add(row["observer"])
    expected = {"observer_a", "observer_b"}
    for sid, branches in sample_branches.items():
        require(branches == expected, f"Unpaired observer coverage for {sid}: {sorted(branches)}")
    require(sorted({r["observer"] for r in pool}) == sorted(expected),
            "This study expects exactly observer_a and observer_b")

    selection = {p: select_rho(pool, fold_of, p, args.mode) for p in PREDICTORS}
    branches = sorted(expected)
    fitted = {obs: {p: fit_predictor([r for r in pool if r["observer"] == obs], p,
                                    selection[p]["rho"], mode=args.mode) for p in PREDICTORS}
              for obs in branches}
    forecasts = build_forecasts(fitted, samples)

    # The attempt directory is the transaction boundary; nothing overwrites.
    args.attempt_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl_exclusive(args.attempt_dir / "predictions.jsonl", forecasts)
    report = {
        "schema_version": "pobs-exp10-fit-v1",
        "dev_rows": len(pool), "observers": branches, "mode": args.mode,
        "selection": {p: {"rho": ("inf" if selection[p]["rho"] == float("inf") else selection[p]["rho"]),
                          "cv_scores": {"inf" if k == float("inf") else k: v
                              for k, v in selection[p]["cv_scores"].items()}}
                      for p in PREDICTORS},
        "forecasts": len(forecasts),
    }
    write_json_exclusive(args.attempt_dir / "report.json", report)
    manifest = {
        "schema_version": "pobs-exp10-fit-manifest-v1",
        "package_version": __version__,
        "inputs_sha256": {"dev_pool": sha_file(args.dev_pool), "samples": sha_file(args.samples),
                          "script": sha_file(Path(__file__).resolve())},
        "outputs_sha256": {"predictions": sha_file(args.attempt_dir / "predictions.jsonl"),
                           "report": sha_file(args.attempt_dir / "report.json")},
        "note": "Exclusive fit attempt binding development labels, evaluation design, code and outputs by hash. Not a prospective lock; freezing happens at pobs prepare --forecasts.",
    }
    write_json_exclusive(args.attempt_dir / "manifest.json", manifest)
    print(json.dumps({p: report["selection"][p]["rho"] for p in PREDICTORS}))


if __name__ == "__main__":
    main()
