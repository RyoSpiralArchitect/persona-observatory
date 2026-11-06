#!/usr/bin/env python3
"""Compute a study analysis report and write it with provenance binding.

Binds the scores input, the evaluation plan, the analysis module, and this
script by SHA-256 inside the written JSON so downstream consumers can verify
exactly which inputs and code produced the report.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from persona_observatory.analysis import e12_report
from persona_observatory.storage import sha_file, write_json_exclusive


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, type=Path, help="scores.json from pobs score")
    ap.add_argument("--design", required=True, type=Path, help="eval design JSONL (sample plan)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", default="e12", choices=["e12"], help="analysis variant")
    args = ap.parse_args()
    scores = json.loads(args.scores.read_text())
    plan = {r["sample_id"]: r for r in map(json.loads, args.design.read_text().splitlines())}
    result = e12_report(scores["per_row"], plan)
    report = {
        "schema_version": "pobs-analysis-report-v1",
        "report_type": args.report,
        "provenance": {
            "scores_sha256": sha_file(args.scores),
            "design_sha256": sha_file(args.design),
            "analysis_module_sha256": sha_file(
                Path(__file__).resolve().parent.parent / "src" / "persona_observatory" / "analysis.py"),
            "script_sha256": sha_file(Path(__file__).resolve()),
        },
        "result": result,
    }
    write_json_exclusive(args.output, report)
    print(json.dumps({"decision": result["primary_decision"], "delta": result["primary"]["delta_ab_mean"]}))


if __name__ == "__main__":
    main()
