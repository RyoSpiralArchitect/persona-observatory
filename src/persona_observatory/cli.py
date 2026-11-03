"""Command-line entry point; help, design and verification need no Torch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .storage import now, read_jsonl, require, sha_file, write_json_exclusive, write_jsonl_exclusive


def parser():
    p = argparse.ArgumentParser(prog="pobs", description="Observable persona research: frozen evidence and portable local runs")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check backend availability without loading model weights")
    doctor.add_argument("--device", default="auto")
    doctor.add_argument("--dtype", default="auto")
    fetch = sub.add_parser("fetch-model", help="Download public pinned files and verify their checksums")
    fetch.add_argument("--config", required=True, type=Path)
    fetch.add_argument("--destination", required=True, type=Path)
    design = sub.add_parser("design", help="Export target IDs/prompts before inference, for externally prepared forecasts")
    design.add_argument("--config", required=True, type=Path)
    design.add_argument("--output", required=True, type=Path)
    prepare = sub.add_parser("prepare", help="Freeze a new run: config, tokens, seeds, source and optional forecasts")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--run-dir", required=True, type=Path)
    prepare.add_argument("--model-path", required=True, type=Path)
    prepare.add_argument("--forecasts", type=Path)
    generate = sub.add_parser("generate", help="Make exactly one generation attempt for a prepared run")
    generate.add_argument("--run-dir", required=True, type=Path)
    generate.add_argument("--model-path", type=Path, help="Relocated model directory; every frozen byte must still match")
    verify = sub.add_parser("verify-run", help="Read-only run integrity and completion check")
    verify.add_argument("--run-dir", required=True, type=Path)
    export = sub.add_parser("export-observer", help="Create a new blinded semantic-coding workspace")
    export.add_argument("--run-dir", required=True, type=Path)
    export.add_argument("--observer", required=True)
    validate = sub.add_parser("validate-observer", help="Validate all semantic codes and optionally lock them")
    validate.add_argument("--run-dir", required=True, type=Path)
    validate.add_argument("--observer", required=True)
    validate.add_argument("--lock", action="store_true")
    score = sub.add_parser("score", help="Score only prelocked external forecasts against locked observations")
    score.add_argument("--run-dir", required=True, type=Path)
    return p


def dispatch(a):
    if a.command == "doctor":
        from .backend import inspect_backend
        return inspect_backend(a.device, a.dtype)
    if a.command == "fetch-model":
        from .models import fetch_model, manifest_from_config
        return fetch_model(a.destination, manifest_from_config(a.config))
    if a.command == "design":
        from .study import load_study, design_rows
        config, questions, settings, _ = load_study(a.config)
        rows = design_rows(config, questions, settings)
        write_jsonl_exclusive(a.output, rows)
        return {"status": "DESIGN_EXPORTED", "samples": len(rows), "output": str(a.output),
                "model_inference_performed": False, "purpose": config["purpose"]}
    from .runs import generate, prepare, verify_run
    if a.command == "prepare":
        return prepare(a.config, a.run_dir, a.model_path, a.forecasts)
    if a.command == "generate":
        return generate(a.run_dir, a.model_path)
    if a.command == "verify-run":
        return verify_run(a.run_dir)
    # Observation and scoring always require a successful, unchanged generation.
    run = a.run_dir.resolve()
    verify_run(run, require_complete=True)
    from .observations import export_observer, validate_observer
    if a.command == "export-observer":
        return export_observer(run, a.observer, read_jsonl(run / "generations.jsonl"))
    if a.command == "validate-observer":
        return validate_observer(run, a.observer, lock=a.lock)
    if a.command == "score":
        from .evaluation import validate_forecasts, score_rows
        from .runs import verify_prepared
        from .storage import read_json
        verify_prepared(run, check_current_code=True)
        require(read_json(run / "run_lock.json")["forecasts_locked"], "No forecasts were locked before generation")
        require(not (run / "scores.json").exists() and not (run / "score_lock.json").exists(),
                "Scores already exist; never overwrite a scored run")
        forecasts = read_jsonl(run / "forecasts.jsonl")
        dims = validate_forecasts(forecasts, [r["sample_id"] for r in read_jsonl(run / "plan.jsonl")])
        labels, observation_hashes = {}, {}
        for observer in dims["observers"]:
            observation_dir = run / "observers" / observer
            require((observation_dir / "observation_lock.json").is_file(), "Missing observation lock: " + observer)
            validate_observer(run, observer)
            labels[observer] = read_jsonl(observation_dir / "labels.jsonl")
            observation_hashes[observer] = sha_file(observation_dir / "observation_lock.json")
        result = score_rows(forecasts, labels)
        write_json_exclusive(run / "scores.json", result)
        write_json_exclusive(run / "score_lock.json", {
            "schema_version": "persona-observatory-score-lock-v1", "scored_utc": now(),
            "run_lock_sha256": sha_file(run / "run_lock.json"), "completion_sha256": sha_file(run / "completion.json"),
            "forecast_lock_sha256": sha_file(run / "forecast_lock.json"), "observation_locks": observation_hashes,
            "scores_sha256": sha_file(run / "scores.json"),
            "note": "No predictor was fit or selected by this scoring command.",
        })
        return {"status": "SCORED", "dimensions": result["dimensions"], "summary": result["summary"]}
    raise ValueError("Unknown command")


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        result = dispatch(a)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if result.get("available") is False or result.get("status") in {"FAILED", "INCOMPLETE"}:
            return 2
        return 0
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
