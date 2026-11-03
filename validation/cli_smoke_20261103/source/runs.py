"""Prepare immutable local runs, perform one generation attempt, and verify it.

Hashes provide local integrity, not external timestamp certification. A prepared
run owns its configuration, tokenized prompts, seeds, code snapshot, and optional
external forecasts. It never overwrites a failed or completed generation attempt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import __version__
from .backend import generate_records, inspect_backend
from .models import verify_model
from .storage import (check_hashes, checked_file, now, read_json, read_jsonl, require,
                      sha_file, write_json_exclusive, write_jsonl_exclusive)
from .study import design_rows, load_study

_ENV_KEYS = ("device", "dtype", "python", "torch_version", "transformers_version", "cuda_runtime_version",
             "platform", "device_name", "device_capability")


def _timestamp(value):
    from datetime import datetime
    require(isinstance(value, str), "Timestamp must be an ISO8601 string")
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "Timestamp must include a timezone")
    return parsed


def _code_files():
    root = Path(__file__).resolve().parent
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*")
            if p.is_file() and p.suffix in {".py", ".md"} and "__pycache__" not in p.parts}


def _snapshot_code(run):
    hashes = {}
    for name, source in sorted(_code_files().items()):
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(source.read_bytes())
        hashes[name] = sha_file(source)
    return hashes


def prepare(config_path, run_dir, model_path, forecasts_path=None):
    config, questions, settings, manifest = load_study(config_path)
    require(config["purpose"] != "prospective" or forecasts_path is not None,
            "A prospective forecast study requires --forecasts before generation; use exploratory for other work")
    rows = design_rows(config, questions, settings)
    forecasts = None
    if forecasts_path is not None:
        from .evaluation import validate_forecasts
        forecasts = read_jsonl(forecasts_path)
        validate_forecasts(forecasts, [row["sample_id"] for row in rows])
    model_path = Path(model_path).resolve()
    verify_model(model_path, manifest)
    backend = inspect_backend(config["runtime"]["device"], config["runtime"]["dtype"])
    require(backend["available"], backend.get("error", "Backend unavailable"))
    requested_runtime = dict(config["runtime"])
    config["runtime"].update(device=backend["device"], dtype=backend["dtype"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    require(bool(tokenizer.chat_template), "A local chat template is required; do not silently invent one")
    generation_config_path = model_path / "generation_config.json"
    generation_config = read_json(generation_config_path if generation_config_path.exists() else model_path / "config.json")
    eos = generation_config.get("eos_token_id")
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    require(all(type(i) is int and i >= 0 for i in eos), "Invalid effective EOS IDs")
    eos = sorted(set(eos))
    effective_pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos, default=0)
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True)
        require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids), "Invalid tokenizer IDs")
        require(tokenizer(prompt, add_special_tokens=False)["input_ids"] == ids, "Chat-template tokenization mismatch")
        row.update(prompt=prompt, prompt_ids=ids, prompt_tokens=len(ids))
    run = Path(run_dir).resolve()
    # A new directory is the transaction boundary. A partial preparation is preserved.
    run.mkdir(parents=True, exist_ok=False)
    config.update(question_bank="inputs/questions.json", settings="inputs/settings.json",
                  model_manifest="inputs/model_manifest.json")
    write_json_exclusive(run / "config.json", config)
    write_json_exclusive(run / "inputs/questions.json", questions)
    write_json_exclusive(run / "inputs/settings.json", settings)
    write_json_exclusive(run / "inputs/model_manifest.json", manifest)
    write_json_exclusive(run / "backend_preflight.json", {**backend, "requested_runtime": requested_runtime,
                                                        "model_forward_performed": False})
    write_json_exclusive(run / "tokenizer.json", {"chat_template": tokenizer.chat_template,
                                                "eos_token_id": tokenizer.eos_token_id,
                                                "pad_token_id": tokenizer.pad_token_id,
                                                "effective_eos_token_ids": eos, "effective_pad_token_id": effective_pad})
    write_jsonl_exclusive(run / "plan.jsonl", rows)
    code_hashes = _snapshot_code(run)
    if forecasts is not None:
        write_jsonl_exclusive(run / "forecasts.jsonl", forecasts)
        write_json_exclusive(run / "forecast_lock.json", {
            "schema_version": "persona-observatory-forecast-lock-v1", "locked_utc": now(),
            "forecasts_sha256": sha_file(run / "forecasts.jsonl"),
            "original_forecasts_sha256": sha_file(forecasts_path),
            "plan_sha256": sha_file(run / "plan.jsonl"), "generation_absent": True,
            "scope": "Externally supplied predictions; their fitting and development provenance are not audited here.",
        })
    inputs = {p.relative_to(run).as_posix(): sha_file(p) for p in run.rglob("*") if p.is_file()}
    write_json_exclusive(run / "run_lock.json", {
        "schema_version": "persona-observatory-run-lock-v1", "prepared_utc": now(), "package_version": __version__,
        "experiment_id": config["experiment_id"], "purpose": config["purpose"], "samples": len(rows),
        "input_hashes": inputs, "code_hashes": code_hashes, "model_path": str(model_path),
        "generation_absent": True, "forecasts_locked": forecasts is not None,
        "note": "Local integrity record; not a public preregistration or independent timestamp certificate.",
    })
    return {"status": "PREPARED", "run_dir": str(run), "samples": len(rows), "device": backend["device"],
            "dtype": backend["dtype"], "purpose": config["purpose"], "forecasts_locked": forecasts is not None,
            "run_lock_sha256": sha_file(run / "run_lock.json")}


def verify_prepared(run_dir, check_current_code=False):
    run = Path(run_dir).resolve()
    lock = read_json(run / "run_lock.json")
    require(lock["schema_version"] == "persona-observatory-run-lock-v1", "Unsupported run lock")
    prepared_time = _timestamp(lock["prepared_utc"])
    check_hashes(run, lock["input_hashes"])
    config, questions, settings, manifest = load_study(run / "config.json")
    require(config["question_bank"] == "inputs/questions.json" and config["settings"] == "inputs/settings.json"
            and config["model_manifest"] == "inputs/model_manifest.json", "Prepared config refers outside its frozen inputs")
    require(lock["experiment_id"] == config["experiment_id"] and lock["purpose"] == config["purpose"]
            and lock["generation_absent"] is True, "Run-lock identity differs from frozen config")
    expected = design_rows(config, questions, settings)
    rows = read_jsonl(run / "plan.jsonl")
    require(len(rows) == lock["samples"] == len(expected), "Plan sample count differs")
    for actual, target in zip(rows, expected):
        require(set(actual) == set(target) | {"prompt", "prompt_ids", "prompt_tokens"}, "Plan fields differ")
        require(all(actual[key] == value for key, value in target.items()), "Plan does not match frozen study")
        require(isinstance(actual["prompt"], str) and bool(actual["prompt"]), "Empty rendered prompt")
        require(isinstance(actual["prompt_ids"], list) and actual["prompt_ids"]
                and all(type(v) is int and v >= 0 for v in actual["prompt_ids"]), "Invalid frozen tokens")
        require(actual["prompt_tokens"] == len(actual["prompt_ids"]), "Prompt token count differs")
    mandatory = {"config.json", "inputs/questions.json", "inputs/settings.json", "inputs/model_manifest.json",
                 "backend_preflight.json", "tokenizer.json", "plan.jsonl"}
    require(isinstance(lock["code_hashes"], dict) and bool(lock["code_hashes"]), "Empty source snapshot inventory")
    code_names = {"source/" + name for name in lock["code_hashes"]}
    actual_code_names = {p.relative_to(run).as_posix() for p in (run / "source").rglob("*") if p.is_file()}
    require(code_names == actual_code_names, "Source snapshot inventory differs")
    expected_inputs = mandatory | code_names
    if lock["forecasts_locked"]:
        expected_inputs |= {"forecasts.jsonl", "forecast_lock.json"}
    require(expected_inputs == set(lock["input_hashes"]), "Prepared-input inventory differs")
    for name, digest in lock["code_hashes"].items():
        require(lock["input_hashes"]["source/" + name] == digest, "Code snapshot identity differs")
    if check_current_code:
        current = {name: sha_file(p) for name, p in _code_files().items()}
        require(current == lock["code_hashes"], "Runner source changed after prepare; create a new run")
    require(type(lock["forecasts_locked"]) is bool, "Invalid forecast boundary")
    if lock["forecasts_locked"]:
        from .evaluation import validate_forecasts
        require({"forecasts.jsonl", "forecast_lock.json"} <= set(lock["input_hashes"]), "Forecast not in run lock")
        forecast_lock = read_json(run / "forecast_lock.json")
        require(forecast_lock["schema_version"] == "persona-observatory-forecast-lock-v1"
                and forecast_lock["forecasts_sha256"] == sha_file(run / "forecasts.jsonl")
                and forecast_lock["plan_sha256"] == sha_file(run / "plan.jsonl")
                and forecast_lock["generation_absent"] is True, "Forecast boundary differs")
        require(_timestamp(forecast_lock["locked_utc"]) <= prepared_time, "Forecast was locked after preparation")
        validate_forecasts(read_jsonl(run / "forecasts.jsonl"), [r["sample_id"] for r in rows])
    else:
        require(not (run / "forecasts.jsonl").exists() and not (run / "forecast_lock.json").exists(),
                "Forecasts were added after prepare")
    require(config["purpose"] != "prospective" or lock["forecasts_locked"], "Prospective study has no forecast lock")
    return run, lock, config, rows, manifest


def generate(run_dir, model_path=None):
    run, lock, config, rows, manifest = verify_prepared(run_dir, check_current_code=True)
    for name in ("attempt.json", "generations.jsonl", "completion.json", "failure.json"):
        require(not (run / name).exists(), "Run already has a generation attempt; no overwrite/resume: " + name)
    model_path = Path(model_path or lock["model_path"]).resolve()
    verify_model(model_path, manifest)
    backend = inspect_backend(config["runtime"]["device"], config["runtime"]["dtype"])
    require(backend["available"], backend.get("error", "Backend unavailable"))
    preflight = read_json(run / "backend_preflight.json")
    require(all(backend.get(k) == preflight.get(k) for k in _ENV_KEYS),
            "Backend/Python/Torch/Transformers changed after prepare; prepare a new run on this machine")
    attempt = {"schema_version": "persona-observatory-attempt-v1", "started_utc": now(),
               "run_lock_sha256": sha_file(run / "run_lock.json"), "model_path": str(model_path),
               "model_verified": True, "backend": backend}
    write_json_exclusive(run / "attempt.json", attempt)
    emitted = 0
    try:
        with (run / "generations.jsonl").open("x", encoding="utf-8", newline="\n") as output:
            def on_record(record):
                nonlocal emitted
                require(emitted < len(rows) and all(record.get(k) == v for k, v in rows[emitted].items()),
                        "Backend output differs from the next frozen plan row")
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                emitted += 1
            summary = generate_records(rows, model_path, config["runtime"], on_record)
        require(emitted == len(rows), "Generation ended without all planned outputs")
        verify_prepared(run, check_current_code=True)
        verify_model(model_path, manifest)
        write_json_exclusive(run / "completion.json", {
            "schema_version": "persona-observatory-completion-v1", "completed_utc": now(), "samples": emitted,
            "run_lock_sha256": sha_file(run / "run_lock.json"), "attempt_sha256": sha_file(run / "attempt.json"),
            "generations_sha256": sha_file(run / "generations.jsonl"), "runtime": summary,
        })
    except BaseException as exc:
        if not (run / "failure.json").exists():
            write_json_exclusive(run / "failure.json", {"failed_utc": now(), "error_type": type(exc).__name__,
                                                        "message": str(exc), "records_written": emitted})
        raise
    return verify_run(run, require_complete=True)


def verify_run(run_dir, require_complete=False):
    run, lock, config, plan, _ = verify_prepared(run_dir)
    result = {"status": "PREPARED", "run_dir": str(run), "samples": len(plan),
              "purpose": config["purpose"], "forecasts_locked": lock["forecasts_locked"],
              "run_lock_sha256": sha_file(run / "run_lock.json")}
    if not (run / "completion.json").exists():
        if (run / "attempt.json").exists():
            attempt = read_json(run / "attempt.json")
            require(attempt["schema_version"] == "persona-observatory-attempt-v1"
                    and attempt["run_lock_sha256"] == result["run_lock_sha256"]
                    and attempt["model_verified"] is True, "Invalid generation-attempt record")
            require(_timestamp(lock["prepared_utc"]) <= _timestamp(attempt["started_utc"]), "Attempt precedes preparation")
            result["status"] = "FAILED" if (run / "failure.json").exists() else "INCOMPLETE"
        else:
            require(not (run / "generations.jsonl").exists() and not (run / "failure.json").exists(),
                    "Orphan generation or failure file without an attempt record")
        require(not require_complete, "A complete generation record is required; current status: " + result["status"])
        return result
    require(not (run / "failure.json").exists(), "Run contains both success and failure records")
    completion = read_json(run / "completion.json")
    attempt = read_json(run / "attempt.json")
    require(completion["schema_version"] == "persona-observatory-completion-v1"
            and attempt["schema_version"] == "persona-observatory-attempt-v1"
            and attempt["model_verified"] is True, "Invalid generation/completion schema")
    require(completion["run_lock_sha256"] == attempt["run_lock_sha256"] == result["run_lock_sha256"], "Run lock changed")
    require(completion["attempt_sha256"] == sha_file(run / "attempt.json"), "Attempt record changed")
    require(completion["generations_sha256"] == sha_file(run / "generations.jsonl"), "Generated output bytes changed")
    records = read_jsonl(run / "generations.jsonl")
    require(len(records) == completion["samples"] == len(plan), "Incomplete generated output inventory")
    special_tokens = read_json(run / "tokenizer.json")
    eos = special_tokens["effective_eos_token_ids"]
    require(completion["runtime"]["eos_token_ids"] == eos
            and completion["runtime"]["pad_token_id"] == special_tokens["effective_pad_token_id"],
            "Runtime special-token IDs differ from preparation")
    preflight = read_json(run / "backend_preflight.json")
    require(all(attempt["backend"].get(k) == preflight.get(k) == completion["runtime"]["backend"].get(k)
                for k in _ENV_KEYS), "Recorded generation backend differs from preparation")
    require(completion["runtime"]["runtime"] == config["runtime"]
            and completion["runtime"]["actual_dtype"] == config["runtime"]["dtype"], "Recorded runtime differs")
    for expected, record in zip(plan, records):
        require(set(record) == set(expected) | {"completion", "generated_token_ids", "generated_tokens", "stop_reason"},
                "Unexpected generation fields")
        require(all(record[k] == v for k, v in expected.items()), "Generated record lost frozen plan metadata")
        ids = record["generated_token_ids"]
        require(isinstance(record["completion"], str) and isinstance(ids, list)
                and all(type(x) is int and x >= 0 for x in ids), "Invalid completion/token list")
        require(type(record["generated_tokens"]) is int and len(ids) == record["generated_tokens"]
                and 1 <= len(ids) <= config["runtime"]["max_new_tokens"], "Invalid generated token count")
        require(record["stop_reason"] in {"eos", "length"}, "Unknown generation stop")
        require(not any(i in eos for i in ids[:-1]), "Generation continued after EOS")
        require((record["stop_reason"] == "eos") == (ids[-1] in eos), "EOS token and stop reason disagree")
        require(record["stop_reason"] != "length" or len(ids) == config["runtime"]["max_new_tokens"], "Early length stop")
    require(_timestamp(lock["prepared_utc"]) <= _timestamp(attempt["started_utc"]) <= _timestamp(completion["completed_utc"]),
            "Run chronology differs")
    if lock["forecasts_locked"]:
        require(_timestamp(read_json(run / "forecast_lock.json")["locked_utc"]) <= _timestamp(lock["prepared_utc"]),
                "Forecasts were locked after preparation")
    result.update(status="COMPLETE", generations_sha256=completion["generations_sha256"],
                  runtime=completion["runtime"])
    return result
