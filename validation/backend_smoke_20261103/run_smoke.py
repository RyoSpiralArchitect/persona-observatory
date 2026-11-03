#!/usr/bin/env python3
"""Real-weights engineering smoke; not new scientific evidence.

From the repository root, with the inference dependencies installed:

    python validation/backend_smoke_20261103/run_smoke.py --model-dir /path/to/model

Completed evidence is never overwritten. To repeat the engineering check, pass
``--output validation/backend_smoke_<new-name>`` and retain the original result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

from persona_observatory.backend import generate_records, validate_runtime


PURPOSE = "engineering_smoke_not_new_scientific_evidence"
NAMESPACE = "persona-observatory/backend-engineering-smoke/20261103"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=repo / "archives/experiment_09_20261103.zip")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("attempt.json", "plan.jsonl", "records.jsonl", "summary.json"):
        if (args.output / name).exists():
            raise FileExistsError(f"Preserve existing smoke evidence: {name}")
    manifest_path = repo / "research/experiment_09/model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model_identity = []
    for entry in manifest["files"]:
        path = args.model_dir / entry["name"]
        actual = {"name": entry["name"], "size_bytes": path.stat().st_size, "sha256": sha_file(path)}
        if actual != entry:
            raise ValueError(f"Local model snapshot mismatch: {entry['name']}")
        model_identity.append(actual)
    source_member = "persona_shrinkage_pilot/data/target_plan.jsonl"
    with zipfile.ZipFile(args.archive) as archive:
        source_plan = archive.read(source_member)
    source = json.loads(source_plan.splitlines()[0])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    prompt_ids = tokenizer.encode(source["prompt"], add_special_tokens=False)
    if prompt_ids != source["prompt_ids"]:
        raise ValueError("Retokenized old prompt differs from the frozen source prompt IDs")
    plan = []
    for replicate in range(2):
        material = f"{NAMESPACE}/replicate/{replicate}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (2**63 - 1)
        plan.append({
            "sample_id": sha_bytes(material + b"/sample")[:20],
            "purpose": PURPOSE,
            "source_question_id": source["question_id"],
            "requested_answerer": source["requested_answerer"],
            "question": source["question"],
            "prompt": source["prompt"],
            "prompt_ids": prompt_ids,
            "seed": seed,
            "replicate_id": replicate,
        })
    runtime = validate_runtime({
        "device": "cpu", "dtype": "float32", "batch_size": 2, "cpu_threads": 4,
        "max_new_tokens": 4, "temperature": 0.8, "top_p": 0.95,
    })
    with (args.output / "plan.jsonl").open("x", encoding="utf-8") as handle:
        for row in plan:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    attempt = {
        "purpose": PURPOSE,
        "started_utc": now(),
        "seed_namespace": NAMESPACE,
        "new_scientific_evidence": False,
        "semantic_labels_assigned": False,
        "source_archive": args.archive.name,
        "source_archive_sha256": sha_file(args.archive),
        "source_plan_member": source_member,
        "source_plan_sha256": sha_bytes(source_plan),
        "source_question_id": source["question_id"],
        "retokenized_prompt_matches_frozen_ids": True,
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "model_files": model_identity,
        "model_manifest_sha256": sha_file(manifest_path),
        "backend_sha256": sha_file(repo / "src/persona_observatory/backend.py"),
        "smoke_script_sha256": sha_file(Path(__file__)),
        "plan_sha256": sha_file(args.output / "plan.jsonl"),
        "runtime": runtime,
    }
    write_json(args.output / "attempt.json", attempt)
    records = []
    with (args.output / "records.jsonl").open("x", encoding="utf-8") as handle:
        def accept(record):
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        result = generate_records(plan, args.model_dir, runtime, accept)
    if len(records) != 2 or result["records"] != 2:
        raise ValueError("Engineering smoke must produce exactly two retained records")
    if any(not 1 <= row["generated_tokens"] <= 4 or
           row["generated_tokens"] != len(row["generated_token_ids"]) for row in records):
        raise ValueError("Engineering smoke output token counts differ from the fixed budget")
    summary = {
        "purpose": PURPOSE,
        "completed_utc": now(),
        "status": "PASS",
        "new_scientific_evidence": False,
        "semantic_labels_assigned": False,
        "attempt_sha256": sha_file(args.output / "attempt.json"),
        "plan_sha256": sha_file(args.output / "plan.jsonl"),
        "records_sha256": sha_file(args.output / "records.jsonl"),
        "generation": result,
        "checks": {
            "all_six_model_files_match_pinned_snapshot": True,
            "old_prompt_retokenized_exactly": True,
            "two_samples_retained": True,
            "four_token_budget_enforced": True,
            "cpu_float32": result["backend"]["device"] == "cpu" and result["actual_dtype"] == "float32",
            "eager_attention": result["backend"]["attention_implementation"] == "eager",
            "kv_decode_exercised": all(row["generated_tokens"] > 1 for row in records),
        },
    }
    if not all(summary["checks"].values()):
        raise ValueError("Engineering smoke did not exercise every required path")
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"status": "PASS", "purpose": PURPOSE,
                      "records": 2, "tokens": [r["generated_tokens"] for r in records],
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
