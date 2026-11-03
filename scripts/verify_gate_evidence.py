#!/usr/bin/env python3
"""Verify the public Experiment 10 evidence and recompute its recorded scores.

No model, inference library, semantic classifier, or network is used. Public
derivatives of model paths and private-context notes cannot prove original bytes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_observatory.evaluation import score_rows
from persona_observatory.observations import validate_observer
from persona_observatory.storage import require


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def same_scores(actual, expected):
    """Allow only tiny floating-point differences between platform math libs."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(same_scores(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(same_scores(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, float):
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= 1e-12
    return actual == expected


def verify(archive, receipt):
    receipt = json.loads(Path(receipt).read_text())
    raw = Path(archive).read_bytes()
    require(digest(raw) == receipt["sha256"] and len(raw) == receipt["bytes"], "Archive differs from receipt")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) == receipt["members"], "Archive inventory differs")
        require(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts
                    and not n.endswith("/") and "\\" not in n for n in names), "Unsafe member path")
        require(sum(i.file_size for i in z.infolist()) < 256 * 1024 * 1024, "Evidence exceeds size bound")
        data = {name: z.read(name) for name in names}
    require(digest(data["evidence_manifest.json"]) == receipt["manifest_sha256"], "Manifest differs")
    manifest = json.loads(data["evidence_manifest.json"])
    require(set(data) == set(manifest["files"]) | {"evidence_manifest.json"}, "Manifest inventory differs")
    for name, entry in manifest["files"].items():
        require(digest(data[name]) == entry["sha256"] and len(data[name]) == entry["bytes"], "Changed file: " + name)
        if entry["bytes_unchanged"]:
            require(entry["source_sha256"] == entry["sha256"], "Source hash differs: " + name)
    require(set(manifest["redactions"]) == {
        "run/run_lock.public.json", "run/attempt.public.json",
        "run/observers/observer_a/observations.public.md", "agent_a/observations.public.md",
    }, "Redaction scope differs")

    def obj(name):
        return json.loads(data[name])

    def rows(name):
        return [json.loads(line) for line in data[name].splitlines() if line.strip()]

    def bound(name, expected):
        require(digest(data[name]) == expected, "Binding differs: " + name)

    run_lock = obj("run/run_lock.public.json")
    for name, expected in run_lock["input_hashes"].items():
        bound("run/" + name, expected)
    for name in ("run/run_lock.public.json", "run/attempt.public.json"):
        require(obj(name)["model_path"] == "<local model directory omitted>", "Unexpected model path")
    completion = obj("run/completion.json")
    require(completion["run_lock_sha256"] == manifest["files"]["run/run_lock.public.json"]["source_sha256"], "Run source binding differs")
    require(completion["attempt_sha256"] == manifest["files"]["run/attempt.public.json"]["source_sha256"], "Attempt source binding differs")
    bound("run/generations.jsonl", completion["generations_sha256"])
    score_lock = obj("run/score_lock.json")
    require(score_lock["run_lock_sha256"] == completion["run_lock_sha256"], "Score run binding differs")
    for name, key in (("completion.json", "completion_sha256"), ("forecast_lock.json", "forecast_lock_sha256"), ("scores.json", "scores_sha256")):
        bound("run/" + name, score_lock[key])

    study = obj("study/study_lock_20261103.json")
    bindings = {"protocol": "study/protocol.json", "fit_lock": "study/fit_lock.json",
                "fit_audit": "study/arithmetic_audit.json", "forecast_lock": "run/forecast_lock.json",
                "observer_assignment": "study/observer_assignment_20261103.json",
                "environment": "study/mps_environment_20261103.json", "summary_script": "scripts/summarize_gate_study.py"}
    for key, name in bindings.items():
        bound(name, study["records"][key]["sha256"])
    require(study["records"]["run_lock"]["sha256"] == completion["run_lock_sha256"], "Study run binding differs")
    require(data["study/forecasts.jsonl"] == data["run/forecasts.jsonl"], "Published forecasts differ")

    labels = {}
    with tempfile.TemporaryDirectory(prefix="pobs-evidence-") as directory:
        run = Path(directory)
        for name, contents in data.items():
            if name.startswith("run/"):
                # The A-note original is private. Validate the public note and
                # all other observer fields, then check the original lock below.
                if name == "run/observers/observer_a/observation_lock.json":
                    continue
                path = run / name.removeprefix("run/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
        for branch in ("a", "b"):
            observer = "observer_" + branch
            base = "run/observers/" + observer + "/"
            agent = "agent_" + branch + "/"
            notes_name = "observations.public.md" if branch == "a" else "observations.md"
            if branch == "a":
                (run / "observers" / observer / "observations.md").write_bytes(data[base + notes_name])
            validation = validate_observer(run, observer)
            original_lock = obj(base + "observation_lock.json")
            if branch == "a":
                require(original_lock["label_rows"] == validation["records"]
                        and original_lock["unique_triples"] == validation["unique_triples"], "Original observation inventory differs")
                for key, value in validation.items():
                    if key.endswith("_sha256"):
                        expected = manifest["files"][base + notes_name]["source_sha256"] if key == "observations_sha256" else value
                        require(original_lock[key] == expected, "Original observation binding differs: " + key)
            bound(base + "observation_lock.json", score_lock["observation_locks"][observer])
            labels[observer] = rows(base + "labels.jsonl")
            for name in ("labels.jsonl", notes_name, "validation_blind.jsonl", "discovery_blind.jsonl", "rubric.md"):
                require(data[base + name] == data[agent + name], "Agent handoff differs: " + name)
            dispatch = obj("study/observer_packets_20261103.json")["observer_packets"][observer]
            bound(agent + "packet_manifest.json", dispatch["agent_packet_manifest_sha256"])
            bound(base + "manifest.json", dispatch["export_manifest_sha256"])
            packet = obj(agent + "packet_manifest.json")
            for name, expected in packet["input_sha256"].items():
                bound(agent + name, expected)
            annotation = obj(agent + "annotation_receipt.json")
            for name, key in (("packet_manifest.json", "input_manifest_sha256"), ("labels.jsonl", "labels_sha256")):
                bound(agent + name, annotation[key])
            require(manifest["files"][agent + notes_name]["source_sha256"] == annotation["notes_sha256"], "Original notes source binding differs")
            manual = {}
            for name, expected in annotation["manual_batch_sha256"].items():
                bound(agent + "manual_batches/" + name, expected)
                for row in rows(agent + "manual_batches/" + name):
                    index = row["case_index"]
                    require(index not in manual, "Duplicate manual case")
                    manual[index] = {k: v for k, v in row.items() if k != "case_index"}
            unique = rows(agent + "unique_blind.jsonl")
            require(set(manual) == set(range(len(unique))), "Incomplete manual inventory")
            mapping = obj(agent + "duplicate_map.json")
            for record in rows(agent + "validation_blind.jsonl"):
                representative = unique[mapping[record["sample_id"]]]
                require(all(record[k] == representative[k] for k in ("requested_answerer", "question", "completion")), "Duplicate mapping differs")
            for label in labels[observer]:
                require({k: v for k, v in label.items() if k != "sample_id"} == manual[mapping[label["sample_id"]]], "Manual expansion differs")

    scores = score_rows(rows("run/forecasts.jsonl"), labels)
    require(same_scores(scores, obj("run/scores.json")), "Recomputed scores differ")
    plan = rows("run/plan.jsonl")
    require(len(plan) == 1152 and len({r["question_id"] for r in plan}) == 96, "Study dimensions differ")
    indexed = {(r["sample_id"], r["observer"], r["predictor"]): r for r in scores["per_row"]}
    report = obj("study/results_20261103.json")
    bound("run/score_lock.json", report["score_lock_sha256"])
    for contrast in report["contrasts"]:
        for observer, expected in contrast["observers"].items():
            for metric, target in expected.items():
                values = [indexed[(r["sample_id"], observer, contrast["left"])][metric]
                          - indexed[(r["sample_id"], observer, contrast["right"])][metric] for r in plan]
                require(abs(math.fsum(values) / len(values) - target) < 1e-12, "Reported contrast differs")
        for metric, target in contrast["ab_mean"].items():
            require(abs(math.fsum(v[metric] for v in contrast["observers"].values()) / 2 - target) < 1e-12, "A/B mean differs")
    primary = report["contrasts"][0]
    baseline = next(c for c in report["contrasts"] if (c["left"], c["right"]) == ("Joint", "Global"))
    gates = {"absolute_brier_improvement": primary["ab_mean"]["gate_brier"] <= -0.005,
             "both_observers_improve": all(v["gate_brier"] < -1e-12 for v in primary["observers"].values()),
             "not_worse_than_global": baseline["ab_mean"]["gate_brier"] <= 1e-12,
             "gate_nll_not_worse": primary["ab_mean"]["gate_nll"] <= 1e-12}
    require(gates == report["advance_gates"] and all(gates.values()) == report["advance"], "Advance decision differs")
    return {"status": "PUBLIC_EVIDENCE_VERIFIED", "members": len(data), "generations": len(plan),
            "forecast_rows": len(scores["per_row"]), "advance": report["advance"], "score_float_tolerance": 1e-12,
            "scope": "File bindings, manual-record expansion, observation consistency and score arithmetic. No certification of semantic validity, actual blind access, original redacted bytes or independent chronology."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.archive, arguments.receipt), indent=2))
