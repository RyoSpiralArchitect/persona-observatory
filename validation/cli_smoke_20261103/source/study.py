"""Render a question/condition grid and pair random streams across conditions."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .backend import validate_runtime
from .models import validate_manifest
from .storage import read_json, require, safe_name

DEFAULT_SYSTEM = "You are {answerer}. Answer the question as yourself in two sentences. Give only your own answer."
DEFAULT_WRAPPER = "Active response instruction for {answerer}:"
CONFIG_REQUIRED = {"schema_version", "experiment_id", "purpose", "seed_namespace", "replicates",
                   "question_bank", "settings", "model_manifest", "runtime"}
CONFIG_OPTIONAL = {"system_template", "wrapper_template", "notes"}


def load_study(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    require(isinstance(config, dict) and CONFIG_REQUIRED <= set(config)
            and set(config) <= CONFIG_REQUIRED | CONFIG_OPTIONAL, "Config fields differ from the study schema")
    require(config["schema_version"] == "persona-observatory-study-v1", "Unsupported study schema")
    safe_name(config["experiment_id"], "experiment_id")
    require(config["purpose"] in {"smoke_replay", "exploratory", "prospective"}, "Unknown study purpose")
    safe_name(config["seed_namespace"], "seed_namespace")
    require(type(config["replicates"]) is int and 1 <= config["replicates"] <= 1000, "Invalid replicate count")
    config["runtime"] = validate_runtime(config["runtime"])
    config.setdefault("system_template", DEFAULT_SYSTEM)
    config.setdefault("wrapper_template", DEFAULT_WRAPPER)
    for key in ("system_template", "wrapper_template"):
        require(isinstance(config[key], str) and bool(config[key].strip()), "Empty prompt template")
        try:
            config[key].format(answerer="Morgan")
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError("Prompt templates accept only {answerer}") from exc
    paths = {}
    for key in ("question_bank", "settings", "model_manifest"):
        require(isinstance(config[key], str) and bool(config[key]), "Invalid input path: " + key)
        paths[key] = (config_path.parent / config[key]).resolve()
    questions, settings = read_json(paths["question_bank"]), read_json(paths["settings"])
    require(isinstance(questions, list) and questions and isinstance(settings, list) and settings,
            "Question and setting banks must be nonempty lists")
    for bank, identifier, required in [
        (questions, "question_id", {"question_id", "question", "topic", "evidence_stratum"}),
        (settings, "setting_id", {"setting_id", "requested_answerer", "policy_text"}),
    ]:
        ids = []
        for row in bank:
            require(isinstance(row, dict) and required <= set(row), "Missing fields in " + identifier + " bank")
            ids.append(safe_name(row[identifier], identifier))
            for key in required:
                require(isinstance(row[key], str) and bool(row[key].strip()), "Empty bank field: " + key)
        require(len(ids) == len(set(ids)), "Duplicate " + identifier)
    count = len(questions) * len(settings) * config["replicates"]
    require(count <= 100000, "Study grid exceeds 100000 outputs; split into explicitly named studies")
    manifest = validate_manifest(read_json(paths["model_manifest"]))
    return config, questions, settings, manifest


def design_rows(config, questions, settings):
    rows = []
    for setting in settings:
        answerer = setting["requested_answerer"]
        for question in questions:
            messages = [
                {"role": "system", "content": config["system_template"].format(answerer=answerer)},
                {"role": "user", "content": config["wrapper_template"].format(answerer=answerer)
                 + "\n> " + setting["policy_text"] + "\n\nQuestion: " + question["question"]},
            ]
            for replica in range(config["replicates"]):
                key = f"{config['seed_namespace']}|{question['question_id']}|{replica}"
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)
                sid = hashlib.sha256((config["experiment_id"] + "|" + setting["setting_id"] + "|" + key).encode()).hexdigest()[:20]
                rows.append({"sample_id": sid, "experiment_id": config["experiment_id"], "purpose": config["purpose"],
                             "question_id": question["question_id"], "topic": question["topic"],
                             "question": question["question"], "evidence_stratum": question["evidence_stratum"],
                             "setting_id": setting["setting_id"], "requested_answerer": answerer,
                             "policy_text": setting["policy_text"], "replicate_id": replica, "seed": seed,
                             "messages": messages})
    require(len(rows) == len({row["sample_id"] for row in rows}), "Sample ID collision")
    streams = {(r["question_id"], r["replicate_id"]): r["seed"] for r in rows}
    require(len(set(streams.values())) == len(streams), "Random-stream collision")
    return rows
