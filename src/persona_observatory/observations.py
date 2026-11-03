"""Blinded exports and immutable, explicit semantic-observation records.

This module validates the eight fields and internal consistency of the frozen
Experiment 9 rubric. It never assigns a judgment from keywords or generates
semantic annotations. A person or a separately instructed observer supplies the
complete labels and notes before a lock is written.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
from pathlib import Path
import re
from typing import Any


FIELDS = (
    "sample_id", "answerer_answer_mode", "answerer_judgment", "answerer_stance",
    "limit_owners", "other_answer", "character_description", "note",
)
ENUMS = {
    "answerer_answer_mode": frozenset({"direct", "reported", "both", "none"}),
    "answerer_judgment": frozenset({"support", "reject", "withhold", "mixed", "none"}),
    "answerer_stance": frozenset({"decisive", "tentative", "mixed", "none"}),
}
OWNERS = frozenset({"answerer_direct", "answerer_reported", "other", "unclear"})
BLIND_FIELDS = ("sample_id", "requested_answerer", "question", "completion", "stop_reason")
RUBRIC_SHA256 = "7a28b5f7101c3e9de0367c064bec34f60154b7e7a4c9934c4685eff381d601a6"
EXPORT_SCHEMA = "persona-observatory-observer-export-v1"
LOCK_SCHEMA = "persona-observatory-observation-lock-v1"
NOTES_MARKER = "REPLACE_OBSERVATION_NOTES_TEMPLATE"
NOTES_TEMPLATE = f"""# Observer notes — replace this template before locking

<!-- {NOTES_MARKER} -->

Read rubric.md, discovery_blind.jsonl, and every record in validation_blind.jsonl.
Copy labels.template.jsonl to labels.jsonl and supply all eight fields for every
sample. The template itself must remain unchanged. Do not inspect conditions,
seeds, forecasts, model states, results, or another observer's work.

Replace this entire file with your own short account of how you reviewed all
records, handled identical texts, and resolved or preserved genuine ambiguity.
Do not infer hidden conditions or use keyword rules to supply semantic labels.
The validator checks that this template was replaced by non-heading prose; it
cannot certify the semantic adequacy of your notes or labels.
"""
_EXPORT_FILES = ("validation_blind.jsonl", "discovery_blind.jsonl", "rubric.md", "labels.template.jsonl")
_LOCKED_FILES = (
    "labels.jsonl", "rubric.md", "observations.md", "discovery_blind.jsonl",
    "validation_blind.jsonl", "labels.template.jsonl", "manifest.json",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp with a timezone")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field} timestamp") from exc
    if result.utcoffset() is None:
        raise ValueError(f"{field} must include its timezone")
    return result


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(values: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in values).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _parse_json(data: bytes, name: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON in {name}: {exc}") from exc


def _rows(data: bytes, name: str) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        row = _parse_json(line, f"{name}:{number}")
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object in {name}:{number}")
        rows.append(row)
    return rows


def _observer_dir(run_dir: str | Path, observer: str) -> Path:
    if not isinstance(observer, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", observer):
        raise ValueError("observer must start with a letter and contain only letters, digits, '_' or '-' (max 64 characters)")
    path = Path(run_dir) / "observers" / observer
    if path.is_symlink():
        raise ValueError("An observer directory must not be a symbolic link")
    return path


def _read_file(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError(f"A locked observation input must not be a symbolic link: {path.name}")
    return path.read_bytes()


def _rubric_bytes() -> bytes:
    data = resources.files("persona_observatory").joinpath("resources", "rubric.md").read_bytes()
    if _sha(data) != RUBRIC_SHA256:
        raise ValueError("Packaged rubric does not match the frozen Experiment 9 rubric")
    return data


def _source_rows(run_dir: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = _read_file(run_dir / "generations.jsonl")
    records = _rows(raw, "generations.jsonl")
    if not records:
        raise ValueError("Cannot export or validate an empty generation set")
    seen: set[str] = set()
    for row in records:
        for field in ("sample_id", "requested_answerer", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Generation {field} must be a nonempty string")
        if row["sample_id"] in seen:
            raise ValueError(f"Duplicate generation sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        if not isinstance(row.get("completion"), str):
            raise ValueError(f"Generation completion must be a string: {row['sample_id']}")
        if row.get("stop_reason") not in ("eos", "length"):
            raise ValueError(f"Unknown generation stop_reason: {row['sample_id']}")
    return raw, records


def _blind_rows(records: list[dict[str, Any]], observer: str, generation_sha: str) -> list[dict[str, Any]]:
    def order(row: dict[str, Any]) -> tuple[str, str]:
        value = f"{generation_sha}\0{observer}\0{row['sample_id']}".encode("utf-8")
        return _sha(value), row["sample_id"]

    return [{field: row[field] for field in BLIND_FIELDS} for row in sorted(records, key=order)]


def _label_templates(blind: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row["sample_id"] if field == "sample_id" else None for field in FIELDS} for row in blind]


def export_observer(
    run_dir: str | Path, observer: str, records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create one new blinded observer export; never modify an existing export.

    The caller must establish that the run completed successfully. The provided
    records must exactly equal the complete ``generations.jsonl`` contents.
    """
    run_dir = Path(run_dir)
    destination = _observer_dir(run_dir, observer)
    generation_raw, source = _source_rows(run_dir)
    if records != source:
        raise ValueError("Export records must match the complete generations.jsonl file exactly")
    generation_sha = _sha(generation_raw)
    blind = _blind_rows(source, observer, generation_sha)
    contents = {
        "validation_blind.jsonl": _jsonl_bytes(blind),
        "discovery_blind.jsonl": b"",
        "rubric.md": _rubric_bytes(),
        "labels.template.jsonl": _jsonl_bytes(_label_templates(blind)),
    }
    manifest = {
        "schema_version": EXPORT_SCHEMA,
        "observer": observer,
        "exported_utc": _now(),
        "generation_file": "generations.jsonl",
        "generations_sha256": generation_sha,
        "records": len(blind),
        "blind_fields": list(BLIND_FIELDS),
        "shuffle_algorithm": "sha256-ranked-v1",
        "input_hashes": {name: _sha(contents[name]) for name in _EXPORT_FILES},
        "notes_template_sha256": _sha(NOTES_TEMPLATE.encode("utf-8")),
    }
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        with (destination / name).open("xb") as handle:
            handle.write(data)
    with (destination / "observations.md").open("x", encoding="utf-8") as handle:
        handle.write(NOTES_TEMPLATE)
    # A changed generation file cannot silently produce a successful export.
    if _read_file(run_dir / "generations.jsonl") != generation_raw:
        raise ValueError("Generation file changed during observer export; preserve this failed export")
    with (destination / "manifest.json").open("xb") as handle:
        handle.write(_json_bytes(manifest))
    return {
        "status": "exported",
        "observer": observer,
        "records": len(blind),
        "directory": str(destination),
        "generations_sha256": generation_sha,
        "manifest_sha256": _sha(_json_bytes(manifest)),
        "labels_assigned": False,
    }


def _validate_label(label: dict[str, Any]) -> None:
    if not isinstance(label, dict) or set(label) != set(FIELDS):
        raise ValueError("Every label must have exactly the eight rubric fields")
    sid = label["sample_id"]
    if not isinstance(sid, str) or not sid:
        raise ValueError("Label sample_id must be a nonempty string")
    for field, allowed in ENUMS.items():
        if not isinstance(label[field], str) or label[field] not in allowed:
            raise ValueError(f"Invalid {field} for {sid}")
    owners = label["limit_owners"]
    if not isinstance(owners, list) or any(not isinstance(owner, str) or owner not in OWNERS for owner in owners):
        raise ValueError(f"Invalid limit_owners for {sid}")
    if len(owners) != len(set(owners)):
        raise ValueError(f"Duplicate limit_owners for {sid}")
    if type(label["other_answer"]) is not bool or type(label["character_description"]) is not bool:
        raise ValueError(f"The two Boolean rubric fields must be actual Booleans for {sid}")
    if not isinstance(label["note"], str) or not label["note"].strip():
        raise ValueError(f"A text-grounded note is required for {sid}")
    none_values = [label[field] == "none" for field in ENUMS]
    if len(set(none_values)) != 1:
        raise ValueError(f"Mode, judgment and stance none values must coincide for {sid}")
    if (label["answerer_judgment"] == "mixed") != (label["answerer_stance"] == "mixed"):
        raise ValueError(f"Judgment mixed iff stance mixed for {sid}")
    if label["answerer_judgment"] == "withhold" and label["answerer_stance"] != "tentative":
        raise ValueError(f"Withhold requires tentative stance for {sid}")


def _six_codes(label: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(label[field] for field in ENUMS) + (
        tuple(sorted(label["limit_owners"])), label["other_answer"], label["character_description"],
    )


def derived_outcome(label: dict[str, Any]) -> dict[str, Any]:
    """Derive the v9 five-category, D and strict-E outcomes from a valid label."""
    _validate_label(label)
    direct = label["answerer_answer_mode"] in ("direct", "both")
    return {
        "category": label["answerer_judgment"] if direct else "no_direct",
        "direct_event": direct,
        "strict_event": direct and "answerer_direct" in label["limit_owners"],
    }


def _validate_notes(raw: bytes) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("observations.md must be UTF-8 text") from exc
    if raw == NOTES_TEMPLATE.encode("utf-8") or NOTES_MARKER in text:
        raise ValueError("Replace the observations.md instructions with your own substantive review notes")
    without_comments = re.sub(r"<!--[\s\S]*?-->", "", text)
    prose = "\n".join(line.strip() for line in without_comments.splitlines()
                       if line.strip() and not line.lstrip().startswith("#"))
    if len(prose) < 20:
        raise ValueError("observations.md must contain substantive non-heading review prose (at least 20 characters)")


def validate_observer(run_dir: str | Path, observer: str, lock: bool = False) -> dict[str, Any]:
    """Validate complete labels and source identity, optionally writing one lock.

    An existing lock is always checked. Passing ``lock=True`` again refuses to
    overwrite it, even when all bytes still match.
    """
    run_dir = Path(run_dir)
    destination = _observer_dir(run_dir, observer)
    generation_raw, records = _source_rows(run_dir)
    generation_sha = _sha(generation_raw)
    manifest_raw = _read_file(destination / "manifest.json")
    manifest = _parse_json(manifest_raw, "manifest.json")
    required_manifest = {
        "schema_version", "observer", "exported_utc", "generation_file", "generations_sha256",
        "records", "blind_fields", "shuffle_algorithm", "input_hashes", "notes_template_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_manifest:
        raise ValueError("Observer export manifest fields differ from the required schema")
    if (manifest["schema_version"] != EXPORT_SCHEMA or manifest["observer"] != observer
            or manifest["generation_file"] != "generations.jsonl"
            or manifest["generations_sha256"] != generation_sha
            or type(manifest["records"]) is not int or manifest["records"] != len(records)
            or manifest["blind_fields"] != list(BLIND_FIELDS)
            or manifest["shuffle_algorithm"] != "sha256-ranked-v1"
            or manifest["notes_template_sha256"] != _sha(NOTES_TEMPLATE.encode("utf-8"))):
        raise ValueError("Observer manifest identity differs from the generation source or export specification")
    if not isinstance(manifest["input_hashes"], dict) or set(manifest["input_hashes"]) != set(_EXPORT_FILES):
        raise ValueError("Observer manifest must bind exactly the four exported inputs")
    exported_time = _timestamp(manifest["exported_utc"], "exported_utc")
    blind = _blind_rows(records, observer, generation_sha)
    expected_export = {
        "validation_blind.jsonl": _jsonl_bytes(blind),
        "discovery_blind.jsonl": b"",
        "rubric.md": _rubric_bytes(),
        "labels.template.jsonl": _jsonl_bytes(_label_templates(blind)),
    }
    raw_files = {name: manifest_raw if name == "manifest.json" else _read_file(destination / name)
                 for name in _LOCKED_FILES}
    for name, expected in expected_export.items():
        if raw_files[name] != expected or manifest["input_hashes"][name] != _sha(raw_files[name]):
            raise ValueError(f"Exported input changed or is inconsistent with its source: {name}")
    labels = _rows(raw_files["labels.jsonl"], "labels.jsonl")
    if len(labels) != len(blind):
        raise ValueError("Labels must cover every blind record exactly once")
    by_id = {row["sample_id"]: row for row in blind}
    labeled_ids: set[str] = set()
    same_triples: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    for label in labels:
        _validate_label(label)
        sid = label["sample_id"]
        if sid in labeled_ids or sid not in by_id:
            raise ValueError(f"Duplicate or unexpected label sample_id: {sid}")
        labeled_ids.add(sid)
        record = by_id[sid]
        triple = (record["requested_answerer"], record["question"], record["completion"])
        code = _six_codes(label)
        if triple in same_triples and same_triples[triple] != code:
            raise ValueError(f"Identical visible triples require identical six codes: {sid}")
        same_triples[triple] = code
    if labeled_ids != set(by_id):
        raise ValueError("Labels do not cover the exact exported sample identifiers")
    _validate_notes(raw_files["observations.md"])
    hashes = {
        "generations_sha256": generation_sha,
        "manifest_sha256": _sha(manifest_raw),
        "labels_sha256": _sha(raw_files["labels.jsonl"]),
        "rubric_sha256": _sha(raw_files["rubric.md"]),
        "observations_sha256": _sha(raw_files["observations.md"]),
        "discovery_blind_sha256": _sha(raw_files["discovery_blind.jsonl"]),
        "validation_blind_sha256": _sha(raw_files["validation_blind.jsonl"]),
        "labels_template_sha256": _sha(raw_files["labels.template.jsonl"]),
    }
    lock_path = destination / "observation_lock.json"
    already_locked = lock_path.exists()
    if already_locked:
        locked = _parse_json(_read_file(lock_path), "observation_lock.json")
        required_lock = {"schema_version", "observer", "locked_utc", "label_rows", "unique_triples", *hashes}
        if (not isinstance(locked, dict) or set(locked) != required_lock
                or locked["schema_version"] != LOCK_SCHEMA or locked["observer"] != observer
                or type(locked["label_rows"]) is not int or locked["label_rows"] != len(labels)
                or type(locked["unique_triples"]) is not int or locked["unique_triples"] != len(same_triples)
                or any(locked[key] != value for key, value in hashes.items())):
            raise ValueError("Locked observation bytes or identity changed; never revise a locked observer")
        if _timestamp(locked["locked_utc"], "locked_utc") < exported_time:
            raise ValueError("Observation lock timestamp precedes observer export")
        if lock:
            raise FileExistsError("Never overwrite an observation lock")
    elif lock:
        # Check all source bytes once more immediately before the exclusive write.
        if _read_file(run_dir / "generations.jsonl") != generation_raw or any(
            _read_file(destination / name) != raw for name, raw in raw_files.items()
        ):
            raise ValueError("Observation inputs changed during validation; lock was not written")
        locked_time = _now()
        if _timestamp(locked_time, "locked_utc") < exported_time:
            raise ValueError("Observation lock timestamp precedes observer export")
        record = {
            "schema_version": LOCK_SCHEMA,
            "observer": observer,
            "locked_utc": locked_time,
            "label_rows": len(labels),
            "unique_triples": len(same_triples),
            **hashes,
        }
        with lock_path.open("xb") as handle:
            handle.write(_json_bytes(record))
    return {
        "status": "passed",
        "observer": observer,
        "records": len(labels),
        "unique_triples": len(same_triples),
        "locked": bool(lock or already_locked),
        **hashes,
    }
