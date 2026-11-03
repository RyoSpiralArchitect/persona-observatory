"""Small, dependency-free primitives for immutable experiment files."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _bad_constant(value):
    raise ValueError("Non-finite JSON value: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_bad_constant)


def read_jsonl(path):
    return [json.loads(line, parse_constant=_bad_constant)
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_text_exclusive(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def write_json_exclusive(path, value):
    write_text_exclusive(path, json_text(value))


def write_jsonl_exclusive(path, rows):
    write_text_exclusive(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def safe_name(value, label="identifier"):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value),
            f"Invalid {label}: use a simple identifier without path separators")
    require(value not in {".", ".."}, "Invalid " + label)
    return value


def relative_path(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value,
            "Expected a portable relative path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and all(x not in {".", "..", ""} for x in value.split("/")),
            "Path must stay inside its root")
    return p


def checked_file(root, relative):
    root = Path(root).resolve()
    rel = relative_path(relative)
    path = root.joinpath(*rel.parts)
    require(path.is_file() and not path.is_symlink(), "Missing or linked file: " + relative)
    require(path.resolve().is_relative_to(root), "File escapes root: " + relative)
    return path


def check_hashes(root, hashes):
    require(isinstance(hashes, dict) and bool(hashes), "Empty file-hash inventory")
    for relative, digest in hashes.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid SHA-256")
        require(sha_file(checked_file(root, relative)) == digest, "Changed file: " + relative)
