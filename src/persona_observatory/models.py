"""Public, revision-pinned model downloads and local byte verification."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid

from .storage import checked_file, read_json, relative_path, require, sha_file


def validate_manifest(manifest):
    require(isinstance(manifest, dict) and set(manifest) == {"model_id", "revision", "files"},
            "Model manifest must contain model_id, revision, files")
    require(isinstance(manifest["model_id"], str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", manifest["model_id"]),
            "Expected a Hugging Face owner/model identifier")
    require(isinstance(manifest["revision"], str) and re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]),
            "Model revision must be a full commit SHA, not main or a floating tag")
    require(isinstance(manifest["files"], list) and manifest["files"], "Model manifest has no files")
    seen = set()
    for row in manifest["files"]:
        require(isinstance(row, dict) and set(row) == {"name", "size_bytes", "sha256"}, "Invalid model file entry")
        relative_path(row["name"])
        require(row["name"].casefold() not in seen, "Duplicate model filename")
        seen.add(row["name"].casefold())
        require(type(row["size_bytes"]) is int and row["size_bytes"] > 0, "Invalid model file size")
        require(isinstance(row["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "Invalid model checksum")
    require("config.json" in seen, "Model config.json must be in the manifest")
    return manifest


def verify_model(path, manifest):
    manifest = validate_manifest(manifest)
    path = Path(path).resolve()
    require(path.is_dir(), "Local model directory does not exist: " + str(path))
    expected = {row["name"] for row in manifest["files"]}
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()
              and ".cache" not in p.relative_to(path).parts}
    require(actual == expected, "Model file inventory differs from its manifest; use a dedicated model directory")
    for row in manifest["files"]:
        p = checked_file(path, row["name"])
        require(p.stat().st_size == row["size_bytes"] and sha_file(p) == row["sha256"],
                "Model size/checksum mismatch: " + row["name"])
    return {"status": "VERIFIED", "model_id": manifest["model_id"], "revision": manifest["revision"],
            "files": len(expected)}


def fetch_model(destination, manifest):
    """Download only named public files; never read implicit HF credentials."""
    manifest = validate_manifest(manifest)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for row in manifest["files"]:
        path = destination.joinpath(*relative_path(row["name"]).parts)
        require(path.resolve().is_relative_to(destination), "Model destination escapes its root")
        if path.exists():
            require(not path.is_symlink() and path.is_file() and path.stat().st_size == row["size_bytes"]
                    and sha_file(path) == row["sha256"], "Existing model file differs; no overwrite: " + row["name"])
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # A unique temporary path preserves interrupted downloads instead of hiding a retry.
        temp = path.with_name(path.name + ".download-" + uuid.uuid4().hex + ".part")
        url = "https://huggingface.co/" + manifest["model_id"] + "/resolve/" + manifest["revision"] + "/" + quote(row["name"], safe="/")
        with urlopen(Request(url, headers={"User-Agent": "persona-observatory/0.1"}), timeout=120) as source, temp.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        require(temp.stat().st_size == row["size_bytes"] and sha_file(temp) == row["sha256"],
                "Downloaded model checksum mismatch; temporary file preserved: " + str(temp))
        # Never silently replace another completed file.
        require(not path.exists(), "Model destination appeared during download")
        temp.rename(path)
    return verify_model(destination, manifest)


def manifest_from_config(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    require(isinstance(config, dict) and isinstance(config.get("model_manifest"), str), "Config needs model_manifest")
    return validate_manifest(read_json(config_path.parent / config["model_manifest"]))
