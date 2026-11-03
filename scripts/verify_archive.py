#!/usr/bin/env python3
"""Read-only integrity checks for the preserved studies; standard library only.

This checks bytes, membership and manifests. It does not validate semantic labels,
rerun inference, or repeat the archived independent arithmetic audit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile


MAX_ARCHIVE_BYTES = 100_000_000
MAX_MEMBER_BYTES = 100_000_000
MAX_UNCOMPRESSED_BYTES = 100_000_000
MAX_MEMBERS = 10_000
MAX_DEPTH = 3
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ArchiveError(ValueError):
    """An archive or its provenance index failed a required integrity check."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_relative(name: str, *, directory: bool = False) -> str:
    """Validate portable relative names without extracting anything."""
    require(isinstance(name, str) and bool(name), "Empty or non-string path")
    require("\x00" not in name and "\\" not in name, "Non-portable path: " + repr(name))
    clean = name[:-1] if directory and name.endswith("/") else name
    require(not clean.startswith("/"), "Absolute path: " + repr(name))
    parts = clean.split("/")
    require(all(part not in ("", ".", "..") and ":" not in part for part in parts),
            "Unsafe path: " + repr(name))
    require(PurePosixPath(clean).as_posix() == clean, "Non-canonical path: " + repr(name))
    return clean


def checked_repo_path(root: Path, relative: str) -> Path:
    safe_relative(relative)
    root = root.resolve()
    candidate = root / relative
    # Symlinks are never necessary for the released archive or browsable copies.
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        require(not current.is_symlink(), "Symlink in evidence path: " + relative)
    require(candidate.resolve().is_relative_to(root), "Evidence path escapes repository")
    return candidate


def read_bounded(path: Path) -> bytes:
    require(path.is_file(), "Missing evidence file: " + str(path))
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, "Oversized evidence file: " + str(path))
    data = path.read_bytes()
    require(len(data) <= MAX_ARCHIVE_BYTES, "Evidence grew beyond size limit")
    return data


def verify_zip_bytes(raw: bytes, *, label: str = "archive", expected_sha256: str | None = None,
                     _depth: int = 0, _cache: dict | None = None) -> dict:
    """Check every member and recursively check unique embedded ZIPs.

    The input must have exactly one top-level directory and FILE_MANIFEST.json
    at that root. The manifest must enumerate every non-directory member except
    itself. Duplicate names, symlinks and unsafe paths are rejected before reads.
    """
    require(_depth <= MAX_DEPTH, "Nested archive depth limit exceeded")
    require(len(raw) <= MAX_ARCHIVE_BYTES, "Archive size limit exceeded")
    archive_sha = digest(raw)
    if expected_sha256 is not None:
        require(SHA256.fullmatch(expected_sha256) is not None, "Invalid expected SHA-256")
        require(archive_sha == expected_sha256, "Archive SHA-256 mismatch: " + label)
    cache = {} if _cache is None else _cache
    if archive_sha in cache:
        return {"sha256": archive_sha, "already_verified": True}
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except (zipfile.BadZipFile, OSError) as exc:
        raise ArchiveError("Invalid ZIP: " + label) from exc

    with archive:
        infos = archive.infolist()
        require(0 < len(infos) <= MAX_MEMBERS, "Archive member count outside limits")
        require(sum(info.file_size for info in infos) <= MAX_UNCOMPRESSED_BYTES,
                "Uncompressed archive size limit exceeded")
        names: set[str] = set()
        portable_names: set[str] = set()
        roots: set[str] = set()
        files: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            name = safe_relative(info.filename, directory=info.is_dir())
            # orig_filename retains NUL suffixes that ZipInfo.filename truncates.
            require(info.orig_filename == info.filename, "Non-canonical ZIP entry name")
            require(name not in names, "Duplicate ZIP member: " + name)
            require(name.casefold() not in portable_names, "Case-colliding ZIP member: " + name)
            names.add(name)
            portable_names.add(name.casefold())
            roots.add(name.split("/")[0])
            require(info.file_size <= MAX_MEMBER_BYTES, "Oversized member: " + name)
            require(not (info.flag_bits & 1), "Encrypted member is unsupported: " + name)
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            expected_kind = stat.S_IFDIR if info.is_dir() else stat.S_IFREG
            require(kind in (0, expected_kind), "Non-regular ZIP member: " + name)
            if not info.is_dir():
                files[name] = info
        require(len(roots) == 1, "Archive must have one top-level directory")
        root = next(iter(roots))
        require(all(name.startswith(root + "/") for name in files), "File at archive root")
        manifest_name = root + "/FILE_MANIFEST.json"
        require(manifest_name in files, "Missing root FILE_MANIFEST.json")
        try:
            manifest = json.loads(archive.read(files[manifest_name]))
        except (json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            raise ArchiveError("Unreadable FILE_MANIFEST.json") from exc
        require(isinstance(manifest, list), "Manifest must be a list")
        expected: dict[str, dict] = {}
        for entry in manifest:
            require(isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"},
                    "Malformed manifest entry")
            relative = safe_relative(entry["path"])
            name = root + "/" + relative
            require(name != manifest_name, "Manifest must not enumerate itself")
            require(name not in expected, "Duplicate manifest entry: " + relative)
            require(type(entry["bytes"]) is int and 0 <= entry["bytes"] <= MAX_MEMBER_BYTES,
                    "Invalid manifest byte count: " + relative)
            require(isinstance(entry["sha256"], str) and SHA256.fullmatch(entry["sha256"]) is not None,
                    "Invalid manifest digest: " + relative)
            expected[name] = entry
        require(set(files) == set(expected) | {manifest_name},
                "ZIP members differ from manifest membership")
        nested = []
        total_bytes = 0
        for name, entry in expected.items():
            info = files[name]
            require(info.file_size == entry["bytes"], "ZIP size differs from manifest: " + name)
            try:
                data = archive.read(info)
            except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                raise ArchiveError("Unreadable member: " + name) from exc
            require(len(data) == entry["bytes"], "Extracted byte count mismatch: " + name)
            require(digest(data) == entry["sha256"], "Member SHA-256 mismatch: " + name)
            total_bytes += len(data)
            if name.endswith(".zip"):
                nested.append({"member": name, "verification": verify_zip_bytes(
                    data, label=label + "!" + name, expected_sha256=entry["sha256"],
                    _depth=_depth + 1, _cache=cache)})
        result = {"status": "VERIFIED", "sha256": archive_sha, "bytes": len(raw),
                  "root": root, "members": len(infos), "manifest_entries": len(manifest),
                  "verified_payload_bytes": total_bytes, "embedded_archives": nested}
        cache[archive_sha] = result
        return result


def verify_repository(root: Path, *, index_path: Path | None = None,
                      archive_override: Path | None = None) -> dict:
    """Verify the released ZIP and all byte-identical browsable research copies."""
    root = root.resolve()
    index_path = root / "archives/index.json" if index_path is None else index_path
    try:
        index = json.loads(read_bounded(index_path))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArchiveError("Unreadable archive index") from exc
    require(index.get("schema_version") == "persona-observatory-archive-index-v1", "Unknown index schema")
    spec = index["archive"]
    source = checked_repo_path(root, spec["path"]) if archive_override is None else archive_override
    raw = read_bounded(source)
    require(len(raw) == spec["bytes"], "Archive byte count differs from index")
    result = verify_zip_bytes(raw, label=spec["path"], expected_sha256=spec["sha256"])
    require(result["root"] == spec["root"] and result["members"] == spec["members"],
            "Archive dimensions differ from index")
    copies = index["browsable_copies"]
    require(isinstance(copies, list), "Invalid browsable-copy index")
    seen: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for entry in index.get("embedded_archives", []):
            member = safe_relative(entry["archive_member"])
            nested = archive.read(member)
            require(len(nested) == entry["bytes"] and digest(nested) == entry["sha256"],
                    "Embedded archive differs from provenance index")
            with zipfile.ZipFile(io.BytesIO(nested)) as child:
                require(len(child.infolist()) == entry["members"], "Embedded member count differs")
                require(entry["root"] + "/FILE_MANIFEST.json" in child.namelist(), "Embedded root differs")
        for entry in copies:
            relative = safe_relative(entry["path"])
            require(relative not in seen, "Duplicate browsable-copy index entry")
            seen.add(relative)
            member = safe_relative(entry["archive_member"])
            copied = read_bounded(checked_repo_path(root, relative))
            require(len(copied) == entry["bytes"] and digest(copied) == entry["sha256"],
                    "Browsable copy differs from index: " + relative)
            require(copied == archive.read(member), "Browsable copy differs from archive: " + relative)
    return {"status": "VERIFIED", "archive": result, "browsable_copies": len(copies),
            "scope": "Byte and membership integrity only; no model download, inference, or semantic reannotation."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root containing archives/index.json")
    parser.add_argument("--index", type=Path, help="Explicit provenance index; defaults to archives/index.json")
    parser.add_argument("--archive", type=Path, help="Alternate location of the same exact archived ZIP")
    args = parser.parse_args(argv)
    try:
        result = verify_repository(args.repo, index_path=args.index, archive_override=args.archive)
    except (ArchiveError, OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
