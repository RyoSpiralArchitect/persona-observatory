"""The evidence checker must reject corrupt and non-portable archives."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_archive.py"
SPEC = importlib.util.spec_from_file_location("archive_checker", SCRIPT)
archive_checker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(archive_checker)


def make_archive(payload=None, *, manifest_edit=None, extra=None):
    payload = {"result.json": b'{"value": 7}\n'} if payload is None else payload
    entries = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
               for name, data in payload.items()]
    if manifest_edit is not None:
        manifest_edit(entries)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr("study/" + name, data)
        archive.writestr("study/FILE_MANIFEST.json", json.dumps(entries))
        if extra:
            archive.writestr(*extra)
    return stream.getvalue()


class ArchiveIntegrityTests(unittest.TestCase):
    def test_nested_manifest_is_checked_and_duplicate_payload_reused(self):
        child = make_archive()
        parent = make_archive({"provenance/child.zip": child, "snapshot/child.zip": child})
        result = archive_checker.verify_zip_bytes(parent)
        nested = result["embedded_archives"]
        self.assertEqual(result["manifest_entries"], 2)
        self.assertEqual(nested[0]["verification"]["manifest_entries"], 1)
        self.assertTrue(nested[1]["verification"]["already_verified"])

    def test_wrong_archive_hash_is_rejected(self):
        with self.assertRaisesRegex(archive_checker.ArchiveError, "Archive SHA-256 mismatch"):
            archive_checker.verify_zip_bytes(make_archive(), expected_sha256="0" * 64)

    def test_member_corruption_is_not_hidden_by_valid_zip_crc(self):
        raw = make_archive(manifest_edit=lambda rows: rows[0].update(sha256="0" * 64))
        with self.assertRaisesRegex(archive_checker.ArchiveError, "Member SHA-256 mismatch"):
            archive_checker.verify_zip_bytes(raw)

    def test_child_manifest_corruption_is_rejected(self):
        child = make_archive(manifest_edit=lambda rows: rows[0].update(sha256="0" * 64))
        with self.assertRaisesRegex(archive_checker.ArchiveError, "Member SHA-256 mismatch"):
            archive_checker.verify_zip_bytes(make_archive({"child.zip": child}))

    def test_unmanifested_file_is_rejected(self):
        with self.assertRaisesRegex(archive_checker.ArchiveError, "manifest membership"):
            archive_checker.verify_zip_bytes(make_archive(extra=("study/extra.txt", "extra")))

    def test_traversal_and_cross_platform_paths_are_rejected(self):
        for path in ("../escape", "/absolute", "dir/../../escape", "C:/escape", "dir\\escape"):
            with self.subTest(path=path), self.assertRaises(archive_checker.ArchiveError):
                archive_checker.verify_zip_bytes(make_archive({path: b"untrusted"}))

    def test_symlink_is_rejected_without_extraction(self):
        info = zipfile.ZipInfo("study/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(archive_checker.ArchiveError, "Non-regular ZIP member"):
            archive_checker.verify_zip_bytes(make_archive(extra=(info, "../../outside")))

    def test_duplicate_and_case_colliding_members_are_rejected(self):
        for name in ("study/result.json", "study/RESULT.JSON"):
            with self.subTest(name=name), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                raw = make_archive(extra=(name, b"conflicting value"))
                with self.assertRaisesRegex(archive_checker.ArchiveError, "(?:Duplicate|Case-colliding)"):
                    archive_checker.verify_zip_bytes(raw)

    def test_browsable_copy_must_match_the_archive(self):
        raw = make_archive()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "archives").mkdir()
            (root / "archives/study.zip").write_bytes(raw)
            (root / "copy.json").write_bytes(b'{"value": 8}\n')
            index = {"schema_version": "persona-observatory-archive-index-v1",
                     "archive": {"path": "archives/study.zip", "sha256": hashlib.sha256(raw).hexdigest(),
                                 "bytes": len(raw), "root": "study", "members": 2},
                     "browsable_copies": [{"path": "copy.json", "archive_member": "study/result.json",
                                          "bytes": 13, "sha256": hashlib.sha256(b'{"value": 7}\n').hexdigest()}]}
            (root / "archives/index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(archive_checker.ArchiveError, "Browsable copy differs"):
                archive_checker.verify_repository(root)


if __name__ == "__main__":
    unittest.main()
