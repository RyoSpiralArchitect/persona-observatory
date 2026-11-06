"""Adversarial checks of attribution-code integrity, not semantic auto-labeling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from persona_observatory.observations import (
    BLIND_FIELDS, FIELDS, NOTES_MARKER, RUBRIC_SHA256,
    derived_outcome, export_observer, validate_observer,
)


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def label(sid, **changes):
    return {
        "sample_id": sid, "answerer_answer_mode": "none", "answerer_judgment": "none",
        "answerer_stance": "none", "limit_owners": [], "other_answer": False,
        "character_description": False, "note": "スキーマ検証のために作成した合成ラベル。", **changes,
    }


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.records = [
            {"sample_id": "opaque-one", "requested_answerer": "Taylor", "question": "Does it work?",
             "completion": "A synthetic example.", "stop_reason": "eos", "private_condition": "cautious", "seed": 12},
            {"sample_id": "opaque-two", "requested_answerer": "Taylor", "question": "Does it work?",
             "completion": "Another synthetic example.", "stop_reason": "length", "private_condition": "confident", "seed": 13},
        ]
        jsonl(self.run / "generations.jsonl", self.records)
        self.observer = "observer_a"
        self.directory = self.run / "observers" / self.observer

    def export(self):
        return export_observer(self.run, self.observer, self.records)

    def complete(self, labels=None):
        jsonl(self.directory / "labels.jsonl", labels or [label(row["sample_id"]) for row in self.records])
        (self.directory / "observations.md").write_text(
            "# Review\n\nEvery complete synthetic test record was reviewed; identical texts were handled together.\n",
            encoding="utf-8",
        )

    def test_export_is_blinded_and_never_assigns_codes(self):
        result = self.export()
        self.assertFalse(result["labels_assigned"])
        blind = [json.loads(line) for line in (self.directory / "validation_blind.jsonl").read_text().splitlines()]
        self.assertTrue(all(set(row) == set(BLIND_FIELDS) for row in blind))
        self.assertNotIn("private_condition", (self.directory / "validation_blind.jsonl").read_text())
        self.assertEqual((self.directory / "discovery_blind.jsonl").read_bytes(), b"")
        self.assertEqual(hashlib.sha256((self.directory / "rubric.md").read_bytes()).hexdigest(), RUBRIC_SHA256)
        templates = [json.loads(line) for line in (self.directory / "labels.template.jsonl").read_text().splitlines()]
        self.assertTrue(all(set(row) == set(FIELDS) for row in templates))
        self.assertTrue(all(row[field] is None for row in templates for field in FIELDS if field != "sample_id"))
        self.assertFalse((self.directory / "labels.jsonl").exists())

    def test_export_requires_complete_source_and_safe_name(self):
        with self.assertRaises(ValueError):
            export_observer(self.run, self.observer, self.records[:1])
        for name in ("../escape", "a/b", ".hidden", "", "a" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                export_observer(self.run, name, self.records)
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_labels_must_cover_exact_identifiers_and_fields(self):
        self.export()
        for labels in ([label("opaque-one")], [label("opaque-one"), label("opaque-one")],
                       [label("opaque-one"), label("unseen")],
                       [label("opaque-one", extra="ignored?"), label("opaque-two")]):
            with self.subTest(labels=labels):
                self.complete(labels)
                with self.assertRaises(ValueError):
                    validate_observer(self.run, self.observer)

    def test_attribution_consistency_rules(self):
        self.export()
        invalid = [
            {"answerer_answer_mode": "direct"},
            {"answerer_answer_mode": "direct", "answerer_judgment": "mixed", "answerer_stance": "decisive"},
            {"answerer_answer_mode": "direct", "answerer_judgment": "withhold", "answerer_stance": "decisive"},
            {"limit_owners": ["other", "other"]}, {"limit_owners": ["named_character"]},
            {"other_answer": 0}, {"character_description": 1}, {"note": "  "},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.complete([label("opaque-one", **changes), label("opaque-two")])
                with self.assertRaises(ValueError):
                    validate_observer(self.run, self.observer)

    def test_identical_triples_require_identical_six_codes_but_notes_may_differ(self):
        self.records[1]["completion"] = self.records[0]["completion"]
        jsonl(self.run / "generations.jsonl", self.records)
        self.export()
        self.complete([label("opaque-one"), label("opaque-two", note="別の説明で同じ六コードを記録した。")])
        self.assertEqual(validate_observer(self.run, self.observer)["unique_triples"], 1)
        self.complete([label("opaque-one"), label("opaque-two", character_description=True)])
        with self.assertRaisesRegex(ValueError, "Identical visible triples"):
            validate_observer(self.run, self.observer)

    def test_instruction_notes_are_not_substantive_observations(self):
        self.export()
        jsonl(self.directory / "labels.jsonl", [label(row["sample_id"]) for row in self.records])
        with self.assertRaisesRegex(ValueError, "Replace"):
            validate_observer(self.run, self.observer)
        (self.directory / "observations.md").write_text("# Complete\n\nOK\n")
        with self.assertRaisesRegex(ValueError, "substantive"):
            validate_observer(self.run, self.observer)
        self.complete()
        self.assertEqual(validate_observer(self.run, self.observer)["status"], "passed")

    def test_altered_generation_or_export_is_detected(self):
        self.export()
        self.complete()
        files = [self.run / "generations.jsonl", self.directory / "validation_blind.jsonl",
                 self.directory / "rubric.md", self.directory / "discovery_blind.jsonl",
                 self.directory / "labels.template.jsonl"]
        for path in files:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    validate_observer(self.run, self.observer)
                path.write_bytes(original)

    def test_changing_export_and_manifest_together_cannot_replace_the_blind(self):
        self.export()
        self.complete()
        path = self.directory / "validation_blind.jsonl"
        blind = [json.loads(line) for line in path.read_text().splitlines()]
        blind[0]["completion"] = "Changed text"
        jsonl(path, blind)
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["input_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "inconsistent with its source"):
            validate_observer(self.run, self.observer)

    def test_lock_verifies_and_refuses_overwrite_or_label_revision(self):
        self.export()
        self.complete()
        result = validate_observer(self.run, self.observer, lock=True)
        self.assertTrue(result["locked"])
        self.assertTrue(validate_observer(self.run, self.observer)["locked"])
        with self.assertRaises(FileExistsError):
            validate_observer(self.run, self.observer, lock=True)
        labels_path = self.directory / "labels.jsonl"
        original = labels_path.read_bytes()
        labels_path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Locked observation"):
            validate_observer(self.run, self.observer)

    def test_note_revision_is_detected_after_lock(self):
        self.export()
        self.complete()
        validate_observer(self.run, self.observer, lock=True)
        path = self.directory / "observations.md"
        path.write_text(path.read_text() + "An extra post-lock sentence.\n")
        with self.assertRaisesRegex(ValueError, "Locked observation"):
            validate_observer(self.run, self.observer)

    def test_derived_outcomes_keep_reported_and_direct_distinct(self):
        direct = label("x", answerer_answer_mode="direct", answerer_judgment="withhold",
                       answerer_stance="tentative", limit_owners=["answerer_direct"])
        self.assertEqual(derived_outcome(direct), {"category": "withhold", "direct_event": True, "strict_event": True})
        reported = {**direct, "answerer_answer_mode": "reported", "limit_owners": ["answerer_reported"]}
        self.assertEqual(derived_outcome(reported), {"category": "no_direct", "direct_event": False, "strict_event": False})
        self.assertEqual(derived_outcome(label("x"))["category"], "no_direct")
        both = {**direct, "answerer_answer_mode": "both",
                "limit_owners": ["answerer_direct", "answerer_reported"]}
        self.assertEqual(derived_outcome(both), {"category": "withhold", "direct_event": True, "strict_event": True})


if __name__ == "__main__":
    unittest.main()
