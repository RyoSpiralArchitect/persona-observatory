"""Independent failure cases for run-record consistency.

These tests construct locally self-consistent hashes around deliberately
contradictory records. A digest check alone must not turn those records into a
valid experiment. The fixture contains no neural-model output or research data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runs as run_fixtures
from persona_observatory import runs
from persona_observatory.storage import read_json, read_jsonl, sha_file
from persona_observatory.study import design_rows, load_study


def write_json(path, value):
    Path(path).write_text(json.dumps(value) + "\n", encoding="utf-8")


class RunRecordReviewTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the root test suite's explicit non-neural fixture setup;
        # no inherited tests are rerun and no production files are changed.
        self.fixture = run_fixtures.RunTests("test_streams_are_paired_and_plan_is_frozen")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.run

    def test_orphan_generations_are_not_reported_as_prepared(self):
        self.fixture.prepare()
        (self.run / "generations.jsonl").write_text('{}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            runs.verify_run(self.run)

    def test_orphan_failure_is_not_reported_as_prepared(self):
        self.fixture.prepare()
        write_json(self.run / "failure.json", {"error_type": "Interrupted"})
        with self.assertRaises(ValueError):
            runs.verify_run(self.run)

    def test_completion_schema_is_part_of_validity(self):
        self.fixture.prepare()
        runs.generate(self.run)
        completion = read_json(self.run / "completion.json")
        completion["schema_version"] = "unrecognized-completion-format"
        write_json(self.run / "completion.json", completion)
        with self.assertRaises(ValueError):
            runs.verify_run(self.run)

    def test_eos_stop_requires_the_effective_eos_token(self):
        self.fixture.prepare()
        runs.generate(self.run)
        records = read_jsonl(self.run / "generations.jsonl")
        records[0]["generated_token_ids"] = [7, 8]
        # Count and claimed reason still look superficially valid; digest is
        # updated so that only a stop-semantics check catches the contradiction.
        (self.run / "generations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        completion = read_json(self.run / "completion.json")
        completion["generations_sha256"] = sha_file(self.run / "generations.jsonl")
        write_json(self.run / "completion.json", completion)
        with self.assertRaises(ValueError):
            runs.verify_run(self.run)

    def test_timezone_free_chronology_is_rejected_even_if_ordered(self):
        self.fixture.prepare()
        runs.generate(self.run)
        lock = read_json(self.run / "run_lock.json")
        attempt = read_json(self.run / "attempt.json")
        completion = read_json(self.run / "completion.json")
        for record, field in ((lock, "prepared_utc"), (attempt, "started_utc"), (completion, "completed_utc")):
            record[field] = datetime.fromisoformat(record[field]).replace(tzinfo=None).isoformat()
        write_json(self.run / "run_lock.json", lock)
        attempt["run_lock_sha256"] = sha_file(self.run / "run_lock.json")
        write_json(self.run / "attempt.json", attempt)
        completion.update(run_lock_sha256=sha_file(self.run / "run_lock.json"),
                          attempt_sha256=sha_file(self.run / "attempt.json"))
        write_json(self.run / "completion.json", completion)
        with self.assertRaises(ValueError):
            runs.verify_run(self.run)

    def test_forecast_chronology_is_checked_before_a_generation_attempt(self):
        config, questions, settings, _ = load_study(self.fixture.root / "config.json")
        rows = design_rows(config, questions, settings)
        forecasts_path = self.fixture.root / "forecasts.jsonl"
        forecasts_path.write_text("".join(json.dumps({
            "sample_id": row["sample_id"], "observer": "observer_a", "predictor": "G",
            "category_probabilities": [0.2] * 5,
        }) + "\n" for row in rows), encoding="utf-8")
        self.fixture.prepare(forecasts_path=forecasts_path)
        lock = read_json(self.run / "run_lock.json")
        forecast_lock = read_json(self.run / "forecast_lock.json")
        forecast_lock["locked_utc"] = (datetime.fromisoformat(lock["prepared_utc"]) + timedelta(seconds=1)).isoformat()
        write_json(self.run / "forecast_lock.json", forecast_lock)
        lock["input_hashes"]["forecast_lock.json"] = sha_file(self.run / "forecast_lock.json")
        write_json(self.run / "run_lock.json", lock)
        with self.assertRaises(ValueError):
            runs.generate(self.run)
        self.assertFalse((self.run / "attempt.json").exists(),
                         "Invalid forecast chronology must fail before an inference attempt is recorded")

    def test_source_inventory_cannot_be_empty_on_read_only_verification(self):
        self.fixture.prepare()
        lock = read_json(self.run / "run_lock.json")
        lock["code_hashes"] = {}
        lock["input_hashes"] = {name: value for name, value in lock["input_hashes"].items()
                                if not name.startswith("source/")}
        write_json(self.run / "run_lock.json", lock)
        with self.assertRaises(ValueError):
            runs.verify_prepared(self.run, check_current_code=False)


if __name__ == "__main__":
    unittest.main()
