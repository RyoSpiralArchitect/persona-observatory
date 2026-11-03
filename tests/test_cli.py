"""End-to-end CLI contract on explicitly synthetic records, without a neural model."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import math
import unittest

import test_runs
from persona_observatory.cli import main
from persona_observatory.storage import read_json, read_jsonl, write_jsonl_exclusive
from persona_observatory.study import load_study, design_rows


class CliLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_runs.RunTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def call(self, *args):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main([str(x) for x in args])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_fixed_external_forecasts_through_scoring_and_repeat_refusal(self):
        f = self.fixture
        f.config["purpose"] = "prospective"
        (f.root / "config.json").write_text(json.dumps(f.config))
        config, questions, settings, _ = load_study(f.root / "config.json")
        targets = design_rows(config, questions, settings)
        predictions = [{"sample_id": row["sample_id"], "observer": "synthetic_observer", "predictor": "uniform_fixture",
                        "category_probabilities": [0.2] * 5} for row in targets]
        path = f.root / "predictions.jsonl"
        write_jsonl_exclusive(path, predictions)
        status, out, err = self.call("prepare", "--config", f.root / "config.json", "--run-dir", f.run,
                                     "--model-path", f.model, "--forecasts", path)
        self.assertEqual(status, 0, err)
        self.assertTrue(json.loads(out)["forecasts_locked"])
        self.assertEqual(self.call("generate", "--run-dir", f.run)[0], 0)
        self.assertEqual(self.call("export-observer", "--run-dir", f.run, "--observer", "synthetic_observer")[0], 0)
        observer = f.run / "observers/synthetic_observer"
        labels = [{"sample_id": row["sample_id"], "answerer_answer_mode": "direct", "answerer_judgment": "support",
                   "answerer_stance": "decisive", "limit_owners": [], "other_answer": False,
                   "character_description": False, "note": "Synthetic coding fixture; not a semantic research observation."}
                  for row in read_jsonl(observer / "validation_blind.jsonl")]
        write_jsonl_exclusive(observer / "labels.jsonl", labels)
        (observer / "observations.md").write_text("All records and labels are explicitly synthetic fixtures for CLI integration testing.\n")
        status, _, err = self.call("validate-observer", "--run-dir", f.run, "--observer", "synthetic_observer", "--lock")
        self.assertEqual(status, 0, err)
        status, _, err = self.call("score", "--run-dir", f.run)
        self.assertEqual(status, 0, err)
        result = read_json(f.run / "scores.json")
        self.assertAlmostEqual(result["summary"][0]["category_brier"], 0.8)
        self.assertAlmostEqual(result["summary"][0]["category_nll"], math.log(5))
        original = (f.run / "scores.json").read_bytes()
        self.assertEqual(self.call("score", "--run-dir", f.run)[0], 2)
        self.assertEqual((f.run / "scores.json").read_bytes(), original)

    def test_observer_export_requires_completed_generation(self):
        self.fixture.prepare()
        status, _, _ = self.call("export-observer", "--run-dir", self.fixture.run, "--observer", "a")
        self.assertEqual(status, 2)
        self.assertFalse((self.fixture.run / "observers").exists())


if __name__ == "__main__":
    unittest.main()
