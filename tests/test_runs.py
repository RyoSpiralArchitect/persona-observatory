"""Exercise file-boundary behavior with an explicit non-neural backend fixture."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_observatory import runs
from persona_observatory.storage import read_json, sha_file
from persona_observatory.study import design_rows, load_study


class TokenizerFixture:
    chat_template = "explicit-test-fixture-not-a-model"
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 3, 4] if tokenize else "fixture prompt"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 3, 4]}


BACKEND = {"available": True, "device": "cpu", "dtype": "float32", "python": "fixture",
           "torch_version": "fixture", "transformers_version": "fixture", "cuda_runtime_version": None}


def fake_generation(plan, model_path, runtime, on_record):
    for row in plan:
        on_record({**row, "completion": "A fixture reply.", "generated_token_ids": [7, 2],
                   "generated_tokens": 2, "stop_reason": "eos"})
    return {"fixture": True, "eos_token_ids": [2], "pad_token_id": 0,
            "backend": BACKEND, "runtime": runtime, "actual_dtype": "float32"}


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{}\n')
        self.manifest = {"model_id": "fixture/model", "revision": "a" * 40,
                         "files": [{"name": "config.json", "size_bytes": 3, "sha256": sha_file(self.model / "config.json")}]}
        self.questions = [{"question_id": "q1", "topic": "fixture", "evidence_stratum": "limited", "question": "Does it work?"}]
        self.settings = [{"setting_id": "neutral", "requested_answerer": "Morgan", "policy_text": "Assess it."},
                         {"setting_id": "cautious", "requested_answerer": "Morgan", "policy_text": "Consider limits."}]
        self.config = {"schema_version": "persona-observatory-study-v1", "experiment_id": "fixture", "purpose": "smoke_replay",
                       "seed_namespace": "fixture-seeds", "replicates": 2, "question_bank": "questions.json",
                       "settings": "settings.json", "model_manifest": "manifest.json", "runtime": {"device": "cpu", "max_new_tokens": 4}}
        for path, value in [("config.json", self.config), ("questions.json", self.questions),
                            ("settings.json", self.settings), ("manifest.json", self.manifest)]:
            (self.root / path).write_text(json.dumps(value), encoding="utf-8")
        auto = types.SimpleNamespace(from_pretrained=lambda *a, **kw: TokenizerFixture())
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoTokenizer=auto)}).start()
        patch.object(runs, "inspect_backend", return_value=BACKEND).start()
        patch.object(runs, "generate_records", side_effect=fake_generation).start()
        self.run = self.root / "runs" / "fixture"

    def prepare(self, **kwargs):
        return runs.prepare(self.root / "config.json", self.run, self.model, **kwargs)

    def test_streams_are_paired_and_plan_is_frozen(self):
        result = self.prepare()
        self.assertEqual(result["samples"], 4)
        _, _, _, rows, _ = runs.verify_prepared(self.run)
        self.assertEqual(rows[0]["seed"], rows[2]["seed"])
        self.assertEqual(rows[1]["seed"], rows[3]["seed"])
        self.assertNotEqual(rows[0]["seed"], rows[1]["seed"])
        self.assertEqual(runs.verify_run(self.run)["status"], "PREPARED")
        with self.assertRaises(FileExistsError): self.prepare()

    def test_single_attempt_and_tampered_outputs(self):
        self.prepare()
        result = runs.generate(self.run)
        self.assertEqual(result["status"], "COMPLETE")
        with self.assertRaises(ValueError): runs.generate(self.run)
        with (self.run / "generations.jsonl").open("a") as f: f.write('\n')
        with self.assertRaises(ValueError): runs.verify_run(self.run)

    def test_config_change_stops_before_attempt(self):
        self.prepare()
        with (self.run / "config.json").open("a") as f: f.write('\n')
        with self.assertRaises(ValueError): runs.generate(self.run)
        self.assertFalse((self.run / "attempt.json").exists())

    def test_model_change_stops_before_attempt(self):
        self.prepare()
        (self.model / "config.json").write_text('{"altered":true}')
        with self.assertRaises(ValueError): runs.generate(self.run)
        self.assertFalse((self.run / "attempt.json").exists())

    def test_current_source_change_stops_before_attempt(self):
        self.prepare()
        with patch.object(runs, "_code_files", return_value={}):
            with self.assertRaises(ValueError): runs.generate(self.run)
        self.assertFalse((self.run / "attempt.json").exists())

    def test_failure_is_preserved_and_not_resumed(self):
        self.prepare()
        def fail(plan, model_path, runtime, on_record):
            fake_generation(plan[:1], model_path, runtime, on_record)
            raise RuntimeError("injected interruption")
        with patch.object(runs, "generate_records", side_effect=fail):
            with self.assertRaises(RuntimeError): runs.generate(self.run)
        self.assertEqual(runs.verify_run(self.run)["status"], "FAILED")
        self.assertEqual(read_json(self.run / "failure.json")["records_written"], 1)
        with self.assertRaises(ValueError): runs.generate(self.run)

    def test_prospective_requires_forecasts(self):
        self.config["purpose"] = "prospective"
        (self.root / "config.json").write_text(json.dumps(self.config))
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.run.exists())

    def test_late_forecast_is_rejected(self):
        self.prepare()
        (self.run / "forecasts.jsonl").write_text('{}\n')
        with self.assertRaises(ValueError): runs.generate(self.run)

    def test_explicit_unknown_config_or_duplicate_question_rejected(self):
        self.config["ignored_typo"] = True
        (self.root / "config.json").write_text(json.dumps(self.config))
        with self.assertRaises(ValueError): load_study(self.root / "config.json")
        del self.config["ignored_typo"]
        (self.root / "config.json").write_text(json.dumps(self.config))
        (self.root / "questions.json").write_text(json.dumps(self.questions * 2))
        with self.assertRaises(ValueError): load_study(self.root / "config.json")


if __name__ == "__main__":
    unittest.main()
