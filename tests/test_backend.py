"""Portable backend contracts; tensor tests run when the torch extra is installed."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest

from persona_observatory.backend import _resolve_backend, _sample_batch, _validate_plan, validate_runtime


def fake_capabilities(cuda=False, mps=False, count=1):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: count),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


class BackendConfigurationTests(unittest.TestCase):
    def test_auto_and_explicit_devices(self):
        self.assertEqual(_resolve_backend(fake_capabilities(), "auto", "auto"), ("cpu", "float32"))
        self.assertEqual(_resolve_backend(fake_capabilities(mps=True), "auto", "auto"), ("mps", "float32"))
        self.assertEqual(_resolve_backend(fake_capabilities(cuda=True, mps=True), "auto", "bfloat16"), ("cuda:0", "bfloat16"))
        self.assertEqual(_resolve_backend(fake_capabilities(cuda=True, count=2), "cuda:1", "float16"), ("cuda:1", "float16"))

    def test_explicit_device_never_silently_falls_back(self):
        for device in ("cuda", "cuda:0", "mps"):
            with self.subTest(device=device), self.assertRaises(RuntimeError):
                _resolve_backend(fake_capabilities(), device, "auto")
        with self.assertRaises(RuntimeError):
            _resolve_backend(fake_capabilities(cuda=True, count=1), "cuda:1", "auto")

    def test_rejects_ignored_settings_and_invalid_numbers(self):
        for settings in ({"batc_size": 2}, {"device": "cdua"}, {"dtype": "int8"},
                         {"batch_size": True}, {"max_new_tokens": 0}, {"temperature": -1},
                         {"temperature": float("nan")}, {"top_p": 0}, {"top_p": 1.1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_runtime(settings)
        self.assertEqual(validate_runtime({"temperature": 0})["temperature"], 0)

    def test_frozen_plan_requirements(self):
        row = {"sample_id": "sample-1", "prompt_ids": [0, 3], "seed": 2**63 - 1}
        _validate_plan([row])
        for plan in ([], [row, row], [{**row, "seed": 2**63}], [{**row, "seed": True}],
                     [{**row, "prompt_ids": []}], [{**row, "prompt_ids": [True]}]):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                _validate_plan(plan)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch extra is not installed")
class BackendTensorTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.runtime = validate_runtime({"max_new_tokens": 6, "temperature": 0.8, "top_p": 0.95})
        self.tokenizer = SimpleNamespace(
            eos_token_id=4, pad_token_id=0,
            decode=lambda ids, skip_special_tokens: " ".join(str(x) for x in ids if x not in (0, 4)),
        )

    def model(self, mode="random"):
        torch = self.torch

        class ToyModel:
            generation_config = SimpleNamespace(eos_token_id=[4])

            def __init__(self):
                self.calls = []

            def __call__(self, input_ids, attention_mask, position_ids, use_cache, past_key_values=None):
                self.calls.append({"input_ids": input_ids.clone(), "mask": attention_mask.clone(), "pos": position_ids.clone()})
                logits = torch.full((len(input_ids), 1, 5), -100.0)
                logits[:, :, 1:4] = 0.0
                if mode == "eos":
                    logits[:, :, 4] = 100.0
                elif mode == "mixed":
                    stopping = input_ids[:, -1] == 1 if past_key_values is None else past_key_values
                    logits[stopping, :, 4] = 100.0
                    logits[~stopping, :, 2] = 100.0
                    return SimpleNamespace(logits=logits, past_key_values=stopping)
                return SimpleNamespace(logits=logits, past_key_values=None)

        return ToyModel()

    def sample(self, model, rows, runtime=None):
        return _sample_batch(model, self.tokenizer, rows, runtime or self.runtime, "cpu", self.torch)

    def test_eos_is_retained_and_distinguished_from_length(self):
        rows = [{"sample_id": "one", "prompt_ids": [2], "seed": 7}]
        eos_result = self.sample(self.model("eos"), rows)[0]
        self.assertEqual(eos_result["generated_token_ids"], [4])
        self.assertEqual(eos_result["generated_tokens"], 1)
        self.assertEqual(eos_result["stop_reason"], "eos")
        self.assertEqual(eos_result["completion"], "")
        capped = self.sample(self.model(), rows)[0]
        self.assertEqual(capped["generated_tokens"], 6)
        self.assertEqual(capped["stop_reason"], "length")

    def test_cpu_rng_is_per_sample_and_independent_of_batch_order(self):
        rows = [
            {"sample_id": "a", "prompt_ids": [1, 2], "seed": 9234},
            {"sample_id": "b", "prompt_ids": [3], "seed": 702},
            {"sample_id": "paired", "prompt_ids": [2, 2, 3], "seed": 9234},
        ]
        batched = {r["sample_id"]: r["generated_token_ids"] for r in self.sample(self.model(), rows)}
        self.torch.manual_seed(129993)
        reversed_batch = {r["sample_id"]: r["generated_token_ids"] for r in self.sample(self.model(), list(reversed(rows)))}
        single = {row["sample_id"]: self.sample(self.model(), [row])[0]["generated_token_ids"] for row in rows}
        self.assertEqual(batched, reversed_batch)
        self.assertEqual(batched, single)
        self.assertEqual(batched["a"], batched["paired"])

    def test_left_padding_positions_and_finished_rows(self):
        rows = [{"sample_id": "stop", "prompt_ids": [2, 1], "seed": 1},
                {"sample_id": "continue", "prompt_ids": [3], "seed": 2}]
        model = self.model("mixed")
        result = self.sample(model, rows)
        self.assertEqual(model.calls[0]["input_ids"].tolist(), [[2, 1], [0, 3]])
        self.assertEqual(model.calls[0]["mask"].tolist(), [[1, 1], [0, 1]])
        self.assertEqual(model.calls[0]["pos"].tolist(), [[0, 1], [0, 0]])
        self.assertEqual(model.calls[1]["pos"].tolist(), [[2], [1]])
        self.assertEqual(result[0]["generated_token_ids"], [4])
        self.assertEqual(result[1]["generated_token_ids"], [2] * 6)

    def test_greedy_and_one_token_eos_cap(self):
        rows = [{"sample_id": "one", "prompt_ids": [2], "seed": 1}]
        runtime = validate_runtime({"max_new_tokens": 1, "temperature": 0})
        self.assertEqual(self.sample(self.model(), rows, runtime)[0]["generated_token_ids"], [1])
        result = self.sample(self.model("eos"), rows, runtime)[0]
        self.assertEqual(result["stop_reason"], "eos")


if __name__ == "__main__":
    unittest.main()
