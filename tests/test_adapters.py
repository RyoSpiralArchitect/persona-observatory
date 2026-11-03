"""Engine dispatch, environment locks and native sampling boundaries."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from persona_observatory import adapters, runs
from persona_observatory.backend import validate_runtime
from persona_observatory.mlx_backend import sample_token, validate_model_config
import test_runs as fixtures


class AdapterTests(unittest.TestCase):
    def test_default_remains_transformers_and_engine_device_pairs_are_explicit(self):
        self.assertEqual(validate_runtime({})["engine"], "transformers")
        self.assertEqual(validate_runtime({"engine": "mlx", "device": "metal"})["device"], "metal")
        for value in ({"engine": "mlxx"}, {"engine": "mlx", "device": "mps"},
                      {"engine": "transformers", "device": "metal"},
                      {"engine": "mlx", "batch_size": 2}, {"engine": "mlx", "cpu_threads": 2}):
            with self.subTest(value=value), self.assertRaises(ValueError): validate_runtime(value)

    def test_dispatch_selects_only_requested_adapter(self):
        with patch("persona_observatory.mlx_backend.inspect_backend", return_value={"engine": "mlx"}) as mlx, \
             patch("persona_observatory.backend.inspect_backend") as torch:
            self.assertEqual(adapters.inspect_backend("metal", "float32", "mlx"), {"engine": "mlx"})
            mlx.assert_called_once_with("metal", "float32")
            torch.assert_not_called()
        with patch("persona_observatory.mlx_backend.generate_records", side_effect=RuntimeError("native failure")), \
             patch("persona_observatory.backend.generate_records") as torch:
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                adapters.generate_records([], ".", {"engine": "mlx"}, lambda row: None)
            torch.assert_not_called()

    def test_custom_code_and_quantized_or_nested_models_are_rejected(self):
        validate_model_config({"model_type": "llama"})
        for value in ({"model_type": "other"}, {"model_type": "llama", "model_file": "custom.py"},
                      {"model_type": "llama", "auto_map": {}}, {"model_type": "llama", "quantization": {}},
                      {"model_type": "llama", "text_config": {}}):
            with self.subTest(value=value), self.assertRaises(ValueError): validate_model_config(value)

    def test_archived_smoke_remains_readable_without_rewriting_runtime(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(runs.verify_run(root / "validation/cli_smoke_20261103")["status"], "COMPLETE")


class AdapterEnvironmentTests(unittest.TestCase):
    setUp = fixtures.RunTests.setUp
    prepare = fixtures.RunTests.prepare
    def test_changed_engine_sampler_or_library_stops_before_attempt(self):
        self.prepare()
        for key in ("engine", "sampler", "mlx_version", "mlx_lm_version", "numpy_version", "mps_fallback"):
            with self.subTest(key=key), patch.object(runs, "inspect_backend", return_value={**fixtures.BACKEND, key: "changed"}):
                with self.assertRaisesRegex(ValueError, "changed after prepare"):
                    runs.generate(self.run)
                self.assertFalse((self.run / "attempt.json").exists())


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional")
class NumpySamplerTests(unittest.TestCase):
    def test_greedy_nucleus_and_nonfinite_contracts(self):
        import numpy as np
        rng = np.random.Generator(np.random.PCG64(123))
        self.assertEqual(sample_token([1, 3, 2], rng, 0, 1), 1)
        for _ in range(12): self.assertEqual(sample_token([1, 3, 2], rng, 1, 0.01), 1)
        with self.assertRaises(RuntimeError): sample_token([0, float("nan")], rng, 1, 1)
        with self.assertRaises(RuntimeError): sample_token([], rng, 1, 1)

    def test_per_sample_streams_do_not_depend_on_order_or_lose_high_seed_bits(self):
        import numpy as np
        def draw(seed):
            rng = np.random.Generator(np.random.PCG64(seed))
            return [sample_token([0, 0, 0, 0], rng, 1, 1) for _ in range(24)]
        seeds = [4, 4 + 2**32, 2**63 - 1]
        forward = {s: draw(s) for s in seeds}
        self.assertEqual(forward, {s: draw(s) for s in reversed(seeds)})
        self.assertNotEqual(forward[4], forward[4 + 2**32])


if __name__ == "__main__": unittest.main()
