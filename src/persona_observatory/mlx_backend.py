"""Native MLX forward adapter for local, unquantized Llama-family weights.

NumPy PCG64 sampling is per sample and independent of the MLX global RNG. This
is a different sampler from the Torch adapter, not a token-identity guarantee.
The original weight files and shared Hugging Face chat tokenizer are unchanged.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import platform
import time

from .backend import _package_version, _validate_plan, validate_runtime


def inspect_backend(device="auto", dtype="auto"):
    validate_runtime({"engine": "mlx", "device": device, "dtype": dtype})
    result = {
        "available": False, "engine": "mlx", "requested_device": device,
        "requested_dtype": dtype, "device": None, "dtype": None,
        "python": platform.python_version(), "platform": platform.platform(),
        "mlx_version": _package_version("mlx"), "mlx_lm_version": _package_version("mlx-lm"),
        "numpy_version": _package_version("numpy"), "transformers_version": _package_version("transformers"),
        "sampling_device": "cpu", "sampler": "numpy-pcg64-float32-nucleus-float64-draw",
        "attention_implementation": "mlx-lm-native", "cross_device_token_equality_guaranteed": False,
    }
    try:
        import mlx.core as mx
        import numpy  # noqa: F401 - dependency availability is part of preflight
        from mlx_lm.utils import load_model  # noqa: F401 - catch incompatible library APIs before an attempt
        if result["mlx_lm_version"] is None or result["transformers_version"] is None:
            raise ImportError("Install the mlx extra in a separate environment")
        available = bool(mx.metal.is_available())
        resolved = ("metal" if available else "cpu") if device == "auto" else device
        if resolved == "metal" and not available:
            raise RuntimeError("MLX Metal was requested but is unavailable; no CPU fallback")
        resolved_dtype = "float32" if dtype == "auto" else dtype
        with mx.stream(mx.gpu if resolved == "metal" else mx.cpu):
            mx.eval(mx.zeros((1,), dtype=getattr(mx, resolved_dtype)))
        result.update(available=True, device=resolved, device_name="mlx:" + resolved,
                      dtype=resolved_dtype, metal_available=available)
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def validate_model_config(config):
    # mlx-lm can import model_file Python code. This adapter never permits it.
    if not isinstance(config, dict) or config.get("model_type") != "llama":
        raise ValueError("MLX adapter currently supports unquantized model_type=llama only")
    forbidden = {"model_file", "auto_map", "quantization", "quantization_config", "text_config"}
    if forbidden & config.keys():
        raise ValueError("MLX adapter refuses custom code, nested architectures, and quantized weights")


def sample_token(logits, rng, temperature, top_p):
    import numpy as np
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 1 or not len(logits) or not np.isfinite(logits).all():
        raise RuntimeError("Non-finite or invalid MLX logits; no substituted output")
    if temperature == 0:
        return int(logits.argmax())
    scaled = logits / temperature
    if not np.isfinite(scaled).all():
        raise RuntimeError("Non-finite temperature-scaled logits")
    indices = np.argsort(-scaled, kind="stable")
    ranked = scaled[indices]
    probabilities = np.exp(ranked - ranked[0])
    probabilities /= probabilities.sum()
    remove = probabilities.cumsum() > top_p
    remove[1:] = remove[:-1].copy()
    remove[0] = False
    probabilities[remove] = 0
    # Generator.choice expects its double-precision probability sum to be one.
    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum()
    return int(indices[rng.choice(len(indices), p=probabilities)])


def generate_records(plan, model_path, runtime, on_record):
    runtime = validate_runtime(runtime)
    if runtime["engine"] != "mlx":
        raise ValueError("MLX adapter requires engine=mlx")
    _validate_plan(plan)
    model_path = Path(model_path)
    if not model_path.is_dir() or not callable(on_record):
        raise ValueError("A local model directory and callable record writer are required")
    config = json.loads((model_path / "config.json").read_text())
    validate_model_config(config)
    report = inspect_backend(runtime["device"], runtime["dtype"])
    if not report["available"]:
        raise RuntimeError(report.get("error", "MLX unavailable"))
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    generation_path = model_path / "generation_config.json"
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else config
    eos = generation_config.get("eos_token_id")
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    if any(type(x) is not int or x < 0 for x in eos):
        raise ValueError("Invalid effective EOS IDs")
    eos = set(eos)
    context_limit = config.get("max_position_embeddings")
    if type(context_limit) is int and any(len(r["prompt_ids"]) + runtime["max_new_tokens"] > context_limit for r in plan):
        raise ValueError("A prompt plus generation budget exceeds the model context limit")
    counts = Counter()
    started = time.perf_counter()
    with mx.stream(mx.gpu if report["device"] == "metal" else mx.cpu):
        model, _ = load_model(model_path, lazy=True, strict=True)
        model.set_dtype(getattr(mx, report["dtype"]))
        model.eval()
        mx.eval(model.parameters())
        dtypes = {str(a.dtype).removeprefix("mlx.core.") for _, a in tree_flatten(model.parameters())}
        if dtypes != {report["dtype"]}:
            raise RuntimeError(f"Loaded MLX parameter dtypes differ: {sorted(dtypes)}")
        for row in plan:
            rng = np.random.Generator(np.random.PCG64(row["seed"]))
            cache = make_prompt_cache(model)
            tokens = mx.array([row["prompt_ids"]])
            emitted = []
            for _ in range(runtime["max_new_tokens"]):
                logits = model(tokens, cache=cache)[:, -1, :].astype(mx.float32)
                mx.eval(logits)
                token = sample_token(np.array(logits)[0], rng, runtime["temperature"], runtime["top_p"])
                emitted.append(token)
                if token in eos:
                    break
                tokens = mx.array([[token]])
            reason = "eos" if emitted[-1] in eos else "length"
            on_record({**row, "completion": tokenizer.decode(emitted, skip_special_tokens=True),
                       "generated_token_ids": emitted, "generated_tokens": len(emitted), "stop_reason": reason})
            counts[reason] += 1
    return {"status": "completed", "records": sum(counts.values()), "stop_reasons": dict(counts),
            "elapsed_seconds": time.perf_counter() - started, "backend": report, "runtime": runtime,
            "actual_dtype": report["dtype"], "model_class": type(model).__name__, "model_type": config["model_type"],
            "eos_token_ids": sorted(eos),
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos, default=0),
            "cpu_threads": None, "batch_size": 1, "weight_conversion_or_quantization": False}
