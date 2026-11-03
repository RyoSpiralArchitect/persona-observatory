"""Local, portable autoregressive inference for new experiments.

This module is deliberately separate from the byte-frozen Experiment 9 runtime.
Weights are loaded only from an existing local directory.  Every sample owns a
CPU random-number generator, including when the forward pass uses MPS or CUDA.
This makes random streams independent of batch order; it does not make floating
point outputs, or sampled tokens, identical across devices or software versions.
"""
from __future__ import annotations

from collections import Counter
from importlib import metadata
import math
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Callable


DEFAULT_RUNTIME = {
    "engine": "transformers",
    "device": "auto",
    "dtype": "auto",
    "max_new_tokens": 96,
    "temperature": 0.8,
    "top_p": 0.95,
    "cpu_threads": 4,
    "batch_size": 1,
}
_DTYPES = {"float32", "float16", "bfloat16"}


def validate_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    """Return explicit runtime settings, rejecting ignored or invalid options."""
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a JSON object")
    unknown = set(runtime) - set(DEFAULT_RUNTIME)
    if unknown:
        raise ValueError("Unknown runtime options: " + ", ".join(sorted(unknown)))
    result = {**DEFAULT_RUNTIME, **runtime}
    device, dtype = result["device"], result["dtype"]
    if result["engine"] not in ("transformers", "mlx"):
        raise ValueError("engine must be transformers or mlx")
    pattern = r"auto|cpu|metal" if result["engine"] == "mlx" else r"auto|cpu|mps|cuda(?::[0-9]+)?"
    if not isinstance(device, str) or not re.fullmatch(pattern, device):
        raise ValueError("device is not supported by the requested engine")
    if not isinstance(dtype, str) or dtype not in _DTYPES | {"auto"}:
        raise ValueError("dtype must be auto, float32, float16, or bfloat16")
    for key in ("max_new_tokens", "cpu_threads", "batch_size"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("temperature", "top_p"):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        result[key] = float(value)
    if result["temperature"] < 0:
        raise ValueError("temperature must be nonnegative; zero selects greedy decoding")
    if not 0 < result["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if result["engine"] == "mlx" and (result["batch_size"] != 1 or result["cpu_threads"] != 4):
        raise ValueError("MLX adapter supports batch_size=1; cpu_threads must remain the unused default 4")
    return result


def _resolve_backend(torch: Any, device: str, dtype: str) -> tuple[str, str]:
    """Resolve explicitly; never silently move an unavailable request to CPU."""
    validate_runtime({"device": device, "dtype": dtype})
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if device == "auto":
        device = "cuda:0" if cuda_available else "mps" if mps_available else "cpu"
    elif device == "cuda":
        device = "cuda:0"
    if device.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError("CUDA was requested but this PyTorch runtime has no available CUDA device")
        if int(device.split(":", 1)[1]) >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device {device} is outside the available device count")
    if device == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but this PyTorch runtime has no available Apple MPS backend")
    return device, "float32" if dtype == "auto" else dtype


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def inspect_backend(device: str = "auto", dtype: str = "auto") -> dict[str, Any]:
    """Describe capabilities and test one tiny allocation, without loading a model.

    A missing dependency or unavailable device is returned as ``available=False``
    with a useful error; invalid option spelling raises ``ValueError``.
    """
    validate_runtime({"device": device, "dtype": dtype})
    result: dict[str, Any] = {
        "engine": "transformers",
        "available": False,
        "requested_device": device,
        "requested_dtype": dtype,
        "device": None,
        "dtype": None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "sampling_device": "cpu",
        "attention_implementation": "eager",
        "sampler": "torch-cpu-float32-multinomial",
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "cross_device_token_equality_guaranteed": False,
    }
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        mps = getattr(torch.backends, "mps", None)
        result["mps_available"] = bool(mps is not None and mps.is_available())
        result["cuda_runtime_version"] = getattr(torch.version, "cuda", None)
        resolved_device, resolved_dtype = _resolve_backend(torch, device, dtype)
        if resolved_device == "mps" and result["mps_fallback"] != "0":
            raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for an explicit MPS run")
        result.update(device=resolved_device, dtype=resolved_dtype)
        # A runtime capability check is more accurate than an OS-version guess.
        torch.empty(1, device=resolved_device, dtype=getattr(torch, resolved_dtype))
        if resolved_device.startswith("cuda:"):
            index = int(resolved_device.split(":", 1)[1])
            result["device_name"] = torch.cuda.get_device_name(index)
            result["device_capability"] = list(torch.cuda.get_device_capability(index))
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if resolved_dtype == "bfloat16" and bf16_supported is not None:
                with torch.cuda.device(index):
                    if not bf16_supported():
                        raise RuntimeError("This CUDA device does not report bfloat16 support")
        else:
            result["device_name"] = resolved_device
        result["available"] = True
    except (ImportError, OSError, RuntimeError, AssertionError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _validate_plan(plan: list[dict[str, Any]]) -> None:
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must contain at least one sample")
    sample_ids: set[str] = set()
    for row in plan:
        if not isinstance(row, dict):
            raise ValueError("Every plan row must be an object")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("Every sample_id must be a unique nonempty string")
        sample_ids.add(sample_id)
        ids = row.get("prompt_ids")
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError(f"Invalid prompt_ids for {sample_id}")
        seed = row.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError(f"seed for {sample_id} must be an integer in [0, 2**63)")


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        return set()
    values = value if isinstance(value, (list, tuple)) else [value]
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError("Model/tokenizer supplies invalid EOS token IDs")
    return set(values)


def _sample_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    runtime: dict[str, Any],
    device: str,
    torch: Any,
) -> list[dict[str, Any]]:
    eos = _eos_ids(model, tokenizer)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = min(eos) if eos else 0
    width = max(len(row["prompt_ids"]) for row in rows)
    input_ids = torch.tensor(
        [[pad] * (width - len(row["prompt_ids"])) + row["prompt_ids"] for row in rows],
        dtype=torch.long, device=device,
    )
    mask = torch.tensor(
        [[0] * (width - len(row["prompt_ids"])) + [1] * len(row["prompt_ids"]) for row in rows],
        dtype=torch.long, device=device,
    )
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    generators = [torch.Generator(device="cpu").manual_seed(row["seed"]) for row in rows]
    emitted: list[list[int]] = [[] for _ in rows]
    done = [False] * len(rows)
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=True)
        for step in range(runtime["max_new_tokens"]):
            # Both the nucleus calculation and random draw use float32 on CPU.
            logits = output.logits[:, -1, :].detach().to(device="cpu", dtype=torch.float32)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Non-finite model logits; stopping without a substituted output")
            if runtime["temperature"] == 0:
                greedy_ids = logits.argmax(dim=-1).tolist()
            else:
                sorted_logits, sorted_indices = (logits / runtime["temperature"]).sort(descending=True, dim=-1)
                probabilities = torch.softmax(sorted_logits, dim=-1)
                remove = probabilities.cumsum(-1) > runtime["top_p"]
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                probabilities = torch.softmax(sorted_logits.masked_fill(remove, -float("inf")), dim=-1)
            next_ids = []
            for index, generator in enumerate(generators):
                if done[index]:
                    next_ids.append(pad)
                    continue
                if runtime["temperature"] == 0:
                    token = int(greedy_ids[index])
                else:
                    chosen = torch.multinomial(probabilities[index], 1, generator=generator)
                    token = int(sorted_indices[index, chosen].item())
                next_ids.append(token)
                emitted[index].append(token)
                done[index] = token in eos
            if all(done) or step == runtime["max_new_tokens"] - 1:
                break
            next_positions = mask.sum(-1).unsqueeze(-1)
            mask = torch.cat([mask, torch.ones((len(rows), 1), dtype=mask.dtype, device=device)], dim=-1)
            output = model(
                input_ids=torch.tensor(next_ids, dtype=torch.long, device=device).unsqueeze(-1),
                attention_mask=mask,
                position_ids=next_positions,
                past_key_values=output.past_key_values,
                use_cache=True,
            )
    return [
        {
            **row,
            "completion": tokenizer.decode(ids, skip_special_tokens=True),
            "generated_token_ids": ids,
            "generated_tokens": len(ids),
            "stop_reason": "eos" if ids and ids[-1] in eos else "length",
        }
        for row, ids in zip(rows, emitted)
    ]


def generate_records(
    plan: list[dict[str, Any]],
    model_path: str | Path,
    runtime: dict[str, Any],
    on_record: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """Generate a frozen plan once and stream each completed record to a writer.

    The caller owns run locks, source/snapshot hashes, append-only persistence,
    failure recording, and observer blinding.  Exceptions propagate; this backend
    never retries a sample, changes precision/device, or overwrites a result.
    """
    runtime = validate_runtime(runtime)
    if runtime["engine"] != "transformers":
        raise ValueError("Use the adapter dispatcher for a non-Transformers engine")
    _validate_plan(plan)
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise ValueError(f"A local model directory is required: {model_path}")
    if not callable(on_record):
        raise ValueError("on_record must be callable")
    report = inspect_backend(runtime["device"], runtime["dtype"])
    if not report["available"]:
        raise RuntimeError(report.get("error", "Requested backend is unavailable"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    torch.set_num_threads(runtime["cpu_threads"])
    # Explicitly disable TF32 for these portable float32 profiles.
    torch.set_float32_matmul_precision("highest")
    if report["device"].startswith("cuda:"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=getattr(torch, report["dtype"]),
        attn_implementation="eager",
    ).to(report["device"]).eval()
    actual_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    if actual_dtype != report["dtype"]:
        raise RuntimeError(f"Loaded model dtype {actual_dtype} differs from requested {report['dtype']}")
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is None:
        context_limit = getattr(model.config, "n_positions", None)
    if type(context_limit) is int and any(
        len(row["prompt_ids"]) + runtime["max_new_tokens"] > context_limit for row in plan
    ):
        raise ValueError(f"A prompt plus generation budget exceeds the model context limit ({context_limit})")
    counts: Counter[str] = Counter()
    for start in range(0, len(plan), runtime["batch_size"]):
        records = _sample_batch(
            model, tokenizer, plan[start : start + runtime["batch_size"]], runtime, report["device"], torch,
        )
        for record in records:
            on_record(record)
            counts[record["stop_reason"]] += 1
    return {
        "status": "completed",
        "records": sum(counts.values()),
        "stop_reasons": dict(counts),
        "elapsed_seconds": time.perf_counter() - started,
        "backend": report,
        "runtime": runtime,
        "actual_dtype": actual_dtype,
        "model_class": type(model).__name__,
        "model_type": getattr(model.config, "model_type", None),
        "eos_token_ids": sorted(_eos_ids(model, tokenizer)),
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(_eos_ids(model, tokenizer), default=0),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
